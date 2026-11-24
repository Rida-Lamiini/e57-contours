#!/usr/bin/env python
"""
e57_to_contours.py  -  E57 point cloud  ->  ground DEM  ->  contour lines (DXF for AutoCAD)

Designed for exterior / topographic scans (terrain + buildings + vegetation + cars).

Pipeline
  1. Stream the E57 in chunks (never loads the whole cloud) and bin it into a fine
     "lowest point" grid (default 0.10 m).  The grid is cached (.npz) so re-runs with
     other settings take seconds instead of minutes.
  2. Remove isolated low noise, coarsen to DEM resolution (default 0.25 m).
  3. Ground filter: Progressive Morphological Filter (Zhang 2003) on the grid ->
     removes buildings, trees, cars, people, walls, etc.
  4. Interpolate the bare-earth surface under removed objects, light smoothing.
  5. Mask building footprints (optional) and areas with no data.
  6. Extract contours (contourpy), simplify + smooth, write DXF (layers, elevations,
     index labels) + PNG preview + ground DEM as GeoTIFF-like .asc (optional).

Usage
  python e57_to_contours.py scan.e57 -o contours.dxf --interval 0.25 --index-every 4
"""
import argparse
import math
import os
import sys
import time

import numpy as np
from scipy import ndimage as ndi
from scipy.interpolate import LinearNDInterpolator


# --------------------------------------------------------------------------- #
# 1. Reading + fine lowest-point grid
# --------------------------------------------------------------------------- #
def build_min_grid(e57_path, cell, chunk=4_000_000, scan_index=0, clip=None):
    import pye57

    f = pye57.E57(e57_path)
    h = f.get_header(scan_index)
    b = h.cartesianBounds
    xmin, xmax = b["xMinimum"].value(), b["xMaximum"].value()
    ymin, ymax = b["yMinimum"].value(), b["yMaximum"].value()
    if clip:
        xmin, ymin, xmax, ymax = clip
    pad = 1.0
    x0, y0 = math.floor(xmin - pad), math.floor(ymin - pad)
    W = int(math.ceil((xmax + pad - x0) / cell)) + 1
    H = int(math.ceil((ymax + pad - y0) / cell)) + 1
    print(f"[read] {h.point_count:,} points, grid {W} x {H} cells @ {cell} m "
          f"({W * H * 4 / 1e6:.0f} MB)")

    # per fine cell: the 3 lowest heights (robust low surface), the highest (verticality), count
    low = np.full((3, H * W), np.inf, dtype=np.float32)
    zmax = np.full(H * W, -np.inf, dtype=np.float32)
    cnt = np.zeros(H * W, dtype=np.uint16)

    data, bufs = f.make_buffers(["cartesianX", "cartesianY", "cartesianZ"], chunk)
    reader = h.points.reader(bufs)
    # pose (identity for most scanners' registered exports, but honour it anyway)
    R = None
    T = np.asarray(h.translation, dtype=np.float64)
    q = np.asarray(h.rotation, dtype=np.float64)
    if not (np.allclose(q, [1, 0, 0, 0]) and np.allclose(T, 0)):
        w, x, y, z = q
        R = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])

    total, t0 = 0, time.time()
    while True:
        n = reader.read()
        if n == 0:
            break
        X = data["cartesianX"][:n]
        Y = data["cartesianY"][:n]
        Z = data["cartesianZ"][:n]
        if R is not None:
            P = np.stack([X, Y, Z], 1) @ R.T + T
            X, Y, Z = P[:, 0], P[:, 1], P[:, 2]
        ix = ((X - x0) / cell).astype(np.int64)
        iy = ((Y - y0) / cell).astype(np.int64)
        ok = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H) & np.isfinite(Z)
        idx = iy[ok] * W + ix[ok]
        z = Z[ok].astype(np.float32)
        if len(idx):
            order = np.lexsort((z, idx))            # by cell, then by height
            si, sz = idx[order], z[order]
            first = np.empty(len(si), bool)
            first[0] = True
            np.not_equal(si[1:], si[:-1], out=first[1:])
            starts = np.flatnonzero(first)
            gid = np.cumsum(first) - 1
            rank = np.arange(len(si)) - starts[gid]
            U = si[first]
            cand = np.full((len(U), 6), np.inf, dtype=np.float32)
            for r in range(3):
                m = rank == r
                cand[gid[m], r] = sz[m]
            cand[:, 3:] = low[:, U].T
            cand.sort(axis=1)
            low[:, U] = cand[:, :3].T
            ends = np.r_[starts[1:], len(si)] - 1
            zmax[U] = np.maximum(zmax[U], sz[ends])
            cnt[U] = np.minimum(cnt[U].astype(np.int64) + (ends - starts + 1), 65535).astype(np.uint16)
        total += n
        el = time.time() - t0
        print(f"\r[read] {total / 1e6:6.1f} M pts  {total / el / 1e6:4.2f} M/s", end="", flush=True)
    reader.close()
    f.close()
    print()
    low[~np.isfinite(low)] = np.nan
    zmax[~np.isfinite(zmax)] = np.nan
    return (low.reshape(3, H, W), zmax.reshape(H, W), cnt.reshape(H, W)), (x0, y0, cell)


def surface_from_layers(layers, surface="robust", vert_thr=0.25):
    """Fine-grid surface + 'vertical' mask from the per-cell statistics."""
    low, zmax, cnt = layers
    z1, z2 = low[0], low[1]
    if surface == "min":
        z = z1.copy()
    else:   # median of the 3 lowest points: rejects single low-noise points
        z = np.where(cnt >= 3, z2, z1)
    z[cnt < 1] = np.nan
    with np.errstate(invalid="ignore"):
        vertical = (cnt >= 3) & ((zmax - z1) > vert_thr)    # walls / foliage inside one 10 cm cell
    return z, vertical


def coarsen_any(mask, f):
    H, W = mask.shape
    H2, W2 = (H + f - 1) // f, (W + f - 1) // f
    pad = np.zeros((H2 * f, W2 * f), bool)
    pad[:H, :W] = mask
    return pad.reshape(H2, f, W2, f).any(axis=(1, 3))


def load_control(path):
    """Lines like: '1  C1  Y(north)  X(east)  Z'  ->  list of (name, E, N, Z)."""
    pts = []
    for ln in open(path, encoding="utf-8", errors="ignore"):
        p = ln.replace(",", " ").replace(";", " ").split()
        if len(p) >= 4:
            try:    # name N E Z   (a leading running index column is tolerated)
                n_, e_, z_ = float(p[-3]), float(p[-2]), float(p[-1])
                pts.append((p[-4], e_, n_, z_))
            except ValueError:
                pass
    return pts


def check_control(dem, supported, x0, y0, res, pts, label):
    from scipy.ndimage import map_coordinates
    H, W = dem.shape
    rows = []
    for name, E, N, Z in pts:
        c, r = (E - x0) / res - 0.5, (N - y0) / res - 0.5
        if 0 <= c < W - 1 and 0 <= r < H - 1:
            ok = bool(supported[int(round(r)), int(round(c))])
            zd = float(map_coordinates(dem, [[r], [c]], order=1, mode="nearest")[0])
            rows.append((name, E, N, Z, zd, zd - Z, ok))
    inside = [r for r in rows if r[6]]
    print(f"[control:{label}] {len(pts)} targets, {len(rows)} inside grid, {len(inside)} on supported ground")
    if inside:
        d = np.array([r[5] for r in inside])
        med = float(np.median(d))
        dd = d - med
        print(f"  dZ (DEM - target): median {med:+.3f}  mean {d.mean():+.3f}  std {d.std():.3f}  "
              f"| after removing median: RMSE {np.sqrt((dd ** 2).mean()):.3f}  "
              f"|dZ|<=5cm {np.mean(np.abs(dd) <= .05) * 100:.0f}%  <=10cm {np.mean(np.abs(dd) <= .10) * 100:.0f}%  "
              f"max {np.abs(dd).max():.3f}")
    return rows


def _bincount_add(cnt, idx):
    bc = np.bincount(idx, minlength=cnt.size)
    np.minimum(cnt.astype(np.uint32) + bc, 65535, out=bc)
    cnt[:] = bc.astype(np.uint16)


# --------------------------------------------------------------------------- #
# 2-4. Cleaning, ground filter, interpolation
# --------------------------------------------------------------------------- #
def coarsen_min(grid, f):
    """Min-pool by integer factor f (NaN aware)."""
    H, W = grid.shape
    H2, W2 = (H + f - 1) // f, (W + f - 1) // f
    pad = np.full((H2 * f, W2 * f), np.nan, dtype=np.float32)
    pad[:H, :W] = grid
    with np.errstate(all="ignore"):
        out = np.nanmin(pad.reshape(H2, f, W2, f), axis=(1, 3))
    return out


def remove_low_noise(g, thresh=0.6, size=5):
    """Drop cells far below their neighbourhood median (multipath / reflections)."""
    filled = g.copy()
    nanm = np.isnan(filled)
    if nanm.all():
        return g
    idx = ndi.distance_transform_edt(nanm, return_distances=False, return_indices=True)
    filled = filled[tuple(idx)]
    med = ndi.median_filter(filled, size=size, mode="nearest")
    bad = (g < med - thresh)
    out = g.copy()
    out[bad] = np.nan
    return out


def fill_nearest(g):
    nanm = np.isnan(g)
    if not nanm.any():
        return g.copy()
    idx = ndi.distance_transform_edt(nanm, return_distances=False, return_indices=True)
    return g[tuple(idx)]


def progressive_morph_filter(dsm, res, max_window=40.0, slope=0.35, dh0=0.25, dh_max=3.0,
                             cell_size_ratio=1.0):
    """
    Zhang et al. 2003 progressive morphological filter on a min-elevation grid.
    Returns boolean mask 'ground' (True = bare earth).
    slope      : expected max terrain slope (rise/run).  Raise for steep embankments.
    dh0, dh_max: initial / maximum elevation-difference thresholds (m).
    max_window : largest object width (m) to remove (biggest building / hangar).
    """
    z = fill_nearest(dsm)
    valid = ~np.isnan(dsm)
    ground = valid.copy()

    # window sizes (in cells), exponential growth
    ws, k = [], 1
    while True:
        w = 2 * k + 1
        ws.append(w)
        if w * res >= max_window:
            break
        k = max(k + 1, int(k * 1.6))
    cur = z.copy()
    prev_w = 1
    for w in ws:
        fp = (w, w)
        opened = ndi.grey_opening(cur, size=fp, mode="nearest")
        dh = min(dh_max, dh0 + slope * (w - prev_w) * res * cell_size_ratio * 0.5 + (0 if prev_w > 1 else 0))
        # accumulate threshold with window growth (classic PMF)
        dh = min(dh_max, dh0 + slope * (w - 1) * res / 2.0)
        ground &= ~((cur - opened) > dh)
        cur = opened
        prev_w = w
    return ground & valid


def interpolate_ground(dsm, ground, res, decim_m=1.0):
    """Linear interpolation (Delaunay) of ground cells over everything else."""
    H, W = dsm.shape
    ys, xs = np.nonzero(ground)
    zs = dsm[ys, xs]
    step = max(1, int(round(decim_m / res)))
    # keep the lowest cell inside every decim block -> stays on the ground
    key = (ys // step) * (W // step + 2) + (xs // step)
    order = np.lexsort((zs, key))
    keep = np.ones(len(order), bool)
    keep[1:] = key[order][1:] != key[order][:-1]
    sel = order[keep]
    pts = np.column_stack([xs[sel], ys[sel]]).astype(np.float64)
    interp = LinearNDInterpolator(pts, zs[sel].astype(np.float64), fill_value=np.nan)
    yy, xx = np.mgrid[0:H, 0:W]
    out = np.full((H, W), np.nan, dtype=np.float32)
    # evaluate only where needed (non-ground) to save time, then paste original ground
    need = ~ground
    out[need] = interp(np.column_stack([xx[need], yy[need]]))
    out[ground] = dsm[ground]
    return fill_nearest(out)


def building_mask(dsm, dem, res, min_height=2.2, min_area=12.0, close=0.6):
    """Large, tall non-ground blobs (buildings / walls) -> excluded from contouring."""
    tall = np.nan_to_num(dsm - dem, nan=0) > min_height
    r = max(1, int(round(close / res)))
    tall = ndi.binary_closing(tall, structure=np.ones((3, 3)), iterations=r)
    tall = ndi.binary_fill_holes(tall)
    lab, n = ndi.label(tall)
    if n == 0:
        return tall
    area = ndi.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1)) * res * res
    keep = np.zeros(n + 1, bool)
    keep[1:] = area >= min_area
    m = keep[lab]
    # slight dilation = eaves overhang
    return ndi.binary_dilation(m, iterations=max(1, int(round(0.3 / res))))


# --------------------------------------------------------------------------- #
# 5. Contours
# --------------------------------------------------------------------------- #
def chaikin(pts, closed, it=2):
    for _ in range(it):
        if len(pts) < 3:
            break
        p = pts if not closed else pts[:-1]
        q = np.roll(p, -1, axis=0)
        a = 0.75 * p + 0.25 * q
        b = 0.25 * p + 0.75 * q
        new = np.empty((len(p) * 2, 2))
        new[0::2], new[1::2] = a, b
        if closed:
            pts = np.vstack([new, new[:1]])
        else:
            pts = np.vstack([p[:1], new[1:-1] if len(new) > 2 else new, p[-1:]])
            # keep ends fixed
    return pts


def extract_contours(dem, mask, x0, y0, res, interval, smooth_it=2, simplify_tol=0.02,
                     min_len=4.0, min_loop_area=2.0):
    import contourpy
    from shapely.geometry import LineString

    H, W = dem.shape
    xs = x0 + (np.arange(W) + 0.5) * res
    ys = y0 + (np.arange(H) + 0.5) * res
    z = np.ma.array(dem, mask=mask)
    gen = contourpy.contour_generator(xs, ys, z, name="serial", line_type="Separate")
    zmin, zmax = float(np.nanmin(dem[~mask])), float(np.nanmax(dem[~mask]))
    lv0 = math.ceil(zmin / interval) * interval
    levels = np.arange(lv0, zmax, interval)
    out = []
    for lv in levels:
        for line in gen.lines(float(lv)):
            pts = np.asarray(line, dtype=np.float64)
            if len(pts) < 3:
                continue
            closed = bool(np.allclose(pts[0], pts[-1]))
            ls = LineString(pts)
            if ls.length < min_len:
                continue
            if closed and len(pts) > 3:
                from shapely.geometry import Polygon
                if Polygon(pts).area < min_loop_area:
                    continue
            if simplify_tol > 0:
                pts = np.asarray(ls.simplify(simplify_tol).coords)
            if smooth_it:
                pts = chaikin(pts, closed, smooth_it)
            out.append((round(float(lv), 4), pts, closed))
    return out


# --------------------------------------------------------------------------- #
# 6. Output
# --------------------------------------------------------------------------- #
def write_dxf(path, contours, interval, index_every, label_min_len=25.0, label_height=0.8,
              masks_polys=None, ctrl=None, plan=None):
    import ezdxf

    doc = ezdxf.new("R2018", setup=True)
    doc.units = ezdxf.units.M
    doc.styles.get("Standard").dxf.font = "arial.ttf"
    msp = doc.modelspace()
    for name, color in (("COURBES_MAITRESSES", 1), ("COURBES_INTERMEDIAIRES", 8),
                        ("COURBES_COTES", 7), ("BATIMENTS_MASQUES", 6)):
        doc.layers.add(name, color=color)
    index_step = interval * index_every
    nlab = 0
    for lv, pts, closed in contours:
        is_index = abs((lv / index_step) - round(lv / index_step)) < 1e-6
        layer = "COURBES_MAITRESSES" if is_index else "COURBES_INTERMEDIAIRES"
        msp.add_lwpolyline([(float(x), float(y)) for x, y in pts], close=False,
                           dxfattribs={"layer": layer, "elevation": float(lv),
                                       "lineweight": 35 if is_index else 13})
        if is_index:
            seg = np.hypot(*np.diff(pts, axis=0).T)
            L = seg.sum()
            if L >= label_min_len:
                cum = np.concatenate([[0], np.cumsum(seg)])
                k = int(np.searchsorted(cum, L / 2))
                k = min(max(k, 1), len(pts) - 1)
                p = pts[k]
                d = pts[min(k + 1, len(pts) - 1)] - pts[k - 1]
                ang = math.degrees(math.atan2(d[1], d[0]))
                if ang > 90 or ang < -90:
                    ang += 180
                msp.add_text(f"{lv:.2f}".rstrip("0").rstrip("."), height=label_height,
                             dxfattribs={"layer": "COURBES_COTES", "rotation": ang,
                                         "insert": (float(p[0]), float(p[1])),
                                         "elevation": float(lv)})
                nlab += 1
    if masks_polys:
        for poly in masks_polys:
            msp.add_lwpolyline([(float(x), float(y)) for x, y in poly], close=True,
                               dxfattribs={"layer": "BATIMENTS_MASQUES"})
    if ctrl:
        doc.layers.add("CIBLES", color=3)
        for name, E, N, Z, zd, dz, ok in ctrl:
            msp.add_point((E, N, Z), dxfattribs={"layer": "CIBLES"})
            rr = label_height * 0.9
            msp.add_circle((E, N), rr, dxfattribs={"layer": "CIBLES"})
            msp.add_line((E - 1.6 * rr, N), (E + 1.6 * rr, N), dxfattribs={"layer": "CIBLES"})
            msp.add_line((E, N - 1.6 * rr), (E, N + 1.6 * rr), dxfattribs={"layer": "CIBLES"})
            msp.add_text(f"{name}  {Z:.3f}", height=label_height * 0.9,
                         dxfattribs={"layer": "CIBLES", "insert": (E + 1.8 * rr, N + 0.5 * rr)})
    if plan:
        import plan_sheet
        info = dict(plan["info"], interval=interval, index_step=index_step, has_targets=bool(ctrl))
        P = plan_sheet.build_plan(doc, msp, plan["bbox"], info, plan["spots"], masks_polys)
        print(f"[plan] sheet {P['sheet']} 1/{P['scale']}, spot heights every {P['spot_step']:g} m "
              f"({len(plan['spots'])}), grid every {P['grid_step']:g} m")
    doc.saveas(path)
    if plan:
        try:   # printable PDF of the layout sheet (vector)
            from ezdxf.addons.drawing import matplotlib as mpl
            pdf = os.path.splitext(path)[0] + ".pdf"
            mpl.qsave(doc.layouts.get("PLAN"), pdf, bg="#ffffff", fg="#000000",
                      size_inches=(P["W"] / 25.4, P["H"] / 25.4))
            print(f"[plan] PDF: {pdf}")
        except Exception as e:  # noqa: BLE001
            print(f"[plan] PDF export skipped ({e})")
    return nlab


def mask_outlines(mask, x0, y0, res, min_area=5.0):
    import contourpy
    gen = contourpy.contour_generator(
        x0 + (np.arange(mask.shape[1]) + 0.5) * res,
        y0 + (np.arange(mask.shape[0]) + 0.5) * res,
        mask.astype(float), name="serial", line_type="Separate")
    polys = []
    for l in gen.lines(0.5):
        l = np.asarray(l)
        if len(l) > 3 and np.allclose(l[0], l[-1]):
            a = 0.5 * abs(np.dot(l[:-1, 0], l[1:, 1]) - np.dot(l[1:, 0], l[:-1, 1]))
            if a >= min_area:
                polys.append(l[:-1])
    return polys


def write_preview(path, dem, mask, contours, x0, y0, res, interval, index_every,
                  bmask=None, nodata_m=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource

    H, W = dem.shape
    ext = (x0, x0 + W * res, y0, y0 + H * res)
    z = np.where(mask, np.nan, dem)
    ls = LightSource(azdeg=315, altdeg=40)
    shade = ls.hillshade(np.nan_to_num(dem, nan=float(np.nanmin(dem))), vert_exag=2, dx=res, dy=res)
    fig, ax = plt.subplots(figsize=(14, 14 * H / W + 0.5), dpi=130)
    ax.imshow(shade, cmap="gray", extent=ext, origin="lower", alpha=0.9)
    if bmask is not None:
        mm = np.ma.array(bmask.astype(float), mask=~bmask)
        ax.imshow(mm, cmap="Reds", extent=ext, origin="lower", alpha=0.45, vmin=0, vmax=1.5)
    nd = np.ma.array(np.ones_like(dem), mask=~nodata_m)
    ax.imshow(nd, cmap="gray_r", extent=ext, origin="lower", alpha=1.0, vmin=0, vmax=1)
    # zoom on the data
    ys, xs = np.nonzero(~nodata_m)
    ax.set_xlim(x0 + xs.min() * res - 5, x0 + xs.max() * res + 5)
    ax.set_ylim(y0 + ys.min() * res - 5, y0 + ys.max() * res + 5)
    step = interval * index_every
    for lv, pts, closed in contours:
        idx = abs(lv / step - round(lv / step)) < 1e-6
        ax.plot(pts[:, 0], pts[:, 1], color="#b00020" if idx else "#555555",
                lw=0.9 if idx else 0.35)
    ax.set_aspect("equal")
    ax.set_title(f"Contours every {interval} m (index every {step:g} m) - red zone = masked buildings")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description="E57 -> ground contours -> DXF")
    ap.add_argument("e57")
    ap.add_argument("-o", "--out", default=None, help="output .dxf (default: next to e57 / cwd)")
    ap.add_argument("--interval", type=float, default=0.25, help="contour interval, m (default 0.25)")
    ap.add_argument("--index-every", type=int, default=4, help="every Nth contour is a bold index contour")
    ap.add_argument("--fine-cell", type=float, default=0.10, help="fine lowest-point grid, m")
    ap.add_argument("--dem-res", type=float, default=0.25, help="DEM resolution, m (multiple of fine-cell)")
    ap.add_argument("--max-object", type=float, default=40.0, help="largest building/object width to strip, m")
    ap.add_argument("--slope", type=float, default=0.35, help="max terrain slope for ground filter (rise/run)")
    ap.add_argument("--smooth", type=float, default=0.6, help="DEM gaussian sigma in metres (0 = none)")
    ap.add_argument("--decim", type=float, default=0.5, help="ground sample spacing for interpolation, m")
    ap.add_argument("--refine", type=int, default=2, help="ground re-classification passes")
    ap.add_argument("--refine-h", type=float, default=0.15, help="ground tolerance in refinement, m")
    ap.add_argument("--reach", type=float, default=2.5,
                    help="bridge ground gaps up to this width, m (larger = more interpolated contours)")
    ap.add_argument("--max-gap", type=float, default=3.0,
                    help="interpolate across data gaps up to this radius, m")
    ap.add_argument("--surface", choices=["robust", "min"], default="robust",
                    help="robust = median of the 3 lowest points per 10 cm cell (default); min = lowest point")
    ap.add_argument("--vert-slope", type=float, default=1.7,
                    help="surface steeper than this (rise/run, 1.7 = 60 deg) is a wall, not ground")
    ap.add_argument("--plan", action="store_true",
                    help="also build a finished plan sheet (layout 'PLAN': title block, scale, legend, spot heights)")
    ap.add_argument("--title", default=None, help="project name for the plan title block")
    ap.add_argument("--mode", choices=["outline", "contours", "both"], default="outline",
                    help="outline = exterior wall outline of the built complex (plan linework, layer Murs); "
                         "contours = elevation contour lines; both")
    ap.add_argument("--tall-h", type=float, default=None,
                    help="outline: structures higher than this above the ground count as built, m")
    ap.add_argument("--no-walls", action="store_true", help="outline: skip the inner wall lines")
    ap.add_argument("--reference", default=None, help="outline: a DXF drawn by hand, to score the result against")
    ap.add_argument("--no-fix", action="store_true", help="skip the automatic contour fixer")
    ap.add_argument("--tol-z", type=float, default=0.08,
                    help="fixer: max height deviation from the surface when smoothing, m")
    ap.add_argument("--join-gap", type=float, default=4.0,
                    help="fixer: join broken contour ends of the same level up to this gap, m")
    ap.add_argument("--ground-las", default=None,
                    help="LAS of ground points from an external classifier (PDAL SMRF/CSF) instead of the built-in filter")
    ap.add_argument("--no-vertical", action="store_true", help="disable vertical-face removal")
    ap.add_argument("--control", default=None, help="control-point txt: idx name N E Z; checks DEM heights")
    ap.add_argument("--keep-buildings", action="store_true", help="do not mask building footprints")
    ap.add_argument("--min-len", type=float, default=8.0, help="drop contours shorter than this, m")
    ap.add_argument("--clip", type=float, nargs=4, metavar=("XMIN", "YMIN", "XMAX", "YMAX"))
    ap.add_argument("--cache", default=None, help="grid cache .npz (default: <out>.grid.npz)")
    ap.add_argument("--rebuild", action="store_true", help="ignore cache, re-read the E57")
    a = ap.parse_args(argv)

    out = a.out or os.path.splitext(os.path.basename(a.e57))[0] + "_contours.dxf"
    cache = a.cache or os.path.splitext(out)[0] + ".grid.npz"
    fine = a.fine_cell
    f = int(round(a.dem_res / fine))
    res = fine * f

    t0 = time.time()
    layers = None
    if os.path.exists(cache) and not a.rebuild:
        z = np.load(cache)
        if "low" in z.files:
            layers = (z["low"], z["zmax"], z["cnt"])
            x0, y0, fine = [float(v) for v in z["org"]]
            f = int(round(a.dem_res / fine)); res = fine * f
            print(f"[cache] loaded {cache}")
        else:
            print("[cache] old cache format (min only) -> re-reading the E57")
    if layers is None:
        layers, (x0, y0, fine) = build_min_grid(a.e57, fine, clip=a.clip)
        np.savez_compressed(cache, low=layers[0], zmax=layers[1], cnt=layers[2],
                            org=np.array([x0, y0, fine]))
        print(f"[cache] saved {cache}")

    # --- surface (robust low surface) + vertical-face mask, then clean + coarsen ---
    fg, _ = surface_from_layers(layers, a.surface)
    fg = remove_low_noise(fg, thresh=0.5, size=7)
    dsm = coarsen_min(fg, f)
    H, W = dsm.shape
    vert_c = np.zeros(dsm.shape, bool)
    if not a.no_vertical:
        # walls / steep faces: surface gradient steeper than --vert-slope (rise/run)
        filled = fill_nearest(dsm)
        gy, gx = np.gradient(filled, res)
        valid = ndi.binary_erosion(np.isfinite(dsm), iterations=2)
        vert_c = (np.hypot(gx, gy) > a.vert_slope) & valid
        vert_c = ndi.binary_dilation(vert_c, iterations=1)
    print(f"[dem] surface={a.surface}  vertical-face cells {vert_c.mean() * 100:.1f}%")
    print(f"[dem] {W} x {H} cells @ {res} m, data coverage {np.isfinite(dsm).mean() * 100:.0f}%")

    # --- ground filter ---
    t = time.time()
    if a.ground_las:
        import laspy
        L = laspy.read(a.ground_las)
        gc = ((np.asarray(L.x) - x0) / res).astype(int)
        gr = ((np.asarray(L.y) - y0) / res).astype(int)
        okp = (gc >= 0) & (gc < W) & (gr >= 0) & (gr < H)
        ground = np.zeros(dsm.shape, bool)
        ground[gr[okp], gc[okp]] = True
        ground &= np.isfinite(dsm)
        # external classifier: a cell is ground if it holds ground points; thin the holes
        ground = ndi.binary_closing(ground, iterations=1) & np.isfinite(dsm)
        print(f"[ground-las] {a.ground_las}")
    else:
        ground = progressive_morph_filter(dsm, res, max_window=a.max_object, slope=a.slope) & ~vert_c
    print(f"[pmf] ground cells {ground.sum() / max(1, np.isfinite(dsm).sum()) * 100:.0f}% "
          f"({time.time() - t:.0f}s)")

    # --- interpolate bare earth ---
    t = time.time()
    dem = interpolate_ground(dsm, ground, res, decim_m=a.decim)
    for _ in range(a.refine):
        # re-classify against the interpolated surface: removes thin walls/kerbs/vehicles
        # the morphological pass kept, and re-admits genuine ground it removed on slopes
        diff = dsm - dem
        ground = np.isfinite(dsm) & (diff < a.refine_h) & (diff > -a.refine_h)
        ground &= ndi.binary_opening(ground, iterations=1) | (np.abs(np.nan_to_num(diff)) < a.refine_h / 3)
        ground &= ~vert_c
        dem =interpolate_ground(dsm, ground, res, decim_m=a.decim)
    if a.smooth > 0:
        dem = ndi.gaussian_filter(dem, sigma=a.smooth / res, mode="nearest")
    print(f"[dem] bare-earth surface built ({time.time() - t:.0f}s)  "
          f"Z {np.nanmin(dem):.2f} .. {np.nanmax(dem):.2f}")

    # --- exterior outline of the built complex (plan linework) ---
    if a.mode in ("outline", "both"):
        import outline as ol
        # plain bare-earth model for the heights (independent of the contour refinement/smoothing)
        g0 = progressive_morph_filter(dsm, res, max_window=a.max_object, slope=a.slope)
        dem0 = interpolate_ground(dsm, g0, res, decim_m=a.decim)
        hag = ol.top_height(layers[1], dem0, f)
        prm = ol.load_params()
        if prm:
            print(f"[outline] using tuned settings from params.json: {prm}")
        rings, _ = ol.extract_outlines(
            hag, x0, y0, res, tall_h=a.tall_h if a.tall_h is not None else prm.get("tall_h", 3.5),
            close_it=int(prm.get("close_it", 3)), open_it=int(prm.get("open_it", 3)),
            smooth_r=prm.get("smooth_r", 0.6),
            reg={k: prm[k] for k in ("min_seg", "merge_tol", "dp_tol") if k in prm})
        wl = None
        if not a.no_walls and rings:
            import walls as wk
            wl = wk.extract_walls(wk.height_above_ground(layers[1], dem0, f), fine, x0, y0, ol._dominant(rings[0]))
        opath = out if a.mode == "outline" else os.path.splitext(out)[0] + "_outline.dxf"
        ol.write_outline_dxf(opath, rings, walls=wl)
        print(f"[outline] DXF: {opath}")
        if a.reference and rings:
            ol.compare_reference(rings, a.reference)
        if a.mode == "outline":
            print(f"[done] total {time.time() - t0:.0f}s")
            return

    # --- masks ---
    nodata = ~np.isfinite(dsm)
    nodata = ndi.binary_opening(nodata, iterations=1)
    far = ndi.distance_transform_edt(nodata) * res > a.max_gap  # holes bigger than max-gap
    bmask = np.zeros_like(far) if a.keep_buildings else building_mask(dsm, dem, res)
    # no contours where the surface is pure interpolation over a wide gap (no real ground seen)
    reach = max(1, int(round(a.reach / res)))
    supported = ndi.binary_dilation(ground, iterations=reach)
    supported = ndi.binary_erosion(supported, iterations=reach)   # = morphological closing
    mask = far | bmask | ~supported
    # trim 1px edge so contours do not hug the data boundary
    print(f"[mask] buildings {bmask.mean() * 100:.1f}%  nodata {far.mean() * 100:.1f}%")
    ctrl_rows = None
    if a.control:
        pts = load_control(a.control)
        ctrl_rows = check_control(dem, supported & ~far, x0, y0, res, pts, a.surface)

    # --- contours ---
    if a.no_fix:
        contours = extract_contours(dem, mask, x0, y0, res, a.interval, min_len=a.min_len,
                                    min_loop_area=a.min_len * 1.5)
    else:
        # raw lines first (no smoothing / simplification, short pieces kept so they can be joined),
        # then the automatic fixer straightens, cleans and completes them
        import contour_fix
        raw = extract_contours(dem, mask, x0, y0, res, a.interval, smooth_it=0, simplify_tol=0,
                               min_len=1.0, min_loop_area=1.0)
        contours = contour_fix.fix_contours(raw, dem, mask, None if a.keep_buildings else bmask,
                                            x0, y0, res, tol_z=a.tol_z, join_gap=a.join_gap,
                                            min_len=a.min_len)
    print(f"[contours] {len(contours)} polylines, levels "
          f"{min(c[0] for c in contours):g}..{max(c[0] for c in contours):g}")

    polys = None if a.keep_buildings else mask_outlines(bmask, x0, y0, res)
    plan, label_h = None, 0.8
    if a.plan and contours:
        import plan_sheet
        allp = np.vstack([c[1] for c in contours])
        bb = (allp[:, 0].min(), allp[:, 1].min(), allp[:, 0].max(), allp[:, 1].max())
        P0 = plan_sheet.plan_params((bb[0] - 8, bb[1] - 8, bb[2] + 8, bb[3] + 8))
        label_h = 1.1 * P0["text_h"]
        spots = plan_sheet.spot_heights(dem, mask, x0, y0, res, P0["spot_step"])
        plan = {"bbox": bb, "spots": spots,
                "info": {"project": a.title or os.path.splitext(os.path.basename(a.e57))[0],
                         "source": os.path.basename(a.e57)}}
    nlab = write_dxf(out, contours, a.interval, a.index_every, label_height=label_h,
                     masks_polys=polys, ctrl=ctrl_rows, plan=plan)
    png = os.path.splitext(out)[0] + "_preview.png"
    write_preview(png, dem, mask, contours, x0, y0, res, a.interval, a.index_every,
                  bmask=None if a.keep_buildings else bmask, nodata_m=far)
    print(f"[done] {out}  ({nlab} labels)  preview: {png}  total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    sys.exit(main())
