"""
slice_walls.py - wall plan from the full point cloud by horizontal slicing.

  1. ground model (bare earth) from the lowest-points grid built by fullcloud.py
  2. three horizontal bands above the ground:   low 0.35-0.80 m | mid 0.95-1.40 m | high 1.70-2.30 m
     points are counted per 10 cm cell; green vegetation (excess-green colour index) is ignored
  3. a WALL cell has points in all three bands (continuous vertical surface, not a bush, car or roof edge)
     a LOW WALL cell has points in the low band only (0.35-0.80 m, nothing above)
  4. clean (close gaps, drop small blobs), vectorise the wall faces, square them up to the building
     directions and write DXF layers  Murs / mur bas / Zellige
"""
import argparse
import json
import math
import os
import sys
import time

import contourpy
import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import map_coordinates
from shapely.geometry import LineString, Polygon

import e57_to_contours as core
import outline as ol

BANDS = {"low": (0.35, 0.80), "mid": (0.95, 1.40), "high": (1.70, 2.30)}


def load_cache(d):
    meta = json.load(open(os.path.join(d, "meta.json")))
    z = np.load(os.path.join(d, "layers.npz"))
    return meta, (z["low"], z["zmax"], z["cnt"])


def ground_model(layers, fine, dem_res=0.2, max_object=40.0, slope=0.35, log=print):
    f = int(round(dem_res / fine))
    res = fine * f
    fg, _ = core.surface_from_layers(layers, "robust")
    fg = core.remove_low_noise(fg, thresh=0.5, size=7)
    dsm = core.coarsen_min(fg, f)
    g = core.progressive_morph_filter(dsm, res, max_window=max_object, slope=slope)
    dem = core.interpolate_ground(dsm, g, res, decim_m=0.5)
    log(f"[ground] DEM {dem.shape[1]} x {dem.shape[0]} @ {res} m, ground cells {g.sum() / max(1, np.isfinite(dsm).sum()) * 100:.0f}%")
    return dem, res


def slice_pass(d, meta, dem, dem_res, block=2_000_000, log=print):
    x0, y0, z0, cell, W, H = meta["x0"], meta["y0"], meta["z0"], meta["cell"], meta["W"], meta["H"]
    n = meta["points"]
    xyz = np.memmap(os.path.join(d, "xyz.f32"), dtype=np.float32, mode="r", shape=(n, 3))
    rgb = np.memmap(os.path.join(d, "rgb.u8"), dtype=np.uint8, mode="r", shape=(n, 3))
    S = {k: np.zeros(H * W, dtype=np.uint16) for k in BANDS}
    green = {k: np.zeros(1, dtype=np.uint16) for k in BANDS}     # not stored (saves memory)
    t0 = time.time()
    for s in range(0, n, block):
        P = np.asarray(xyz[s:s + block])
        C = np.asarray(rgb[s:s + block]).astype(np.int16)
        # point height above the ground model (bilinear)
        r = P[:, 1] / dem_res - 0.5                          # xyz is stored relative to (x0, y0, z0)
        c = (P[:, 0]) / dem_res - 0.5
        hag = P[:, 2] + z0 - map_coordinates(dem, [r, c], order=1, mode="nearest")
        exg = (2 * C[:, 1] - C[:, 0] - C[:, 2]) / np.maximum(C.sum(1), 1)
        veg = exg > 0.10
        ix = (P[:, 0] / cell).astype(np.int64)
        iy = (P[:, 1] / cell).astype(np.int64)
        ok = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
        for k, (lo, hi) in BANDS.items():
            m = ok & (hag >= lo) & (hag < hi)
            idx = (iy[m] * W + ix[m])[~veg[m]]                # vegetation (green) points are ignored
            if len(idx):
                u, cn = np.unique(idx, return_counts=True)     # sparse update: no full-size temporaries
                S[k][u] = np.minimum(S[k][u].astype(np.int32) + cn, 65535).astype(np.uint16)
        el = time.time() - t0
        print(f"\r[slice] {min(s + block, n) / 1e6:6.1f} / {n / 1e6:.0f} M pts  {min(s + block, n) / el / 1e6:4.1f} M/s", end="", flush=True)
    print()
    return {k: v.reshape(H, W) for k, v in S.items()}, {k: v for k, v in green.items()}


def wall_masks(S, G, cell, k_mid=2, k_other=1, min_area=1.2, close_it=1):
    mid, low, high = S["mid"], S["low"], S["high"]
    wall = (mid >= k_mid) & (low >= k_other) & (high >= k_other)
    wall = ndi.binary_closing(wall, structure=np.ones((3, 3)), iterations=close_it)
    lab, n = ndi.label(wall)
    if n:
        area = ndi.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1)) * cell * cell
        keep = np.zeros(n + 1, bool)
        keep[1:] = area >= min_area
        wall = keep[lab]
    lowwall = (low >= 2) & (mid == 0) & (high == 0) & ~ndi.binary_dilation(wall, iterations=2)
    lowwall = ndi.binary_opening(lowwall, structure=np.ones((2, 2)))
    lab, n = ndi.label(lowwall)
    if n:
        area = ndi.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1)) * cell * cell
        keep = np.zeros(n + 1, bool)
        keep[1:] = area >= min_area
        lowwall = keep[lab]
    return wall, lowwall


def vectorise(mask, x0, y0, cell, base, min_ring=1.5, log=print):
    """all boundaries of the mask (outer faces + room interiors) -> squared-up closed polylines"""
    H, W = mask.shape
    xs = x0 + (np.arange(W) + 0.5) * cell
    ys = y0 + (np.arange(H) + 0.5) * cell
    g = contourpy.contour_generator(xs, ys, mask.astype(float), name="serial", line_type="Separate")
    rings, bad = [], 0
    for line in g.lines(0.5):
        r = np.asarray(line)
        if len(r) < 8 or LineString(r).length < min_ring * 2:
            continue
        reg = ol.regularise(r, base=base, dp_tol=0.07, snap_tol=math.radians(8), min_seg=0.35, merge_tol=0.12)
        P = Polygon(reg)
        if not P.is_valid or P.area < 0.15:
            reg = np.asarray(LineString(r).simplify(0.06).coords)
            bad += 1
        rings.append(reg)
    log(f"[vector] {len(rings)} polylines ({bad} kept as simplified because squaring-up self-intersected)")
    return rings


def write_dxf(path, wall_rings, low_rings, ref_dxf=None):
    import ezdxf
    doc = ezdxf.new("R2018", setup=True)
    doc.units = ezdxf.units.M
    for name, color in (("Murs", 7), ("mur bas", 130), ("Zellige", 6)):
        doc.layers.add(name, color=color)
    msp = doc.modelspace()
    for lay, rings in (("Murs", wall_rings), ("mur bas", low_rings)):
        for r in rings:
            r = np.asarray(r)
            pts = r[:-1] if np.allclose(r[0], r[-1]) else r
            msp.add_lwpolyline([(float(x), float(y)) for x, y in pts], close=True, dxfattribs={"layer": lay})
    doc.saveas(path)


def preview(path, wall, lowwall, rings, low_rings, x0, y0, cell, ref_dxf=None, window=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    H, W = wall.shape
    ext = (x0, x0 + W * cell, y0, y0 + H * cell)
    fig, ax = plt.subplots(figsize=(14, 14), dpi=75)
    img = np.zeros((H, W, 4), np.float32)
    img[wall] = (0.12, 0.47, 0.71, 0.55)
    img[lowwall] = (1.0, 0.5, 0.05, 0.55)
    ax.imshow(img, extent=ext, origin="lower")
    for r in rings:
        ax.plot(r[:, 0], r[:, 1], color="#08306b", lw=0.7)
    for r in low_rings:
        ax.plot(r[:, 0], r[:, 1], color="#d94801", lw=0.7)
    if ref_dxf:
        import ezdxf
        for e in ezdxf.readfile(ref_dxf).modelspace():
            if e.dxftype() == "LWPOLYLINE":
                p = np.array(list(e.get_points("xy")))
                ax.plot(p[:, 0], p[:, 1], color="red", lw=1.4, ls="--")
    if window:
        ax.set_xlim(window[0], window[1])
        ax.set_ylim(window[2], window[3])
    ax.set_aspect("equal")
    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cache_dir")
    ap.add_argument("-o", "--out", default="out/full/Plan_murs.dxf")
    ap.add_argument("--reference", default=None)
    ap.add_argument("--k-mid", type=int, default=2, help="min points per 10 cm cell in the 1 m band")
    ap.add_argument("--rebuild-slices", action="store_true")
    a = ap.parse_args(argv)
    meta, layers = load_cache(a.cache_dir)
    x0, y0, cell = meta["x0"], meta["y0"], meta["cell"]
    sp = os.path.join(a.cache_dir, "slices.npz")
    if os.path.exists(sp) and not a.rebuild_slices:
        z = np.load(sp)
        S = {k: z["S_" + k] for k in BANDS}
        G = {k: z["G_" + k] for k in BANDS}
        print(f"[slice] loaded {sp}")
    else:
        dem, dres = ground_model(layers, cell)
        S, G = slice_pass(a.cache_dir, meta, dem, dres)
        np.savez_compressed(sp, **{"S_" + k: v for k, v in S.items()}, **{"G_" + k: v for k, v in G.items()})
    wall, lowwall = wall_masks(S, G, cell, k_mid=a.k_mid)
    print(f"[walls] wall cells {wall.sum():,} ({wall.sum() * cell * cell:.0f} m2), low-wall cells {lowwall.sum():,}")
    # building direction from the long edges of the largest wall boundary
    g = contourpy.contour_generator(x0 + (np.arange(wall.shape[1]) + .5) * cell, y0 + (np.arange(wall.shape[0]) + .5) * cell,
                                    wall.astype(float), name="serial", line_type="Separate")
    big = max(g.lines(0.5), key=len)
    base = ol._dominant(np.asarray(big))
    print(f"[walls] main direction {math.degrees(base):.1f} deg")
    rings = vectorise(wall, x0, y0, cell, base)
    low_rings = vectorise(lowwall, x0, y0, cell, base, min_ring=1.0)
    write_dxf(a.out, rings, low_rings)
    print(f"[done] {a.out}  (Murs {len(rings)}, mur bas {len(low_rings)})")
    preview(os.path.splitext(a.out)[0] + "_apercu.png", wall, lowwall, rings, low_rings, x0, y0, cell, a.reference,
            window=(251090, 251255, 118690, 118850) if a.reference else None)


if __name__ == "__main__":
    sys.exit(main())
