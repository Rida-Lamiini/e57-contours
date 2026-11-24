"""
dl_export.py - run on THIS PC: turn the scan caches + your drawings into a deep-learning dataset.

  data/stack.npz   : image layers at 10 cm (float16), channels listed in meta.json
  data/labels.npz  : 0 background / 1 wall / 2 low wall / 3 zellige / 255 = not labelled (ignored by the loss)
  data/meta.json   : origin, cell size, channel names

Labels come from DXF drawings:
  --outline-truth : drawings that only contain the OUTER outline (e.g. your first hand drawing).  Only a band of
                    +-4 m around those lines is supervised (inner walls are NOT claimed to be absent).
  --full-truth    : complete wall drawings (e.g. saved from the review tool). Everything inside the drawing's
                    area (+10 m) is supervised: no line there = no wall.
"""
import argparse
import glob
import json
import math
import os
import sys
import time

import ezdxf
import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import map_coordinates
from skimage.draw import line as draw_line
from skimage.draw import polygon as draw_poly

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import slice_walls as sw          # noqa: E402
import walls as wk                # noqa: E402

CHANNELS = ["height_max", "slice_low", "slice_mid", "slice_high", "density", "warmth", "brightness", "green", "valid"]


def colour_pass(d, meta, dem, dres, block=3_000_000):
    n, W, H, cell = meta["points"], meta["W"], meta["H"], meta["cell"]
    xyz = np.memmap(os.path.join(d, "xyz.f32"), dtype=np.float32, mode="r", shape=(n, 3))
    rgb = np.memmap(os.path.join(d, "rgb.u8"), dtype=np.uint8, mode="r", shape=(n, 3))
    sums = np.zeros((3, H * W), np.float32)
    cnt = np.zeros(H * W, np.float32)
    t0 = time.time()
    for s in range(0, n, block):
        P = np.asarray(xyz[s:s + block])
        C = np.asarray(rgb[s:s + block]).astype(np.float32)
        hag = P[:, 2] + meta["z0"] - map_coordinates(dem, [P[:, 1] / dres - 0.5, P[:, 0] / dres - 0.5], order=1, mode="nearest")
        ix, iy = (P[:, 0] / cell).astype(np.int64), (P[:, 1] / cell).astype(np.int64)
        ok = (hag >= 0.3) & (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
        idx = iy[ok] * W + ix[ok]
        if len(idx):
            u, inv = np.unique(idx, return_inverse=True)
            cnt[u] += np.bincount(inv).astype(np.float32)
            for k in range(3):
                sums[k][u] += np.bincount(inv, weights=C[ok, k]).astype(np.float32)
        print(f"\r[colour] {min(s + block, n) / 1e6:6.1f} / {n / 1e6:.0f} M pts  {time.time() - t0:.0f}s", end="", flush=True)
    print()
    m = np.maximum(cnt, 1)
    return sums[0] / m, sums[1] / m, sums[2] / m, cnt


def raster_truth(path, x0, y0, cell, shape, thick_m=0.35):
    """DXF -> label image (1 wall, 2 low wall, 3 zellige) and the bounding polygon points of everything drawn"""
    H, W = shape
    lab = np.zeros(shape, np.uint8)
    pts_all = []
    d = ezdxf.readfile(path)
    for e in d.modelspace():
        if e.dxftype() == "LWPOLYLINE":
            p = np.array(list(e.get_points("xy")), float)
            closed = bool(e.closed)
        elif e.dxftype() == "LINE":
            p = np.array([[e.dxf.start.x, e.dxf.start.y], [e.dxf.end.x, e.dxf.end.y]], float)
            closed = False
        else:
            continue
        cls = {"Murs": 1, "Murs_interieurs": 1, "0": 1, "mur bas": 2, "Zellige": 3}.get(e.dxf.layer, 1)
        q = np.vstack([p, p[:1]]) if closed else p
        pts_all.append(p)
        layer = np.zeros(shape, bool)
        for a, b in zip(q[:-1], q[1:]):
            r0, c0 = (a[1] - y0) / cell, (a[0] - x0) / cell
            r1, c1 = (b[1] - y0) / cell, (b[0] - x0) / cell
            rr, cc = draw_line(int(round(r0)), int(round(c0)), int(round(r1)), int(round(c1)))
            ok = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
            layer[rr[ok], cc[ok]] = True
        layer = ndi.binary_dilation(layer, iterations=max(1, int(round(thick_m / cell / 2))))
        lab[layer & (lab == 0)] = cls
    return lab, (np.vstack(pts_all) if pts_all else np.zeros((0, 2)))


def build_labels(shape, x0, y0, cell, outline_truth, full_truth, log=print):
    H, W = shape
    labels = np.full(shape, 255, np.uint8)
    n_pos = 0
    for path, full in [(p, False) for p in outline_truth] + [(p, True) for p in full_truth]:
        lab, pts = raster_truth(path, x0, y0, cell, shape)
        if not len(pts):
            continue
        if full:   # supervise the whole area covered by the drawing (+10 m)
            hull = np.zeros(shape, bool)
            from shapely.geometry import MultiPoint
            hp = MultiPoint(pts).convex_hull.buffer(10.0)
            ex = np.array(hp.exterior.coords)
            rr, cc = draw_poly((ex[:, 1] - y0) / cell, (ex[:, 0] - x0) / cell, shape)
            hull[rr, cc] = True
            region = hull
        else:      # only a band around the drawn lines
            region = ndi.binary_dilation(lab > 0, iterations=int(4.0 / cell))
        cur = labels.copy()
        cur[region & (cur == 255)] = 0
        cur[lab > 0] = lab[lab > 0]
        labels = cur
        n_pos += int((lab > 0).sum())
        log(f"[labels] {os.path.basename(path)} ({'full' if full else 'outline-only'}): "
            f"{int((lab > 0).sum()):,} line cells, supervised area {region.sum() * cell * cell:.0f} m2")
    return labels


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", default=os.path.join(HERE, "out", "full"), help="folder made by fullcloud.py")
    ap.add_argument("--out", default=os.path.join(HERE, "dl_package", "data"))
    ap.add_argument("--outline-truth", nargs="*", default=[
        "//GENAS/Public/ARCHIVE2026/132_HMILIT/132_BdTerrain/132_SCAN19112026/Public (GENAS).dxf"])
    ap.add_argument("--full-truth", nargs="*", default=None,
                    help="default: every DXF in the 'training' folder")
    ap.add_argument("--no-colour", action="store_true")
    a = ap.parse_args(argv)
    full_truth = a.full_truth if a.full_truth is not None else sorted(glob.glob(os.path.join(HERE, "training", "*.dxf")))
    os.makedirs(a.out, exist_ok=True)

    meta, layers = sw.load_cache(a.full)
    x0, y0, cell, W, H = meta["x0"], meta["y0"], meta["cell"], meta["W"], meta["H"]
    print(f"[export] grid {W} x {H} @ {cell} m", flush=True)
    dem, dres = sw.ground_model(layers, cell)
    f = int(round(dres / cell))
    zs = np.load(os.path.join(a.full, "slices.npz"))
    cnt = layers[2]

    # crop to the data
    rows, cols = np.nonzero(cnt > 0)
    r0, r1, c0, c1 = (int(v) for v in (rows.min(), rows.max() + 1, cols.min(), cols.max() + 1))
    sl = (slice(r0, r1), slice(c0, c1))
    h, w = r1 - r0, c1 - c0
    ox, oy = x0 + c0 * cell, y0 + r0 * cell
    print(f"[export] cropped to {w} x {h} cells, origin {ox:.1f}, {oy:.1f}", flush=True)

    stack = np.zeros((len(CHANNELS), h, w), np.float16)
    hag = wk.height_above_ground(layers[1], dem, f)[sl]
    stack[0] = np.clip(np.nan_to_num(hag, nan=-1.0), -1, 10) / 10.0
    for k, name in ((1, "low"), (2, "mid"), (3, "high")):
        stack[k] = np.log1p(zs["S_" + name][sl].astype(np.float32)) / 5.0
    stack[4] = np.log1p(cnt[sl].astype(np.float32)) / 8.0
    if not a.no_colour:
        R, G, B, cc = colour_pass(a.full, meta, dem, dres)
        R, G, B, cc = (x.reshape(H, W)[sl] for x in (R, G, B, cc))
        has = cc > 0
        stack[5] = np.where(has, (R - B) / 255.0, 0)
        stack[6] = np.where(has, (R + G + B) / 765.0, 0)
        stack[7] = np.where(has, (2 * G - R - B) / np.maximum(R + G + B, 1), 0)
    stack[8] = (cnt[sl] > 0).astype(np.float16)

    labels = build_labels((h, w), ox, oy, cell, a.outline_truth, full_truth)
    np.savez_compressed(os.path.join(a.out, "stack.npz"), stack=stack)
    np.savez_compressed(os.path.join(a.out, "labels.npz"), labels=labels)
    json.dump({"x0": float(ox), "y0": float(oy), "cell": float(cell), "channels": CHANNELS, "shape": [int(h), int(w)],
               "outline_truth": [os.path.basename(p) for p in a.outline_truth],
               "full_truth": [os.path.basename(p) for p in full_truth]},
              open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    sup = (labels != 255).sum()
    print(f"[export] done: stack {stack.shape} ({stack.nbytes / 1e6:.0f} MB), supervised cells {sup:,} "
          f"({sup * cell * cell:.0f} m2), wall cells {(labels == 1).sum():,}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
