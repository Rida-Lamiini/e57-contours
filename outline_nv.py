"""
outline_nv.py - exterior outline from the FULL cloud, ignoring vegetation by colour.

  top height above ground = highest NON-GREEN point per cell (tree crowns drop out, roofs/walls stay)
  then the same closing / fill / squaring-up as outline.py
"""
import argparse
import json
import os
import sys
import time

import numpy as np

import outline as ol
import slice_walls as sw


def nonveg_top(d, meta, warm_min=20, block=3_000_000):
    n, W, H = meta["points"], meta["W"], meta["H"]
    xyz = np.memmap(os.path.join(d, "xyz.f32"), dtype=np.float32, mode="r", shape=(n, 3))
    rgb = np.memmap(os.path.join(d, "rgb.u8"), dtype=np.uint8, mode="r", shape=(n, 3))
    top = np.full(H * W, -np.inf, dtype=np.float32)
    cell, t0, nveg = meta["cell"], time.time(), 0
    for s in range(0, n, block):
        P = np.asarray(xyz[s:s + block])
        C = np.asarray(rgb[s:s + block]).astype(np.int16)
        keep = (C[:, 0] - C[:, 2]) >= warm_min        # warm plaster/stone (R-B); foliage here is neutral grey
        nveg += int((~keep).sum())
        P = P[keep]
        ix = (P[:, 0] / cell).astype(np.int64)
        iy = (P[:, 1] / cell).astype(np.int64)
        ok = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
        idx, z = iy[ok] * W + ix[ok], P[ok, 2]
        order = np.argsort(z, kind="stable")          # ascending -> highest written last
        top[idx[order]] = np.maximum(top[idx[order]], z[order])
        print(f"\r[nonveg] {min(s + block, n) / 1e6:6.1f} / {n / 1e6:.0f} M pts", end="", flush=True)
    print(f"\n[nonveg] {nveg / n * 100:.1f}% of points are green vegetation (ignored)")
    top[~np.isfinite(top)] = np.nan
    return top.reshape(H, W) + meta["z0"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cache_dir")
    ap.add_argument("-o", "--out", default="out/full/Outline_nv.dxf")
    ap.add_argument("--reference", default=None)
    ap.add_argument("--tall-h", type=float, default=3.5)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--warm-min", type=int, default=20)
    a = ap.parse_args(argv)
    meta, layers = sw.load_cache(a.cache_dir)
    x0, y0, cell = meta["x0"], meta["y0"], meta["cell"]
    dem, dres = sw.ground_model(layers, cell)
    f = int(round(dres / cell))
    tp = os.path.join(a.cache_dir, f"warm_top_{a.warm_min}.npy")
    if os.path.exists(tp) and not a.rebuild:
        top = np.load(tp)
    else:
        top = nonveg_top(a.cache_dir, meta, warm_min=a.warm_min)
        np.save(tp, top)
    hag = ol.top_height(top - 0.0, dem, f)
    rings, _ = ol.extract_outlines(hag, x0, y0, dres, tall_h=a.tall_h, min_area=60.0)
    ol.write_outline_dxf(a.out, rings)
    print(f"[outline] DXF: {a.out}")
    if a.reference and rings:
        ol.compare_reference(rings, a.reference)


if __name__ == "__main__":
    sys.exit(main())
