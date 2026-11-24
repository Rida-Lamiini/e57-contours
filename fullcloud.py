"""
fullcloud.py - pass 1 over a multi-scan E57: lowest-points grid (ground model input) + a compact local
point cache (XYZ float32 relative to an origin + RGB uint8) so later passes never touch the network again.

  python fullcloud.py <file.e57> <cache_dir> [--max-points N]   (N only for quick tests)
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np


def update_layers(low, zmax, cnt, idx, z):
    """low: (3, H*W) three lowest heights per cell, zmax: highest, cnt: points per cell (all in place)"""
    order = np.lexsort((z, idx))
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


def to_u8(a):
    a = np.asarray(a)
    if a.dtype.kind == "f":
        mx = float(a.max()) if a.size else 1.0
        a = a * (255.0 if mx <= 1.0 else 1.0)
    return np.clip(a, 0, 255).astype(np.uint8)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("e57")
    ap.add_argument("cache_dir")
    ap.add_argument("--cell", type=float, default=0.10)
    ap.add_argument("--chunk", type=int, default=2_000_000)
    ap.add_argument("--max-points", type=int, default=0, help="stop after N points per scan (tests)")
    ap.add_argument("--scans", type=int, nargs="*", default=None)
    a = ap.parse_args(argv)
    import pye57

    os.makedirs(a.cache_dir, exist_ok=True)
    f = pye57.E57(a.e57)
    scans = a.scans if a.scans is not None else list(range(f.scan_count))
    hs = {i: f.get_header(i) for i in scans}
    bx = np.array([[h.cartesianBounds[k].value() for k in
                    ("xMinimum", "xMaximum", "yMinimum", "yMaximum", "zMinimum", "zMaximum")] for h in hs.values()])
    x0, y0, z0 = math.floor(bx[:, 0].min() - 1), math.floor(bx[:, 2].min() - 1), math.floor(bx[:, 4].min() - 1)
    cell = a.cell
    W = int(math.ceil((bx[:, 1].max() + 1 - x0) / cell)) + 1
    H = int(math.ceil((bx[:, 3].max() + 1 - y0) / cell)) + 1
    total = sum(h.point_count for h in hs.values())
    print(f"[full] {len(scans)} scans, {total:,} points, grid {W} x {H} @ {cell} m "
          f"(~{W * H * 18 / 1e6:.0f} MB), origin {x0},{y0},{z0}", flush=True)
    low = np.full((3, H * W), np.inf, dtype=np.float32)
    zmax = np.full(H * W, -np.inf, dtype=np.float32)
    cnt = np.zeros(H * W, dtype=np.uint16)

    fx = open(os.path.join(a.cache_dir, "xyz.f32"), "wb")
    fc = open(os.path.join(a.cache_dir, "rgb.u8"), "wb")
    scan_of = open(os.path.join(a.cache_dir, "scan.u8"), "wb")
    names = ["cartesianX", "cartesianY", "cartesianZ", "colorRed", "colorGreen", "colorBlue"]
    done, t0 = 0, time.time()
    for si in scans:
        h = hs[si]
        have_rgb = all(n in h.point_fields for n in names[3:])
        flds = names if have_rgb else names[:3]
        data, bufs = f.make_buffers(flds, a.chunk)
        rd = h.points.reader(bufs)
        got = 0
        while True:
            n = rd.read()
            if n == 0:
                break
            if a.max_points and got + n > a.max_points:
                n = a.max_points - got
            X, Y, Z = (data[k][:n] for k in names[:3])
            ok = np.isfinite(Z)
            ix = ((X - x0) / cell).astype(np.int64)
            iy = ((Y - y0) / cell).astype(np.int64)
            ok &= (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
            idx = iy[ok] * W + ix[ok]
            if len(idx):
                update_layers(low, zmax, cnt, idx, Z[ok].astype(np.float32))
            xyz = np.column_stack([X[ok] - x0, Y[ok] - y0, Z[ok] - z0]).astype(np.float32)
            xyz.tofile(fx)
            if have_rgb:
                np.column_stack([to_u8(data[k][:n][ok]) for k in names[3:]]).tofile(fc)
            else:
                np.zeros((int(ok.sum()), 3), np.uint8).tofile(fc)
            np.full(int(ok.sum()), si, np.uint8).tofile(scan_of)
            got += n
            done += n
            el = time.time() - t0
            print(f"\r[full] scan {si:2d}  {done / 1e6:7.1f} / {total / 1e6:.0f} M pts  {done / el / 1e6:4.2f} M/s  "
                  f"eta {max(total - done, 0) / max(done / el, 1) / 60:4.1f} min", end="", flush=True)
            if a.max_points and got >= a.max_points:
                break
        rd.close()
    print()
    for fh in (fx, fc, scan_of):
        fh.close()
    low[~np.isfinite(low)] = np.nan
    zmax[~np.isfinite(zmax)] = np.nan
    np.savez_compressed(os.path.join(a.cache_dir, "layers.npz"), low=low.reshape(3, H, W),
                        zmax=zmax.reshape(H, W), cnt=cnt.reshape(H, W), org=np.array([x0, y0, cell]), z0=z0)
    json.dump({"x0": x0, "y0": y0, "z0": z0, "cell": cell, "W": W, "H": H, "points": done,
               "e57": a.e57}, open(os.path.join(a.cache_dir, "meta.json"), "w"))
    print(f"[full] done: {done:,} points cached, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
