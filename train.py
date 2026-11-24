"""
train.py - tune the outline pipeline to YOUR drawings.

  truth  : your hand drawing (--truth) plus every DXF the review tool saved into the 'training' folder
  tuned  : tall_h, close_it, open_it, smooth_r, min_seg, merge_tol, dp_tol
  score  : area overlap (IoU) - 0.02 x mean edge distance (m) - 0.0005 x |vertices - truth vertices|
  result : params.json  (read automatically by 'e57_to_contours.py --mode outline')

This is calibration (a handful of numbers), not deep learning: it cannot learn new shapes such as rooms,
but it makes the automatic run start as close as possible to what you keep correcting.
"""
import argparse
import glob
import json
import os
import random
import sys
import time

import ezdxf
import numpy as np
from shapely.geometry import LineString, Polygon

import e57_to_contours as core
import outline as ol

HERE = os.path.dirname(os.path.abspath(__file__))
SPACE = {                       # name: (low, high, integer?)
    "tall_h": (2.2, 5.0, False), "close_it": (1, 6, True), "open_it": (0, 6, True),
    "smooth_r": (0.2, 1.4, False), "min_seg": (1.0, 4.0, False), "merge_tol": (0.4, 1.6, False),
    "dp_tol": (0.2, 0.8, False),
}
DEFAULT = {"tall_h": 3.5, "close_it": 3, "open_it": 3, "smooth_r": 0.6, "min_seg": 2.0, "merge_tol": 0.9, "dp_tol": 0.4}


def load_truth(paths):
    out = []
    for p in paths:
        d = ezdxf.readfile(p)
        polys = [np.array(list(e.get_points("xy"))) for e in d.modelspace()
                 if e.dxftype() == "LWPOLYLINE" and e.dxf.layer in ("Murs", "Murs_interieurs", "0") and len(e) >= 4]
        if not polys:
            continue
        ring = max(polys, key=lambda q: abs(Polygon(q).area))
        P = Polygon(ring)
        P = P if P.is_valid else P.buffer(0)
        out.append((os.path.basename(p), ring, P))
    return out


def score_rings(rings, truths):
    tot, det = [], []
    for name, tr, TP in truths:
        # my ring = the one overlapping this truth most
        def poly(r):
            q = Polygon(r)
            return q if q.is_valid else q.buffer(0)
        mine = max(rings, key=lambda r: poly(r).intersection(TP).area)
        P = poly(mine)
        iou = P.intersection(TP).area / max(P.union(TP).area, 1e-9)
        a, b = LineString(mine), LineString(tr)
        dm = np.median([b.distance(a.interpolate(i / 300, normalized=True)) for i in range(300)])
        dr = np.median([a.distance(b.interpolate(i / 300, normalized=True)) for i in range(300)])
        s = iou - 0.02 * 0.5 * (dm + dr) - 0.0005 * abs(len(mine) - len(tr))
        tot.append(s)
        det.append((iou, dm, dr, len(mine), len(tr)))
    return float(np.mean(tot)), det


def evaluate(params, hag, x0, y0, res, truths):
    try:
        rings, _ = ol.extract_outlines(
            hag, x0, y0, res, tall_h=params["tall_h"], close_it=int(params["close_it"]), open_it=int(params["open_it"]),
            smooth_r=params["smooth_r"], log=lambda *a: None,
            reg={"min_seg": params["min_seg"], "merge_tol": params["merge_tol"], "dp_tol": params["dp_tol"]})
        if not rings:
            return -1.0, None
        return score_rings(rings, truths)
    except Exception:
        return -1.0, None


def sample(rng):
    p = {}
    for k, (lo, hi, isint) in SPACE.items():
        v = rng.uniform(lo, hi)
        p[k] = int(round(v)) if isint else round(v, 3)
    return p


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.join(HERE, "out", "v2", "Contour.grid.npz"))
    ap.add_argument("--truth", nargs="*", default=[], help="hand-drawn DXF files")
    ap.add_argument("--random", type=int, default=45, help="random trials")
    ap.add_argument("--refine", type=int, default=2, help="coordinate-refinement rounds")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args(argv)
    paths = list(a.truth) + sorted(glob.glob(os.path.join(HERE, "training", "*.dxf")))
    truths = load_truth(paths)
    if not truths:
        print("no training drawings found (pass --truth or save from the review tool)")
        return 1
    print(f"[train] {len(truths)} training drawing(s): {[t[0] for t in truths]}", flush=True)

    z = np.load(a.cache)
    layers = (z["low"], z["zmax"], z["cnt"])
    x0, y0, fine = [float(v) for v in z["org"]]
    f = 2
    res = fine * f
    fg, _ = core.surface_from_layers(layers, "robust")
    fg = core.remove_low_noise(fg, 0.5, 7)
    dsm = core.coarsen_min(fg, f)
    g = core.progressive_morph_filter(dsm, res, max_window=40.0, slope=0.35)
    dem = core.interpolate_ground(dsm, g, res, decim_m=0.5)
    hag = ol.top_height(layers[1], dem, f)

    t0 = time.time()
    base_s, base_d = evaluate(DEFAULT, hag, x0, y0, res, truths)
    print(f"[train] baseline score {base_s:.4f}  (IoU {np.mean([d[0] for d in base_d]) * 100:.1f}%)", flush=True)
    rng = random.Random(a.seed)
    best, best_s, best_d = dict(DEFAULT), base_s, base_d
    hist = [{"params": dict(DEFAULT), "score": base_s}]
    for i in range(a.random):
        p = sample(rng)
        s, d = evaluate(p, hag, x0, y0, res, truths)
        hist.append({"params": p, "score": s})
        if s > best_s:
            best, best_s, best_d = p, s, d
            print(f"[train] trial {i + 1:3d}: new best {s:.4f}  (IoU {np.mean([x[0] for x in d]) * 100:.1f}%)  "
                  f"{time.time() - t0:.0f}s", flush=True)
    for r in range(a.refine):                       # coordinate refinement around the best
        for k, (lo, hi, isint) in SPACE.items():
            step = (hi - lo) * (0.12 if r == 0 else 0.05)
            for d_ in (-step, step):
                p = dict(best)
                v = min(hi, max(lo, best[k] + d_))
                p[k] = int(round(v)) if isint else round(v, 3)
                if p[k] == best[k]:
                    continue
                s, d = evaluate(p, hag, x0, y0, res, truths)
                hist.append({"params": p, "score": s})
                if s > best_s:
                    best, best_s, best_d = p, s, d
                    print(f"[train] refine {k}={p[k]}: {s:.4f}  (IoU {np.mean([x[0] for x in d]) * 100:.1f}%)", flush=True)
    json.dump(best, open(ol.PARAMS_PATH, "w"), indent=1)
    json.dump(hist, open(os.path.join(HERE, "train_log.json"), "w"))
    print(f"[train] done in {time.time() - t0:.0f}s: score {base_s:.4f} -> {best_s:.4f}")
    for name, d in zip([t[0] for t in truths], best_d or []):
        print(f"  {name}: IoU {d[0] * 100:.1f}%  edge distance {d[1]:.2f}/{d[2]:.2f} m  vertices {d[3]} (truth {d[4]})")
    print(f"[train] best settings saved to {ol.PARAMS_PATH}:", best)


if __name__ == "__main__":
    sys.exit(main())
