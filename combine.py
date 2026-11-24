"""
combine.py - best of both methods.

  shape   : exterior outline from the clipped cloud (roof/tall-structure based, robust, closes the complex)
  detail  : full-cloud horizontal slice (continuous wall faces) used to
            1. move every outline edge onto the real wall face where a wall is visible (snapping)
            2. add the inner wall lines that lie inside / right next to the complex
"""
import argparse
import math
import os
import sys

import ezdxf
import numpy as np
from scipy import ndimage as ndi
from shapely.geometry import LineString, Polygon

import outline as ol
import slice_walls as sw


def ring_area(r):
    x, y = r[:, 0], r[:, 1]
    return 0.5 * float(np.dot(x[:-1], y[1:]) - np.dot(x[1:], y[:-1]))


def snap_ring(ring, cells, reach=1.2, min_cov=0.35, face="outer", min_cells=8, log=print):
    """shift each edge of the ring onto the wall face found in the slice cells (N x 2 array of cell centres)"""
    R = np.asarray(ring)
    if np.allclose(R[0], R[-1]):
        R = R[:-1]
    n = len(R)
    ccw = ring_area(np.vstack([R, R[:1]])) > 0
    lines, snapped, shifts, sup_len, tot_len = [], 0, [], 0.0, 0.0
    for i in range(n):
        a, b = R[i], R[(i + 1) % n]
        v = b - a
        L = float(np.hypot(*v))
        d = v / L
        nrm = np.array([d[1], -d[0]]) if ccw else np.array([-d[1], d[0]])      # outward normal
        tot_len += L
        line = {"p": a.copy(), "d": d, "n": nrm, "L": L, "moved": 0.0}
        if L >= 1.5 and len(cells):
            rel = cells - a
            s = rel @ d
            t = rel @ nrm
            m = (s >= 0) & (s <= L) & (np.abs(t) <= reach)
            if m.sum() >= min_cells:
                s, t = s[m], t[m]
                bins = np.arange(-reach, reach + 0.05, 0.05)
                h, _ = np.histogram(t, bins=bins)
                h = np.convolve(h, np.ones(3), mode="same")
                best = None
                for k in np.argsort(-h)[:6]:
                    if h[k] < 3:
                        break
                    tc = (bins[k] + bins[k + 1]) / 2
                    cl = np.abs(t - tc) <= 0.35
                    sb = np.unique(np.floor(s[cl] / 0.5).astype(int))
                    cov = len(sb) / max(1, math.ceil(L / 0.5))
                    if cov >= min_cov and (best is None or cov > best[0]):
                        best = (cov, tc, cl)
                if best:
                    cov, tc, cl = best
                    tn = float(np.percentile(t[cl], 95)) if face == "outer" else float(np.median(t[cl]))
                    line["p"] = a + nrm * tn
                    line["moved"] = tn
                    snapped += 1
                    shifts.append(abs(tn))
                    sup_len += L
        lines.append(line)

    def inter(l1, l2):
        d1, d2 = l1["d"], l2["d"]
        den = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(den) < 1e-3:
            return None
        dp = l2["p"] - l1["p"]
        return l1["p"] + d1 * ((dp[0] * d2[1] - dp[1] * d2[0]) / den)

    out = []
    for i in range(n):
        l1, l2 = lines[i - 1], lines[i]
        q = inter(l1, l2)
        orig = R[i]
        if q is None or np.hypot(*(q - orig)) > 2.5:
            q = orig + 0.5 * (l1["n"] * l1["moved"] + l2["n"] * l2["moved"])     # fall back: shift the vertex
        out.append(q)
    out = np.array(out)
    log(f"[snap] {snapped}/{n} edges moved onto a visible wall face ({sup_len / tot_len * 100:.0f}% of the outline length), "
        f"mean shift {np.mean(shifts) if shifts else 0:.2f} m, max {max(shifts) if shifts else 0:.2f} m")
    return np.vstack([out, out[:1]])


def load_outlines(dxf, min_area=150.0):
    d = ezdxf.readfile(dxf)
    rings = []
    for e in d.modelspace().query("LWPOLYLINE"):
        if e.dxf.layer != "Murs":
            continue
        r = np.array(list(e.get_points("xy")))
        r = np.vstack([r, r[:1]])
        if abs(ring_area(r)) >= min_area:
            rings.append(r)
    return rings


def inner_walls(wall, x0, y0, cell, outline, base, log=print):
    """wall faces inside / next to the complex, away from the outline itself, long and thin only"""
    P = Polygon(outline).buffer(0)
    H, W = wall.shape
    # keep wall cells inside the complex (+1.5 m) and farther than 0.7 m from the outline
    yy, xx = np.mgrid[0:H, 0:W]
    minx, miny, maxx, maxy = P.bounds
    r0, r1 = int((miny - 3 - y0) / cell), int((maxy + 3 - y0) / cell)
    c0, c1 = int((minx - 3 - x0) / cell), int((maxx + 3 - x0) / cell)
    sub = np.zeros_like(wall)
    sub[r0:r1, c0:c1] = wall[r0:r1, c0:c1]
    from skimage.draw import polygon as rpoly
    inside = np.zeros_like(wall)
    pb = P.buffer(1.5)
    ex = np.array(pb.exterior.coords)
    rr, cc = rpoly((ex[:, 1] - y0) / cell - 0.5, (ex[:, 0] - x0) / cell - 0.5, wall.shape)
    inside[rr, cc] = True
    edge = np.zeros_like(wall)
    ls = np.array(P.exterior.coords)
    for a, b in zip(ls[:-1], ls[1:]):
        k = max(int(np.hypot(*(b - a)) / (cell / 2)), 2)
        px = np.linspace(a[0], b[0], k)
        py = np.linspace(a[1], b[1], k)
        edge[np.clip(((py - y0) / cell).astype(int), 0, H - 1), np.clip(((px - x0) / cell).astype(int), 0, W - 1)] = True
    near_edge = ndi.binary_dilation(edge, iterations=int(0.7 / cell))
    m = sub & inside & ~near_edge
    m = ndi.binary_opening(m, structure=np.ones((2, 2)))
    lab, n = ndi.label(m, structure=np.ones((3, 3)))
    keep = np.zeros(n + 1, bool)
    objs = ndi.find_objects(lab)
    for k, sl in enumerate(objs, start=1):
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        L = max(h, w) * cell
        area = (lab[sl] == k).sum() * cell * cell
        # long & thin: a wall, not a bush/car/pole
        if area >= 0.8 and (area / max(L, 1e-3)) <= 0.9 and L >= 3.0:
            keep[k] = True
    m = keep[lab]
    rings = sw.vectorise(m, x0, y0, cell, base, min_ring=2.0, log=lambda *a: None)
    rings = [r for r in rings if LineString(r).length >= 4.0]
    log(f"[inner] {len(rings)} inner wall outlines kept")
    return rings


def compare(rings, ref_dxf, label):
    return ol.compare_reference(rings, ref_dxf, log=lambda s: print(f"  {label}: {s}"))


def write(path, outlines, inner):
    doc = ezdxf.new("R2018", setup=True)
    doc.units = ezdxf.units.M
    for name, color in (("Murs", 7), ("Murs_interieurs", 5), ("mur bas", 130), ("Zellige", 6)):
        doc.layers.add(name, color=color)
    msp = doc.modelspace()
    for lay, rings in (("Murs", outlines), ("Murs_interieurs", inner)):
        for r in rings:
            r = np.asarray(r)
            pts = r[:-1] if np.allclose(r[0], r[-1]) else r
            msp.add_lwpolyline([(float(x), float(y)) for x, y in pts], close=True, dxfattribs={"layer": lay})
    doc.saveas(path)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--outline", default="out/v2/Outline.dxf")
    ap.add_argument("--slices", default="out/full")
    ap.add_argument("-o", "--out", default="out/v2/Plan_murs_combine.dxf")
    ap.add_argument("--reference", default=None)
    ap.add_argument("--face", choices=["outer", "center", "both"], default="both")
    a = ap.parse_args(argv)

    meta, _ = sw.load_cache(a.slices)
    x0, y0, cell = meta["x0"], meta["y0"], meta["cell"]
    z = np.load(os.path.join(a.slices, "slices.npz"))
    S = {k: z["S_" + k] for k in sw.BANDS}
    wall, _low = sw.wall_masks(S, None, cell)
    ys, xs = np.nonzero(wall)
    cells = np.column_stack([x0 + (xs + 0.5) * cell, y0 + (ys + 0.5) * cell])
    print(f"[combine] wall-face cells from the slice: {len(cells):,}")

    rings = load_outlines(a.outline)
    main_ring = max(rings, key=lambda r: abs(ring_area(r)))
    base = ol._dominant(main_ring)
    variants = {}
    for face in (["outer", "center"] if a.face == "both" else [a.face]):
        snapped = [snap_ring(r, cells, face=face) if abs(ring_area(r)) > 1500 else r for r in rings]
        variants[face] = snapped
    if a.reference:
        print("[score] against your drawing:")
        print("  before snapping:", end=" ")
        ol.compare_reference(rings, a.reference)
        for face, sr in variants.items():
            print(f"  snapped to {face} face:", end=" ")
            ol.compare_reference(sr, a.reference)
    face = "outer" if "outer" in variants else next(iter(variants))
    final = variants[face]
    big = max(final, key=lambda r: abs(ring_area(r)))
    inner = inner_walls(wall, x0, y0, cell, big, base)
    write(a.out, final, inner)
    print(f"[done] {a.out}  (Murs {len(final)}, Murs_interieurs {len(inner)})")
    return None


if __name__ == "__main__":
    sys.exit(main())
