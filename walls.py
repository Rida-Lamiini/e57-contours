"""
walls.py - wall lines (plan view) from the scan: thin, straight, raised structures.

  1. height above ground per 10 cm cell (highest point - bare-earth model)
  2. thin raised cells (blobs such as roofs / trees removed by a morphological opening), skeleton
  3. probabilistic Hough transform -> straight segments
  4. merge collinear pieces, snap to the building's dominant directions, merge again
  5. classify by median height: tall walls (Murs) / low walls (mur bas); drop short stubs
"""
import math

import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import map_coordinates
from skimage.morphology import disk, skeletonize
from skimage.transform import probabilistic_hough_line


def height_above_ground(zmax_fine, dem, f):
    H, W = zmax_fine.shape
    up = np.kron(dem, np.ones((f, f), dtype=np.float32))[:H, :W]
    return zmax_fine - up


def thin_raised_mask(hag, fine, min_h=0.6, blob_w=1.0):
    M = np.nan_to_num(hag, nan=0.0) >= min_h
    r = max(2, int(round(blob_w / 2 / fine)))
    blobs = ndi.binary_dilation(ndi.binary_opening(M, structure=disk(r)), structure=disk(2))
    return M & ~blobs


def hough_segments(mask, fine, x0, y0, min_len=3.0, gap=0.8, thresh=10, seed=1):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return []
    r0, r1, c0, c1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    sk = skeletonize(mask[r0:r1, c0:c1])
    th = np.linspace(-np.pi / 2, np.pi / 2, 1800, endpoint=False)
    segs = probabilistic_hough_line(sk, threshold=thresh, line_length=int(min_len / fine),
                                    line_gap=int(gap / fine), theta=th, rng=seed)
    return [((x0 + (ca + c0 + .5) * fine, y0 + (ra + r0 + .5) * fine),
             (x0 + (cb + c0 + .5) * fine, y0 + (rb + r0 + .5) * fine)) for (ca, ra), (cb, rb) in segs]


def _fit(P):
    c = P.mean(0)
    _, _, vt = np.linalg.svd(P - c)
    d = vt[0]
    t = (P - c) @ d
    return c, d, t.min(), t.max()


def merge_collinear(segs, ang_tol=math.radians(2.5), off_tol=0.35, gap_tol=2.0):
    items = [{"pts": [np.array(a), np.array(b)]} for a, b in segs if math.dist(a, b) > 0]
    changed = True
    while changed:
        changed = False
        fits = [_fit(np.array(it["pts"])) for it in items]
        order = sorted(range(len(items)), key=lambda i: -(fits[i][3] - fits[i][2]))
        used, new = [False] * len(items), []
        for i in order:
            if used[i]:
                continue
            used[i] = True
            cur = list(items[i]["pts"])
            c, d, t0, t1 = fits[i]
            for j in order:
                if used[j]:
                    continue
                c2, d2, u0, u1 = fits[j]
                if abs(math.atan2(d[0] * d2[1] - d[1] * d2[0], d @ d2)) > ang_tol:
                    continue
                v = c2 - c
                if abs(d[0] * v[1] - d[1] * v[0]) > off_tol:
                    continue
                P2 = c2 + np.outer([u0, u1], d2)
                tt = (P2 - c) @ d
                if max(tt.min() - t1, t0 - tt.max(), 0) > gap_tol:
                    continue
                cur += items[j]["pts"]
                used[j] = True
                changed = True
            new.append({"pts": cur})
        items = new
    out = []
    for it in items:
        c, d, t0, t1 = _fit(np.array(it["pts"]))
        out.append((tuple(c + d * t0), tuple(c + d * t1)))
    return out


def snap(segs, base, tol=math.radians(5)):
    """keep only segments parallel / perpendicular to the building (oblique ones are foliage, poles...)"""
    dirs = [base % math.pi, (base + math.pi / 2) % math.pi]
    out = []
    for a, b in segs:
        ang = math.atan2(b[1] - a[1], b[0] - a[0]) % math.pi
        best = min(dirs, key=lambda d: min(abs(ang - d), math.pi - abs(ang - d)))
        if min(abs(ang - best), math.pi - abs(ang - best)) <= tol:
            c = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
            L = math.dist(a, b) / 2
            u = (math.cos(best), math.sin(best))
            out.append(((c[0] - u[0] * L, c[1] - u[1] * L), (c[0] + u[0] * L, c[1] + u[1] * L)))
    return out


def support_fraction(a, b, near, x0, y0, fine, step=0.2):
    """share of the line that has raised thin cells within a few cm (continuous wall, not scattered noise)"""
    L = math.dist(a, b)
    n = max(int(L / step), 2)
    xs, ys = np.linspace(a[0], b[0], n), np.linspace(a[1], b[1], n)
    r = np.clip(np.round((ys - y0) / fine - 0.5).astype(int), 0, near.shape[0] - 1)
    c = np.clip(np.round((xs - x0) / fine - 0.5).astype(int), 0, near.shape[1] - 1)
    return float(near[r, c].mean())


def classify(segs, hag, x0, y0, fine, near=None, tall=1.8, min_tall=4.0, min_low=6.0, min_support=0.6):
    """median height along each segment -> ('tall'|'low', a, b); short stubs are dropped"""
    out = []
    # highest value within +-1 cell across the wall (the skeleton can be off by a cell) - computed once
    h = ndi.maximum_filter(np.nan_to_num(hag, nan=0.0), size=3)
    for a, b in segs:
        L = math.dist(a, b)
        n = max(int(L / 0.3), 2)
        xs, ys = np.linspace(a[0], b[0], n), np.linspace(a[1], b[1], n)
        r, c = (ys - y0) / fine - 0.5, (xs - x0) / fine - 0.5
        v = map_coordinates(h, [r, c], order=0, mode="nearest")
        med = float(np.median(v))
        if near is not None and support_fraction(a, b, near, x0, y0, fine) < min_support:
            continue
        if med >= tall and L >= min_tall:
            out.append(("tall", a, b, med))
        elif 0.5 <= med < tall and L >= min_low:
            out.append(("low", a, b, med))
    return out


def extract_walls(hag, fine, x0, y0, base, log=print):
    mask = thin_raised_mask(hag, fine)
    near = ndi.binary_dilation(mask, structure=disk(3))          # within ~0.3 m of a raised thin cell
    segs = hough_segments(mask, fine, x0, y0)
    n0 = len(segs)
    segs = merge_collinear(segs)
    segs = snap(segs, base)
    segs = merge_collinear(segs, off_tol=0.25)
    cl = classify(segs, hag, x0, y0, fine, near=near)
    nt = sum(1 for c in cl if c[0] == "tall")
    L = sum(math.dist(c[1], c[2]) for c in cl)
    log(f"[walls] {n0} raw pieces -> {len(cl)} wall lines ({nt} tall, {len(cl) - nt} low), {L:.0f} m")
    return cl
