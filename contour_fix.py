"""
contour_fix.py - automatic clean-up of raw contour lines.

What it does (in this order), per contour level:
  1. drop edge artefacts   : short lines that just hug the data/mask boundary (whiskers)
  2. join broken lines     : end points of the same level within --join-gap, only when the
                             directions agree (no zig-zag bridging, never through buildings)
  3. adaptive smoothing    : each line is smoothed as much as possible while it stays within a
                             vertical tolerance of the real surface (so flat-area wobble is removed,
                             real bends on slopes are kept)
  4. straightening         : near-straight runs become true straight segments (Douglas-Peucker)
  5. drop tiny leftovers, resolve crossings between levels, report fidelity metrics
"""
import math

import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import map_coordinates, gaussian_filter1d
from shapely.geometry import LineString
from shapely.strtree import STRtree


# ------------------------------------------------------------------ helpers
def _length(p):
    return float(np.hypot(*np.diff(p, axis=0).T).sum()) if len(p) > 1 else 0.0


def _resample(p, step, closed):
    seg = np.hypot(*np.diff(p, axis=0).T)
    s = np.concatenate([[0], np.cumsum(seg)])
    L = s[-1]
    if L < step * 2:
        return p
    n = max(int(round(L / step)), 3)
    t = np.linspace(0, L, n + (0 if closed else 1), endpoint=not closed)
    out = np.column_stack([np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])])
    if closed:
        out = np.vstack([out, out[:1]])
    else:
        out[0], out[-1] = p[0], p[-1]
    return out


def _smooth(p, sigma_m, step, closed):
    """Gaussian smoothing along the curve; open lines keep their end points."""
    sg = sigma_m / step
    if closed:
        q = p[:-1]
        sm = np.column_stack([gaussian_filter1d(q[:, k], sg, mode="wrap") for k in (0, 1)])
        return np.vstack([sm, sm[:1]])
    sm = np.column_stack([gaussian_filter1d(p[:, k], sg, mode="nearest") for k in (0, 1)])
    # blend back to the original near the ends so the line does not shrink / detach
    s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(p, axis=0).T))])
    d = np.minimum(s, s[-1] - s)
    w = np.clip(d / (2.0 * sigma_m), 0, 1)[:, None]
    return p * (1 - w) + sm * w


class Surface:
    def __init__(self, dem, x0, y0, res, mask, bmask):
        self.dem, self.x0, self.y0, self.res = dem, x0, y0, res
        self.edge = ndi.distance_transform_edt(~mask) * res          # distance to masked area
        self.bmask = bmask

    def _rc(self, p):
        return [(p[:, 1] - self.y0) / self.res - 0.5, (p[:, 0] - self.x0) / self.res - 0.5]

    def z(self, p):
        return map_coordinates(self.dem, self._rc(p), order=1, mode="nearest")

    def edge_dist(self, p):
        return map_coordinates(self.edge, self._rc(p), order=1, mode="nearest")

    def in_building(self, p):
        if self.bmask is None:
            return np.zeros(len(p), bool)
        r, c = self._rc(p)
        r = np.clip(np.round(r).astype(int), 0, self.bmask.shape[0] - 1)
        c = np.clip(np.round(c).astype(int), 0, self.bmask.shape[1] - 1)
        return self.bmask[r, c]


# ------------------------------------------------------------------ steps
def drop_edge_artifacts(lines, surf, edge_tol=0.6, frac=0.7, max_len=25.0):
    keep, dropped = [], 0
    for lv, p, closed in lines:
        L = _length(p)
        if L < max_len and not closed:
            near = (surf.edge_dist(p) < edge_tol).mean()
            if near > frac:
                dropped += 1
                continue
        keep.append((lv, p, closed))
    return keep, dropped


def _tangent(p, at_end, span=1.5):
    # outward tangent: from a point ~span m inside the line to the chosen end point
    seg = np.hypot(*np.diff(p, axis=0).T)
    s = np.concatenate([[0], np.cumsum(seg)])
    if at_end:
        i = int(np.searchsorted(s, s[-1] - span))
        v = p[-1] - p[min(i, len(p) - 2)]
    else:
        i = int(np.searchsorted(s, span))
        v = p[0] - p[min(max(i, 1), len(p) - 1)]
    n = np.hypot(*v)
    return v / n if n > 1e-9 else v


def join_lines(lines, surf, gap=4.0, max_angle=50.0, step=0.2):
    cosmax = math.cos(math.radians(max_angle))
    by_level = {}
    for lv, p, closed in lines:
        by_level.setdefault(lv, []).append((p, closed))
    out, joins = [], 0
    for lv, items in by_level.items():
        closed_ones = [(lv, p, True) for p, c in items if c]
        open_ = [p for p, c in items if not c]
        merged = True
        while merged and len(open_) > 0:
            merged = False
            ends = []
            for i, p in enumerate(open_):
                ends.append((i, 0, p[0], _tangent(p, False)))
                ends.append((i, 1, p[-1], _tangent(p, True)))
            best = None
            for a in range(len(ends)):
                ia, ea, pa, ta = ends[a]
                for b in range(a + 1, len(ends)):
                    ib, eb, pb, tb = ends[b]
                    v = pb - pa
                    d = float(np.hypot(*v))
                    if d > gap or (best and d >= best[0]):
                        continue
                    if ia == ib:
                        if _length(open_[ia]) < 3 * max(d, 1.0):
                            continue
                    elif d > 0.3:
                        u = v / d
                        if np.dot(ta, u) < cosmax or np.dot(tb, -u) < cosmax:
                            continue
                    elif d <= 0.3:
                        pass
                    # bridge must not run through a building / masked area
                    nb = max(int(d / step), 2)
                    br = np.column_stack([np.linspace(pa[0], pb[0], nb), np.linspace(pa[1], pb[1], nb)])
                    if surf.in_building(br).mean() > 0.2:
                        continue
                    best = (d, a, b)
            if best:
                _, a, b = best
                ia, ea, pa, _ = ends[a]
                ib, eb, pb, _ = ends[b]
                if ia == ib:
                    p = open_[ia]
                    closed_ones.append((lv, np.vstack([p, p[:1]]), True))
                    open_.pop(ia)
                else:
                    A, B = open_[ia], open_[ib]
                    A = A if ea == 1 else A[::-1]       # A ends at the joining point
                    B = B if eb == 0 else B[::-1]       # B starts at the joining point
                    joined = np.vstack([A, B])
                    for k in sorted((ia, ib), reverse=True):
                        open_.pop(k)
                    open_.append(joined)
                joins += 1
                merged = True
        out.extend(closed_ones)
        out.extend((lv, p, False) for p in open_)
    return out, joins


def smooth_and_straighten(lines, surf, tol_z=0.08, h_tol=0.25, max_shift=1.5, step=0.2, straight_tol=0.04,
                          sigmas=(0.4, 0.8, 1.3, 2.0, 3.0, 4.5)):
    res, stats = [], {"sigma": [], "err95": [], "verts_before": 0, "verts_after": 0}
    for lv, p, closed in lines:
        stats["verts_before"] += len(p)
        pr = _resample(p, step, closed)
        best, best_s = pr, 0.0
        for s in sigmas:
            if s > _length(pr) / 4 and not closed:
                break
            q = _smooth(pr, s, step, closed)
            disp = np.hypot(*(q - pr).T)
            dz = np.abs(surf.z(q) - lv)
            # a vertex is fine if it stays within the height tolerance OR moved only a little sideways
            good = (dz <= tol_z) | (disp <= h_tol)
            ok = good.mean() >= 0.95 and disp.max() <= max_shift
            if not ok:
                break
            best, best_s = q, s
        if best_s == 0.0:
            best = _smooth(pr, 0.4, step, closed) if _length(pr) > 3 else pr
        ls = LineString(best)
        simp = np.asarray(ls.simplify(straight_tol, preserve_topology=False).coords)
        if closed:
            simp[-1] = simp[0]
            if len(simp) < 4:
                simp = best
        stats["sigma"].append(best_s)
        stats["err95"].append(float(np.percentile(np.abs(surf.z(simp) - lv), 95)))
        stats["shift"] = stats.get("shift", []) + [float(np.hypot(*(simp[:1] - p[:1]).T).max())]
        stats["verts_after"] += len(simp)
        res.append((lv, simp, closed))
    return res, stats


def resolve_crossings(lines, short=12.0):
    """Contours of different levels must not cross. Remove the short offender, report the rest."""
    geoms = [LineString(p) for _, p, _ in lines]
    tree = STRtree(geoms)
    bad = {}
    for i, g in enumerate(geoms):
        for j in tree.query(g):
            if j <= i or lines[i][0] == lines[j][0]:
                continue
            if g.crosses(geoms[j]):
                bad.setdefault(i, set()).add(j)
                bad.setdefault(j, set()).add(i)
    drop = set()
    for i in sorted(bad, key=lambda k: geoms[k].length):
        if geoms[i].length < short and any(j not in drop for j in bad[i]):
            drop.add(i)
    remaining = sum(1 for i in bad if i not in drop and any(j not in drop for j in bad[i])) // 2
    return [l for k, l in enumerate(lines) if k not in drop], len(drop), remaining


# ------------------------------------------------------------------ driver
def roughness(lines, step=1.0):
    """mean absolute turning angle (deg) per metre -> lower = smoother / straighter"""
    tot, L = 0.0, 0.0
    for _, p, closed in lines:
        q = _resample(p, step, closed)
        if len(q) < 4:
            continue
        v = np.diff(q, axis=0)
        a = np.arctan2(v[:, 1], v[:, 0])
        d = np.abs((np.diff(a) + np.pi) % (2 * np.pi) - np.pi)
        tot += np.degrees(d).sum()
        L += _length(q)
    return tot / L if L else 0.0


def fix_contours(contours, dem, mask, bmask, x0, y0, res, tol_z=0.08, join_gap=4.0,
                 min_len=8.0, log=print):
    surf = Surface(dem, x0, y0, res, mask, bmask)
    n0, rough0 = len(contours), roughness(contours)
    lines, d_edge = drop_edge_artifacts(contours, surf)
    lines, joins = join_lines(lines, surf, gap=join_gap)
    lines, st = smooth_and_straighten(lines, surf, tol_z=tol_z)
    lines = [l for l in lines if _length(l[1]) >= min_len]
    lines, d_cross, left = resolve_crossings(lines)
    rough1 = roughness(lines)
    sg = np.array(st["sigma"]) if st["sigma"] else np.array([0.0])
    log(f"[fix] {n0} lines -> {len(lines)}  | edge artefacts removed {d_edge}, joined {joins} gaps, "
        f"crossing offenders removed {d_cross}, crossings left {left}")
    log(f"[fix] roughness {rough0:.1f} -> {rough1:.1f} deg/m  | vertices {st['verts_before']} -> {st['verts_after']}  "
        f"| smoothing used median {np.median(sg):.1f} m  | height fidelity (95th pct) median "
        f"{np.median(st['err95']) * 100 if st['err95'] else 0:.1f} cm (tolerance {tol_z * 100:.0f} cm)")
    return lines
