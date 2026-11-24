"""
outline.py - exterior outline (plan-view contour) of the built complex, as clean rectilinear polylines.

  1. 'tall structure' raster  : highest point above ground >= tall_h   (roofs + walls, not the ground)
  2. close small gaps, fill the courtyard, drop thin appendages (foliage, poles)
  3. boundary of each large component
  4. regularise: Douglas-Peucker, snap edges to the building's dominant directions, rebuild the
     corners as line intersections, drop tiny jogs.
"""
import math

import contourpy
import numpy as np
from scipy import ndimage as ndi
from shapely.geometry import LineString, Polygon


def foliage_mask(hag, tall_h, win=7, std_max=0.8):
    """trees: tall cells whose neighbouring tall cells vary a lot in height (roofs and walls are smooth)"""
    T = np.nan_to_num(hag, nan=0.0) >= tall_h
    h = np.where(T, np.nan_to_num(hag, nan=0.0), 0.0)
    n = ndi.uniform_filter(T.astype(float), win)
    m1 = ndi.uniform_filter(h, win) / np.maximum(n, 1e-6)
    m2 = ndi.uniform_filter(h * h, win) / np.maximum(n, 1e-6)
    std = np.sqrt(np.maximum(m2 - m1 * m1, 0))
    return T & (std > std_max) & (n > 0.5)


def occupancy(hag, tall_h=3.0, close_it=3, open_it=3, no_foliage=False):
    t = np.nan_to_num(hag, nan=0.0) >= tall_h
    if no_foliage:
        fol = ndi.binary_dilation(foliage_mask(hag, tall_h), iterations=2)
        t &= ~fol
    t = ndi.binary_closing(t, structure=np.ones((3, 3)), iterations=close_it)
    t = ndi.binary_fill_holes(t)
    if open_it:
        t = ndi.binary_opening(t, iterations=open_it)
    return t


def boundaries(mask, x0, y0, res, min_area=40.0):
    lab, n = ndi.label(mask)
    out = []
    H, W = mask.shape
    xs = x0 + (np.arange(W) + 0.5) * res
    ys = y0 + (np.arange(H) + 0.5) * res
    for k in range(1, n + 1):
        m = lab == k
        area = m.sum() * res * res
        if area < min_area:
            continue
        g = contourpy.contour_generator(xs, ys, np.pad(m, 1).astype(float)[1:-1, 1:-1],
                                        name="serial", line_type="Separate")
        lines = g.lines(0.5)
        if lines:
            ring = np.asarray(max(lines, key=len))
            out.append((area, ring))
    out.sort(key=lambda t: -t[0])
    return out


def _dominant(ring, k=2):
    ring = np.asarray(LineString(ring).simplify(0.6).coords)       # long straight edges only
    seg = np.diff(ring, axis=0)
    L = np.hypot(seg[:, 0], seg[:, 1])
    a = np.mod(np.arctan2(seg[:, 1], seg[:, 0]), math.pi / 2)
    h, edges = np.histogram(a, bins=90, range=(0, math.pi / 2), weights=L)
    h = np.convolve(np.r_[h[-2:], h, h[:2]], np.ones(5) / 5, mode="same")[2:-2]
    i = int(np.argmax(h))
    return (edges[i] + edges[i + 1]) / 2


def _wrap(a):
    return (a + math.pi / 2) % math.pi - math.pi / 2


def regularise(ring, base=None, dp_tol=0.40, snap_tol=math.radians(14), min_seg=2.0, merge_tol=0.9, force_rect=False):
    """Rectilinear regularisation: every edge is snapped to the dominant directions when it is close
    to one, short edges (jogs, bumps) are removed, parallel neighbours are merged, and the corners
    are rebuilt as intersections of the fitted lines."""
    ring = np.asarray(ring)
    pts = ring[:-1] if np.allclose(ring[0], ring[-1]) else ring
    if base is None:
        base = _dominant(ring)
    dirs = [base % math.pi, (base + math.pi / 2) % math.pi]
    s = np.asarray(LineString(np.vstack([pts, pts[:1]])).simplify(dp_tol).coords)[:-1]
    if len(s) < 4:
        return np.vstack([s, s[:1]])
    # one fitted line per edge: anchor point, unit direction, weight, direction index (or -1)
    lines = []
    for i in range(len(s)):
        a, b = s[i], s[(i + 1) % len(s)]
        v = b - a
        L = float(np.hypot(*v))
        if L < 1e-9:
            continue
        ang = math.atan2(v[1], v[0]) % math.pi
        k = min((0, 1), key=lambda j: abs(_wrap(ang - dirs[j])))
        if force_rect or abs(_wrap(ang - dirs[k])) <= snap_tol:
            d = np.array([math.cos(dirs[k]), math.sin(dirs[k])])
        else:
            d, k = v / L, -1
        lines.append({"p": (a + b) / 2, "d": d, "w": L, "k": k, "v": i})

    if force_rect:
        # two neighbouring edges with the same direction (a jog): add the perpendicular connector between them
        withc = []
        for i, ln in enumerate(lines):
            withc.append(ln)
            nxt = lines[(i + 1) % len(lines)]
            if ln["k"] == nxt["k"]:
                k2 = 1 - ln["k"]
                withc.append({"p": s[(ln["v"] + 1) % len(s)], "d": np.array([math.cos(dirs[k2]), math.sin(dirs[k2])]),
                              "w": 1e-3, "k": k2, "v": -1})
        lines = withc

    def inter(l1, l2):
        d1, d2 = l1["d"], l2["d"]
        den = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(den) < 1e-3:
            return None
        dp = l2["p"] - l1["p"]
        t = (dp[0] * d2[1] - dp[1] * d2[0]) / den
        return l1["p"] + d1 * t

    def offset(l1, l2):                       # perpendicular distance between two parallel lines
        dp = l2["p"] - l1["p"]
        return abs(l1["d"][0] * dp[1] - l1["d"][1] * dp[0])

    def merge(l1, l2):
        w = l1["w"] + l2["w"]
        d = l1["d"] if l1["d"] @ l2["d"] > 0 else -l2["d"]
        d = l1["d"]
        p = (l1["p"] * l1["w"] + l2["p"] * l2["w"]) / w
        return {"p": p, "d": d, "w": w, "k": l1["k"]}

    def verts(ls):
        out = []
        for i in range(len(ls)):
            q = inter(ls[i - 1], ls[i])
            out.append(q if q is not None else ls[i]["p"])
        return np.array(out)

    for _ in range(200):
        n = len(lines)
        if n < 4:
            break
        # merge parallel neighbours that are (almost) on the same line
        did = False
        for i in range(n):
            l1, l2 = lines[i], lines[(i + 1) % n]
            if l1["k"] >= 0 and l1["k"] == l2["k"] and offset(l1, l2) <= merge_tol:
                lines[i] = merge(l1, l2)
                del lines[(i + 1) % n if (i + 1) % n > i else 0]
                did = True
                break
        if did:
            continue
        V = verts(lines)
        el = np.hypot(*(np.roll(V, -1, axis=0) - V).T)          # edge i runs V[i] -> V[i+1]
        j = int(np.argmin(el))
        if el[j] >= min_seg:
            break
        # the shortest edge is line j; remove it, its neighbours then meet (or merge if parallel)
        del lines[j]
        n = len(lines)
        a, b = lines[(j - 1) % n], lines[j % n]
        if a["k"] >= 0 and a["k"] == b["k"] and offset(a, b) <= 2.5:
            lines[(j - 1) % n] = merge(a, b)
            del lines[j % n if j % n != (j - 1) % n else 0]
    V = verts(lines)
    # drop duplicate / collinear vertices
    out = []
    for i in range(len(V)):
        a, b, c = V[i - 1], V[i], V[(i + 1) % len(V)]
        cr = abs((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]))
        if np.hypot(*(b - a)) > 0.05 and cr > 0.05:
            out.append(b)
    V = np.array(out) if len(out) >= 3 else V
    return np.vstack([V, V[:1]])


def smooth_ring(ring, r=0.6):
    """morphological opening + closing of the polygon: removes thin spurs/bumps and fills hairline notches"""
    P = Polygon(ring)
    P = P if P.is_valid else P.buffer(0)
    Q = P.buffer(-r).buffer(r)                       # opening: spurs narrower than 2r disappear
    if Q.is_empty:
        Q = P
    Q = Q.buffer(r).buffer(-r)                       # closing: gaps narrower than 2r are filled
    if Q.geom_type == "MultiPolygon":
        Q = max(Q.geoms, key=lambda g: g.area)
    if Q.is_empty or Q.geom_type != "Polygon":
        return np.asarray(ring)
    return np.asarray(Q.exterior.coords)


def top_height(zmax_fine, dem, f):
    """highest point above ground per DEM cell (f x f fine cells)"""
    H, W = dem.shape
    zm = np.full((H * f, W * f), np.nan, np.float32)
    zm[:zmax_fine.shape[0], :zmax_fine.shape[1]] = zmax_fine
    with np.errstate(all="ignore"):
        top = np.nanmax(zm.reshape(H, f, W, f), axis=(1, 3))
    return top - dem


def _valid(ring):
    P = Polygon(ring)
    return P.is_valid and P.area > 0


import json as _json
import os as _os

PARAMS_PATH = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "params.json")


def load_params():
    """settings tuned by train.py (empty dict if not trained yet)"""
    try:
        return _json.load(open(PARAMS_PATH))
    except Exception:
        return {}


def extract_outlines(hag, x0, y0, res, tall_h=3.0, close_it=3, open_it=3, min_area=40.0, no_foliage=False,
                     rect=True, smooth_r=0.6, reg=None, log=print):
    reg_kw, reg = reg, None
    m = occupancy(hag, tall_h, close_it, open_it, no_foliage)
    comps = boundaries(m, x0, y0, res, min_area)
    if not comps:
        return [], m
    base = _dominant(comps[0][1])
    out, fixed = [], 0
    for area, ring in comps:
        ring = smooth_ring(ring, smooth_r) if smooth_r else ring
        reg = regularise(ring, base=base, force_rect=rect, **(reg_kw or {}))
        if not _valid(reg):                       # self-intersection: retry coarser, then fall back
            for ms in (2.6, 3.4):
                reg = regularise(ring, base=base, min_seg=ms, merge_tol=1.2, force_rect=rect)
                if _valid(reg):
                    break
            else:
                reg = regularise(ring, base=base)             # un-forced
                if not _valid(reg):
                    reg = np.asarray(LineString(ring).simplify(0.5).coords)
            fixed += 1
        if Polygon(reg).area >= min_area * 0.5:
            out.append(reg)
    log(f"[outline] {len(out)} outline(s) (largest {comps[0][0]:.0f} m2), main direction "
        f"{math.degrees(base):.1f} deg, vertices {sum(len(o) for o in out)}, repaired {fixed}")
    return out, m


def write_outline_dxf(path, rings, layer="Murs", walls=None):
    """plain 2D drawing like a hand-drawn plan: closed polylines on layer 'Murs' (+ the usual extra layers)"""
    import ezdxf
    doc = ezdxf.new("R2018", setup=True)
    doc.units = ezdxf.units.M
    for name, color in (("Murs", 7), ("mur bas", 130), ("Zellige", 6)):
        doc.layers.add(name, color=color)
    msp = doc.modelspace()
    for ring in rings:
        r = np.asarray(ring)
        pts = r[:-1] if np.allclose(r[0], r[-1]) else r
        msp.add_lwpolyline([(float(x), float(y)) for x, y in pts], close=True, dxfattribs={"layer": layer})
    if walls:
        doc.layers.add("Murs_interieurs", color=5)
        for kind, a, b, _h in walls:
            lay = "Murs_interieurs" if kind == "tall" else "mur bas"
            msp.add_lwpolyline([(float(a[0]), float(a[1])), (float(b[0]), float(b[1]))],
                               dxfattribs={"layer": lay})
    doc.saveas(path)
    return doc


def compare_reference(rings, ref_dxf, log=print):
    """overlap + edge distance between my outline (largest ring) and a reference DXF's longest polyline"""
    import ezdxf
    d = ezdxf.readfile(ref_dxf)
    polys = [np.array(list(e.get_points("xy"))) for e in d.modelspace() if e.dxftype() == "LWPOLYLINE"]
    ref = max(polys, key=lambda q: len(q))
    refL, refP = LineString(ref), Polygon(ref)
    refP = refP if refP.is_valid else refP.buffer(0)
    def _poly(r):
        q = Polygon(r)
        return q if q.is_valid else q.buffer(0)
    mine = max(rings, key=lambda r: _poly(r).intersection(refP).area)      # the outline that overlaps the reference most
    P = _poly(mine)
    iou = P.intersection(refP).area / P.union(refP).area
    b = LineString(mine)
    dm = np.array([refL.distance(b.interpolate(i / 600, normalized=True)) for i in range(600)])
    dr = np.array([b.distance(refL.interpolate(i / 600, normalized=True)) for i in range(600)])
    log(f"[reference] overlap (IoU) {iou * 100:.1f}%  | my edge -> reference: median {np.median(dm):.2f} m, "
        f"90% within {np.percentile(dm, 90):.1f} m | reference -> my edge: median {np.median(dr):.2f} m, "
        f"90% within {np.percentile(dr, 90):.1f} m | vertices {len(mine)} (reference {len(ref)})")
    return iou
