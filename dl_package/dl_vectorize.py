"""
dl_vectorize.py - wall probability map -> clean CAD lines (no PyTorch needed).

  wall / low-wall mask  -> skeleton -> centre lines -> simplified -> squared to the building directions
  zellige mask          -> filled areas -> outlines
Output: DXF with layers Murs, mur bas, Zellige.
"""
import math

import contourpy
import ezdxf
import numpy as np
from scipy import ndimage as ndi
from shapely.geometry import LineString
from skimage.morphology import skeletonize

import outline as ol


def clean_mask(mask, cell, min_len=1.5, close_m=0.3):
    m = ndi.binary_closing(mask, structure=np.ones((3, 3)), iterations=max(1, int(round(close_m / cell / 2))))
    lab, n = ndi.label(m, structure=np.ones((3, 3)))
    if n:
        sizes = ndi.sum(m, lab, index=np.arange(1, n + 1)) * cell        # ~ length of a thin wall in metres
        keep = np.zeros(n + 1, bool)
        keep[1:] = sizes >= min_len
        m = keep[lab]
    return m


def trace_skeleton(skel):
    """skeleton pixels -> list of pixel paths between end/junction pixels"""
    sk = skel.astype(np.uint8)
    nb = ndi.convolve(sk, np.ones((3, 3), np.uint8), mode="constant") - sk
    pts = set(map(tuple, np.argwhere(sk)))
    nodes = {p for p in pts if nb[p] != 2}
    seen_edge = set()
    paths = []

    def neighbours(p):
        r, c = p
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                q = (r + dr, c + dc)
                if (dr or dc) and q in pts:
                    yield q

    visited = set()
    for s in nodes:
        for q in neighbours(s):
            if (s, q) in seen_edge:
                continue
            path, prev, cur = [s, q], s, q
            seen_edge.add((s, q))
            while cur not in nodes:
                nxt = [x for x in neighbours(cur) if x != prev and x not in path[-3:]]
                if not nxt:
                    break
                prev, cur = cur, nxt[0]
                path.append(cur)
            seen_edge.add((path[-1], path[-2]))
            paths.append(path)
            visited.update(path)
    # closed loops (no end / junction pixel at all, e.g. a full outline): trace them from any unvisited pixel
    for s in sorted(pts - visited):
        if s in visited:
            continue
        path, prev, cur = [s], None, s
        while True:
            nxt = [x for x in neighbours(cur) if x != prev and x not in visited and x not in path]
            if not nxt:
                break
            prev, cur = cur, nxt[0]
            path.append(cur)
        if len(path) > 2:
            path.append(s)                       # close the loop
        paths.append(path)
        visited.update(path)
    return paths


def square_polyline(P, theta, tol_deg=7.0):
    """edges that are nearly parallel to a building direction are made exactly parallel"""
    c, s = math.cos(theta), math.sin(theta)
    R = np.array([[c, s], [-s, c]])
    o = P.mean(0)
    Q = (P - o) @ R.T
    t = math.tan(math.radians(tol_deg))
    for _ in range(2):
        for i in range(len(Q) - 1):
            du, dv = Q[i + 1, 0] - Q[i, 0], Q[i + 1, 1] - Q[i, 1]
            if abs(dv) <= t * abs(du):
                m = (Q[i, 1] + Q[i + 1, 1]) / 2
                Q[i, 1] = Q[i + 1, 1] = m
            elif abs(du) <= t * abs(dv):
                m = (Q[i, 0] + Q[i + 1, 0]) / 2
                Q[i, 0] = Q[i + 1, 0] = m
    return Q @ R + o


def centre_lines(mask, x0, y0, cell, theta, simplify_m=0.12, min_len=1.5):
    sk = skeletonize(mask)
    out = []
    for path in trace_skeleton(sk):
        P = np.array([[x0 + (c + 0.5) * cell, y0 + (r + 0.5) * cell] for r, c in path])
        if len(P) < 2:
            continue
        L = LineString(P)
        if L.length < min_len:
            continue
        S = np.asarray(L.simplify(simplify_m).coords)
        out.append(square_polyline(S, theta))
    return out


def area_outlines(mask, x0, y0, cell, theta, min_area=1.0):
    H, W = mask.shape
    xs = x0 + (np.arange(W) + 0.5) * cell
    ys = y0 + (np.arange(H) + 0.5) * cell
    g = contourpy.contour_generator(xs, ys, mask.astype(float), name="serial", line_type="Separate")
    rings = []
    for line in g.lines(0.5):
        r = np.asarray(line)
        if len(r) < 8:
            continue
        reg = ol.regularise(r, base=theta, dp_tol=0.08, snap_tol=math.radians(8), min_seg=0.4, merge_tol=0.15)
        if abs(ol.Polygon(reg).area) >= min_area:
            rings.append(reg)
    return rings


def building_direction(mask, x0, y0, cell):
    """dominant wall direction (radians) from the longest edges of the mask"""
    H, W = mask.shape
    g = contourpy.contour_generator(x0 + (np.arange(W) + 0.5) * cell, y0 + (np.arange(H) + 0.5) * cell,
                                    mask.astype(float), name="serial", line_type="Separate")
    lines = g.lines(0.5)
    if not lines:
        return 0.0
    return ol._dominant(np.asarray(max(lines, key=len)))


def write_dxf(path, wall_lines, low_lines, zellige_rings):
    doc = ezdxf.new("R2018", setup=True)
    doc.units = ezdxf.units.M
    for name, color in (("Murs", 7), ("mur bas", 130), ("Zellige", 6)):
        doc.layers.add(name, color=color)
    msp = doc.modelspace()
    for layer, items in (("Murs", wall_lines), ("mur bas", low_lines)):
        for P in items:
            msp.add_lwpolyline([(float(x), float(y)) for x, y in P], dxfattribs={"layer": layer})
    for r in zellige_rings:
        pts = r[:-1] if np.allclose(r[0], r[-1]) else r
        msp.add_lwpolyline([(float(x), float(y)) for x, y in pts], close=True, dxfattribs={"layer": "Zellige"})
    doc.saveas(path)


def prob_to_dxf(prob, x0, y0, cell, out, thr=0.5, log=print):
    """prob: (4,H,W) class probabilities"""
    wall = prob[1] >= thr
    low = prob[2] >= thr
    zel = prob[3] >= thr
    theta = building_direction(ndi.binary_dilation(wall, iterations=3), x0, y0, cell)
    log(f"[vector] building direction {math.degrees(theta):.1f} deg")
    wall_lines = centre_lines(clean_mask(wall, cell), x0, y0, cell, theta)
    low_lines = centre_lines(clean_mask(low, cell), x0, y0, cell, theta)
    zel_rings = area_outlines(clean_mask(zel, cell, min_len=1.0), x0, y0, cell, theta)
    write_dxf(out, wall_lines, low_lines, zel_rings)
    log(f"[vector] {len(wall_lines)} wall lines, {len(low_lines)} low-wall lines, {len(zel_rings)} zellige areas -> {out}")
    return wall_lines, low_lines, zel_rings
