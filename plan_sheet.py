"""
plan_sheet.py - turn the contour drawing into a finished topographic plan (AutoCAD layout sheet).

Adds to the DXF:
  * model space : coordinate grid (crosses + E/N labels), spot heights, hatched building footprints
  * layout 'PLAN': sheet frame, viewport at a standard scale, title block, north arrow,
                   graphic scale bar and legend
Sheet size and scale are chosen automatically (smallest A-size / standard scale that fits).
"""
import datetime
import math
import os

import numpy as np

SHEETS = {"A3": (420, 297), "A2": (594, 420), "A1": (841, 594), "A0": (1189, 841)}
SCALES = [200, 250, 500, 1000, 1500, 2000, 2500, 5000]
MARGIN = 10.0        # mm between sheet edge and frame
STRIP = 88.0         # mm, right-hand strip (north arrow, scale bar, legend, title block)


def choose_sheet(bw, bh, pad=1.08):
    """smallest sheet + standard scale whose drawing window holds a bw x bh (m) area"""
    # readable scales first: smallest sheet that fits at <=1/500, else at <=1/1000, else any scale
    for max_sc, max_sheet in ((500, 3), (1000, 3), (10 ** 9, 4)):
        for name, (W, H) in list(SHEETS.items())[:max_sheet]:        # A3..A1 first, A0 only as a last resort
            vw, vh = W - 2 * MARGIN - STRIP, H - 2 * MARGIN
            for sc in SCALES:
                if sc > max_sc:
                    break
                if bw * pad * 1000 / sc <= vw and bh * pad * 1000 / sc <= vh:
                    return name, W, H, sc
    W, H = SHEETS["A0"]
    return "A0", W, H, SCALES[-1]


def _nice(v):
    """nearest 1-2-5 number (log scale)"""
    e = 10 ** math.floor(math.log10(v))
    return min((m * e for m in (1, 2, 5, 10)), key=lambda c: abs(math.log(c / v)))


def plan_params(bbox, spot_mm=22.0):
    xmin, ymin, xmax, ymax = bbox
    name, W, H, sc = choose_sheet(xmax - xmin, ymax - ymin)
    k = sc / 1000.0                                  # metres of ground per mm of paper
    return {"sheet": name, "W": W, "H": H, "scale": sc, "k": k,
            "text_h": 2.0 * k,                       # 2 mm text on paper
            "spot_step": _nice(spot_mm * k),         # ~2.2 cm between spot heights on paper
            "grid_step": _nice(100 * k),             # ~10 cm between grid crosses
            "bbox": bbox}


def spot_heights(dem, mask, x0, y0, res, step):
    """regular grid of spot heights on valid (non-masked) ground"""
    H, W = dem.shape
    xs = np.arange(math.ceil(x0 / step) * step, x0 + W * res, step)
    ys = np.arange(math.ceil(y0 / step) * step, y0 + H * res, step)
    pts = []
    for y in ys:
        r = int((y - y0) / res)
        for x in xs:
            c = int((x - x0) / res)
            if 0 <= r < H and 0 <= c < W and not mask[r, c] and np.isfinite(dem[r, c]):
                # reject points next to a masked cell (unreliable edge)
                r0, r1, c0, c1 = max(r - 2, 0), r + 3, max(c - 2, 0), c + 3
                if not mask[r0:r1, c0:c1].any():
                    pts.append((float(x), float(y), float(dem[r, c])))
    return pts


# --------------------------------------------------------------------------- model space
def add_model_space_items(doc, msp, P, spots, bmask_polys):
    th = P["text_h"]
    for name, color in (("COTES_POINTS", 7), ("GRILLE", 9), ("BATIMENTS_HACHURES", 6)):
        if name not in doc.layers:
            doc.layers.add(name, color=color)
    # spot heights: small cross + height
    s = 0.5 * P["k"]
    for x, y, z in spots:
        msp.add_line((x - s, y), (x + s, y), dxfattribs={"layer": "COTES_POINTS"})
        msp.add_line((x, y - s), (x, y + s), dxfattribs={"layer": "COTES_POINTS"})
        msp.add_text(f"{z:.2f}", height=0.8 * th,
                     dxfattribs={"layer": "COTES_POINTS", "insert": (x + 0.6 * P["k"], y + 0.3 * P["k"])})
    # building footprints: light hatch
    for poly in bmask_polys or []:
        h = msp.add_hatch(color=6, dxfattribs={"layer": "BATIMENTS_HACHURES"})
        h.set_pattern_fill("ANSI31", scale=max(0.05, 3.5 * P["k"] / 3.175))
        h.paths.add_polyline_path([(float(x), float(y)) for x, y in poly], is_closed=True)
    # coordinate grid crosses + labels on the window edge
    xmin, ymin, xmax, ymax = P["win"]
    g, c = P["grid_step"], 2.0 * P["k"]
    for x in np.arange(math.ceil(xmin / g) * g, xmax, g):
        for y in np.arange(math.ceil(ymin / g) * g, ymax, g):
            msp.add_line((x - c, y), (x + c, y), dxfattribs={"layer": "GRILLE"})
            msp.add_line((x, y - c), (x, y + c), dxfattribs={"layer": "GRILLE"})
        msp.add_text(f"E {x:.0f}", height=th, dxfattribs={
            "layer": "GRILLE", "insert": (x + 0.5 * th, ymin + 1.0 * P["k"]), "rotation": 90})
    for y in np.arange(math.ceil(ymin / g) * g, ymax, g):
        msp.add_text(f"N {y:.0f}", height=th, dxfattribs={
            "layer": "GRILLE", "insert": (xmin + 1.0 * P["k"], y + 0.5 * th)})


# --------------------------------------------------------------------------- paper space
def _rect(ps, x0, y0, x1, y1, layer="CARTOUCHE", lw=35):
    ps.add_lwpolyline([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], close=True,
                      dxfattribs={"layer": layer, "lineweight": lw})


def _text(ps, s, x, y, h=3.0, layer="CARTOUCHE", **kw):
    t = ps.add_text(s, height=h, dxfattribs={"layer": layer, "insert": (x, y), **kw})
    return t


def add_layout(doc, P, info):
    W, H, sc = P["W"], P["H"], P["scale"]
    for name, color in (("CARTOUCHE", 7), ("VIEWPORTS", 8)):
        if name not in doc.layers:
            doc.layers.add(name, color=color)
    doc.layers.get("VIEWPORTS").dxf.plot = 0
    lay = doc.layouts.new("PLAN")
    lay.page_setup(size=(W, H), margins=(0, 0, 0, 0), units="mm")
    ps = lay

    # frames
    _rect(ps, 3, 3, W - 3, H - 3, lw=13)
    fx0, fy0, fx1, fy1 = MARGIN, MARGIN, W - MARGIN, H - MARGIN
    _rect(ps, fx0, fy0, fx1, fy1, lw=50)
    vx1 = fx1 - STRIP
    ps.add_line((vx1, fy0), (vx1, fy1), dxfattribs={"layer": "CARTOUCHE", "lineweight": 35})

    # viewport (drawing window)
    vw, vh = vx1 - fx0, fy1 - fy0
    xmin, ymin, xmax, ymax = P["bbox"]
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    vp = ps.add_viewport(center=((fx0 + vx1) / 2, (fy0 + fy1) / 2), size=(vw, vh),
                         view_center_point=(cx, cy), view_height=vh * P["k"])
    vp.dxf.layer = "VIEWPORTS"
    vp.dxf.status = 1
    # drawing window in model coordinates (for the grid labels)
    half_w, half_h = vw * P["k"] / 2, vh * P["k"] / 2
    P["win"] = (cx - half_w, cy - half_h, cx + half_w, cy + half_h)

    sx0, sx1 = vx1 + 4, fx1 - 4                # strip inner limits
    sw = sx1 - sx0
    y = fy1 - 6

    # north arrow
    ax, ay = sx0 + sw / 2, y - 20
    ps.add_lwpolyline([(ax, ay + 14), (ax - 5, ay - 8), (ax, ay - 3), (ax + 5, ay - 8)], close=True,
                      dxfattribs={"layer": "CARTOUCHE", "lineweight": 35})
    _text(ps, "N", ax - 2, ay + 16, 6, "CARTOUCHE")
    y -= 48

    # graphic scale bar (length = nice number of metres, ~40 mm on paper)
    L = _nice(40 * P["k"])
    seg = L / 4
    bx = sx0 + 6
    _text(ps, f"Echelle 1/{sc}", sx0 + 6, y, 4.5)
    y -= 8
    for i in range(4):
        w = seg / P["k"]
        ps.add_lwpolyline([(bx + i * w, y), (bx + (i + 1) * w, y), (bx + (i + 1) * w, y - 2.5), (bx + i * w, y - 2.5)],
                          close=True, dxfattribs={"layer": "CARTOUCHE", "lineweight": 25})
        if i % 2 == 0:
            ps.add_hatch(color=7, dxfattribs={"layer": "CARTOUCHE"}).paths.add_polyline_path(
                [(bx + i * w, y), (bx + (i + 1) * w, y), (bx + (i + 1) * w, y - 2.5), (bx + i * w, y - 2.5)], is_closed=True)
    for i in range(5):
        _text(ps, f"{seg * i:g}", bx + i * seg / P["k"] - 1.5, y - 7, 2.5)
    _text(ps, "m", bx + L / P["k"] + 3, y - 7, 2.5)
    y -= 20

    # legend
    _text(ps, "LEGENDE", sx0 + 2, y, 4)
    y -= 7
    items = [
        ("COURBES_MAITRESSES", f"Courbe maitresse (tous les {info['index_step']:g} m)", 1, 35),
        ("COURBES_INTERMEDIAIRES", f"Courbe intermediaire ({info['interval']:g} m)", 8, 13),
    ]
    for lay_name, label, color, lw in items:
        ps.add_line((sx0 + 2, y + 1), (sx0 + 16, y + 1), dxfattribs={"layer": lay_name, "lineweight": lw})
        _text(ps, label, sx0 + 19, y, 2.5)
        y -= 6
    ps.add_line((sx0 + 5, y + 1), (sx0 + 13, y + 1), dxfattribs={"layer": "COTES_POINTS"})
    ps.add_line((sx0 + 9, y - 3), (sx0 + 9, y + 5), dxfattribs={"layer": "COTES_POINTS"})
    _text(ps, "Point coté (altitude en m)", sx0 + 19, y, 2.5)
    y -= 8
    h = ps.add_hatch(color=6, dxfattribs={"layer": "BATIMENTS_HACHURES"})
    h.set_pattern_fill("ANSI31", scale=0.6)
    h.paths.add_polyline_path([(sx0 + 2, y - 2), (sx0 + 16, y - 2), (sx0 + 16, y + 4), (sx0 + 2, y + 4)], is_closed=True)
    _text(ps, "Bâtiment / objet masqué", sx0 + 19, y, 2.5)
    y -= 8
    if info.get("has_targets"):
        ps.add_circle((sx0 + 9, y + 1), 2.2, dxfattribs={"layer": "CIBLES"})
        ps.add_line((sx0 + 5.5, y + 1), (sx0 + 12.5, y + 1), dxfattribs={"layer": "CIBLES"})
        ps.add_line((sx0 + 9, y - 2.5), (sx0 + 9, y + 4.5), dxfattribs={"layer": "CIBLES"})
        _text(ps, "Cible topographique", sx0 + 19, y, 2.5)
        y -= 8

    # title block (bottom of the strip)
    tb_h = 80
    ty1 = fy0 + tb_h
    _rect(ps, vx1, fy0, fx1, ty1, lw=35)
    rows = [("PLAN TOPOGRAPHIQUE", 4.6, 0), ("COURBES DE NIVEAU", 3.4, 0)]
    yy = ty1 - 8
    for s, h, _ in rows:
        _text(ps, s, sx0, yy, h)
        yy -= h + 3
    ps.add_line((vx1, yy + 1), (fx1, yy + 1), dxfattribs={"layer": "CARTOUCHE"})
    yy -= 5
    lines = [
        f"Projet : {info['project']}",
        f"Source : {info['source']}",
        f"Echelle : 1/{sc}   Format : {P['sheet']}",
        f"Equidistance : {info['interval']:g} m",
        f"Date : {info['date']}",
        "Coordonnées : système du scan (E/N),",
        "altitudes Z du scan.",
        "Établi automatiquement à partir",
        "du nuage de points.",
    ]
    for s in lines:
        _text(ps, s, sx0, yy, 2.3)
        yy -= 4.6
    return lay


def build_plan(doc, msp, bbox, info, spots, bmask_polys, margin_m=8.0):
    xmin, ymin, xmax, ymax = bbox
    bbox = (xmin - margin_m, ymin - margin_m, xmax + margin_m, ymax + margin_m)
    P = plan_params(bbox)
    info = dict(info)
    info.setdefault("date", datetime.date.today().strftime("%d/%m/%Y"))
    add_layout(doc, P, info)
    add_model_space_items(doc, msp, P, spots, bmask_polys)
    return P
