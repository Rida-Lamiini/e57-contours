"""
review_tool.py - review and fix the wall plan by hand.

Shows the DXF over the scan (height map) and your reference drawing; edit with the mouse:
  Edit    : drag a vertex (square corners are kept), Delete = remove vertex / line
  Line    : click points, Enter / right-click = finish (Close option makes a closed polygon)
  Room    : two clicks = rectangle aligned to the building, new closed polyline on the chosen layer
  Add     : two clicks = rectangle merged INTO the polygon under the first click (adds a block)
  Cut     : two clicks = rectangle removed FROM the polygon under the first click (cuts a notch)
  Erase   : click a line to delete it
Undo / Redo, per-layer visibility, snapping to vertices, ortho (building directions) while drawing.
"""
import copy
import math
import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import ezdxf
import numpy as np
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

HERE = os.path.dirname(os.path.abspath(__file__))

LAYERS = ["Murs", "Murs_interieurs", "mur bas", "Zellige"]
COLORS = {"Murs": "#08306b", "Murs_interieurs": "#2ca02c", "mur bas": "#ff7f0e", "Zellige": "#d62728"}
ACI = {"Murs": 7, "Murs_interieurs": 5, "mur bas": 130, "Zellige": 6}


# =========================================================================== model (no GUI)
class Frame:
    """building frame: u along the main direction theta, v perpendicular"""
    def __init__(self, theta, origin):
        self.t, self.o = theta, np.asarray(origin, float)
        c, s = math.cos(theta), math.sin(theta)
        self.R = np.array([[c, s], [-s, c]])          # world -> (u, v)

    def to_uv(self, P):
        return (np.asarray(P, float) - self.o) @ self.R.T

    def to_xy(self, Q):
        return np.asarray(Q, float) @ self.R + self.o


def clean_collinear(P, closed, tol=0.02):
    P = np.asarray(P, float)
    if len(P) < 3:
        return P
    keep = []
    n = len(P)
    for i in range(n):
        if not closed and i in (0, n - 1):
            keep.append(P[i])
            continue
        a, b, c = P[i - 1], P[i], P[(i + 1) % n]
        dup = np.hypot(*(b - a)) < 1e-6
        cr = abs((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]))
        if not dup and cr > tol:
            keep.append(b)
    return np.array(keep) if len(keep) >= (3 if closed else 2) else P


def move_vertex_square(P, i, new, frame, closed):
    """move vertex i to `new`; its two neighbours slide so that the edges stay parallel to the building directions"""
    Q = frame.to_uv(P)
    n = len(Q)
    nu, nv = frame.to_uv([new])[0]
    nb = []
    if closed or i > 0:
        nb.append((i - 1) % n)
    if closed or i < n - 1:
        nb.append((i + 1) % n)
    for j in nb:
        horizontal = abs(Q[i, 0] - Q[j, 0]) >= abs(Q[i, 1] - Q[j, 1])       # edge mostly along u -> same v
        if horizontal:
            Q[j, 1] = nv
        else:
            Q[j, 0] = nu
    Q[i] = (nu, nv)
    return frame.to_xy(Q)


def rect_from_corners(c1, c2, frame):
    q1, q2 = frame.to_uv([c1])[0], frame.to_uv([c2])[0]
    u0, u1 = sorted((q1[0], q2[0]))
    v0, v1 = sorted((q1[1], q2[1]))
    return frame.to_xy([(u0, v0), (u1, v0), (u1, v1), (u0, v1)])


def ortho_point(last, cur, frame):
    q0, q1 = frame.to_uv([last])[0], frame.to_uv([cur])[0]
    if abs(q1[0] - q0[0]) >= abs(q1[1] - q0[1]):
        q1[1] = q0[1]
    else:
        q1[0] = q0[0]
    return frame.to_xy([q1])[0]


def boolean_rect(ring, rect, op):
    """union / difference of a polygon ring with a rectangle -> new ring (largest piece)"""
    P = Polygon(ring)
    P = P if P.is_valid else P.buffer(0)
    R = Polygon(rect)
    res = P.union(R) if op == "add" else P.difference(R)
    if res.is_empty:
        return None
    if res.geom_type == "MultiPolygon":
        res = max(res.geoms, key=lambda g: g.area)
    if len(res.interiors):                      # rectangle fully inside: that would be a hole, not a notch/block
        return "inside"
    return np.asarray(res.exterior.coords)[:-1]


class Doc:
    def __init__(self):
        self.ents = []                       # {"layer", "pts" (N,2), "closed"}
        self.undo, self.redo = [], []

    def snapshot(self):
        return copy.deepcopy(self.ents)

    def push(self):
        self.undo.append(self.snapshot())
        self.redo.clear()
        if len(self.undo) > 200:
            self.undo.pop(0)

    def do_undo(self):
        if self.undo:
            self.redo.append(self.snapshot())
            self.ents = self.undo.pop()
            return True
        return False

    def do_redo(self):
        if self.redo:
            self.undo.append(self.snapshot())
            self.ents = self.redo.pop()
            return True
        return False

    # ---- io
    def load(self, path):
        d = ezdxf.readfile(path)
        self.ents = []
        for e in d.modelspace():
            t = e.dxftype()
            if t == "LWPOLYLINE":
                p = np.array(list(e.get_points("xy")), float)
                if len(p) >= 2:
                    self.ents.append({"layer": e.dxf.layer, "pts": p, "closed": bool(e.closed)})
            elif t == "LINE":
                self.ents.append({"layer": e.dxf.layer, "closed": False,
                                  "pts": np.array([[e.dxf.start.x, e.dxf.start.y], [e.dxf.end.x, e.dxf.end.y]])})
        self.undo.clear()
        self.redo.clear()

    def save(self, path):
        doc = ezdxf.new("R2018", setup=True)
        doc.units = ezdxf.units.M
        names = list(dict.fromkeys(LAYERS + [e["layer"] for e in self.ents]))
        for nm in names:
            if nm not in doc.layers:
                doc.layers.add(nm, color=ACI.get(nm, 7))
        msp = doc.modelspace()
        for e in self.ents:
            msp.add_lwpolyline([(float(x), float(y)) for x, y in e["pts"]], close=e["closed"],
                               dxfattribs={"layer": e["layer"]})
        doc.saveas(path)

    def main_frame(self):
        best = None
        for e in self.ents:
            if e["layer"] == "Murs" and len(e["pts"]) >= 3:
                a = abs(Polygon(e["pts"]).area)
                if best is None or a > best[0]:
                    best = (a, e["pts"])
        pts = best[1] if best else (self.ents[0]["pts"] if self.ents else np.array([[0, 0], [1, 0]]))
        seg = np.diff(np.vstack([pts, pts[:1]]), axis=0)
        L = np.hypot(seg[:, 0], seg[:, 1])
        a = np.mod(np.arctan2(seg[:, 1], seg[:, 0]), math.pi / 2)
        h, edges = np.histogram(a, bins=90, range=(0, math.pi / 2), weights=L)
        th = (edges[int(np.argmax(h))] + edges[int(np.argmax(h)) + 1]) / 2
        return Frame(th, pts.mean(0))


# =========================================================================== background
def make_background(cache_npz, out_npz, log=print):
    """height-above-ground map of the scan, used as a backdrop"""
    sys.path.insert(0, HERE)
    import e57_to_contours as core
    import outline as ol
    from scipy import ndimage as ndi
    z = np.load(cache_npz)
    layers = (z["low"], z["zmax"], z["cnt"])
    x0, y0, fine = [float(v) for v in z["org"]]
    fg, _ = core.surface_from_layers(layers, "robust")
    fg = core.remove_low_noise(fg, 0.5, 7)
    f = 2
    res = fine * f
    dsm = core.coarsen_min(fg, f)
    g = core.progressive_morph_filter(dsm, res, max_window=40, slope=0.35)
    dem = core.interpolate_ground(dsm, g, res, decim_m=0.5)
    hag = ol.top_height(layers[1], dem, f)
    H, W = hag.shape
    v = np.clip(np.nan_to_num(hag, nan=-1), -1, 8)
    import matplotlib
    rgba = matplotlib.colormaps["Greys"](np.clip((v + 1) / 9, 0, 1) * 0.85 + 0.1)
    rgba = (rgba * 255).astype(np.uint8)
    rgba[np.isnan(hag)] = (235, 235, 235, 255)
    # strong tall/thin features read as dark
    np.savez_compressed(out_npz, img=rgba[::-1], extent=np.array([x0, x0 + W * res, y0, y0 + H * res]))
    log(f"[bg] background saved: {out_npz}")


# =========================================================================== GUI
class FakeEvent:
    def __init__(self, ax, x, y, button=1, key=None, dbl=False):
        self.inaxes, self.xdata, self.ydata, self.button, self.key, self.dblclick = ax, x, y, button, key, dbl
        px, py = ax.transData.transform((x, y))
        self.x, self.y = px, py


class App(tk.Tk):
    def __init__(self, dxf=None, ref=None, bg=None):
        super().__init__()
        import matplotlib
        matplotlib.use("TkAgg")
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        from matplotlib.figure import Figure
        self.title("Wall plan review")
        self.geometry("1380x900")
        self.doc = Doc()
        self.frame = Frame(0.0, (0, 0))
        self.path = None
        self.ref_lines = []
        self.bg = None
        self.sel = None                      # (entity index, vertex index or None)
        self.drag = None
        self.pending = []                    # points of the object being drawn
        self.mode = tk.StringVar(value="edit")
        self.layer = tk.StringVar(value="Murs")
        self.ortho = tk.BooleanVar(value=True)
        self.snap = tk.BooleanVar(value=True)
        self.square = tk.BooleanVar(value=True)
        self.close = tk.BooleanVar(value=False)
        self.show_bg = tk.BooleanVar(value=True)
        self.show_ref = tk.BooleanVar(value=True)
        self.vis = {n: tk.BooleanVar(value=True) for n in LAYERS}
        self.status = tk.StringVar(value="Open a DXF to start.")
        self._build(Figure, FigureCanvasTkAgg, NavigationToolbar2Tk)
        if bg:
            self.load_bg(bg)
        if ref:
            self.load_ref(ref)
        if dxf:
            self.open(dxf)

    # ------------------------------------------------------------------ layout
    def _build(self, Figure, FigureCanvasTkAgg, NavigationToolbar2Tk):
        top = ttk.Frame(self, padding=4)
        top.pack(fill="x")
        for val, lab in (("edit", "Edit"), ("line", "Line"), ("room", "Room"), ("add", "Add block"),
                         ("cut", "Cut notch"), ("erase", "Erase")):
            ttk.Radiobutton(top, text=lab, value=val, variable=self.mode, command=self._mode_changed,
                            style="Toolbutton").pack(side="left", padx=1)
        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=6)
        ttk.Label(top, text="Layer").pack(side="left")
        ttk.Combobox(top, textvariable=self.layer, values=LAYERS, width=16, state="readonly").pack(side="left", padx=3)
        for var, lab in ((self.ortho, "Ortho"), (self.snap, "Snap"), (self.square, "Square drag"), (self.close, "Close line")):
            ttk.Checkbutton(top, text=lab, variable=var).pack(side="left", padx=3)
        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=6)
        for lab, cmd in (("Undo", self.undo), ("Redo", self.redo), ("Open…", self.open_dialog),
                         ("Save", self.save), ("Save as…", self.save_as)):
            ttk.Button(top, text=lab, command=cmd, width=8).pack(side="left", padx=1)
        row2 = ttk.Frame(self, padding=(4, 0))
        row2.pack(fill="x")
        ttk.Label(row2, text="Show:").pack(side="left")
        for n in LAYERS:
            ttk.Checkbutton(row2, text=n, variable=self.vis[n], command=self.redraw).pack(side="left", padx=3)
        ttk.Checkbutton(row2, text="Scan background", variable=self.show_bg, command=self.redraw).pack(side="left", padx=8)
        ttk.Checkbutton(row2, text="Your drawing (red)", variable=self.show_ref, command=self.redraw).pack(side="left")
        ttk.Button(row2, text="Load reference…", command=self.load_ref_dialog).pack(side="left", padx=8)

        self.fig = Figure(figsize=(12, 8), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self, pack_toolbar=False)
        self.toolbar.pack(side="bottom", fill="x")
        ttk.Label(self, textvariable=self.status, anchor="w", relief="sunken").pack(side="bottom", fill="x")
        c = self.canvas
        c.mpl_connect("button_press_event", self.on_press)
        c.mpl_connect("motion_notify_event", self.on_move)
        c.mpl_connect("button_release_event", self.on_release)
        c.mpl_connect("key_press_event", self.on_key)
        self.bind("<Control-z>", lambda e: self.undo())
        self.bind("<Control-y>", lambda e: self.redo())
        self.bind("<Control-s>", lambda e: self.save())
        self.bind("<Delete>", lambda e: self.delete_selected())
        self.bind("<Return>", lambda e: self.finish_line())
        self.bind("<Escape>", lambda e: self.cancel())
        self._hint()

    HINTS = {"edit": "Edit: drag a vertex; click a line to select; Delete removes the vertex (or line).",
             "line": "Line: click points; Enter or right-click finishes.",
             "room": "Room: click two opposite corners.",
             "add": "Add block: click two corners (first click inside the polygon to extend).",
             "cut": "Cut notch: click two corners of the notch (first click inside the polygon).",
             "erase": "Erase: click a line to delete it."}

    def _hint(self):
        self.status.set(self.HINTS[self.mode.get()])

    def _mode_changed(self):
        self.pending.clear()
        self.drag = None
        self._hint()
        self.redraw()

    # ------------------------------------------------------------------ data
    def load_bg(self, path):
        if os.path.exists(path):
            z = np.load(path)
            self.bg = (z["img"], z["extent"])

    def load_ref(self, path):
        self.ref_lines = []
        d = ezdxf.readfile(path)
        for e in d.modelspace():
            if e.dxftype() == "LWPOLYLINE":
                self.ref_lines.append(np.array(list(e.get_points("xy"))))
            elif e.dxftype() == "LINE":
                self.ref_lines.append(np.array([[e.dxf.start.x, e.dxf.start.y], [e.dxf.end.x, e.dxf.end.y]]))
        self.redraw()

    def load_ref_dialog(self):
        p = filedialog.askopenfilename(filetypes=[("DXF", "*.dxf")])
        if p:
            self.load_ref(p)

    def open_dialog(self):
        p = filedialog.askopenfilename(filetypes=[("DXF", "*.dxf")])
        if p:
            self.open(p)

    def open(self, path):
        self.doc.load(path)
        self.path = path
        self.frame = self.doc.main_frame()
        self.sel = None
        self.redraw(fit=True)
        self.status.set(f"{os.path.basename(path)}: {len(self.doc.ents)} lines, building direction "
                        f"{math.degrees(self.frame.t):.1f} deg")

    def _keep_for_training(self, path):
        import shutil
        import time as _t
        d = os.path.join(HERE, "training")
        os.makedirs(d, exist_ok=True)
        shutil.copy(path, os.path.join(d, f"{os.path.splitext(os.path.basename(path))[0]}_{_t.strftime('%Y%m%d_%H%M%S')}.dxf"))

    def save(self):
        if not self.path:
            return self.save_as()
        out = self.path if self.path.endswith("_revu.dxf") else os.path.splitext(self.path)[0] + "_revu.dxf"
        self.doc.save(out)
        self.path = out
        self._keep_for_training(out)
        self.status.set(f"Saved {out}  (a copy was kept in the training folder)")

    def save_as(self):
        p = filedialog.asksaveasfilename(defaultextension=".dxf", filetypes=[("DXF", "*.dxf")])
        if p:
            self.doc.save(p)
            self.path = p
            self._keep_for_training(p)
            self.status.set(f"Saved {p}  (a copy was kept in the training folder)")

    # ------------------------------------------------------------------ drawing
    def redraw(self, fit=False):
        ax = self.ax
        xl, yl = ax.get_xlim(), ax.get_ylim()
        ax.clear()
        if self.show_bg.get() and self.bg is not None:
            img, ext = self.bg
            ax.imshow(img, extent=tuple(ext), origin="upper", zorder=0, interpolation="nearest")
        if self.show_ref.get():
            for r in self.ref_lines:
                ax.plot(r[:, 0], r[:, 1], color="red", lw=1.2, ls="--", zorder=1)
        for i, e in enumerate(self.doc.ents):
            if not self.vis.get(e["layer"], tk.BooleanVar(value=True)).get():
                continue
            p = e["pts"]
            q = np.vstack([p, p[:1]]) if e["closed"] else p
            sel = self.sel is not None and self.sel[0] == i
            ax.plot(q[:, 0], q[:, 1], color=COLORS.get(e["layer"], "k"), lw=2.4 if sel else 1.3, zorder=3)
            if sel or self.mode.get() == "edit":
                ax.plot(p[:, 0], p[:, 1], "s", ms=4 if not sel else 5, mfc="white" if not sel else "#ffd700",
                        mec=COLORS.get(e["layer"], "k"), zorder=4)
        if self.pending:
            q = np.array(self.pending)
            ax.plot(q[:, 0], q[:, 1], "o-", color="#d62728", ms=4, lw=1, zorder=5)
        if fit or not (xl != (0.0, 1.0)):
            big = [e["pts"] for e in self.doc.ents if e["layer"] == "Murs" and len(e["pts"]) >= 3]
            pts = [max(big, key=lambda q: abs(Polygon(q).area))] if big else [e["pts"] for e in self.doc.ents]
            if pts:
                a = np.vstack(pts)
                m = 8
                ax.set_xlim(a[:, 0].min() - m, a[:, 0].max() + m)
                ax.set_ylim(a[:, 1].min() - m, a[:, 1].max() + m)
        else:
            ax.set_xlim(xl)
            ax.set_ylim(yl)
        ax.set_aspect("equal", adjustable="datalim")
        self.canvas.draw_idle()

    # ------------------------------------------------------------------ geometry helpers
    def _tol(self, px=9):
        x0, x1 = self.ax.get_xlim()
        w = self.canvas.get_tk_widget().winfo_width() or 1000
        return px * abs(x1 - x0) / max(w, 1)

    def nearest_vertex(self, x, y):
        best = None
        for i, e in enumerate(self.doc.ents):
            if not self.vis.get(e["layer"], tk.BooleanVar(value=True)).get():
                continue
            d = np.hypot(e["pts"][:, 0] - x, e["pts"][:, 1] - y)
            k = int(np.argmin(d))
            if best is None or d[k] < best[0]:
                best = (float(d[k]), i, k)
        return best

    def nearest_entity(self, x, y):
        from shapely.geometry import LineString, Point
        best = None
        for i, e in enumerate(self.doc.ents):
            if not self.vis.get(e["layer"], tk.BooleanVar(value=True)).get():
                continue
            p = e["pts"]
            g = LineString(np.vstack([p, p[:1]]) if e["closed"] else p)
            d = g.distance(Point(x, y))
            if best is None or d < best[0]:
                best = (d, i)
        return best

    def containing_polygon(self, x, y):
        from shapely.geometry import Point
        best = None
        for i, e in enumerate(self.doc.ents):
            if e["closed"] and len(e["pts"]) >= 3:
                P = Polygon(e["pts"])
                P = P if P.is_valid else P.buffer(0)
                if P.contains(Point(x, y)) and (best is None or P.area < best[0]):
                    best = (P.area, i)
        if best is None:                                  # nearest closed polygon edge as a fallback
            d = None
            for i, e in enumerate(self.doc.ents):
                if e["closed"] and len(e["pts"]) >= 3:
                    from shapely.geometry import LineString, Point as Pt
                    dd = LineString(np.vstack([e["pts"], e["pts"][:1]])).distance(Pt(x, y))
                    if d is None or dd < d[0]:
                        d = (dd, i)
            if d and d[0] <= 6 * self._tol():
                return d[1]
            return None
        return best[1]

    def snap_pt(self, x, y):
        if self.snap.get():
            nv = self.nearest_vertex(x, y)
            if nv and nv[0] <= self._tol(12):
                return self.doc.ents[nv[1]]["pts"][nv[2]].copy()
        return np.array([x, y])

    # ------------------------------------------------------------------ mouse
    def _busy(self):
        return bool(getattr(self.toolbar, "mode", ""))

    def on_press(self, ev):
        if self._busy() or ev.inaxes is not self.ax or ev.xdata is None:
            return
        m = self.mode.get()
        x, y = ev.xdata, ev.ydata
        if m == "edit":
            if ev.button == 1:
                nv = self.nearest_vertex(x, y)
                if nv and nv[0] <= self._tol():
                    self.sel = (nv[1], nv[2])
                    self.doc.push()
                    self.drag = (nv[1], nv[2])
                else:
                    ne = self.nearest_entity(x, y)
                    self.sel = (ne[1], None) if ne and ne[0] <= self._tol() * 1.5 else None
                self.redraw()
        elif m == "line":
            if ev.button == 3:
                return self.finish_line()
            pt = self.snap_pt(x, y)
            if self.pending and self.ortho.get():
                pt = ortho_point(self.pending[-1], pt, self.frame)
            self.pending.append(pt)
            self.redraw()
        elif m in ("room", "add", "cut"):
            pt = self.snap_pt(x, y)
            if not self.pending:
                self.pending = [pt]
                self.owner = self.containing_polygon(x, y) if m in ("add", "cut") else None
                self.status.set("Now click the opposite corner.")
            else:
                self.apply_rect(self.pending[0], pt, m)
                self.pending = []
                self._hint()
            self.redraw()
        elif m == "erase" and ev.button == 1:
            ne = self.nearest_entity(x, y)
            if ne and ne[0] <= self._tol() * 1.5:
                self.doc.push()
                self.doc.ents.pop(ne[1])
                self.sel = None
                self.redraw()

    def on_move(self, ev):
        if ev.inaxes is not self.ax or ev.xdata is None:
            return
        if self.drag and not self._busy():
            i, k = self.drag
            e = self.doc.ents[i]
            pt = self.snap_pt(ev.xdata, ev.ydata) if False else np.array([ev.xdata, ev.ydata])
            if self.square.get():
                e["pts"] = move_vertex_square(e["pts"], k, pt, self.frame, e["closed"])
            else:
                e["pts"][k] = pt
            self.redraw()
        self.status.set(f"{self.HINTS[self.mode.get()]}   x {ev.xdata:.2f}  y {ev.ydata:.2f}")

    def on_release(self, ev):
        if self.drag:
            i, k = self.drag
            e = self.doc.ents[i]
            e["pts"] = clean_collinear(e["pts"], e["closed"])
            self.drag = None
            self.sel = (i, None)
            self.redraw()

    def on_key(self, ev):
        if ev.key == "delete":
            self.delete_selected()
        elif ev.key == "enter":
            self.finish_line()
        elif ev.key == "escape":
            self.cancel()

    # ------------------------------------------------------------------ operations
    def apply_rect(self, c1, c2, mode):
        if np.hypot(*(np.asarray(c1) - np.asarray(c2))) < 0.05:
            return
        rect = rect_from_corners(c1, c2, self.frame)
        if mode == "room":
            self.doc.push()
            self.doc.ents.append({"layer": self.layer.get(), "pts": rect, "closed": True})
            self.sel = (len(self.doc.ents) - 1, None)
            self.status.set("Room added.")
            return
        i = getattr(self, "owner", None)
        if i is None:
            self.status.set("Click first INSIDE (or next to) a closed polygon.")
            return
        new = boolean_rect(self.doc.ents[i]["pts"], rect, mode)
        if isinstance(new, str):
            self.status.set("The rectangle is fully inside the polygon: drag it across the edge so it forms a notch / block.")
            return
        if new is None or len(new) < 3:
            self.status.set("Nothing left of the polygon after the cut.")
            return
        self.doc.push()
        self.doc.ents[i]["pts"] = clean_collinear(new, True)
        self.sel = (i, None)
        self.status.set("Block added." if mode == "add" else "Notch cut.")

    def finish_line(self):
        if self.mode.get() == "line" and len(self.pending) >= 2:
            self.doc.push()
            self.doc.ents.append({"layer": self.layer.get(), "pts": np.array(self.pending),
                                  "closed": bool(self.close.get() and len(self.pending) >= 3)})
            self.sel = (len(self.doc.ents) - 1, None)
        self.pending = []
        self.redraw()

    def cancel(self):
        self.pending = []
        self.drag = None
        self.redraw()

    def delete_selected(self):
        if not self.sel:
            return
        i, k = self.sel
        if i >= len(self.doc.ents):
            return
        e = self.doc.ents[i]
        self.doc.push()
        if k is not None and len(e["pts"]) > (3 if e["closed"] else 2):
            e["pts"] = np.delete(e["pts"], k, axis=0)
            self.sel = (i, None)
        else:
            self.doc.ents.pop(i)
            self.sel = None
        self.redraw()

    def undo(self):
        if self.doc.do_undo():
            self.sel = None
            self.redraw()

    def redo(self):
        if self.doc.do_redo():
            self.sel = None
            self.redraw()


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("dxf", nargs="?", default=os.path.join(HERE, "out", "v2", "Plan_murs_combine.dxf"))
    ap.add_argument("--reference", default=None, help="your hand drawing, shown in red")
    ap.add_argument("--cache", default=os.path.join(HERE, "out", "v2", "Contour.grid.npz"))
    a = ap.parse_args(argv)
    bgp = os.path.join(HERE, "out", "v2", "review_bg.npz")
    if not os.path.exists(bgp) and os.path.exists(a.cache):
        print("building the scan background (first run only)…")
        make_background(a.cache, bgp)
    App(a.dxf, a.reference, bgp).mainloop()


if __name__ == "__main__":
    sys.exit(main())
