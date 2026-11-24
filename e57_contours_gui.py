"""
E57 -> Contours (DXF)  -  desktop GUI around e57_to_contours.py
Run:  .venv\\Scripts\\pythonw.exe e57_contours_gui.py   (or double-click run_gui.bat)
"""
import contextlib
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e57_to_contours as core  # noqa: E402


class QueueWriter:
    def __init__(self, q):
        self.q = q

    def write(self, s):
        if s:
            self.q.put(s)

    def flush(self):
        pass


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("E57 -> Contours (DXF)")
        self.geometry("1280x820")
        self.minsize(1000, 640)
        self.q = queue.Queue()
        self.worker = None
        self.preview_img = None
        self.preview_src = None
        self.vars = {}
        self._build()
        self.after(100, self._pump)

    # ------------------------------------------------------------------ UI
    def _build(self):
        left = ttk.Frame(self, padding=10)
        left.pack(side="left", fill="y")
        right = ttk.Frame(self, padding=(0, 10, 10, 10))
        right.pack(side="left", fill="both", expand=True)

        # files
        f = ttk.LabelFrame(left, text="Files", padding=8)
        f.pack(fill="x")
        self.vars["e57"] = tk.StringVar()
        self.vars["out"] = tk.StringVar()
        ttk.Label(f, text="E57 point cloud").grid(row=0, column=0, sticky="w")
        ttk.Entry(f, textvariable=self.vars["e57"], width=46).grid(row=1, column=0, sticky="ew")
        ttk.Button(f, text="Browse…", command=self._pick_e57).grid(row=1, column=1, padx=(4, 0))
        ttk.Label(f, text="Output DXF").grid(row=2, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(f, textvariable=self.vars["out"], width=46).grid(row=3, column=0, sticky="ew")
        ttk.Button(f, text="Save as…", command=self._pick_out).grid(row=3, column=1, padx=(4, 0))

        # what to produce
        m = ttk.LabelFrame(left, text="Output", padding=8)
        m.pack(fill="x", pady=(8, 0))
        self.vars["mode"] = tk.StringVar(value="outline")
        for i, (val, lab) in enumerate((("outline", "Exterior outline (walls plan)"),
                                        ("contours", "Elevation contours"), ("both", "Both"))):
            ttk.Radiobutton(m, text=lab, value=val, variable=self.vars["mode"]).grid(row=i, column=0, sticky="w")
        self._num(m, 3, "tall_h", "Built if higher than (m)", "", "Outline: roofs/walls above ground")
        self.vars["inner_walls"] = tk.BooleanVar(value=True)
        ttk.Checkbutton(m, text="Also draw inner wall lines (Murs_interieurs / mur bas)",
                        variable=self.vars["inner_walls"]).grid(row=4, column=0, columnspan=3, sticky="w")

        # contour settings
        c = ttk.LabelFrame(left, text="Contours", padding=8)
        c.pack(fill="x", pady=(8, 0))
        self._num(c, 0, "interval", "Interval (m)", 0.25, "Vertical spacing between contours")
        self._num(c, 1, "index_every", "Bold line every N", 4, "Index (maitresse) contour every N lines")
        self._num(c, 2, "smooth", "DEM smoothing (m)", 0.6, "Higher = smoother, less detail")
        self._num(c, 3, "min_len", "Min. length (m)", 8.0, "Drop contours shorter than this")
        self._num(c, 4, "tol_z", "Auto-fix height tol. (m)", 0.08, "Bigger = smoother / straighter lines")
        self._num(c, 5, "join_gap", "Join gaps up to (m)", 4.0, "Reconnect broken contour ends")
        self.vars["plan"] = tk.BooleanVar(value=True)
        ttk.Checkbutton(c, text="Build finished plan sheet (layout + PDF)",
                        variable=self.vars["plan"]).grid(row=7, column=0, columnspan=3, sticky="w", pady=(4, 0))
        self.vars["title"] = tk.StringVar()
        ttk.Label(c, text="Plan title / project").grid(row=8, column=0, sticky="w")
        ttk.Entry(c, textvariable=self.vars["title"], width=24).grid(row=8, column=1, columnspan=2, sticky="w", padx=(8, 0))
        self.vars["no_fix"] = tk.BooleanVar(value=False)
        ttk.Checkbutton(c, text="Disable auto-fix (raw grid lines)",
                        variable=self.vars["no_fix"]).grid(row=6, column=0, columnspan=3, sticky="w", pady=(4, 0))

        # ground filter
        g = ttk.LabelFrame(left, text="Ground / objects", padding=8)
        g.pack(fill="x", pady=(8, 0))
        self._num(g, 0, "max_object", "Largest object (m)", 40.0, "Widest building/hangar to remove")
        self._num(g, 1, "slope", "Max terrain slope", 0.35, "Rise/run; raise for steep embankments")
        self._num(g, 2, "reach", "Bridge gaps up to (m)", 2.5, "No contours across wider ground gaps")
        self._num(g, 3, "dem_res", "DEM cell (m)", 0.2, "Multiple of fine cell (0.1)")
        self.vars["keep_buildings"] = tk.BooleanVar(value=False)
        ttk.Checkbutton(g, text="Keep buildings (don't mask footprints)",
                        variable=self.vars["keep_buildings"]).grid(row=4, column=0, columnspan=2, sticky="w", pady=(4, 0))
        self.vars["rebuild"] = tk.BooleanVar(value=False)
        ttk.Checkbutton(g, text="Re-read E57 (ignore cache)",
                        variable=self.vars["rebuild"]).grid(row=5, column=0, columnspan=2, sticky="w")

        # clip
        k = ttk.LabelFrame(left, text="Clip area (optional, same CRS as the scan)", padding=8)
        k.pack(fill="x", pady=(8, 0))
        for i, (key, lab) in enumerate((("xmin", "X min"), ("ymin", "Y min"), ("xmax", "X max"), ("ymax", "Y max"))):
            self.vars[key] = tk.StringVar()
            ttk.Label(k, text=lab).grid(row=i // 2, column=(i % 2) * 2, sticky="e", padx=(0, 3))
            ttk.Entry(k, textvariable=self.vars[key], width=13).grid(row=i // 2, column=(i % 2) * 2 + 1, padx=(0, 8), pady=1)

        # actions
        b = ttk.Frame(left)
        b.pack(fill="x", pady=(12, 0))
        self.run_btn = ttk.Button(b, text="▶  Generate contours", command=self._run)
        self.run_btn.pack(fill="x", ipady=6)
        row = ttk.Frame(b)
        row.pack(fill="x", pady=(6, 0))
        ttk.Button(row, text="Open DXF", command=self._open_dxf).pack(side="left", expand=True, fill="x")
        ttk.Button(row, text="Show folder", command=self._open_folder).pack(side="left", expand=True, fill="x", padx=(4, 0))
        self.progress = ttk.Progressbar(left, mode="indeterminate")
        self.progress.pack(fill="x", pady=(8, 0))

        # right: preview + log
        pv = ttk.LabelFrame(right, text="Preview", padding=4)
        pv.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(pv, bg="#222", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda e: self._draw_preview())
        lg = ttk.LabelFrame(right, text="Log", padding=4)
        lg.pack(fill="x", pady=(8, 0))
        self.log = tk.Text(lg, height=9, font=("Consolas", 9), state="disabled", wrap="none")
        self.log.pack(fill="x")

    def _num(self, parent, row, key, label, default, tip):
        v = tk.StringVar(value=str(default))
        self.vars[key] = v
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w")
        e = ttk.Entry(parent, textvariable=v, width=10)
        e.grid(row=row, column=1, padx=(8, 0), pady=1)
        ttk.Label(parent, text=tip, foreground="#777").grid(row=row, column=2, sticky="w", padx=(8, 0))

    # ------------------------------------------------------------- actions
    def _pick_e57(self):
        p = filedialog.askopenfilename(title="Select E57", filetypes=[("E57 point cloud", "*.e57"), ("All", "*.*")])
        if p:
            self.vars["e57"].set(p)
            if not self.vars["out"].get():
                base = os.path.splitext(os.path.basename(p))[0] + "_contours.dxf"
                self.vars["out"].set(os.path.join(HERE, "out", base))

    def _pick_out(self):
        p = filedialog.asksaveasfilename(title="Output DXF", defaultextension=".dxf", filetypes=[("DXF", "*.dxf")])
        if p:
            self.vars["out"].set(p)

    def _open_dxf(self):
        p = self.vars["out"].get()
        if os.path.exists(p):
            os.startfile(p)
        else:
            messagebox.showinfo("Open DXF", "The DXF does not exist yet.")

    def _open_folder(self):
        p = self.vars["out"].get()
        d = os.path.dirname(p) if p else HERE
        if os.path.isdir(d):
            subprocess.Popen(["explorer", "/select,", os.path.normpath(p)] if os.path.exists(p) else ["explorer", d])

    def _args(self):
        v = {k: x.get().strip() if isinstance(x, tk.StringVar) else x.get() for k, x in self.vars.items()}
        if not v["e57"] and not (v["out"] and os.path.exists(os.path.splitext(v["out"])[0] + ".grid.npz")):
            raise ValueError("Select an E57 file.")
        out = v["out"] or os.path.join(HERE, "out", "contours.dxf")
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        a = [v["e57"] or "cached.e57", "-o", out,
             "--interval", v["interval"], "--index-every", str(int(float(v["index_every"]))),
             "--smooth", v["smooth"], "--min-len", v["min_len"], "--max-object", v["max_object"],
             "--slope", v["slope"], "--reach", v["reach"], "--dem-res", v["dem_res"]]
        a += ["--mode", v["mode"]]
        if v["tall_h"]:
            a += ["--tall-h", v["tall_h"]]
        if not v["inner_walls"]:
            a.append("--no-walls")
        a += ["--tol-z", v["tol_z"], "--join-gap", v["join_gap"]]
        if v["no_fix"]:
            a.append("--no-fix")
        if v["plan"]:
            a.append("--plan")
            if v["title"]:
                a += ["--title", v["title"]]
        if v["keep_buildings"]:
            a.append("--keep-buildings")
        if v["rebuild"]:
            a.append("--rebuild")
        clip = [v[k] for k in ("xmin", "ymin", "xmax", "ymax")]
        if any(clip):
            if not all(clip):
                raise ValueError("Fill all four clip values, or leave them all empty.")
            a += ["--clip"] + clip
            a.append("--rebuild")  # a clip changes the grid extent
        float(v["interval"])  # validate early
        return a, out

    def _run(self):
        if self.worker and self.worker.is_alive():
            return
        try:
            argv, out = self._args()
        except Exception as e:
            messagebox.showerror("Settings", str(e))
            return
        self.vars["out"].set(out)
        self._log_clear()
        self.run_btn.state(["disabled"])
        self.progress.start(12)
        self.worker = threading.Thread(target=self._work, args=(argv, out), daemon=True)
        self.worker.start()

    def _work(self, argv, out):
        ok = True
        try:
            with contextlib.redirect_stdout(QueueWriter(self.q)):
                core.main(argv)
        except SystemExit as e:
            ok = e.code in (0, None)
            if not ok:
                self.q.put(f"\n[error] invalid arguments (code {e.code})\n")
        except Exception as e:  # noqa: BLE001
            ok = False
            import traceback
            self.q.put("\n[error] " + "".join(traceback.format_exception_only(type(e), e)))
            self.q.put(traceback.format_exc())
        self.q.put(("__done__", ok, out))

    # ---------------------------------------------------------------- log
    def _log_clear(self):
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")
        self._cr = False

    def _log_write(self, s):
        self.log.configure(state="normal")
        for i, part in enumerate(s.replace("\r\n", "\n").split("\r")):
            if i > 0:  # a carriage return: overwrite the current line (progress counters)
                self.log.delete("end-1c linestart", "end-1c")
            self.log.insert("end", part)
        self.log.see("end")
        self.log.configure(state="disabled")

    def _pump(self):
        try:
            while True:
                m = self.q.get_nowait()
                if isinstance(m, tuple) and m[0] == "__done__":
                    self.progress.stop()
                    self.run_btn.state(["!disabled"])
                    if m[1]:
                        self._load_preview(os.path.splitext(m[2])[0] + "_preview.png")
                else:
                    self._log_write(m)
        except queue.Empty:
            pass
        self.after(100, self._pump)

    # ------------------------------------------------------------ preview
    def _load_preview(self, path):
        if os.path.exists(path):
            self.preview_src = Image.open(path)
            self._draw_preview()

    def _draw_preview(self):
        if self.preview_src is None:
            return
        cw, ch = max(50, self.canvas.winfo_width()), max(50, self.canvas.winfo_height())
        im = self.preview_src.copy()
        im.thumbnail((cw, ch), Image.LANCZOS)
        self.preview_img = ImageTk.PhotoImage(im)
        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2, image=self.preview_img)


if __name__ == "__main__":
    App().mainloop()
