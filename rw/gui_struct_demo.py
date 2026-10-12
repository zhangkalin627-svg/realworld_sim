"""结构演示页（公开版）：参数化桌子/机架，选型材、材料、载荷 → 三维梁有限元 → 变形、应力、屈曲、地脚。"""
import math
from PySide6 import QtCore, QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np
from .beam import Frame, Section
from .matlib import MAT
from .gui_world import html, spin

MATS = ["Q235", "45", "6061", "6063", "SUS304", "PP", "POM"]


def build_table(L, W, H, tube, mat_leg, mat_top, load_kg, load_x, load_y):
    f = Frame()
    sec = Section.rhs(tube, tube, max(tube / 20, 1.5))
    corners = [(0, 0), (L, 0), (L, W), (0, W)]
    for (x, y) in corners:
        ids = f.beam((x, y, 0), (x, y, H), sec, MAT[mat_leg], "立柱", nseg=4)
        f.fix[ids[0]] = [True, True, True, False, False, False]
    for z, nm in ((H, "上框"), (H * 0.25, "下框")):
        f.beam((0, 0, z), (L, 0, z), sec, MAT[mat_top], nm, nseg=6)
        f.beam((0, W, z), (L, W, z), sec, MAT[mat_top], nm, nseg=6)
        f.beam((0, 0, z), (0, W, z), sec, MAT[mat_top], nm, nseg=6)
        f.beam((L, 0, z), (L, W, z), sec, MAT[mat_top], nm, nseg=6)
    f.beam((0, W / 2, H), (L, W / 2, H), sec, MAT[mat_top], "中梁", nseg=6)
    n = f.nearest((load_x, load_y, H))
    f.add_load(n, F=(0, 0, -load_kg * 9.81))
    return f, sec


class StructDemoPage(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        lay = QtWidgets.QHBoxLayout(self)
        left = QtWidgets.QVBoxLayout(); form = QtWidgets.QFormLayout()
        g = QtWidgets.QGroupBox("机架参数（改了立刻重算）"); g.setLayout(form); left.addWidget(g)
        self.w = {}
        def add(k, lab, wdg):
            form.addRow(lab, wdg); self.w[k] = wdg
            (wdg.valueChanged if hasattr(wdg, "valueChanged") else wdg.currentIndexChanged).connect(self.update_)
        add("L", "长 mm", spin(200, 6000, 1000, 50, 0)); add("W", "宽 mm", spin(200, 6000, 800, 50, 0))
        add("H", "高 mm", spin(200, 4000, 900, 50, 0)); add("t", "方管边长 mm", spin(10, 300, 60, 10, 0))
        for k, lab in (("ml", "立柱材料"), ("mt", "横梁材料")):
            c = QtWidgets.QComboBox(); [c.addItem(MAT[m]["name"], m) for m in MATS]; add(k, lab, c)
        add("P", "台面集中载荷 kg", spin(0, 20000, 300, 50, 0))
        add("x", "载荷位置 X mm", spin(0, 6000, 500, 50, 0)); add("y", "载荷位置 Y mm", spin(0, 6000, 400, 50, 0))
        self.out = QtWidgets.QTextBrowser(); left.addWidget(self.out, 1)
        lw = QtWidgets.QWidget(); lw.setLayout(left)
        sa = QtWidgets.QScrollArea(); sa.setWidgetResizable(True); sa.setWidget(lw)
        sa.setFixedWidth(420); sa.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        lay.addWidget(sa)
        self.fig = Figure(figsize=(8, 6)); self.cv = FigureCanvasQTAgg(self.fig); lay.addWidget(self.cv, 1)
        self.update_()

    def update_(self, *a):
        v = {k: (w.currentData() if isinstance(w, QtWidgets.QComboBox) else w.value()) for k, w in self.w.items()}
        # 载荷坐标校验（测试反馈）：超出台面时"吸附最近节点"会给出看似合格但没意义的结果——夹回台面并明确警告
        L, W = v["L"], v["W"]
        x, y = v["x"], v["y"]
        self._load_warn = ""
        if not (0 <= x <= L and 0 <= y <= W):
            x2, y2 = min(max(x, 0.0), L), min(max(y, 0.0), W)
            self._load_warn = (f"⚠ 载荷位置 ({x:.0f}, {y:.0f}) 超出机架台面 {L:.0f}×{W:.0f} mm，"
                               f"已按 ({x2:.0f}, {y2:.0f}) 计算。越界坐标的结果没有物理意义，请修正坐标。")
            x, y = x2, y2
        f, sec = build_table(L, W, v["H"], v["t"], v["ml"], v["mt"], v["P"], x, y)
        f.solve()
        warn_html = (f'<p style="color:#b26a00;background:#fff4e5;padding:6px;border-left:3px solid #f0a030">{self._load_warn}</p>'
                     if self._load_warn else "")
        U = f.u[:, :3]; P = np.array(f.nodes)
        dmax = float(np.abs(U).max()); smax = max(e["sigma"] for e in f.elems)
        res = []
        if f.singular or dmax > 1e4:
            self.out.setHtml(warn_html + html([("FAIL", "结构是个机构（有杆件没连上/约束不够），会垮，算不出稳定变形")])); return
        legs = [e for e in f.elems if e["name"] == "立柱"]
        Nmax = max(-e["N"] for e in legs) if legs else 0
        m = MAT[v["ml"]]
        Pcr = math.pi**2 * m["E"] / 1e6 * min(sec.Iy, sec.Iz) / (2 * v["H"]) ** 2
        allow = min(MAT[v["ml"]].get("creep_fy", MAT[v["ml"]]["fy"] / 1.5), MAT[v["mt"]].get("creep_fy", MAT[v["mt"]]["fy"] / 1.5)) / 1e6
        res.append(("OK" if smax < allow else "FAIL", f"最大应力 {smax:.1f} MPa，许用 {allow:.0f} MPa（塑料按长期蠕变许用）"))
        res.append(("OK" if Nmax <= 0 or Pcr / Nmax > 3 else "FAIL", f"立柱屈曲安全系数 {Pcr/max(Nmax,1e-9):.1f}（≥3 合格，底部铰支偏保守）"))
        span = max(v["L"], v["W"])
        res.append(("OK" if dmax < span / 1000 else ("WARN" if dmax < span / 300 else "FAIL"), f"最大变形 {dmax:.3f} mm（设备台面一般要求 ≤ 跨度/1000 = {span/1000:.2f} mm）"))
        feet = [f.reactions[n][2] / 9.81 for n in f.fix]
        res.append(("OK" if min(feet) > 0 else "FAIL", f"地脚受力 {min(feet):.0f}~{max(feet):.0f} kg" + ("（有地脚受拉 = 会翘起/倾倒）" if min(feet) <= 0 else "")))
        mass = sum(e["rho"] * e["sec"].A * e["L"] for e in f.elems)
        self.out.setHtml(warn_html + f"<b>机架自重约 {mass:.1f} kg，型材 {sec.name}</b><br><br>" + html(res))
        fig = self.fig; fig.clear(); ax = fig.add_subplot(111, projection="3d")
        sc = 0.08 * span / max(dmax, 1e-9)
        cm = __import__("matplotlib").pyplot.get_cmap("turbo")
        segs = [[P[e["n1"]] + U[e["n1"]] * sc, P[e["n2"]] + U[e["n2"]] * sc] for e in f.elems]
        ax.add_collection3d(Line3DCollection([[P[e["n1"]], P[e["n2"]]] for e in f.elems], colors="#ccc", linestyles="--", linewidths=1))
        ax.add_collection3d(Line3DCollection(segs, colors=[cm(e["sigma"] / max(smax, 1e-9)) for e in f.elems], linewidths=3))
        ax.set_xlim(0, v["L"]); ax.set_ylim(0, v["W"]); ax.set_zlim(0, v["H"] * 1.1); ax.set_box_aspect((v["L"], v["W"], v["H"] * 1.1))
        ax.set_axis_off(); ax.set_title(f"变形放大 {sc:.0f} 倍，颜色=应力（最大 {smax:.1f} MPa）")
        self.cv.draw()
