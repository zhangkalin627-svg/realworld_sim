"""⑤ 材料库：元素周期表（118 种）/ 常用牌号 / 自己配合金。"""
import numpy as np
from PySide6 import QtCore, QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib
import matplotlib.colors as mcolors

from . import viz  # noqa: F401  注册中文字体
from .matdb.elements import ELEMENTS, EL, describe, category, STRUCT_ZH
from .matdb.alloys import GRADES
from .matdb.mixing import mix, report, wt_to_at, miedema_dH, binary, KIND_ZH

COLOR = {"OK": "#1a7f37", "WARN": "#b26a00", "FAIL": "#c62828"}
CAT_COLOR = {"碱金属": "#ffb3a7", "碱土金属": "#ffd8a8", "过渡金属": "#cfe3ff", "主族金属": "#d3f0d0", "类金属": "#e6e0a8",
             "非金属": "#fff3a8", "惰性气体": "#e3d4ff", "镧系/锕系": "#ffd0e8"}
PROPS = [("类别", None, False), ("密度 kg/m³", "density", True), ("熔点 °C", "melt_C", False),
         ("电导率 (1/电阻率)", "cond", True), ("导热 W/(m·K)", "k", True), ("弹性模量 GPa", "E", True),
         ("晶体结构", "struct", False), ("参考价 USD/kg", "price", True)]
STRUCT_COLOR = {"BCC": "#9ecae1", "FCC": "#fdae6b", "HEX": "#a1d99b", "DIA": "#bcbddc", "CUB": "#d9d9d9", "SC": "#fdd0a2",
                "TET": "#c7e9c0", "ORC": "#fcbba1", "MCL": "#dadaeb", "RHL": "#fee391"}


def html(v):
    import html as _h
    return "<br>".join(f'<span style="color:{COLOR[l]}"><b>[{l}]</b> {_h.escape(t)}</span>' for l, t in v)


def pos(e):
    """元素在 18 列周期表里的 (行, 列)。镧系/锕系放在下面两行。"""
    Z = e["Z"]
    if 57 <= Z <= 71: return 8, 2 + (Z - 57)
    if 89 <= Z <= 103: return 9, 2 + (Z - 89)
    return e["period"] - 1, (e["group"] or 3) - 1


def prop_val(e, key):
    if key == "cond":
        return None if not e["resist"] else 1 / e["resist"]
    if key == "density" and e["density_src"] in ("predicted", "gas_25C"):
        return None if e["density_src"] == "predicted" else e["density"]
    return e.get(key)


class PeriodicPage(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        lay = QtWidgets.QVBoxLayout(self)
        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("颜色按："))
        self.prop = QtWidgets.QComboBox(); [self.prop.addItem(n) for n, _, _ in PROPS]
        self.prop.currentIndexChanged.connect(self.recolor); top.addWidget(self.prop)
        self.legend = QtWidgets.QLabel(); top.addWidget(self.legend, 1)
        lay.addLayout(top)
        grid = QtWidgets.QGridLayout(); grid.setSpacing(2)
        self.btn = {}
        for e in ELEMENTS:
            r, c = pos(e)
            zh = "鉝" if e["Z"] == 116 else e["zh"]   # 𫟷 字体里没有，用旧写法
            b = QtWidgets.QPushButton(f"{e['Z']}\n{e['sym']}\n{zh}")
            b.setMinimumSize(54, 58); b.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
            b.clicked.connect(lambda _=False, s=e["sym"]: self.show_el(s))
            grid.addWidget(b, r + (1 if r >= 8 else 0), c)
            self.btn[e["sym"]] = b
        grid.addWidget(QtWidgets.QLabel(""), 8, 0)
        grid.addWidget(QtWidgets.QLabel("镧系 →"), 9, 1)
        grid.addWidget(QtWidgets.QLabel("锕系 →"), 10, 1)
        gw = QtWidgets.QWidget(); gw.setLayout(grid)
        lay.addWidget(gw, 3)
        self.info = QtWidgets.QTextEdit(); self.info.setReadOnly(True); self.info.setMaximumHeight(220)
        lay.addWidget(self.info, 1)
        self.recolor(); self.show_el("Fe")

    def recolor(self, *a):
        name, key, logscale = PROPS[self.prop.currentIndex()]
        if key is None:
            for s, b in self.btn.items():
                b.setStyleSheet(self._css(CAT_COLOR.get(category(s), "#eeeeee")))
            self.legend.setText("  ".join(f'<span style="background:{c}">&nbsp;{k}&nbsp;</span>' for k, c in CAT_COLOR.items()))
            return
        if key == "struct":
            for s, b in self.btn.items():
                b.setStyleSheet(self._css(STRUCT_COLOR.get(EL[s]["struct"], "#f4f4f4")))
            self.legend.setText("  ".join(f'<span style="background:{STRUCT_COLOR[k]}">&nbsp;{STRUCT_ZH[k]}&nbsp;</span>'
                                          for k in ("BCC", "FCC", "HEX", "DIA", "RHL", "ORC", "TET", "MCL")) + "  灰=无数据")
            return
        vals = {s: prop_val(EL[s], key) for s in self.btn}
        v = [x for x in vals.values() if x is not None and x > 0]
        norm = mcolors.LogNorm(min(v), max(v)) if logscale else mcolors.Normalize(min(v), max(v))
        cmap = matplotlib.colormaps["viridis"]
        for s, b in self.btn.items():
            x = vals[s]
            if x is None or x <= 0:
                b.setStyleSheet(self._css("#f4f4f4", "#aaaaaa"))
            else:
                rgba = cmap(norm(x)); col = mcolors.to_hex(rgba)
                fg = "#ffffff" if (0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]) < 0.55 else "#000000"
                b.setStyleSheet(self._css(col, fg))
        self.legend.setText(f"{name}：深紫 = 低 {min(v):.3g} → 黄 = 高 {max(v):.3g}" + ("（对数刻度）" if logscale else "")
                            + "；灰 = 没有可靠实测数据")

    @staticmethod
    def _css(bg, fg="#000000"):
        return f"QPushButton{{background:{bg};color:{fg};border:1px solid #888;font-size:11px;}}"

    def show_el(self, s):
        self.info.setPlainText(describe(s))


class GradePage(QtWidgets.QWidget):
    COLS = [("牌号", None), ("标准", "std"), ("状态", "state"), ("密度 kg/m³", "rho"), ("E GPa", "E"), ("屈服 MPa", "Rp"),
            ("抗拉 MPa", "Rm"), ("伸长 %", "A"), ("导热 W/mK", "k"), ("电阻率 μΩ·cm", "resist"), ("熔化 °C", "melt"),
            ("强度数值", "basis")]

    def __init__(self):
        super().__init__()
        lay = QtWidgets.QVBoxLayout(self)
        lab = QtWidgets.QLabel("常用牌号的实测/标准性能（“标准最小值”可用于设计；“手册典型值”只作参考，以材质证明书为准）。"
                               "强度栏标明是标准最小值还是典型值；密度、导热、电阻率、熔化温度均为手册典型值。"
                               "双击一行 → 到“自己配合金”页按它的成分配一次，看混合法则和实测差多少。")
        lab.setWordWrap(True); lay.addWidget(lab)
        t = QtWidgets.QTableWidget(len(GRADES), len(self.COLS))
        t.setHorizontalHeaderLabels([c for c, _ in self.COLS])
        for i, (k, g) in enumerate(GRADES.items()):
            for j, (_, key) in enumerate(self.COLS):
                v = g["name"] if key is None else g.get(key)
                if key == "resist" and v: v = f"{v * 1e8:.3g}"
                elif key == "melt" and v: v = f"{v[0]}" if v[0] == v[1] else f"{v[0]}–{v[1]}"
                elif key == "Rm" and isinstance(v, tuple): v = f"{v[0]}–{v[1]}"
                elif key == "basis": v = "标准最小值" if v == "min" else "手册典型值"
                it = QtWidgets.QTableWidgetItem("—" if v is None else str(v))
                if g.get("note"): it.setToolTip(g["note"])
                t.setItem(i, j, it)
        t.resizeColumnsToContents(); t.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        t.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        lay.addWidget(t, 1); self.table = t
        self.note = QtWidgets.QLabel(""); self.note.setWordWrap(True); lay.addWidget(self.note)
        t.currentCellChanged.connect(self._note)

    def _note(self, r, *a):
        if r < 0: return
        g = GRADES[list(GRADES)[r]]
        self.note.setText("说明：" + g.get("note", "—") + "    名义成分 wt%：" +
                          "，".join(f"{EL[e]['zh']}{v:g}" for e, v in g["comp"].items() if v > 0))


PRESETS = [("铁 70 + 银 30（不互溶）", {"Fe": 70, "Ag": 30}), ("铜 62 + 锌 38（=H62 黄铜）", {"Cu": 62, "Zn": 38}),
           ("铜 70 + 镍 30（白铜，任意互溶）", {"Cu": 70, "Ni": 30}), ("钨 70 + 铜 30（粉末冶金）", {"W": 70, "Cu": 30}),
           ("铝 50 + 镍 50（金属间化合物）", {"Al": 50, "Ni": 50}), ("铁 99.2 + 碳 0.8（碳钢）", {"Fe": 99.2, "C": 0.8}),
           ("铁 71 + 铬 19 + 镍 10（类似 304）", {"Fe": 71, "Cr": 19, "Ni": 10}),
           ("金 75 + 银 12.5 + 铜 12.5（18K）", {"Au": 75, "Ag": 12.5, "Cu": 12.5}), ("铁 90 + 氧 10", {"Fe": 90, "O": 10})]


class MixPage(QtWidgets.QWidget):
    N = 8

    def __init__(self):
        super().__init__()
        lay = QtWidgets.QHBoxLayout(self)
        left = QtWidgets.QVBoxLayout()
        self.preset = QtWidgets.QComboBox(); self.preset.addItem("— 例子 —")
        [self.preset.addItem(n) for n, _ in PRESETS]
        self.preset.currentIndexChanged.connect(self.load_preset)
        left.addWidget(self.preset)
        g = QtWidgets.QGroupBox("配比（质量 %，自动归一化）"); f = QtWidgets.QGridLayout(g)
        self.rows = []
        syms = [f"{e['sym']} {e['zh']}" for e in ELEMENTS]
        for i in range(self.N):
            cb = QtWidgets.QComboBox(); cb.addItem("（空）"); cb.addItems(syms)
            sp = QtWidgets.QDoubleSpinBox(); sp.setRange(0, 100); sp.setDecimals(3); sp.setSingleStep(1)
            cb.currentIndexChanged.connect(self.update_); sp.valueChanged.connect(self.update_)
            f.addWidget(cb, i // 2, 2 * (i % 2)); f.addWidget(sp, i // 2, 2 * (i % 2) + 1); self.rows.append((cb, sp))
        left.addWidget(g)
        self.out = QtWidgets.QTextBrowser(); left.addWidget(self.out, 1)
        lw = QtWidgets.QWidget(); lw.setLayout(left); lw.setFixedWidth(560); lay.addWidget(lw)
        self.fig = Figure(figsize=(7, 5)); self.cv = FigureCanvasQTAgg(self.fig); lay.addWidget(self.cv, 1)
        self._busy = False
        self.set_comp({"Fe": 70, "Ag": 30})

    def load_preset(self, i):
        if i > 0:
            self.set_comp(PRESETS[i - 1][1])

    def set_comp(self, comp):
        self._busy = True
        items = sorted(((k, v) for k, v in comp.items() if v > 0), key=lambda kv: -kv[1])[:self.N]
        for i, (cb, sp) in enumerate(self.rows):
            if i < len(items):
                cb.setCurrentIndex(EL[items[i][0]]["Z"]); sp.setValue(items[i][1])
            else:
                cb.setCurrentIndex(0); sp.setValue(0)
        self._busy = False
        self.update_()

    def comp(self):
        c = {}
        for cb, sp in self.rows:
            if cb.currentIndex() > 0 and sp.value() > 0:
                s = ELEMENTS[cb.currentIndex() - 1]["sym"]
                c[s] = c.get(s, 0) + sp.value()
        return c

    def update_(self, *a):
        if self._busy:
            return
        c = self.comp()
        if not c:
            self.out.setPlainText("选元素、填质量百分比"); return
        r = mix(c)
        at = "，".join(f"{EL[e]['zh']} {v:.1f}" for e, v in r["at"].items())
        self.out.setHtml(f"<b>原子百分比：</b>{at}<br><br>" + html(report(r)))
        self.plot(r)

    def plot(self, r):
        f = self.fig; f.clear()
        els = sorted(r["wt"], key=r["wt"].get, reverse=True)
        if len(els) >= 2:
            A, B = els[0], els[1]
            xs = np.linspace(0, 100, 101)
            ax = f.add_subplot(211)
            if EL[A]["density"] and EL[B]["density"] and "gas_25C" not in (EL[A]["density_src"], EL[B]["density_src"]):
                rho = np.array([1 / ((100 - x) / 100 / EL[A]["density"] + x / 100 / EL[B]["density"]) for x in xs])
                ax.fill_between(xs, rho * 0.97, rho * 1.035, color="0.85", label="实际合金常见范围 ±3%")
                ax.plot(xs, rho, "k-", label="理想混合密度")
                xb = r["wt"][B] / (r["wt"][A] + r["wt"][B]) * 100
                ax.plot([xb], [np.interp(xb, xs, rho)], "ro", label="当前配比")
                ax.set_ylabel("密度 kg/m³"); ax.legend(fontsize=8, loc="best")
            ax.set_title(f"{EL[A]['zh']}–{EL[B]['zh']} 二元（横轴 = {EL[B]['zh']} 质量 %）", fontsize=10)
            ax2 = f.add_subplot(212)
            H = []
            for x in xs:
                at = wt_to_at({A: max(100 - x, 1e-6), B: max(x, 1e-6)})
                H.append(miedema_dH(A, B, at[A] / 100))
            kind, txt, src = binary(A, B)
            if H[50] is not None:
                ax2.plot(xs, H, "b-"); ax2.axhline(0, color="0.5", lw=0.8)
                ax2.set_ylabel("混合焓 kJ/mol")
                ax2.text(0.02, 0.92, ">0 互相排斥（易分层）   <0 互相吸引（易成化合物）", transform=ax2.transAxes, fontsize=8, va="top")
            else:
                ax2.text(0.5, 0.5, "缺 Miedema 参数（非金属/气体/超重元素）", ha="center", transform=ax2.transAxes)
            ax2.set_title(f"判断：{KIND_ZH[kind]}  [依据：{src}]", fontsize=9, color="#c62828" if kind in ("liq_gap", "compound", "inert", "na") else "black")
            ax2.set_xlabel(f"{EL[B]['zh']} 质量 %")
        else:
            ax = f.add_subplot(111); ax.text(0.5, 0.5, "纯元素", ha="center", transform=ax.transAxes)
        f.tight_layout(); self.cv.draw()


class MaterialsPage(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        lay = QtWidgets.QVBoxLayout(self)
        t = QtWidgets.QTabWidget()
        self.pt = PeriodicPage(); self.gr = GradePage(); self.mx = MixPage()
        t.addTab(self.pt, "元素周期表（118 种）"); t.addTab(self.gr, "常用牌号 实测性能"); t.addTab(self.mx, "自己配合金")
        lay.addWidget(t); self.tabs = t
        self.gr.table.cellDoubleClicked.connect(lambda r, c: (self.mx.set_comp(GRADES[list(GRADES)[r]]["comp"]), t.setCurrentIndex(2)))
