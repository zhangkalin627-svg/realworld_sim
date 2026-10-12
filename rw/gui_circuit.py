"""⑥ 电路/示波器：选示例或打开 KiCad 网表 → ngspice 仿真 → 虚拟示波器（4 通道、10×8 格、触发、光标、自动测量、FFT）。"""
import glob, os, traceback
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from . import viz  # noqa: F401  注册中文字体
from .circuit import examples as EX, kicad as KI
from .circuit.spice import find_ngspice, SpiceError
from .circuit.scope import Scope, nice_steps, eng, COLORS, HDIV, VDIV, WINDOWS

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRIVATE_DIR = os.path.join(HERE, "models", "private_boards")
PUBLIC_NET_DIR = os.path.join(HERE, "models", "netlists")
VSTEPS = nice_steps(1e-3, 500)
TSTEPS = nice_steps(1e-9, 10)
MEAS = ["Vpp", "Vmax", "Vmin", "Vavg", "Vrms", "频率", "周期", "占空比", "上升时间", "下降时间"]


def unit_of(src):
    return "A" if src.strip().lower().startswith("i(") else "V"


class SimWorker(QtCore.QObject):
    done = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self, circuit, analysis):
        super().__init__()
        self.c, self.a = circuit, analysis

    @QtCore.Slot()
    def run(self):
        try:
            self.done.emit(self.c.run(self.a))
        except SpiceError as e:
            self.failed.emit(str(e))
        except Exception:
            self.failed.emit(traceback.format_exc())


class ScopeCanvas(FigureCanvasQTAgg):
    clicked = QtCore.Signal(float, float, int)  # x(格) y(格) 鼠标键

    def __init__(self):
        self.fig = Figure(figsize=(8, 6), facecolor="#1b1d22")
        super().__init__(self.fig)
        self.mpl_connect("button_press_event", self._click)
        self.ax = None

    def _click(self, ev):
        if ev.inaxes is self.ax and ev.xdata is not None:
            self.clicked.emit(float(ev.xdata), float(ev.ydata), int(ev.button))


class ChannelRow(QtWidgets.QWidget):
    changed = QtCore.Signal()

    def __init__(self, n):
        super().__init__()
        g = QtWidgets.QGridLayout(self); g.setContentsMargins(2, 2, 2, 2); g.setHorizontalSpacing(4)
        self.on = QtWidgets.QCheckBox(f"CH{n}")
        self.on.setStyleSheet(f"QCheckBox {{ color: black; background:{COLORS[n-1]}; padding:2px 6px; border-radius:3px; font-weight:bold }}")
        self.src = QtWidgets.QComboBox(); self.src.setEditable(True); self.src.setMinimumWidth(120)
        self.src.setToolTip("网络名；两网络差分写成 a-b；电流写成 i(V1)")
        self.vdiv = QtWidgets.QComboBox()
        for v in VSTEPS:
            self.vdiv.addItem(eng(v, "V/格", 3), v)
        self.off = QtWidgets.QDoubleSpinBox(); self.off.setRange(-1e4, 1e4); self.off.setDecimals(3); self.off.setSuffix(" V")
        self.off.setToolTip("偏移：把波形整体上下移（和真机一样以电压计）")
        self.cpl = QtWidgets.QComboBox(); self.cpl.addItems(["DC", "AC"])
        g.addWidget(self.on, 0, 0); g.addWidget(self.src, 0, 1, 1, 3)
        g.addWidget(QtWidgets.QLabel("伏/格"), 1, 0); g.addWidget(self.vdiv, 1, 1)
        g.addWidget(QtWidgets.QLabel("偏移"), 1, 2); g.addWidget(self.off, 1, 3)
        g.addWidget(QtWidgets.QLabel("耦合"), 2, 0); g.addWidget(self.cpl, 2, 1)
        for w in (self.on,):
            w.toggled.connect(self.changed)
        self.src.currentTextChanged.connect(self.changed)
        self.vdiv.currentIndexChanged.connect(self.changed)
        self.off.valueChanged.connect(self.changed)
        self.cpl.currentIndexChanged.connect(self.changed)

    def set_vdiv(self, v):
        i = int(np.argmin([abs(np.log(x / v)) for x in VSTEPS]))
        self.vdiv.setCurrentIndex(i)


class CircuitPage(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.scope = Scope(4)
        self.wave = None
        self.circuit = None
        self._block = False
        lay = QtWidgets.QHBoxLayout(self)

        # ---------------- 左：电路
        left = QtWidgets.QVBoxLayout()
        gb = QtWidgets.QGroupBox("电路")
        f = QtWidgets.QVBoxLayout(gb)
        self.pick = QtWidgets.QComboBox()
        self.fill_examples()
        f.addWidget(self.pick)
        row = QtWidgets.QHBoxLayout()
        bo = QtWidgets.QPushButton("打开网表…"); bo.clicked.connect(self.open_file); row.addWidget(bo)
        bn = QtWidgets.QPushButton("查看网表"); bn.clicked.connect(self.show_netlist); row.addWidget(bn)
        f.addLayout(row)
        self.theory = QtWidgets.QLabel(""); self.theory.setWordWrap(True)
        self.theory.setStyleSheet("color:#333; background:#f3f5f8; padding:4px; border-radius:4px")
        f.addWidget(self.theory)
        f.addWidget(QtWidgets.QLabel("分析命令（可改）："))
        self.ana = QtWidgets.QLineEdit()
        f.addWidget(self.ana)
        self.btn = QtWidgets.QPushButton("▶  仿真（ngspice）"); self.btn.setMinimumHeight(36); self.btn.clicked.connect(self.simulate)
        f.addWidget(self.btn)
        self.status = QtWidgets.QLabel(""); self.status.setWordWrap(True); f.addWidget(self.status)
        left.addWidget(gb)
        gm = QtWidgets.QGroupBox("自动测量（屏幕内波形）")
        fm = QtWidgets.QVBoxLayout(gm)
        self.table = QtWidgets.QTableWidget(len(MEAS), 4)
        self.table.setVerticalHeaderLabels(MEAS)
        self.table.setHorizontalHeaderLabels([f"CH{i}" for i in range(1, 5)])
        for i in range(4):
            it = self.table.horizontalHeaderItem(i); it.setForeground(QtGui.QColor("black")); it.setBackground(QtGui.QColor(COLORS[i]))
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.table.verticalHeader().setDefaultSectionSize(22)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        fm.addWidget(self.table)
        left.addWidget(gm, 1)
        lw = QtWidgets.QWidget(); lw.setLayout(left); lw.setFixedWidth(400)
        lay.addWidget(lw)

        # ---------------- 中：示波器屏幕
        self.cv = ScopeCanvas()
        self.cv.clicked.connect(self.on_click)
        lay.addWidget(self.cv, 1)

        # ---------------- 右：面板
        self.ctl = QtWidgets.QTabWidget(); self.ctl.setFixedWidth(330)
        # 垂直
        pv = QtWidgets.QWidget(); v = QtWidgets.QVBoxLayout(pv)
        self.rows = []
        for n in range(1, 5):
            r = ChannelRow(n); r.changed.connect(self.from_ui); self.rows.append(r); v.addWidget(r)
            line = QtWidgets.QFrame(); line.setFrameShape(QtWidgets.QFrame.HLine); v.addWidget(line)
        ba = QtWidgets.QPushButton("Auto Set（自动设置）"); ba.clicked.connect(self.autoset); v.addWidget(ba)
        v.addStretch(1)
        self.ctl.addTab(pv, "垂直")
        # 水平/触发
        ph = QtWidgets.QWidget(); h = QtWidgets.QFormLayout(ph)
        self.dial = QtWidgets.QDial(); self.dial.setRange(0, len(TSTEPS) - 1); self.dial.setNotchesVisible(True)
        self.dial.setFixedHeight(90); self.dial.setWrapping(False)
        self.tdiv = QtWidgets.QComboBox()
        for t in TSTEPS:
            self.tdiv.addItem(eng(t, "s/格", 3), t)
        self.dial.valueChanged.connect(self.tdiv.setCurrentIndex)
        self.tdiv.currentIndexChanged.connect(self.dial.setValue)
        self.tdiv.currentIndexChanged.connect(self.from_ui)
        h.addRow("时基旋钮", self.dial)
        h.addRow("时基", self.tdiv)
        self.hpos = QtWidgets.QDoubleSpinBox(); self.hpos.setRange(-1000, 1000); self.hpos.setDecimals(2); self.hpos.setSuffix(" 格")
        self.hpos.setToolTip("触发点相对屏幕中心的位置；正值 = 看触发之后更远的波形")
        self.hpos.valueChanged.connect(self.from_ui)
        h.addRow("水平位置", self.hpos)
        self.hpos_lab = QtWidgets.QLabel(""); h.addRow("", self.hpos_lab)
        h.addRow(QtWidgets.QLabel("<b>触发（边沿）</b>"))
        self.tsrc = QtWidgets.QComboBox(); self.tsrc.addItems([f"CH{i}" for i in range(1, 5)]); self.tsrc.currentIndexChanged.connect(self.from_ui)
        h.addRow("触发源", self.tsrc)
        self.slope = QtWidgets.QComboBox(); self.slope.addItems(["上升沿 ↑", "下降沿 ↓"]); self.slope.currentIndexChanged.connect(self.from_ui)
        h.addRow("边沿", self.slope)
        self.level = QtWidgets.QDoubleSpinBox(); self.level.setRange(-1e5, 1e5); self.level.setDecimals(3); self.level.setSuffix(" V")
        self.level.valueChanged.connect(self.from_ui)
        h.addRow("电平", self.level)
        self.tmode = QtWidgets.QComboBox(); self.tmode.addItems(["Auto", "Normal", "Single"]); self.tmode.currentIndexChanged.connect(self.from_ui)
        h.addRow("模式", self.tmode)
        rb = QtWidgets.QHBoxLayout()
        self.run_btn = QtWidgets.QPushButton("▶ 运行"); self.run_btn.setCheckable(True); self.run_btn.toggled.connect(self.toggle_run)
        self.run_btn.setToolTip("逐个触发点刷新（仿真记录回放，相当于示波器连续采集）")
        bs = QtWidgets.QPushButton("单次"); bs.clicked.connect(self.single)
        rb.addWidget(self.run_btn); rb.addWidget(bs)
        h.addRow(rb)
        self.ctl.addTab(ph, "水平/触发")
        # 光标
        pc = QtWidgets.QWidget(); c = QtWidgets.QFormLayout(pc)
        self.cur_t = QtWidgets.QCheckBox("时间光标 t1/t2"); self.cur_v = QtWidgets.QCheckBox("电压光标 V1/V2")
        self.cur_t.toggled.connect(self.from_ui); self.cur_v.toggled.connect(self.from_ui)
        c.addRow(self.cur_t)
        self.t1 = QtWidgets.QDoubleSpinBox(); self.t2 = QtWidgets.QDoubleSpinBox()
        for s, val in ((self.t1, 2.0), (self.t2, 6.0)):
            s.setRange(0, 10); s.setDecimals(3); s.setSingleStep(0.1); s.setSuffix(" 格"); s.setValue(val); s.valueChanged.connect(self.from_ui)
        c.addRow("t1（左键点屏幕）", self.t1); c.addRow("t2（右键点屏幕）", self.t2)
        c.addRow(self.cur_v)
        self.vch = QtWidgets.QComboBox(); self.vch.addItems([f"CH{i}" for i in range(1, 5)]); self.vch.currentIndexChanged.connect(self.from_ui)
        c.addRow("电压光标所属通道", self.vch)
        self.v1 = QtWidgets.QDoubleSpinBox(); self.v2 = QtWidgets.QDoubleSpinBox()
        for s, val in ((self.v1, -1.0), (self.v2, 1.0)):
            s.setRange(-4, 4); s.setDecimals(3); s.setSingleStep(0.1); s.setSuffix(" 格"); s.setValue(val); s.valueChanged.connect(self.from_ui)
        c.addRow("V1", self.v1); c.addRow("V2", self.v2)
        self.cur_out = QtWidgets.QLabel("—"); self.cur_out.setWordWrap(True)
        self.cur_out.setStyleSheet("font-family: monospace; background:#f3f5f8; padding:4px")
        c.addRow(self.cur_out)
        self.ctl.addTab(pc, "光标")
        # FFT
        pf = QtWidgets.QWidget(); ff = QtWidgets.QFormLayout(pf)
        self.fft_on = QtWidgets.QCheckBox("显示 FFT（屏幕下方）"); self.fft_on.toggled.connect(self.from_ui)
        ff.addRow(self.fft_on)
        self.fft_ch = QtWidgets.QComboBox(); self.fft_ch.addItems([f"CH{i}" for i in range(1, 5)]); self.fft_ch.currentIndexChanged.connect(self.from_ui)
        ff.addRow("信号源", self.fft_ch)
        self.win = QtWidgets.QComboBox()
        for k, vv in WINDOWS.items():
            self.win.addItem(k, vv)
        self.win.setCurrentIndex(1); self.win.currentIndexChanged.connect(self.from_ui)
        ff.addRow("窗函数", self.win)
        self.fmax = QtWidgets.QDoubleSpinBox(); self.fmax.setRange(0, 1e6); self.fmax.setDecimals(1); self.fmax.setSuffix(" kHz")
        self.fmax.setSpecialValueText("自动"); self.fmax.valueChanged.connect(self.from_ui)
        ff.addRow("频率上限", self.fmax)
        self.fft_span = QtWidgets.QComboBox(); self.fft_span.addItems(["屏幕内波形", "整段记录"]); self.fft_span.currentIndexChanged.connect(self.from_ui)
        ff.addRow("数据", self.fft_span)
        self.fft_out = QtWidgets.QLabel("—"); self.fft_out.setWordWrap(True)
        self.fft_out.setStyleSheet("font-family: monospace; background:#f3f5f8; padding:4px")
        ff.addRow(self.fft_out)
        ff.addRow(QtWidgets.QLabel("dBV = 20·lg(Vrms / 1 V)；分辨率 = 1/(屏幕时间)"))
        self.ctl.addTab(pf, "FFT")
        lay.addWidget(self.ctl)

        self.timer = QtCore.QTimer(self); self.timer.timeout.connect(lambda: self.redraw(advance=True))
        self.pick.currentIndexChanged.connect(self.load_choice)
        self.load_choice()
        if not find_ngspice():
            self.status.setText("<span style='color:#c62828'>没找到 ngspice：Ubuntu 上运行 sudo apt install ngspice</span>")

    # ------------------------------------------------ 电路选择
    def fill_examples(self):
        self.pick.blockSignals(True); self.pick.clear()
        for k, n in EX.NAMES.items():
            self.pick.addItem("示例：" + n, ("ex", k))
        for p in sorted(glob.glob(os.path.join(PUBLIC_NET_DIR, "*.cir")) + glob.glob(os.path.join(PUBLIC_NET_DIR, "*.net"))):
            self.pick.addItem("KiCad 网表：" + os.path.basename(p), ("file", p))
        for p in sorted(glob.glob(os.path.join(PRIVATE_DIR, "*"))):
            if os.path.splitext(p)[1].lower() in (".cir", ".net", ".sp", ".spice"):
                self.pick.addItem("私有板：" + os.path.basename(p), ("file", p))
        self.pick.blockSignals(False)

    def open_file(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "打开 KiCad 网表", PRIVATE_DIR if os.path.isdir(PRIVATE_DIR) else HERE,
                                                     "网表 (*.cir *.net *.sp *.spice);;所有文件 (*)")
        if p:
            self.load_file(p)

    def load_choice(self, *_):
        kind, val = self.pick.currentData()
        if kind == "ext":          # 外部送来的波形（例如 ⑦ 单片机/PLC 联合仿真），没有电路可仿真
            self.circuit = None; return
        if kind == "ex":
            e = EX.get(val)
            self.example, self.circuit = e, e.circuit
            t = e.tran
            self.ana.setText(f".tran {t['step']:g} {t['stop']:g} 0 {t.get('tmax', t['step']):g}" + (" uic" if t.get("uic") else ""))
            self.theory.setText("理论值：" + e.theory)
        else:
            self.load_file(val)

    def load_file(self, p):
        try:
            c = KI.load(p)
        except Exception as ex:
            self.status.setText(f"<span style='color:#c62828'>导入失败：{ex}</span>"); return
        self.example, self.circuit = None, c
        self.ana.setText(KI.default_analysis(c, ".tran 1u 10m"))
        w = getattr(c, "warnings", [])
        self.theory.setText(f"导入 {os.path.basename(p)}：{len(c.elements)} 个元件，{len(c.nodes())} 个网络。"
                            + ("<br><span style='color:#b26a00'>" + "<br>".join(w[:6]) + "</span>" if w else ""))

    def show_netlist(self):
        if not self.circuit:
            return
        d = QtWidgets.QDialog(self); d.setWindowTitle("网表"); d.resize(700, 500)
        t = QtWidgets.QPlainTextEdit(self.circuit.netlist(self.ana.text())); t.setReadOnly(True)
        QtWidgets.QVBoxLayout(d).addWidget(t); d.exec()

    # ------------------------------------------------ 仿真
    def simulate(self):
        if not self.circuit:
            return
        self.btn.setEnabled(False); self.status.setText("ngspice 计算中 …")
        self.th = QtCore.QThread(); self.wk = SimWorker(self.circuit, self.ana.text().strip())
        self.wk.moveToThread(self.th); self.th.started.connect(self.wk.run)
        self.wk.done.connect(self.on_done); self.wk.failed.connect(self.on_fail)
        self.wk.done.connect(self.th.quit); self.wk.failed.connect(self.th.quit)
        self.th.start()

    def on_fail(self, msg):
        self.btn.setEnabled(True)
        self.status.setText("<span style='color:#c62828'>仿真失败</span>")
        QtWidgets.QMessageBox.warning(self, "ngspice", msg[:3000])

    def on_done(self, w):
        self.btn.setEnabled(True)
        self.wave = w
        if not w.xname.startswith("time"):
            self.status.setText(f"得到 {w.plotname}（{len(w.x)} 点）。示波器只显示瞬态（.tran）结果。"); return
        self.scope.load(w)
        nets = w.nets()
        self._block = True
        for r in self.rows:
            cur = r.src.currentText(); r.src.clear(); r.src.addItems([""] + nets); r.src.setEditText(cur)
        if self.example is not None:
            self.apply_preset(self.example.scope)
        else:
            vs = [n for n in nets if not n.startswith("i(")][:4]
            for i, r in enumerate(self.rows):
                r.on.setChecked(i < len(vs)); r.src.setEditText(vs[i] if i < len(vs) else "")
            self._block = False; self.from_ui(); self.autoset()
            self.status.setText(f"完成：{len(w.x)} 个时间点，记录 {eng(w.x[-1] - w.x[0], 's')}（已 Auto Set）"); return
        self._block = False
        self.status.setText(f"完成：{len(w.x)} 个时间点，记录 {eng(w.x[-1] - w.x[0], 's')}")
        self.from_ui()

    def apply_preset(self, p):
        for i, r in enumerate(self.rows):
            if i < len(p["ch"]):
                src, vd, off = p["ch"][i]
                r.on.setChecked(True); r.src.setEditText(src); r.set_vdiv(vd); r.off.setValue(off); r.cpl.setCurrentIndex(0)
            else:
                r.on.setChecked(False)
        self.tdiv.setCurrentIndex(int(np.argmin([abs(np.log(x / p["tdiv"])) for x in TSTEPS])))
        tdv = self.tdiv.currentData()
        self.hpos.setValue(p.get("hpos", 0) / tdv)
        ch, sl, lv = p["trig"]
        self.tsrc.setCurrentIndex(ch - 1); self.slope.setCurrentIndex(0 if sl == "rise" else 1); self.level.setValue(lv)
        self.tmode.setCurrentIndex(0)

    def autoset(self):
        if self.scope.t is None:
            return
        self.from_ui(draw=False)
        self.scope.autoset()
        self._block = True
        for r, c in zip(self.rows, self.scope.ch):
            r.set_vdiv(c.vdiv); r.off.setValue(c.offset)
        self.tdiv.setCurrentIndex(int(np.argmin([abs(np.log(x / self.scope.tdiv)) for x in TSTEPS])))
        self.hpos.setValue(0); self.tsrc.setCurrentIndex(self.scope.trig.source - 1); self.slope.setCurrentIndex(0)
        self.level.setValue(self.scope.trig.level)
        self._block = False
        self.from_ui()

    # ------------------------------------------------ 面板 → 引擎
    def from_ui(self, *_, draw=True):
        if self._block:
            return
        s = self.scope
        for r, c in zip(self.rows, s.ch):
            c.on, c.source = r.on.isChecked(), r.src.currentText().strip()
            u = unit_of(c.source)
            if getattr(r, "_unit", None) != u:  # 电流通道显示 A/格
                r._unit = u
                for i, v in enumerate(VSTEPS):
                    r.vdiv.setItemText(i, eng(v, u + "/格", 3))
                r.off.setSuffix(" " + u)
            c.vdiv, c.offset, c.coupling = r.vdiv.currentData(), r.off.value(), r.cpl.currentText()
        s.tdiv = self.tdiv.currentData() or 1e-3
        s.hpos = self.hpos.value() * s.tdiv
        self.hpos_lab.setText(f"= {eng(s.hpos, 's')}（触发点在中心{'右' if s.hpos < 0 else '左'}侧 {abs(self.hpos.value()):.2f} 格）")
        s.trig.source = self.tsrc.currentIndex() + 1
        s.trig.slope = "rise" if self.slope.currentIndex() == 0 else "fall"
        s.trig.level = self.level.value()
        mode = self.tmode.currentText()
        if mode != s.trig.mode:
            s.trig.mode = mode; s.arm()
        self.level.setSuffix(" " + unit_of(s.ch[s.trig.source - 1].source))
        if draw:
            self.redraw()

    def toggle_run(self, on):
        self.run_btn.setText("■ 停止" if on else "▶ 运行")
        if on:
            self.timer.start(400)
        else:
            self.timer.stop()

    def single(self):
        self.tmode.setCurrentIndex(2); self.scope.arm(); self.redraw()

    def on_click(self, x, y, button):
        if button == 1:
            self.cur_t.setChecked(True); self.t1.setValue(x)
        elif button == 3:
            self.cur_t.setChecked(True); self.t2.setValue(x)

    # ------------------------------------------------ 画屏幕
    def redraw(self, advance=False):
        s = self.scope
        fig = self.cv.fig; fig.clear()
        fft = self.fft_on.isChecked() and s.t is not None
        if fft:
            gs = fig.add_gridspec(2, 1, height_ratios=[3, 1.6], hspace=0.28, left=0.04, right=0.96, top=0.95, bottom=0.07)
            ax = fig.add_subplot(gs[0]); axf = fig.add_subplot(gs[1])
        else:
            ax = fig.add_axes([0.04, 0.07, 0.92, 0.86])
        self.cv.ax = ax
        ax.set_facecolor("#06090d")
        ax.set_xlim(0, HDIV); ax.set_ylim(-VDIV / 2, VDIV / 2)
        ax.set_xticks(range(HDIV + 1)); ax.set_yticks(range(-VDIV // 2, VDIV // 2 + 1))
        ax.set_xticklabels([]); ax.set_yticklabels([])
        ax.tick_params(length=0)
        ax.grid(True, color="#3a4250", lw=0.6, ls=":")
        ax.axhline(0, color="#56606f", lw=0.8); ax.axvline(HDIV / 2, color="#56606f", lw=0.8)
        for k in np.arange(0, HDIV, 0.2):
            ax.plot([k, k], [-0.06, 0.06], color="#56606f", lw=0.6)
        for k in np.arange(-VDIV / 2, VDIV / 2, 0.2):
            ax.plot([4.94, 5.06], [k, k], color="#56606f", lw=0.6)
        for sp in ax.spines.values():
            sp.set_color("#56606f")
        if s.t is None:
            ax.text(5, 0, "选电路 → 点“仿真”", color="#9aa4b2", ha="center", va="center", fontsize=14)
            self.cv.draw(); return
        fr = s.acquire(advance=advance)
        # 波形
        for c in s.ch:
            if not (c.on and c.source) or c.n not in fr["traces"]:
                continue
            x, y = fr["traces"][c.n]
            yd = np.clip((y + c.offset) / c.vdiv, -VDIV / 2 - 0.3, VDIV / 2 + 0.3)
            ax.plot(x / s.tdiv, yd, color=c.color, lw=1.2)
            g = np.clip(c.offset / c.vdiv, -VDIV / 2, VDIV / 2)
            ax.annotate(f"{c.n}", xy=(0, g), xytext=(-0.05, g), color="black", fontsize=8, ha="right", va="center",
                        annotation_clip=False, bbox=dict(boxstyle="rarrow,pad=0.15", fc=c.color, ec="none"))
        # 触发位置/电平
        tc = s.ch[s.trig.source - 1]
        if fr.get("tt") is not None:
            xt = HDIV / 2 - s.hpos / s.tdiv
            if 0 <= xt <= HDIV:
                ax.text(xt, VDIV / 2 - 0.02, "▼T", color="#ff9f1a", ha="center", va="top", fontsize=9)
        if tc.source:
            yl = (s.trig.level + tc.offset) / tc.vdiv
            if -VDIV / 2 <= yl <= VDIV / 2:
                ax.annotate("◀T", xy=(HDIV, yl), xytext=(HDIV + 0.05, yl), color=tc.color, va="center", fontsize=9, annotation_clip=False)
                ax.axhline(yl, color=tc.color, lw=0.5, ls=(0, (2, 6)), alpha=0.6)
        # 光标
        cur = s.cursors
        cur["t1"] = cur["t2"] = cur["v1"] = cur["v2"] = None
        txt = []
        if self.cur_t.isChecked():
            cur["t1"], cur["t2"] = self.t1.value() * s.tdiv, self.t2.value() * s.tdiv
            for v in (self.t1.value(), self.t2.value()):
                ax.axvline(v, color="#ffffff", lw=0.9, ls="--")
        if self.cur_v.isChecked():
            vc = s.ch[self.vch.currentIndex()]
            cur["v1"], cur["v2"] = self.v1.value() * vc.vdiv - vc.offset, self.v2.value() * vc.vdiv - vc.offset
            for v in (self.v1.value(), self.v2.value()):
                ax.axhline(v, color=vc.color, lw=0.9, ls="--")
        ro = s.cursor_readout(fr)
        if "Δt" in ro:
            txt += [f"t1={eng(cur['t1'] + fr['t0'], 's')}  t2={eng(cur['t2'] + fr['t0'], 's')}",
                    f"Δt={eng(ro['Δt'], 's')}   1/Δt={eng(ro['1/Δt'], 'Hz')}"]
            for c in s.ch:
                if f"CH{c.n}@t1" in ro:
                    u = unit_of(c.source)
                    txt.append(f"CH{c.n}: {eng(ro[f'CH{c.n}@t1'], u)} → {eng(ro[f'CH{c.n}@t2'], u)}")
        if "ΔV" in ro:
            u = unit_of(s.ch[self.vch.currentIndex()].source)
            txt.append(f"V1={eng(cur['v1'], u)}  V2={eng(cur['v2'], u)}  ΔV={eng(ro['ΔV'], u)}")
        self.cur_out.setText("\n".join(txt) if txt else "—")
        if txt:
            ax.text(0.1, VDIV / 2 - 0.15, "\n".join(txt), color="white", fontsize=8, va="top", family="monospace",
                    bbox=dict(fc="#000000", ec="#56606f", alpha=0.75))
        # 上/下标注
        tn = fr.get("trig_no")
        top = f"{fr['status']}" + (f"  #{tn[0] + 1}/{tn[1]}" if tn else "") + \
              f"    {eng(s.tdiv, 's/格')}    T={eng(fr['t0'] + HDIV / 2 * s.tdiv, 's')}(中心)" + \
              f"    触发 CH{s.trig.source} {'↑' if s.trig.slope == 'rise' else '↓'} {eng(s.trig.level, unit_of(tc.source))}  {s.trig.mode}"
        ax.set_title(top, color="#d6dbe3", fontsize=9, loc="left")
        bot = "    ".join(f"CH{c.n} {c.source}  {eng(c.vdiv, unit_of(c.source) + '/格', 3)} {c.coupling}"
                           for c in s.ch if c.on and c.source)
        ax.set_xlabel(bot, color="#d6dbe3", fontsize=8.5)
        # 测量表
        for j, c in enumerate(s.ch):
            m = s.measure(c.n, fr) if c.on and c.source else {}
            u = unit_of(c.source)
            for i, k in enumerate(MEAS):
                v = m.get(k)
                if v is None:
                    t = "—"
                elif k == "频率":
                    t = eng(v, "Hz")
                elif k == "占空比":
                    t = f"{100 * v:.2f} %"
                elif k in ("周期", "上升时间", "下降时间"):
                    t = eng(v, "s")
                else:
                    t = eng(v, u)
                self.table.setItem(i, j, QtWidgets.QTableWidgetItem(t))
        # FFT
        if fft:
            n = self.fft_ch.currentIndex() + 1
            c = s.ch[n - 1]
            axf.set_facecolor("#06090d")
            for sp in axf.spines.values():
                sp.set_color("#56606f")
            axf.tick_params(colors="#c8ced8", labelsize=8)
            axf.grid(True, color="#3a4250", lw=0.6, ls=":")
            if c.source and (n in fr["traces"] or self.fft_span.currentIndex() == 1):
                f, db, vr = s.fft(n, self.win.currentData(), "screen" if self.fft_span.currentIndex() == 0 else "all", fr)
                m = s.measure(n, fr)
                fm = self.fmax.value() * 1e3
                if fm <= 0:
                    fm = 20 * m["频率"] if m.get("频率") else f[-1]
                fm = min(fm, f[-1])
                k = f <= fm
                axf.plot(f[k], db[k], color=c.color, lw=1.0)
                axf.set_xlim(0, fm)
                axf.xaxis.set_major_formatter(lambda v, _: eng(v, "Hz", 3))
                top_db = np.max(db[k][1:]) if k.sum() > 2 else 0
                axf.set_ylim(top_db - 100, top_db + 10)
                axf.set_ylabel("dBV", color="#c8ced8", fontsize=8)
                axf.set_title(f"FFT  CH{n}  {self.win.currentText()}窗   纵轴 dBV   横轴 0~{eng(fm, 'Hz')}   分辨率 {eng(f[1], 'Hz')}", color="#d6dbe3", fontsize=8.5, loc="left")
                # 前 5 个峰
                idx = np.argsort(vr[1:][f[1:] <= fm])[::-1]
                ff = f[1:][f[1:] <= fm]; vv = vr[1:][f[1:] <= fm]; dd = db[1:][f[1:] <= fm]
                peaks = []
                for i in idx:
                    if all(abs(ff[i] - p) > 3 * f[1] for p, _, _ in peaks):
                        peaks.append((ff[i], vv[i], dd[i]))
                    if len(peaks) >= 5:
                        break
                peaks.sort()
                self.fft_out.setText("\n".join(f"{eng(p, 'Hz'):>10}  {d:7.2f} dBV  {eng(v, 'Vrms')}" for p, v, d in peaks)
                                     + f"\n直流 {eng(vr[0], unit_of(c.source))}")
                for p, v, d in peaks:
                    axf.plot([p], [d], "v", color="white", ms=4)
            else:
                axf.text(0.5, 0.5, "FFT 通道未打开", transform=axf.transAxes, color="#9aa4b2", ha="center")
        self.cv.draw()
