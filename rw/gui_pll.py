# -*- coding: utf-8 -*-
"""⑧b 锁相演示：鼠标点出逆变器 PLL 锁相四案 + 动画回放 + φ_ref 滑块看参数影响。
行为级模型（Stage+Controller，1 kHz 控制周期，与 validate_pll.py 同一口径）；回放已算好的
仿真结果（每案 <1 s，改 φ_ref 后自动重算重播），画面与结论均标"仿真结果"。"""
import bisect
import html as _h
from PySide6 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from .machine.power import Spec, Stage, Controller

CASES = {
    "A": dict(name="A 启动捕获（冷态工件，含 ideal 对照）",
              desc="冷工件上电：PLL 从 0.775·f_max 高位扫频捕获谐振点，相位升到 φ_ref 闭环锁定；"
                   "对照 ideal 通道（知谐振点的基线）的输出功率。",
              T=lambda t: 20.0, k=lambda t: 0.949173, t_end=2.0, ideal=True),
    "B": dict(name="B 居里跟踪（工件 20→900 °C）",
              desc="工件升温过居里点，磁导率坍塌使谐振频率上移约 2%；PLL 全程跟踪不失锁、不跌入容性区。",
              T=lambda t: 20 + 880 * t / 3.0, k=lambda t: 0.949173, t_end=3.0),
    "C": dict(name="C 负载耦合阶跃（k 0.949→0.70 @1.5 s）",
              desc="工件插入深度突变等效耦合系数阶跃，谐振点跳变；PLL 应在 0.5 s 内重新锁定。",
              T=lambda t: 850.0, k=lambda t: 0.949173 if t < 1.5 else 0.70, t_end=3.0),
    "D": dict(name="D 电容失效 → 失锁跳闸（C 减半 @1.5 s）",
              desc="谐振电容失效减半，谐振频率跃过当前开关频率，负载变容性（ZVS 丢失会烧开关管）；"
                   "PLL 检测到 φ<0 持续 50 ms 应 LOSS 锁存跳闸封门极。",
              T=lambda t: 850.0, k=lambda t: 0.949173, t_end=3.0,
              fault_at=1.5, fault_C=0.667e-6 * 0.5),
}


def drive(track, T_prog, k_prog, t_end, fault_at=None, fault_C=None, phi_ref=0.22):
    """与 validate_pll.py 同一直驱口径；rows=(t, f, phi, P, f0, pll_state, fault)。"""
    sp = Spec(track_mode=track, phi_ref=phi_ref); st = Stage(sp); ctl = Controller(sp)
    Vbus = sp.V_src()
    dt = 1e-3; t = 0.0; rows = []
    ctl.tick(dt, False, Vbus, None, None)
    while t < t_end:
        T = T_prog(t); k = k_prog(t)
        if fault_at and t >= fault_at:
            sp.C_res = fault_C
        s = ctl.tick(dt, True, Vbus,
                     lambda f: st.solve(Vbus, f, T, k, 40.0, ctl.m), lambda: st.f0(T, k))
        if s:
            rows.append((t, ctl.f_q, getattr(ctl, "phi", None), s["P_out"], s["f0"], ctl.pll, ctl.fault))
        t += dt
    return rows


def run_case(key, phi_ref=0.22):
    c = CASES[key]
    rows = drive("pll", c["T"], c["k"], c["t_end"], c.get("fault_at"), c.get("fault_C"), phi_ref)
    out = {"rows": rows}
    if c.get("ideal"):
        out["ideal_rows"] = drive("ideal", c["T"], c["k"], c["t_end"])
    lock_t = next((r[0] for r in rows if r[5] == "locked"), None)
    phis = [r[2] for r in rows if r[2] is not None]
    phi_end, phi_min = rows[-1][2], (min(phis) if phis else None)
    v = []
    if key == "A":
        p_pll, p_id = rows[-1][3], out["ideal_rows"][-1][3]
        v.append((lock_t is not None and lock_t < 1.0, f"锁定时间 {lock_t:.3f} s（要求 <1 s）" if lock_t else "未锁定"))
        v.append((abs(phi_end - phi_ref) < 0.03, f"锁定相位 φ={phi_end:.3f} rad（目标 {phi_ref:.2f}±0.03）"))
        v.append((phi_min > -0.05, f"捕获最深 φ_min={phi_min:.3f} rad（不得深入容性，要求 >−0.05）"))
        v.append((p_pll >= 0.9 * p_id, f"锁定功率 {p_pll/1e3:.1f} kW = ideal 对照 {p_id/1e3:.1f} kW 的 {p_pll/p_id*100:.0f}%（要求 ≥90%，同受 I_lim 限流）"))
    elif key == "B":
        v.append((phi_min > 0.0, f"全程 φ_min={phi_min:.3f} rad（不得跌破容性）"))
        v.append((abs(phi_end - phi_ref) < 0.05, f"居里后 φ={phi_end:.3f} rad（目标 {phi_ref:.2f}±0.05）"))
    elif key == "C":
        relock = next((r[0] - 1.5 for r in rows if r[0] > 1.5 and r[2] is not None and abs(r[2] - phi_ref) < 0.03), None)
        v.append((relock is not None and relock < 0.5, f"阶跃后重锁 {relock*1e3:.0f} ms（要求 <500 ms）" if relock is not None else "阶跃后未重锁"))
    else:
        trip = next((r[0] for r in rows if r[6]), None)
        v.append((trip is not None and trip - 1.5 < 0.2 and rows[-1][6] == "LOSS",
                  f"电容失效后 {((trip - 1.5)*1e3):.0f} ms LOSS 跳闸封门极（要求 <200 ms）" if trip else "未跳闸"))
    out["verdicts"] = v
    out["lock_t"] = lock_t
    return out


class _Worker(QtCore.QThread):
    done = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self, key, phi_ref):
        super().__init__(); self.key, self.phi_ref = key, phi_ref

    def run(self):
        try:
            self.done.emit(run_case(self.key, self.phi_ref))
        except Exception as e:
            import traceback
            self.failed.emit(f"{e}\n{traceback.format_exc()[-600:]}")


class PLLDemoPage(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.worker = None; self.res = None
        self.tr = 0.0; self.playing = False
        lay = QtWidgets.QHBoxLayout(self)
        left = QtWidgets.QVBoxLayout()
        gb = QtWidgets.QGroupBox("逆变器锁相环（PLL）频率跟踪演示 —— 以下为行为级模型仿真结果，非实测")
        gb.setStyleSheet("QGroupBox{font-weight:bold}")
        fl = QtWidgets.QFormLayout(gb)
        intro = QtWidgets.QLabel(
            "感应淬火电源的谐振频率随工件温度/插入深度实时漂移，开关频率必须始终咬在谐振点略上方"
            "（感性小角 φ_ref，保证 ZVS 软开关）。PLL 不知谐振点，只看负载相位闭环调频。"
            "四个场景对应 validate_pll.py 验收四案；指令级固件在环版本见 validate_fw_pll.py。")
        intro.setWordWrap(True); intro.setStyleSheet("color:#333; background:#f3f5f8; padding:4px; border-radius:4px")
        intro.setMinimumHeight(intro.fontMetrics().lineSpacing() * 9)
        fl.addRow(intro)
        self.sel = QtWidgets.QComboBox()
        for k, c in CASES.items():
            self.sel.addItem(c["name"], k)
        self.sel.currentIndexChanged.connect(self._sel)
        fl.addRow("场景", self.sel)
        self.desc = QtWidgets.QLabel(); self.desc.setWordWrap(True); self.desc.setStyleSheet("color:#555")
        fl.addRow(self.desc)
        # φ_ref 滑块：改完自动重算重播（每案 <1 s，近实时看参数影响）
        self.pref = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.pref.setRange(10, 40); self.pref.setValue(22)
        self.pref_lab = QtWidgets.QLabel("0.22 rad")
        self.pref.valueChanged.connect(lambda v: self.pref_lab.setText(f"{v/100:.2f} rad"))
        self.pref.sliderReleased.connect(self._rerun)
        hb = QtWidgets.QHBoxLayout(); hb.addWidget(self.pref, 1); hb.addWidget(self.pref_lab)
        fl.addRow("目标相位 φ_ref", hb)
        self.btn = QtWidgets.QPushButton("▶  运行锁相演示"); self.btn.setMinimumHeight(34)
        self.btn.clicked.connect(self.simulate)
        fl.addRow(self.btn)
        self.status = QtWidgets.QLabel("选场景后点运行（每案 <2 s）；改 φ_ref 滑块自动重算。"); self.status.setWordWrap(True)
        fl.addRow(self.status)
        # 回放控制条
        rb = QtWidgets.QHBoxLayout()
        self.pbtn = QtWidgets.QPushButton("▶"); self.pbtn.setFixedWidth(36); self.pbtn.clicked.connect(self.toggle)
        self.rbtn = QtWidgets.QPushButton("⟲"); self.rbtn.setFixedWidth(36); self.rbtn.clicked.connect(self.replay)
        self.speed = QtWidgets.QComboBox(); self.speed.addItems(["0.5×", "1×", "2×", "4×", "8×"]); self.speed.setCurrentIndex(1)
        rb.addWidget(self.pbtn); rb.addWidget(self.rbtn); rb.addWidget(self.speed)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.slider.setRange(0, 1000)
        self.slider.sliderPressed.connect(self._pause); self.slider.sliderMoved.connect(self._seek)
        rb.addWidget(self.slider, 1)
        fl.addRow("回放", rb)
        self.ptr = QtWidgets.QLabel("—"); self.ptr.setStyleSheet("font-family:monospace; color:#333")
        fl.addRow(self.ptr)
        left.addWidget(gb)
        self.verd = QtWidgets.QTextBrowser(); self.verd.setMinimumHeight(200)
        left.addWidget(self.verd, 1)
        lw = QtWidgets.QWidget(); lw.setLayout(left); lw.setFixedWidth(440); lay.addWidget(lw)
        self.fig = Figure(figsize=(9, 7.5)); self.cv = FigureCanvasQTAgg(self.fig)
        lay.addWidget(self.cv, 1)
        self.timer = QtCore.QTimer(self); self.timer.setInterval(50); self.timer.timeout.connect(self._tick)
        self._clock = QtCore.QElapsedTimer()
        self._sel()

    def _sel(self):
        self.desc.setText(CASES[self.sel.currentData()]["desc"])

    def simulate(self):
        if self.worker is not None and self.worker.isRunning():
            return
        self._pause()
        self.btn.setEnabled(False); self.status.setText("计算中…")
        self.worker = _Worker(self.sel.currentData(), self.pref.value() / 100.0)
        self.worker.done.connect(self._done); self.worker.failed.connect(self._fail)
        self.worker.start()

    def _rerun(self):
        if self.res is not None:      # 已有结果时改滑块 = 重算当前场景
            self.simulate()

    def _fail(self, msg):
        self.btn.setEnabled(True)
        self.status.setText(f"<span style='color:#c62828'>运行出错：{_h.escape(msg[:300])}</span>")

    def _done(self, res):
        self.res = res
        self.btn.setEnabled(True)
        key = self.sel.currentData()
        rows = res["rows"]
        COLOR = {True: "#1a7f37", False: "#c62828"}
        self.status.setText(f"仿真 {CASES[key]['t_end']:.1f} s 完成（{len(rows)} 个控制周期）。")
        self.verd.setHtml(
            "<br>".join(f'<span style="color:{COLOR[ok]}"><b>[{"PASS" if ok else "FAIL"}]</b> {_h.escape(t)}</span>'
                        for ok, t in res["verdicts"]) +
            "<br><br><span style='color:#555'>行为级模型仿真结果；验收脚本 validate_pll.py 同一口径。指令级 AVR 固件在环互证见 validate_fw_pll.py（v0.32）。</span>")
        self._setup_plot(key, res)
        self.replay()

    # ---------------------------------------------------------- 画
    def _setup_plot(self, key, res):
        rows = res["rows"]
        self._t = [r[0] for r in rows]
        self._f = [r[1] for r in rows]; self._f0 = [r[4] for r in rows]
        self._phi = [r[2] if r[2] is not None else float("nan") for r in rows]
        self._P = [r[3] / 1e3 for r in rows]
        phi_ref = self.pref.value() / 100.0
        fig = self.fig; fig.clear()
        self.axs = list(fig.subplots(3, 1, sharex=True))
        ax1, ax2, ax3 = self.axs
        ax1.plot(self._t, self._f, lw=0.8, alpha=0.18, color="#1f77b4")
        ax1.plot(self._t, self._f0, lw=0.8, alpha=0.18, color="#d62728")
        self.lf, = ax1.plot([], [], lw=1.4, color="#1f77b4", label="开关频率 f（PLL 输出）")
        self.lf0, = ax1.plot([], [], lw=1.2, ls="--", color="#d62728", label="谐振频率 f0（负载真值）")
        ax1.set_ylabel("Hz"); ax1.legend(loc="center right", fontsize=8); ax1.grid(alpha=0.3)
        ax1.set_title(CASES[key]["name"] + "（仿真结果）", fontsize=11)
        ax2.plot(self._t, self._phi, lw=0.8, alpha=0.18, color="#b26a00")
        self.lphi, = ax2.plot([], [], lw=1.4, color="#b26a00", label="负载相位 φ")
        ax2.axhline(phi_ref, color="#1a7f37", ls=":", lw=1.2, label=f"φ_ref = {phi_ref:.2f} rad")
        ax2.axhline(0, color="#c62828", ls=":", lw=1.0, label="φ=0（容性红线）")
        if key in ("C", "D"):
            for ax in self.axs:
                ax.axvline(1.5, color="#888", ls="--", lw=1.0)
            ax2.text(1.51, ax2.get_ylim()[1], "扰动注入", color="#666", fontsize=8, va="top")
        ax2.set_ylabel("rad"); ax2.legend(loc="lower right", fontsize=8); ax2.grid(alpha=0.3)
        ax3.plot(self._t, self._P, lw=0.8, alpha=0.18, color="#6a3fb5")
        self.lP, = ax3.plot([], [], lw=1.4, color="#6a3fb5", label="输出功率（PLL）")
        if res.get("ideal_rows"):
            ti = [r[0] for r in res["ideal_rows"]]; Pi = [r[3] / 1e3 for r in res["ideal_rows"]]
            ax3.plot(ti, Pi, lw=1.0, ls="--", color="#888", label="ideal 对照")
        ax3.set_ylabel("kW"); ax3.set_xlabel("t / s"); ax3.legend(loc="center right", fontsize=8); ax3.grid(alpha=0.3)
        self.vlines = [ax.axvline(0, color="#333", lw=1.0) for ax in self.axs]
        fig.tight_layout()
        self.tr = 0.0
        self._draw()

    def _draw(self):
        n = bisect.bisect_right(self._t, self.tr)
        self.lf.set_data(self._t[:n], self._f[:n])
        self.lf0.set_data(self._t[:n], self._f0[:n])
        self.lphi.set_data(self._t[:n], self._phi[:n])
        self.lP.set_data(self._t[:n], self._P[:n])
        for vl in self.vlines:
            vl.set_xdata([self.tr, self.tr])
        if n:
            i = n - 1
            self.ptr.setText(f"t={self.tr:5.2f}s   f={self._f[i]:7.0f} Hz   f0={self._f0[i]:7.0f} Hz   "
                             f"φ={self._phi[i]:+.3f} rad   P={self._P[i]:6.1f} kW")
        self.slider.blockSignals(True)
        self.slider.setValue(int(self.tr / self._t[-1] * 1000))
        self.slider.blockSignals(False)
        self.cv.draw_idle()

    # ---------------------------------------------------------- 播放控制
    def toggle(self):
        if self.res is None:
            return
        if self.playing:
            self._pause()
        else:
            if self.tr >= self._t[-1] - 1e-9:
                self.tr = 0.0
            self.playing = True; self.pbtn.setText("⏸")
            self._clock.start(); self.timer.start()

    def _pause(self):
        self.playing = False; self.pbtn.setText("▶"); self.timer.stop()

    def replay(self):
        if self.res is None:
            return
        self.tr = 0.0; self._draw()
        if not self.playing:
            self.toggle()

    def _seek(self, v):
        self.tr = v / 1000.0 * self._t[-1]
        self._draw()

    def _tick(self):
        sp = float(self.speed.currentText().rstrip("×"))
        self.tr += self._clock.restart() / 1000.0 * sp
        if self.tr >= self._t[-1]:
            self.tr = self._t[-1]
            self._draw(); self._pause()
            return
        self._draw()
