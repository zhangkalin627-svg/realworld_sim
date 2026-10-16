# -*- coding: utf-8 -*-
"""⑧b 锁相演示：鼠标点出逆变器 PLL 锁相四案（启动捕获/居里跟踪/负载阶跃/失锁跳闸）。
行为级模型（Stage+Controller，1 kHz 控制周期，与 validate_pll.py 同一口径），曲线与结论均为仿真结果。"""
import html as _h
from PySide6 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from .machine.power import Spec, Stage, Controller

PHI_REF = 0.22  # rad（ZVS 感性裕度目标，与固件 REF=18 码同义）

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


def drive(track, T_prog, k_prog, t_end, fault_at=None, fault_C=None):
    """与 validate_pll.py 同一直驱口径；返回 rows=(t, f, phi, P, f0, pll_state, fault)。"""
    sp = Spec(track_mode=track); st = Stage(sp); ctl = Controller(sp)
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


def run_case(key):
    c = CASES[key]
    rows = drive("pll", c["T"], c["k"], c["t_end"], c.get("fault_at"), c.get("fault_C"))
    out = {"rows": rows}
    if c.get("ideal"):
        out["ideal_rows"] = drive("ideal", c["T"], c["k"], c["t_end"])
    # 结论（与 validate_pll.py 断言同口径）
    lock_t = next((r[0] for r in rows if r[5] == "locked"), None)
    phis = [r[2] for r in rows if r[2] is not None]
    phi_end, phi_min = rows[-1][2], (min(phis) if phis else None)
    v = []
    if key == "A":
        p_pll, p_id = rows[-1][3], out["ideal_rows"][-1][3]
        v.append((lock_t is not None and lock_t < 1.0, f"锁定时间 {lock_t:.3f} s（要求 <1 s）" if lock_t else "未锁定"))
        v.append((abs(phi_end - PHI_REF) < 0.03, f"锁定相位 φ={phi_end:.3f} rad（目标 {PHI_REF}±0.03）"))
        v.append((phi_min > -0.05, f"捕获最深 φ_min={phi_min:.3f} rad（不得深入容性，要求 >−0.05）"))
        v.append((p_pll >= 0.9 * p_id, f"锁定功率 {p_pll/1e3:.1f} kW = ideal 对照 {p_id/1e3:.1f} kW 的 {p_pll/p_id*100:.0f}%（要求 ≥90%，同受 I_lim 限流）"))
    elif key == "B":
        v.append((phi_min > 0.0, f"全程 φ_min={phi_min:.3f} rad（不得跌破容性）"))
        v.append((abs(phi_end - PHI_REF) < 0.05, f"居里后 φ={phi_end:.3f} rad（目标 {PHI_REF}±0.05）"))
    elif key == "C":
        relock = next((r[0] - 1.5 for r in rows if r[0] > 1.5 and r[2] is not None and abs(r[2] - PHI_REF) < 0.03), None)
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

    def __init__(self, key):
        super().__init__(); self.key = key

    def run(self):
        try:
            self.done.emit(run_case(self.key))
        except Exception as e:
            import traceback
            self.failed.emit(f"{e}\n{traceback.format_exc()[-600:]}")


class PLLDemoPage(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.worker = None; self.res = None
        lay = QtWidgets.QHBoxLayout(self)
        left = QtWidgets.QVBoxLayout()
        gb = QtWidgets.QGroupBox("逆变器锁相环（PLL）频率跟踪演示 —— 以下为行为级模型仿真结果，非实测")
        gb.setStyleSheet("QGroupBox{font-weight:bold}")
        fl = QtWidgets.QFormLayout(gb)
        intro = QtWidgets.QLabel(
            "感应淬火电源的谐振频率随工件温度/插入深度实时漂移，开关频率必须始终咬在谐振点略上方"
            "（感性小角 φ_ref≈0.22 rad，保证 ZVS 软开关）。PLL 不知谐振点，只看负载相位闭环调频。"
            "四个场景对应 validate_pll.py 验收四案；指令级固件在环版本见 validate_fw_pll.py。")
        intro.setWordWrap(True); intro.setStyleSheet("color:#333; background:#f3f5f8; padding:4px; border-radius:4px")
        intro.setMinimumHeight(intro.fontMetrics().lineSpacing() * 9)   # 换行 label 的 sizeHint 不知行高，防裁切
        fl.addRow(intro)
        self.sel = QtWidgets.QComboBox()
        for k, c in CASES.items():
            self.sel.addItem(c["name"], k)
        self.sel.currentIndexChanged.connect(self._sel)
        fl.addRow("场景", self.sel)
        self.desc = QtWidgets.QLabel(); self.desc.setWordWrap(True); self.desc.setStyleSheet("color:#555")
        fl.addRow(self.desc)
        self.btn = QtWidgets.QPushButton("▶  运行锁相演示"); self.btn.setMinimumHeight(34)
        self.btn.clicked.connect(self.simulate)
        fl.addRow(self.btn)
        self.status = QtWidgets.QLabel("选场景后点运行（每案 <2 s）。"); self.status.setWordWrap(True)
        fl.addRow(self.status)
        left.addWidget(gb)
        self.verd = QtWidgets.QTextBrowser(); self.verd.setMinimumHeight(240)
        left.addWidget(self.verd, 1)
        lw = QtWidgets.QWidget(); lw.setLayout(left); lw.setFixedWidth(440); lay.addWidget(lw)
        self.fig = Figure(figsize=(9, 7.5)); self.cv = FigureCanvasQTAgg(self.fig)
        lay.addWidget(self.cv, 1)
        self._sel()

    def _sel(self):
        self.desc.setText(CASES[self.sel.currentData()]["desc"])

    def simulate(self):
        if self.worker is not None and self.worker.isRunning():
            return
        self.btn.setEnabled(False); self.status.setText("计算中…")
        self.worker = _Worker(self.sel.currentData())
        self.worker.done.connect(self._done); self.worker.failed.connect(self._fail)
        self.worker.start()

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
        self._plot(key, res)

    def _plot(self, key, res):
        rows = res["rows"]
        t = [r[0] for r in rows]; f = [r[1] for r in rows]; f0 = [r[4] for r in rows]
        phi = [r[2] if r[2] is not None else float("nan") for r in rows]
        P = [r[3] / 1e3 for r in rows]
        fig = self.fig; fig.clear()
        ax1, ax2, ax3 = fig.subplots(3, 1, sharex=True)
        ax1.plot(t, f, lw=1.2, label="开关频率 f（PLL 输出）")
        ax1.plot(t, f0, lw=1.0, ls="--", label="谐振频率 f0（负载真值）")
        ax1.set_ylabel("Hz"); ax1.legend(loc="center right", fontsize=8); ax1.grid(alpha=0.3)
        ax1.set_title(CASES[key]["name"] + "（仿真结果）", fontsize=11)
        ax2.plot(t, phi, lw=1.2, color="#b26a00", label="负载相位 φ")
        ax2.axhline(PHI_REF, color="#1a7f37", ls=":", lw=1.2, label=f"φ_ref = {PHI_REF} rad")
        ax2.axhline(0, color="#c62828", ls=":", lw=1.0, label="φ=0（容性红线）")
        if key in ("C", "D"):
            ax2.axvline(1.5, color="#888", ls="--", lw=1.0)
            ax2.text(1.51, ax2.get_ylim()[1], "扰动注入", color="#666", fontsize=8, va="top")
        ax2.set_ylabel("rad"); ax2.legend(loc="lower right", fontsize=8); ax2.grid(alpha=0.3)
        ax3.plot(t, P, lw=1.2, color="#6a3fb5", label="输出功率（PLL）")
        if res.get("ideal_rows"):
            ti = [r[0] for r in res["ideal_rows"]]; Pi = [r[3] / 1e3 for r in res["ideal_rows"]]
            ax3.plot(ti, Pi, lw=1.0, ls="--", color="#888", label="ideal 对照")
        ax3.set_ylabel("kW"); ax3.set_xlabel("t / s"); ax3.legend(loc="center right", fontsize=8); ax3.grid(alpha=0.3)
        fig.tight_layout()
        self.cv.draw()
