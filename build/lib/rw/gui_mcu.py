# -*- coding: utf-8 -*-
"""⑦ 单片机/PLC：AVR 子集单片机 + FX PLC + 半桥电路 联合仿真。
左：固件源码（汇编，可改）/ PLC 指令表（可改）/ 引脚分配；中：联合仿真时序图；右：时间光标、MCU 引脚与寄存器、PLC 梯形图与 I/O、串口终端。
“送到示波器”把当前时间窗的波形送进 ⑥ 示波器。"""
import re, traceback, types
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from . import viz  # noqa: F401  注册中文字体
from .mcu import firmware as FW
from .mcu.cosim import heater_demo, level_at
from .mcu.asm import AsmError
from .mcu.periph import MCUError

VIEWS = ["全程", "启动握手（25–40 ms）", "软启动结束（125–160 ms）", "过流保护（短路 ±0.3 ms）", "PLC 报警联锁（短路后 0–20 ms）", "复位 + 再启动"]
LAMP_X = [("X000", "启动"), ("X001", "停止(常闭)"), ("X002", "MCU 就绪"), ("X003", "过流报警"), ("X004", "软启动完成"), ("X005", "报警复位")]
LAMP_Y = [("Y000", "启动加热"), ("Y001", "报警灯"), ("Y002", "加热中"), ("Y003", "故障复位")]


class Worker(QtCore.QObject):
    done = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self, kw):
        super().__init__(); self.kw = kw

    @QtCore.Slot()
    def run(self):
        try:
            self.done.emit(heater_demo(**self.kw))
        except (AsmError, MCUError) as e:
            self.failed.emit(str(e))
        except Exception:
            self.failed.emit(traceback.format_exc())


# ---------------------------------------------------------------- 梯形图（从指令表画）
def parse_rungs(il):
    rungs, cur, prev_out = [], None, True
    for ln in il.splitlines():
        s = ln.split(";")[0].strip()
        if not s:
            continue
        p = s.split(); op = p[0].upper(); a = p[1:]
        if op == "END":
            break
        if op in ("LD", "LDI", "LDP", "LDF"):
            if prev_out or cur is None:
                cur = dict(series=[], ors=[], outs=[], complex=False, text=[]); rungs.append(cur)
            else:
                cur["complex"] = True
            cur["series"].append((op, a[0])); prev_out = False
        elif op in ("AND", "ANI", "ANDP", "ANDF"):
            cur["series"].append((op, a[0])); prev_out = False
        elif op in ("OR", "ORI", "ORP", "ORF"):
            cur["ors"].append((len(cur["series"]), op, a[0])); prev_out = False
        elif op in ("ANB", "ORB", "MPS", "MRD", "MPP", "STL", "RET"):
            if cur: cur["complex"] = True
            prev_out = op in ("STL", "RET")
        else:
            cur["outs"].append((op, a)); prev_out = True
        if cur is not None:
            cur["text"].append(s)
    return rungs


class LadderView(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.rungs = []; self.state = {}
        self.setMinimumHeight(200)

    def set_program(self, il):
        self.rungs = parse_rungs(il)
        h = sum(self._rows(r) * 30 + 12 for r in self.rungs) + 10
        self.setMinimumHeight(h); self.update()

    def set_state(self, st):
        self.state = st or {}; self.update()

    def _rows(self, r):
        return max(1 + len(r["ors"]), len(r["outs"]), 1)

    def _on(self, op, dev):
        v = bool(self.state.get(dev, False))
        return (not v) if op in ("LDI", "ANI", "ORI") else v

    def paintEvent(self, e):
        p = QtGui.QPainter(self); p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.fillRect(self.rect(), QtGui.QColor("#fbfcfe"))
        W = self.width(); y = 8
        f = p.font(); f.setPointSize(8); p.setFont(f)
        green, grey, ink = QtGui.QColor("#2e9d4a"), QtGui.QColor("#c9ced6"), QtGui.QColor("#333")
        xr, xo = 10, W - 112
        for r in self.rungs:
            rows = self._rows(r); H = rows * 30
            p.setPen(QtGui.QPen(ink, 2)); p.drawLine(xr, y, xr, y + H)
            if r["complex"]:
                p.setPen(ink); p.drawText(xr + 8, y + 18, "（含 ANB/ORB/MPS 的回路：只显示指令） " + " ".join(r["text"])[:60])
                y += H + 12; continue
            step = max(46, min(70, (xo - 30) // max(1, len(r["series"]))))
            ym = y + 15
            def contact(x, yy, op, dev):
                on = self._on(op, dev)
                p.setPen(QtGui.QPen(green if on else ink, 2))
                p.drawLine(x, yy - 8, x, yy + 8); p.drawLine(x + 10, yy - 8, x + 10, yy + 8)
                if op in ("LDI", "ANI", "ORI"):
                    p.drawLine(x - 1, yy + 8, x + 11, yy - 8)
                if op.endswith("P") or op.endswith("F"):
                    p.drawText(x + 1, yy + 4, "↑" if op.endswith("P") else "↓")
                p.setPen(ink); p.drawText(x - 8, yy - 10, dev)
            x = 30
            p.setPen(QtGui.QPen(ink, 1)); p.drawLine(xr, ym, 30, ym)
            xs = []
            for op, dev in r["series"]:
                contact(x, ym, op, dev)
                p.setPen(QtGui.QPen(ink, 1)); p.drawLine(x + 10, ym, x + step, ym)
                xs.append(x); x += step
            for k, (idx, op, dev) in enumerate(r["ors"]):
                yy = ym + 30 * (k + 1)
                xa = xr; xb = xs[idx] if idx < len(xs) else x
                p.setPen(QtGui.QPen(ink, 1))
                p.drawLine(xa, yy, xa + 20, yy); contact(xa + 20, yy, op, dev)
                p.drawLine(xa + 30, yy, xb - 6, yy); p.drawLine(xb - 6, yy, xb - 6, ym)
            p.setPen(QtGui.QPen(ink, 1)); p.drawLine(x, ym, xo, ym)
            for k, (op, a) in enumerate(r["outs"]):
                yy = ym + 30 * k
                if k:
                    p.drawLine(xo - 10, ym, xo - 10, yy); p.drawLine(xo - 10, yy, xo, yy)
                dev = a[0] if a else ""
                on = bool(self.state.get(dev, False))
                p.setPen(QtGui.QPen(green if on else ink, 2))
                label = f"( {dev} )" if op == "OUT" and len(a) == 1 else f"[{op} {' '.join(a)}]"
                p.drawText(xo + 2, yy + 4, label)
            p.setPen(QtGui.QPen(ink, 2)); p.drawLine(W - 6, y, W - 6, y + H)
            y += H + 12
        p.end()


class Lamp(QtWidgets.QLabel):
    def __init__(self, dev, txt):
        super().__init__(f"{dev}\n{txt}"); self.setAlignment(QtCore.Qt.AlignCenter); self.setFixedSize(76, 36)
        self.set(False)

    def set(self, on, red=False):
        c = ("#e53935" if red else "#43a047") if on else "#e3e6ea"
        self.setStyleSheet(f"background:{c}; color:{'white' if on else '#333'}; border-radius:5px; font-size:10px")


# ---------------------------------------------------------------- 页面
class MCUPage(QtWidgets.QWidget):
    def __init__(self, scope_page=None, switch_to_scope=None):
        super().__init__()
        self.scope_page, self.switch_to_scope = scope_page, switch_to_scope
        self.cs = None
        lay = QtWidgets.QHBoxLayout(self)

        # ---------- 左
        left = QtWidgets.QVBoxLayout()
        gb = QtWidgets.QGroupBox("联合仿真（通用演示）")
        gb.setStyleSheet("QGroupBox{font-weight:bold}")
        fl = QtWidgets.QFormLayout(gb)
        intro = QtWidgets.QLabel("PLC 发“启动加热” → 单片机 PWM 驱动半桥（软启动）→ ADC 读电流 → 过流时封锁并报警给 PLC → PLC 停机")
        intro.setWordWrap(True); intro.setStyleSheet("color:#333; background:#f3f5f8; padding:4px; border-radius:4px")
        fl.addRow(intro)
        self.fwsel = QtWidgets.QComboBox(); self.fwsel.addItems(["汇编固件（AVR 指令子集，周期精确）", "行为级固件（Python，时间为估算）"])
        self.fwsel.currentIndexChanged.connect(self._fw_changed)
        fl.addRow("单片机固件", self.fwsel)
        self.tend = QtWidgets.QDoubleSpinBox(); self.tend.setRange(0.2, 5); self.tend.setValue(1.6); self.tend.setSuffix(" s")
        fl.addRow("仿真时长", self.tend)
        self.fault_on = QtWidgets.QCheckBox("注入负载短路"); self.fault_on.setChecked(True)
        self.tf = QtWidgets.QDoubleSpinBox(); self.tf.setRange(0.05, 4); self.tf.setDecimals(3); self.tf.setValue(0.45); self.tf.setSuffix(" s")
        self.rf = QtWidgets.QDoubleSpinBox(); self.rf.setRange(0.05, 10); self.rf.setValue(0.4); self.rf.setSuffix(" Ω")
        h = QtWidgets.QHBoxLayout(); h.addWidget(self.fault_on); h.addWidget(QtWidgets.QLabel("时刻")); h.addWidget(self.tf); h.addWidget(QtWidgets.QLabel("短路电阻")); h.addWidget(self.rf)
        fl.addRow(h)
        self.btn = QtWidgets.QPushButton("▶  联合仿真"); self.btn.setMinimumHeight(34); self.btn.clicked.connect(self.simulate)
        fl.addRow(self.btn)
        self.status = QtWidgets.QLabel("负载：半桥 48 V、RL 4 Ω / 1 mH、采样 0.2 V/A；PLC 扫描 2 ms、输入滤波 10 ms"); self.status.setWordWrap(True)
        fl.addRow(self.status)
        left.addWidget(gb)
        self.src_tabs = QtWidgets.QTabWidget()
        mono = QtGui.QFont("DejaVu Sans Mono"); mono.setStyleHint(QtGui.QFont.Monospace); mono.setPointSize(8)
        self.asm = QtWidgets.QPlainTextEdit(FW.HEATER_ASM); self.asm.setFont(mono); self.asm.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        self.il = QtWidgets.QPlainTextEdit(FW.PLC_IL); self.il.setFont(mono)
        pm = QtWidgets.QPlainTextEdit(FW.PINMAP + "\n\n汇编器支持的指令：\n" + _instr_list()); pm.setReadOnly(True); pm.setFont(mono)
        self.src_tabs.addTab(self.asm, "固件源码（汇编，可改）")
        self.src_tabs.addTab(self.il, "PLC 程序（FX 指令表）")
        self.src_tabs.addTab(pm, "引脚分配")
        left.addWidget(self.src_tabs, 1)
        lw = QtWidgets.QWidget(); lw.setLayout(left); lw.setFixedWidth(430)
        lay.addWidget(lw)

        # ---------- 中：时序图
        mid = QtWidgets.QVBoxLayout()
        hv = QtWidgets.QHBoxLayout()
        hv.addWidget(QtWidgets.QLabel("视图"))
        self.view = QtWidgets.QComboBox(); self.view.addItems(VIEWS); self.view.currentIndexChanged.connect(self.redraw)
        hv.addWidget(self.view, 1)
        self.to_scope = QtWidgets.QPushButton("送到示波器 ⑥"); self.to_scope.clicked.connect(self.send_scope); self.to_scope.setEnabled(False)
        hv.addWidget(self.to_scope)
        mid.addLayout(hv)
        self.fig = Figure(figsize=(7, 7)); self.cv = FigureCanvasQTAgg(self.fig)
        mid.addWidget(self.cv, 1)
        hs = QtWidgets.QHBoxLayout()
        hs.addWidget(QtWidgets.QLabel("时间光标"))
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.slider.setRange(0, 1000); self.slider.valueChanged.connect(self.cursor_moved)
        hs.addWidget(self.slider, 1)
        self.tlab = QtWidgets.QLabel("—"); self.tlab.setFixedWidth(120); hs.addWidget(self.tlab)
        mid.addLayout(hs)
        mw = QtWidgets.QWidget(); mw.setLayout(mid)
        lay.addWidget(mw, 1)

        # ---------- 右
        self.rt = QtWidgets.QTabWidget(); self.rt.setFixedWidth(400)
        # MCU
        pmcu = QtWidgets.QWidget(); v = QtWidgets.QVBoxLayout(pmcu)
        v.addWidget(QtWidgets.QLabel("<b>引脚（光标时刻）</b>"))
        self.pins = QtWidgets.QTableWidget(0, 4); self.pins.setHorizontalHeaderLabels(["引脚", "功能", "方向", "电平"])
        self.pins.verticalHeader().setVisible(False); self.pins.verticalHeader().setDefaultSectionSize(18)
        self.pins.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.pins.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        v.addWidget(self.pins, 3)
        v.addWidget(QtWidgets.QLabel("<b>寄存器（仿真结束时）</b>"))
        self.regs = QtWidgets.QPlainTextEdit(); self.regs.setReadOnly(True); self.regs.setFont(mono); self.regs.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        v.addWidget(self.regs, 2)
        v.addWidget(QtWidgets.QLabel("<b>串口终端（9615 波特）</b>"))
        self.term = QtWidgets.QPlainTextEdit(); self.term.setReadOnly(True); self.term.setFont(mono)
        v.addWidget(self.term, 1)
        self.rt.addTab(pmcu, "单片机")
        # PLC
        pplc = QtWidgets.QWidget(); v2 = QtWidgets.QVBoxLayout(pplc)
        g = QtWidgets.QGridLayout(); self.lamps = {}
        for i, (d, t) in enumerate(LAMP_X):
            l = Lamp(d, t); self.lamps[d] = l; g.addWidget(l, i // 4, i % 4)
        for i, (d, t) in enumerate(LAMP_Y):
            l = Lamp(d, t); self.lamps[d] = l; g.addWidget(l, 2 + i // 4, i % 4)
        v2.addWidget(QtWidgets.QLabel("<b>输入映像 X / 输出 Y（光标时刻）</b>")); v2.addLayout(g)
        v2.addWidget(QtWidgets.QLabel("<b>梯形图（由指令表生成，绿色 = 导通/得电）</b>"))
        self.ladder = LadderView(); sa = QtWidgets.QScrollArea(); sa.setWidget(self.ladder); sa.setWidgetResizable(True)
        v2.addWidget(sa, 1)
        self.rt.addTab(pplc, "PLC 梯形图/IO")
        # 结果
        pres = QtWidgets.QWidget(); v3 = QtWidgets.QVBoxLayout(pres)
        self.res = QtWidgets.QTableWidget(0, 2); self.res.setHorizontalHeaderLabels(["项目", "仿真值"])
        self.res.verticalHeader().setVisible(False); self.res.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self.res.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        self.res.setWordWrap(True)
        v3.addWidget(self.res)
        self.rt.addTab(pres, "关键时刻")
        lay.addWidget(self.rt)
        self.ladder.set_program(FW.PLC_IL)
        self.fig.text(0.5, 0.5, "点“▶ 联合仿真”", ha="center", va="center", fontsize=14, color="#888")

    def _fw_changed(self, i):
        self.asm.setReadOnly(i == 1)
        if i == 1:
            from .mcu.behav import PY_SOURCE_NOTE
            import inspect
            from .mcu import behav
            self.asm.setPlainText("# " + PY_SOURCE_NOTE + "\n\n" + inspect.getsource(behav.HeaterFirmwarePy))
        else:
            self.asm.setPlainText(FW.HEATER_ASM)

    # ------------------------------------------------ 仿真
    def simulate(self):
        if self.fwsel.currentIndex() == 1:
            from .mcu.behav import HeaterFirmwarePy
            fw = HeaterFirmwarePy()
        else:
            fw = self.asm.toPlainText()
        tf = self.tf.value() if self.fault_on.isChecked() else 1e9
        kw = dict(t_end=self.tend.value(), fault_at=tf, fault_R=self.rf.value(), firmware=fw, plc_il=self.il.toPlainText())
        self.btn.setEnabled(False); self.status.setText("联合仿真计算中 …（1.6 s 约需 2–3 s）")
        self.th = QtCore.QThread(); self.wk = Worker(kw); self.wk.moveToThread(self.th)
        self.th.started.connect(self.wk.run)
        self.wk.done.connect(self.on_done); self.wk.failed.connect(self.on_fail)
        self.wk.done.connect(self.th.quit); self.wk.failed.connect(self.th.quit)
        self.th.start()

    def on_fail(self, msg):
        self.btn.setEnabled(True); self.status.setText("<span style='color:#c62828'>失败：</span>" + msg.splitlines()[-1][:200])
        QtWidgets.QMessageBox.warning(self, "联合仿真", msg[:3000])

    def on_done(self, cs):
        self.cs = cs; self.btn.setEnabled(True); self.to_scope.setEnabled(True)
        self.s = cs.summary()
        m = cs.mcu
        self.status.setText(f"完成：仿真 {m.t():.3f} s，执行 {m.instr_count} 条指令"
                            + ("（行为级：中断次数）" if cs.prog is None else "") + f"，{len(m.irq_log)} 次中断，PLC {len(cs.plc_snap)} 次状态变化")
        self.regs.setPlainText(m.regs_text() + "\n" + self._periph_text())
        self.term.setPlainText("\n".join(f"{t*1e3:9.3f} ms  {ch!r}" for t, ch in cs.uart_text()))
        self.ladder.set_program(cs.plc_il)
        self.fill_results()
        self.redraw()
        self.slider.setValue(self.slider.value())
        self.cursor_moved()

    def _periph_text(self):
        m = self.cs.mcu; t0 = m.t0; a = m.adc; u = m.uart
        return (f"TCCR0A={t0.tccra:02X} TCCR0B={t0.tccrb:02X} TCNT0={t0.cnt:02X} OCR0A={t0.ocr[0]:02X} OCR0B={t0.ocr[1]:02X} TIFR0={t0.tifr:02X}\n"
                f"ADMUX={a.admux:02X} ADCSRA={a.adcsra:02X} ADC={a.result}（{a.result*5/1024:.3f} V）  UBRR0={u.ubrr} → {u.baud():.1f} 波特")

    def fill_results(self):
        s = self.s; rows = []
        def ms(x): return "—" if x is None else f"{x*1e3:.4f} ms"
        def d(a, b, unit="ms"):
            if a is None or b is None: return "—"
            return f"{(b-a)*1e3:.4f} ms" if unit == "ms" else f"{(b-a)*1e6:.3f} µs"
        rows += [("按下启动 X000", ms(s.get("t_btn"))), ("PLC Y000 输出（含 10 ms 滤波 + 扫描）", d(s.get("t_btn"), s.get("t_Y000_on"))),
                 ("Y000 → MCU 打开 EN（0.2 ms 输出延时 + 中断）", d(s.get("t_Y000_on"), s.get("t_EN_on"))),
                 ("软启动到 50 %（PB0 到达）", d(s.get("t_EN_on"), s.get("t_reached")))]
        if "t_fault" in s:
            rows += [("短路时刻", ms(s["t_fault"])), ("短路前电流", f"{s['i_before']:.2f} A"),
                     ("电流越过阈值 %.2f A" % s["i_th"], d(s["t_fault"], s.get("t_cross"), "us")),
                     ("ADC 采到超限 → 关 EN", d(s.get("t_sample_over"), s.get("t_EN_off"), "us")),
                     ("越过阈值 → 关 EN（保护响应）", d(s.get("t_cross"), s.get("t_EN_off"), "us")),
                     ("峰值电流", f"{s['i_peak']:.2f} A"),
                     ("MCU 报警 PD7 → PLC 撤销 Y000", d(s.get("t_PD7_on"), s.get("t_Y000_off"))),
                     ("PLC 报警灯 Y001", ms(s.get("t_Y001_on")))]
        self.res.setRowCount(len(rows))
        for i, (a, b) in enumerate(rows):
            self.res.setItem(i, 0, QtWidgets.QTableWidgetItem(a)); self.res.setItem(i, 1, QtWidgets.QTableWidgetItem(b))
        self.res.resizeRowsToContents()

    # ------------------------------------------------ 画图
    def window(self):
        cs, s = self.cs, self.s
        T = cs.mcu.t(); k = self.view.currentIndex()
        tf = s.get("t_fault")
        if k == 1: return 0.025, 0.040
        if k == 2: return 0.125, 0.160
        if k == 3 and tf: return tf - 0.1e-3, tf + 0.3e-3
        if k == 4 and tf: return tf - 1e-3, tf + 20e-3
        if k == 5 and tf: return 0.74, min(T, 1.0)
        return 0.0, T

    def redraw(self, *_):
        if not self.cs:
            return
        cs = self.cs; t0, t1 = self.window()
        w = cs.waves(t0, t1, 6000)
        t = w.x; tm = t * 1e3
        self.fig.clear()
        ax = self.fig.subplots(4, 1, sharex=True, gridspec_kw=dict(height_ratios=[2.2, 2.2, 1.2, 2.4], hspace=0.22))
        self.fig.subplots_adjust(left=0.15, right=0.965, top=0.95, bottom=0.07)
        def dig(a, names, title):
            for i, (key, lab, col) in enumerate(names):
                y = (w[key] > 0.5 * w[key].max() if w[key].max() > 0 else w[key] * 0).astype(float)
                a.fill_between(tm, i * 1.4, i * 1.4 + y, step="post", color=col, alpha=0.35, lw=0)
                a.step(tm, i * 1.4 + y, where="post", color=col, lw=1.1)
            a.set_yticks([i * 1.4 + 0.5 for i in range(len(names))]); a.set_yticklabels([n[1] for n in names], fontsize=8)
            a.set_ylim(-0.3, len(names) * 1.4); a.set_title(title, fontsize=9, loc="left", pad=2)
        dig(ax[0], [("PLC_X000", "X000 启动", "#1e88e5"), ("PLC_Y000", "Y000 启动", "#43a047"), ("PLC_X003", "X003 报警", "#e53935"),
                    ("PLC_Y001", "Y001 报警灯", "#c62828"), ("PLC_Y002", "Y002 加热中", "#8e24aa")], "PLC（FX 指令表，扫描 2 ms）")
        dig(ax[1], [("EN", "PD6 驱动 EN", "#43a047"), ("PWM_IN", "PD5 PWM", "#fb8c00"), ("MCU_故障", "PD7 报警", "#e53935"),
                    ("MCU_到达", "PB0 到达", "#8e24aa"), ("TXD", "PD1 TXD", "#546e7a")], "单片机引脚")
        # 占空比（每个 PWM 周期从边沿量出）
        e = np.array(cs.pin_hist("PD5")) if cs.pin_hist("PD5") else np.zeros((0, 2))
        if len(e) > 2:
            r = e[e[:, 1] == 1, 0]; f = e[e[:, 1] == 0, 0]
            dd, tt = [], []
            for a0, a1 in zip(r[:-1], r[1:]):
                ff = f[(f > a0) & (f <= a1)]
                if len(ff) and a1 - a0 < 1e-3:
                    dd.append((ff[0] - a0) / (a1 - a0) * 100); tt.append(a0)
            if tt:
                tt = np.array(tt); dd = np.array(dd); m = (tt >= t0) & (tt <= t1)
                ax[2].step(tt[m] * 1e3, dd[m], where="post", color="#fb8c00", lw=1.2)
        ax[2].set_ylabel("占空比 %", fontsize=8); ax[2].set_ylim(-3, 60); ax[2].grid(alpha=0.3)
        ax[3].plot(tm, w["i_load"], color="#1565c0", lw=0.9, label="负载电流（快速电路模型）")
        ax[3].axhline(self.s["i_th"], color="#e53935", ls="--", lw=1, label=f"过流阈值 {self.s['i_th']:.2f} A")
        ad = [(ts, code) for (ts, td, ch, v, code) in cs.adc_samples() if t0 <= ts <= t1 and ch == 0]
        if ad and len(ad) < 3000:
            ax[3].plot([a[0] * 1e3 for a in ad], [a[1] * 5 / 1024 * cs.plant.amps_per_volt() for a in ad], ".", ms=3, color="#2e7d32", label="ADC 采样（换算成 A）")
        ax[3].set_ylabel("A", fontsize=8); ax[3].grid(alpha=0.3); ax[3].legend(fontsize=7, loc="upper right")
        ax[3].set_xlabel("时间 (ms)")
        for a in ax:
            a.tick_params(labelsize=8)
        ax[3].set_xlim(tm[0], tm[-1])
        self.axes = ax
        self.cursor_lines = [a.axvline(np.nan, color="k", lw=0.8, ls=":") for a in ax]
        self.cursor_moved()

    def cursor_t(self):
        if not self.cs:
            return 0.0
        t0, t1 = self.window()
        return t0 + (t1 - t0) * self.slider.value() / 1000.0

    def cursor_moved(self, *_):
        if not self.cs:
            return
        t = self.cursor_t(); self.tlab.setText(f"t = {t*1e3:.4f} ms")
        for l in getattr(self, "cursor_lines", []):
            l.set_xdata([t * 1e3, t * 1e3])
        self.cv.draw_idle()
        cs = self.cs; m = cs.mcu
        from .mcu.defs import PIN_FUNCS
        rows = []
        for L, p in m.ports.items():
            for b in range(p.n):
                nm = f"P{L}{b}"
                h = cs.pin_hist(nm)
                lv = level_at(h, t, 0) if h else p.level(b)
                rows.append((nm, PIN_FUNCS.get(nm, ""), "输出" if p.is_output(b) else "输入", lv))
        self.pins.setRowCount(len(rows))
        for i, (a, b, c, d) in enumerate(rows):
            for j, x in enumerate((a, b, c, str(d))):
                it = QtWidgets.QTableWidgetItem(x)
                if j == 3:
                    it.setBackground(QtGui.QColor("#a5d6a7" if d else "#eceff1"))
                self.pins.setItem(i, j, it)
        snap = level_at(cs.plc_snap, t, {})
        for dev, lamp in self.lamps.items():
            on = bool(snap.get(dev, False))
            lamp.set(on, red=dev in ("X003", "Y001"))
        self.ladder.set_state(snap)

    # ------------------------------------------------ 示波器
    def send_scope(self):
        if not self.cs or self.scope_page is None:
            return
        t0, t1 = self.window()
        if t1 - t0 > 5e-3 or self.view.currentIndex() == 3:     # 全程太长：取“关 EN”前后 ±0.25 ms
            te = self.s.get("t_EN_off") or self.s.get("t_fault") or 0.4
            t0, t1 = te - 0.25e-3, te + 0.25e-3
        w = self.cs.waves(t0, t1, 20000)
        sp = self.scope_page
        tdiv = (t1 - t0) / 10
        preset = dict(ch=[("PWM_IN", 5.0, -15.0), ("EN", 5.0, -5.0), ("i_load", 2.0, -12.0), ("MCU_故障", 5.0, 10.0)],
                      tdiv=tdiv, hpos=0.0, trig=(2, "fall", 2.5))
        sp.example = types.SimpleNamespace(scope=preset)
        sp.circuit = None
        sp.theory.setText("来自 ⑦ 单片机/PLC 联合仿真（%.3f–%.3f ms）：CH1 PWM、CH2 驱动 EN（下降沿触发）、CH3 负载电流（按 V 显示，1 V = 1 A）、CH4 MCU 过流报警" % (t0 * 1e3, t1 * 1e3))
        if sp.pick.findData(("ext", "mcu")) < 0:
            sp.pick.blockSignals(True); sp.pick.insertItem(0, "来自 ⑦ 单片机/PLC 联合仿真", ("ext", "mcu")); sp.pick.blockSignals(False)
        sp.pick.blockSignals(True); sp.pick.setCurrentIndex(sp.pick.findData(("ext", "mcu"))); sp.pick.blockSignals(False)
        sp.ana.setText("（波形来自联合仿真，不用 ngspice）")
        sp.on_done(w)
        sp.example = None
        if self.switch_to_scope:
            self.switch_to_scope()


def _instr_list():
    from .mcu.asm import FORMS
    names = sorted(FORMS)
    return "\n".join("  " + " ".join(names[i:i + 12]) for i in range(0, len(names), 12))
