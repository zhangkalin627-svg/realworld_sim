# -*- coding: utf-8 -*-
"""⑧ 整机联动：上电 → PLC → 变频器 → 主回路 → 机械 → 加热 → 喷淋，一次跑完整条链，改 PLC 程序/注入故障/看结果。"""
import html as _h
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from .machine import scenarios as SC
from .machine.plc_prog import MACHINE_IL, IO_MAP
from .machine.viz import plot_machine, STEP_NAME

COLOR = {"OK": "#1a7f37", "WARN": "#b26a00", "FAIL": "#c62828"}


class _Worker(QtCore.QThread):
    done = QtCore.Signal(object, object)
    failed = QtCore.Signal(str)

    def __init__(self, key, kw):
        super().__init__(); self.key, self.kw = key, kw

    def run(self):
        try:
            m, S = SC.run(self.key, **self.kw)
            self.done.emit(m, S)
        except Exception as e:  # 程序写错等
            import traceback
            self.failed.emit(f"{e}\n{traceback.format_exc()[-600:]}")


class MachinePage(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.m = None; self.S = None; self.worker = None
        lay = QtWidgets.QHBoxLayout(self)
        left = QtWidgets.QVBoxLayout()
        gb = QtWidgets.QGroupBox("整机联动（通用 100 kW 感应淬火电源，20–40 kHz，演示参数；【满功率】/【部分功率】场景含器件温升与 8 h 外推）")
        gb.setStyleSheet("QGroupBox{font-weight:bold}")
        fl = QtWidgets.QFormLayout(gb)
        intro = QtWidgets.QLabel("按下启动 → PLC 开冷却水 → 预充电 → 合旁路 → 变频器就绪 → 线圈下降 → 高频加热(红外测温) → 喷淋淬火 → 线圈上升；"
                                 "全链路用同一条时间轴，故障会一路传下去。")
        intro.setWordWrap(True); intro.setStyleSheet("color:#333; background:#f3f5f8; padding:4px; border-radius:4px")
        fl.addRow(intro)
        self.sel = QtWidgets.QComboBox()
        for k, d in SC.SCENARIOS.items():
            self.sel.addItem(d["name"], k)
        self.sel.currentIndexChanged.connect(self._sel)
        fl.addRow("场景", self.sel)
        self.desc = QtWidgets.QLabel(); self.desc.setWordWrap(True); self.desc.setStyleSheet("color:#555")
        fl.addRow(self.desc)
        self.air = QtWidgets.QDoubleSpinBox(); self.air.setRange(0.1, 0.8); self.air.setSingleStep(0.05); self.air.setValue(0.55); self.air.setSuffix(" MPa")
        fl.addRow("气源压力", self.air)
        self.ua = QtWidgets.QDoubleSpinBox(); self.ua.setRange(0, 3000); self.ua.setSingleStep(100); self.ua.setValue(500); self.ua.setSuffix(" W/K")
        fl.addRow("换热器 UA", self.ua)
        self.tank = QtWidgets.QDoubleSpinBox(); self.tank.setRange(5, 80); self.tank.setValue(25); self.tank.setSuffix(" °C")
        fl.addRow("水箱初始温度", self.tank)
        self.auto = QtWidgets.QCheckBox("改参数后自动重算（约 20 s，均为仿真结果）")
        self.auto.setChecked(True)
        fl.addRow(self.auto)
        self._deb = QtCore.QTimer(self); self._deb.setSingleShot(True); self._deb.setInterval(2000)
        self._deb.timeout.connect(self._auto_start)
        self._pending = False
        for wgt in (self.air, self.ua, self.tank):
            wgt.valueChanged.connect(self._kick)
        self.btn = QtWidgets.QPushButton("▶  运行整机联动"); self.btn.setMinimumHeight(34); self.btn.clicked.connect(self.simulate)
        fl.addRow(self.btn)
        self.status = QtWidgets.QLabel("选场景后点运行。PLC 程序在右侧「PLC 程序」页可改（改完再点运行）。"); self.status.setWordWrap(True)
        fl.addRow(self.status)
        left.addWidget(gb)
        self.verd = QtWidgets.QTextBrowser(); self.verd.setMinimumHeight(260)
        left.addWidget(self.verd, 1)
        lw = QtWidgets.QWidget(); lw.setLayout(left); lw.setFixedWidth(440); lay.addWidget(lw)
        self.tabs = QtWidgets.QTabWidget()
        self.fig = Figure(figsize=(9, 8)); self.cv = FigureCanvasQTAgg(self.fig)
        w1 = QtWidgets.QWidget(); v1 = QtWidgets.QVBoxLayout(w1); v1.setContentsMargins(2, 2, 2, 2)
        v1.addWidget(self.cv, 1)
        # 回放游标：拖动或播放查看任意时刻整机状态（均为仿真结果）
        bar = QtWidgets.QHBoxLayout()
        self.btn_play = QtWidgets.QPushButton("▶ 播放"); self.btn_play.setFixedWidth(70); self.btn_play.setEnabled(False)
        self.btn_play.clicked.connect(self._toggle_play)
        self.speed = QtWidgets.QComboBox(); self.speed.addItems(["0.5×", "1×", "2×", "4×", "8×"]); self.speed.setCurrentIndex(1); self.speed.setFixedWidth(64)
        self.tcur = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.tcur.setRange(0, 1000); self.tcur.setEnabled(False)
        self.tcur.valueChanged.connect(self._slider)
        self.tlab = QtWidgets.QLabel("t = —"); self.tlab.setFixedWidth(96)
        bar.addWidget(self.btn_play); bar.addWidget(self.speed); bar.addWidget(self.tcur, 1); bar.addWidget(self.tlab)
        v1.addLayout(bar)
        self.cread = QtWidgets.QLabel("（仿真完成后，拖动游标或点播放，查看任意时刻整机状态；均为仿真结果）")
        self.cread.setWordWrap(True); self.cread.setStyleSheet("color:#333; background:#f3f5f8; padding:4px; border-radius:4px")
        v1.addWidget(self.cread)
        self.tabs.addTab(w1, "整机曲线")
        self._T = None; self._cur_lines = []; self._tc = 0.0
        self._play_timer = QtCore.QTimer(self); self._play_timer.setInterval(40)
        self._play_timer.timeout.connect(self._tick)
        self.fig2 = Figure(figsize=(9, 6)); self.cv2 = FigureCanvasQTAgg(self.fig2)
        self.tabs.addTab(self.cv2, "器件温升 / 8 h 外推")
        self.tbl = QtWidgets.QTableWidget(); self.tabs.addTab(self.tbl, "每件工件结果")
        self.evt = QtWidgets.QPlainTextEdit(); self.evt.setReadOnly(True); self.evt.setFont(QtGui.QFont("monospace", 10))
        self.tabs.addTab(self.evt, "事件时间线")
        w = QtWidgets.QWidget(); pl = QtWidgets.QVBoxLayout(w)
        io = QtWidgets.QLabel(IO_MAP.replace("\n", "<br>").replace(" ", "&nbsp;")); io.setTextFormat(QtCore.Qt.RichText)
        io.setStyleSheet("background:#f6f6f6; padding:4px"); pl.addWidget(io)
        self.ed = QtWidgets.QPlainTextEdit(MACHINE_IL); self.ed.setFont(QtGui.QFont("monospace", 10)); pl.addWidget(self.ed, 1)
        rb = QtWidgets.QPushButton("恢复示例程序"); rb.clicked.connect(lambda: self.ed.setPlainText(MACHINE_IL)); pl.addWidget(rb)
        self.tabs.addTab(w, "PLC 程序（可改）")
        lay.addWidget(self.tabs, 1)
        self._sel()

    # ---------- 回放游标 ----------
    def _setup_cursor(self):
        t = np.asarray(self.m.log["t"], float)
        if t.size < 2:
            return
        self._T = t
        self._cur_lines = [a.axvline(t[0], color="#1565c0", lw=1.5, alpha=0.85, zorder=5) for a in self.fig.axes]
        self.tcur.setEnabled(True); self.btn_play.setEnabled(True)
        self._tc = float(t[0])
        self.tcur.blockSignals(True); self.tcur.setValue(0); self.tcur.blockSignals(False)
        self._scrub(self._tc)

    def _slider(self, v):
        if self._T is None:
            return
        self._scrub(float(self._T[0]) + (float(self._T[-1]) - float(self._T[0])) * v / 1000.0)

    def _toggle_play(self):
        if self._T is None:
            return
        if self._play_timer.isActive():
            self._stop_play()
        else:
            if self._tc >= float(self._T[-1]) - 1e-9:
                self._tc = float(self._T[0])
            self._play_timer.start(); self.btn_play.setText("⏸ 暂停")

    def _stop_play(self):
        self._play_timer.stop()
        if hasattr(self, "btn_play"):
            self.btn_play.setText("▶ 播放")

    def _tick(self):
        if self._T is None:
            self._stop_play(); return
        t0, t1 = float(self._T[0]), float(self._T[-1])
        spd = [0.5, 1, 2, 4, 8][self.speed.currentIndex()]
        self._tc += 0.04 * spd * (t1 - t0) / 60.0   # 1× 约 60 s 扫完全程
        if self._tc >= t1:
            self._scrub(t1)
            self._stop_play()
            self._tc = t0
            self.tcur.blockSignals(True); self.tcur.setValue(0); self.tcur.blockSignals(False)
            self._scrub(t0)
            return
        self.tcur.blockSignals(True)
        self.tcur.setValue(int(round(1000.0 * (self._tc - t0) / max(t1 - t0, 1e-9))))
        self.tcur.blockSignals(False)
        self._scrub(self._tc)

    def _scrub(self, tc):
        t = self._T
        if t is None:
            return
        self._tc = tc
        for ln in self._cur_lines:
            ln.set_xdata([tc, tc])
        i = int(np.searchsorted(t, tc, "right")) - 1
        i = max(0, min(i, len(t) - 1))
        L = self.m.log
        g = lambda k: float(np.asarray(L[k], float)[i])
        def num(k, fmt="{:.0f}"):
            v = g(k)
            return fmt.format(v) if np.isfinite(v) else "—"   # 无工件等时段如实显示 —
        step_i = int(np.asarray(L["step"], float)[i])
        fault_on = int(np.asarray(L["fault"], float)[i]) > 0
        self.tlab.setText(f"t = {tc:.2f} s")
        self.cread.setText(
            f"步进【{STEP_NAME.get(step_i, str(step_i))}】｜母线 {g('Vbus'):.0f} V｜开关频率 {g('f'):.1f} kHz｜"
            f"输出 {num('P_out','{:.1f}')} kW → 工件 {num('P_wp','{:.1f}')} kW｜工件表面 {num('T_wp')} °C｜线圈 {num('T_coil')} °C｜"
            f"水箱 {g('T_tank'):.1f} °C｜冷却水 {g('cw'):.1f} / 喷淋 {g('spray'):.1f} L/min｜"
            f"线圈位置 {'上' if g('pos') < 0.5 else '下'}" + ("｜⚠ 变频器故障 ON" if fault_on else "") +
            "　（均为仿真结果）")
        self.cv.draw_idle()

    def _kick(self, *_):
        if self.auto.isChecked() and self.m is not None:
            self._deb.start()

    def _auto_start(self):
        if self.worker is not None and self.worker.isRunning():
            self._pending = True
            return
        self.status.setText("参数已改，自动重算中 …")
        self.simulate()

    def _sel(self):
        k = self.sel.currentData()
        self.desc.setText(SC.SCENARIOS[k]["desc"])
        self.ua.setValue(0 if k == "longrun_no_hex" else 500)

    def simulate(self):
        if self.worker is not None and self.worker.isRunning():
            return
        self._deb.stop()
        self._stop_play()
        self.tcur.setEnabled(False); self.btn_play.setEnabled(False)
        k = self.sel.currentData()
        kw = dict(p_air=self.air.value(), UA=self.ua.value(), T_tank0=self.tank.value())
        il = self.ed.toPlainText()
        if il.strip() != MACHINE_IL.strip():
            kw["il"] = il
        self.btn.setEnabled(False); self.status.setText("计算中…")
        self.worker = _Worker(k, kw)
        self.worker.done.connect(self._done); self.worker.failed.connect(self._fail)
        self.worker.start()

    def _fail(self, msg):
        self.btn.setEnabled(True)
        self.status.setText(f"<span style='color:#c62828'>运行出错（多半是 PLC 程序写错）：{_h.escape(msg[:300])}</span>")

    def _done(self, m, S):
        self.m, self.S = m, S
        self.btn.setEnabled(True)
        self.status.setText(f"仿真 {S['t_end']:.1f} s 完成：合格 {S['ok']}，NG {S['ng']}，中断报废 {S['scrap']}")
        self.verd.setHtml("<br>".join(f'<span style="color:{COLOR[l]}"><b>[{l}]</b> {_h.escape(t)}</span>' for l, t in S["verdicts"]) +
                          "<br><br><span style='color:#555'>上电时间线：母线到 80% 额定 " +
                          (f"{S['t_bus90']:.2f} s" if S["t_bus90"] else "—") + "，变频器就绪 " + (f"{S['t_ready']:.2f} s" if S["t_ready"] else "—") +
                          f"；输出峰值电流 {S['I_pk_max']:.0f} A</span>")
        plot_machine(self.fig, m); self.cv.draw()
        self._setup_cursor()
        self._plot_thermal(m, S)
        if self._pending:
            self._pending = False
            QtCore.QTimer.singleShot(0, self._auto_start)
        rows = m.done
        self.tbl.clear(); self.tbl.setColumnCount(8); self.tbl.setRowCount(len(rows))
        self.tbl.setHorizontalHeaderLabels(["工件", "判定", "加热 s", "等效加热 s", "峰值 °C", "等效淬火 s", "HRC(表)", "说明"])
        for i, r in enumerate(rows):
            vals = [f"P{r['id']}", r["verdict"], f"{r['t_heat']:.2f}", f"{r['tau']:.2f}", f"{r['T_peak']:.0f}", f"{r['tq']:.2f}", f"{r['HRC']:.0f}", r["reason"]]
            for j, v in enumerate(vals):
                it = QtWidgets.QTableWidgetItem(v)
                if j == 1:
                    it.setForeground(QtGui.QColor("#1a7f37" if v == "OK" else "#c62828"))
                self.tbl.setItem(i, j, it)
        self.tbl.resizeColumnsToContents()
        self.evt.setPlainText("\n".join(f"{t:8.2f} s  {s}" for t, s in m.events))

    def _plot_thermal(self, m, S):
        """详细仿真段的器件温度 + （若有）8 h 外推。"""
        import numpy as np
        f = self.fig2; f.clear()
        L = m.log
        if m.net is None or not L.get("Tj"):
            f.text(0.5, 0.5, "该场景没有器件温升网络", ha="center"); self.cv2.draw(); return
        lr = S.get("lr")
        ax = f.add_subplot(2 if lr else 1, 1, 1)
        t = np.array(L["t"])
        for k, lab in (("Tj", "IGBT 结温"), ("T_plate", "IGBT 基板"), ("T_w", "冷却进水"), ("T_tr", "变压器"), ("T_cap", "谐振电容"), ("T_coil", "线圈")):
            ax.plot(t, L[k], label=lab)
        ax.set_xlabel("t / s（详细仿真）"); ax.set_ylabel("°C"); ax.grid(alpha=.3); ax.legend(fontsize=8, ncol=3)
        if lr:
            se = lr["series"]; ax2 = f.add_subplot(2, 1, 2)
            for k, lab in (("Tj", "IGBT 结温(均值)"), ("T_plate", "IGBT 基板"), ("T_w", "冷却进水"), ("T_tr", "变压器"), ("T_cap", "谐振电容"),
                           ("T_can", "电解电容芯"), ("T_cab", "柜内"), ("T_coil", "线圈"), ("T_tank", "淬火水箱")):
                ax2.plot(se["h"], se[k], label=lab)
            if lr["trip"]:
                ax2.axvline(lr["trip"][0], color="r", ls="--"); ax2.text(lr["trip"][0], ax2.get_ylim()[1], " " + lr["trip"][2], color="r", va="top", fontsize=8)
            ax2.set_xlabel("h（按平均损耗外推，估算）"); ax2.set_ylabel("°C"); ax2.grid(alpha=.3); ax2.legend(fontsize=8, ncol=3)
        f.tight_layout(); self.cv2.draw()
