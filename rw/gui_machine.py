# -*- coding: utf-8 -*-
"""⑧ 整机联动：上电 → PLC → 变频器 → 主回路 → 机械 → 加热 → 喷淋，一次跑完整条链，改 PLC 程序/注入故障/看结果。"""
import html as _h
from PySide6 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from .machine import scenarios as SC
from .machine.plc_prog import MACHINE_IL, IO_MAP
from .machine.viz import plot_machine

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
        gb = QtWidgets.QGroupBox("整机联动（通用 10 kW 级 30 kHz 感应淬火机，演示参数）")
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
        self.tabs.addTab(self.cv, "整机曲线")
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

    def _sel(self):
        k = self.sel.currentData()
        self.desc.setText(SC.SCENARIOS[k]["desc"])
        self.ua.setValue(0 if k == "longrun_no_hex" else 500)

    def simulate(self):
        if self.worker is not None and self.worker.isRunning():
            return
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
