"""真实的世界 v0.7：感应加热淬火 + 结构校核 + PLC 虚拟调试 + 物理沙盒 + 材料库 + 电路/示波器 + 单片机/PLC 联合仿真（桌面程序）
运行:  python3 app.py
"""
import sys, warnings, traceback
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("QtAgg")
from PySide6 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from rw.induction import InductionQuenchSim, BottleNeckGeometry, find_power
from rw import materials as M, viz
from rw.version import VERSION
from rw import theme

MEDIA = [(k, M.QUENCHANTS[k]["name"]) for k in ("water_spray", "polymer_spray", "water_dip", "oil_dip", "air")]


class Worker(QtCore.QObject):
    progress = QtCore.Signal(int, str)
    done = QtCore.Signal(object, object)
    failed = QtCore.Signal(str)

    def __init__(self, params):
        super().__init__()
        self.p = params

    @QtCore.Slot()
    def run(self):
        try:
            p = dict(self.p)
            geom = BottleNeckGeometry(**p.pop("geom"))
            target = p.pop("target")
            ck = p.pop("conc_key")
            if ck:
                from rw.em.conc import Concentrator
                p["conc"] = Concentrator(ck)
            if target is not None:
                self.progress.emit(2, f"自动找功率：让外表面加热到 {target:.0f}°C …")
                p["power_kw"] = find_power(target, geom=geom, freq=p["freq"], t_heat=p["t_heat"],
                                           mur_cold=p["mur_cold"], nonlinear=p["nonlinear"], conc=p.get("conc"))
            self.progress.emit(10, f"计算中：{p['power_kw']:.2f} kW …")
            sim = InductionQuenchSim(geom=geom, **p)
            res = sim.run(progress=lambda x: self.progress.emit(10 + int(88 * x), "加热 → 淬火 …"))
            self.done.emit(sim, res)
        except Exception:
            self.failed.emit(traceback.format_exc())


class Canvas(FigureCanvasQTAgg):
    def __init__(self):
        self.fig = Figure(figsize=(6, 6))
        super().__init__(self.fig)


class InductionPage(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        lay = QtWidgets.QHBoxLayout(self)

        # ---------------- 左侧参数
        left = QtWidgets.QVBoxLayout()
        gw = QtWidgets.QGroupBox("工件 / 线圈 (mm)")
        fg = QtWidgets.QFormLayout(gw)
        self.e = {}
        for key, lab, val, lo, hi in [
            ("neck_od", "瓶口外径", 22, 5, 200), ("wall", "壁厚", 3, 0.5, 30), ("neck_len", "瓶颈长", 50, 10, 300),
            ("zone", "淬火区长", 20, 2, 100), ("coil_id", "线圈内径", 30, 6, 300), ("coil_h", "线圈高度", 25, 2, 200)]:
            sp = QtWidgets.QDoubleSpinBox(); sp.setRange(lo, hi); sp.setValue(val); sp.setDecimals(1)
            fg.addRow(lab, sp); self.e[key] = sp
        left.addWidget(gw)

        gh = QtWidgets.QGroupBox("加热")
        fh = QtWidgets.QFormLayout(gh)
        self.freq = QtWidgets.QComboBox(); self.freq.setEditable(True)
        self.freq.addItems(["5", "10", "20", "30", "50", "100", "200"]); self.freq.setCurrentText("20")
        fh.addRow("频率 (kHz)", self.freq)
        self.mode = QtWidgets.QComboBox(); self.mode.addItems(["给定目标温度（自动调功率）", "给定功率"])
        fh.addRow("方式", self.mode)
        self.target = QtWidgets.QDoubleSpinBox(); self.target.setRange(200, 1600); self.target.setValue(900)
        fh.addRow("外表面目标 °C", self.target)
        self.power = QtWidgets.QDoubleSpinBox(); self.power.setRange(0.1, 500); self.power.setValue(6.0)
        fh.addRow("工件吸收功率 kW", self.power)
        self.theat = QtWidgets.QDoubleSpinBox(); self.theat.setRange(0.1, 60); self.theat.setValue(3.0)
        fh.addRow("加热时间 s", self.theat)
        self.mur = QtWidgets.QDoubleSpinBox(); self.mur.setRange(1, 200); self.mur.setValue(20)
        fh.addRow("冷态有效磁导率 μr", self.mur)
        self.nl = QtWidgets.QCheckBox("非线性 B-H + 居里点 (更准, 慢约 3 倍)")
        fh.addRow(self.nl)
        self.conc = QtWidgets.QComboBox()
        self.conc.addItem("无导磁体", None)
        from rw.em.bh import FLUX_MATERIALS
        for k, mt in FLUX_MATERIALS.items():
            self.conc.addItem(mt.name, k)
        fh.addRow("线圈导磁体", self.conc)
        left.addWidget(gh)

        gq = QtWidgets.QGroupBox("冷却")
        fq = QtWidgets.QFormLayout(gq)
        self.medium = QtWidgets.QComboBox()
        for k, n in MEDIA:
            self.medium.addItem(n, k)
        fq.addRow("介质", self.medium)
        self.tdelay = QtWidgets.QDoubleSpinBox(); self.tdelay.setRange(0, 30); self.tdelay.setValue(0.0)
        fq.addRow("断电到喷液延时 s", self.tdelay)
        self.tq = QtWidgets.QDoubleSpinBox(); self.tq.setRange(0.1, 60); self.tq.setValue(3.0)
        fq.addRow("淬火时间 s", self.tq)
        self.inner = QtWidgets.QDoubleSpinBox(); self.inner.setRange(0, 1); self.inner.setSingleStep(0.1); self.inner.setValue(0.0)
        fq.addRow("内孔也被冷却 (0~1)", self.inner)
        left.addWidget(gq)

        self.btn = QtWidgets.QPushButton("▶  开始模拟")
        self.btn.setMinimumHeight(40)
        self.btn.clicked.connect(self.start)
        left.addWidget(self.btn)
        self.bar = QtWidgets.QProgressBar(); left.addWidget(self.bar)
        self.status = QtWidgets.QLabel("就绪"); self.status.setWordWrap(True); left.addWidget(self.status)
        self.out = QtWidgets.QTextEdit(); self.out.setReadOnly(True); left.addWidget(self.out, 1)
        lw = QtWidgets.QWidget(); lw.setLayout(left)
        sa = QtWidgets.QScrollArea(); sa.setWidgetResizable(True); sa.setWidget(lw)
        sa.setFixedWidth(352); sa.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)   # 小窗口(如 1280×800)左侧栏可滚动，按钮/结果一定够得着
        lay.addWidget(sa)

        # ---------------- 右侧图
        self.tabs = QtWidgets.QTabWidget()
        self.cv = {}
        for k, n in [("glow", "加热后外观(火色)"), ("temp", "温度场"), ("hrc", "硬度分布"), ("curve", "温度曲线"), ("line", "硬度-深度"), ("mur", "磁导率分布")]:
            c = Canvas(); self.cv[k] = c; self.tabs.addTab(c, n)
        lay.addWidget(self.tabs, 1)

    def params(self):
        geom = {k: self.e[k].value() for k in self.e}
        geom["coil_z0"] = geom["neck_len"] - geom["zone"] - (geom["coil_h"] - geom["zone"]) / 2 - 0.5
        return dict(
            geom=geom, freq=float(self.freq.currentText()) * 1e3,
            target=self.target.value() if self.mode.currentIndex() == 0 else None,
            power_kw=self.power.value(), t_heat=self.theat.value(), mur_cold=self.mur.value(),
            medium=self.medium.currentData(), t_delay=self.tdelay.value(), t_quench=self.tq.value(),
            t_air=4.0, inner_quench=self.inner.value(),
            nonlinear=self.nl.isChecked() or self.conc.currentData() is not None, conc_key=self.conc.currentData())

    def start(self):
        self.btn.setEnabled(False)
        self.th = QtCore.QThread()
        self.wk = Worker(self.params())
        self.wk.moveToThread(self.th)
        self.th.started.connect(self.wk.run)
        self.wk.progress.connect(lambda v, s: (self.bar.setValue(v), self.status.setText(s)))
        self.wk.done.connect(self.show_result)
        self.wk.failed.connect(self.show_err)
        self.wk.done.connect(self.th.quit); self.wk.failed.connect(self.th.quit)
        self.th.start()

    def show_err(self, tb):
        self.btn.setEnabled(True)
        self.status.setText("出错了")
        self.out.setPlainText(tb)

    def show_result(self, sim, r):
        self.btn.setEnabled(True)
        self.bar.setValue(100)
        self.status.setText("完成")
        self.sim = sim
        txt = [
            f"频率 {r['freq_kHz']:.0f} kHz，工件吸收 {r['power_kW']:.2f} kW",
            f"透入深度：冷态 {r['skin_cold_mm']:.2f} mm，热态(>居里点) {r['skin_hot_mm']:.2f} mm",
            f"加热结束：外表面 {r['T_surf_end_heat']:.0f}°C，内孔 {r['T_inner_end_heat']:.0f}°C，最高 {r['T_max']:.0f}°C",
            f"线圈电流约 {r['coil_current_A']:.0f} A(有效值)，线圈铜损约 {r['coil_loss_kW']:.2f} kW，电效率约 {r['efficiency']*100:.0f}%",
            f"淬火：{r['medium']}，t8/5 外表面 {r['t85_surface']:.2f} s / 内孔 {r['t85_inner']:.2f} s",
            f"硬度(估算)：外表面 {r['HRC_surface']:.0f} HRC，内孔 {r['HRC_inner']:.0f} HRC",
            f"有效硬化层 (≥50 HRC)：{r['case_depth_mm']:.2f} mm" + ("，已淬透" if r['through_hardened'] else ""),
        ]
        if r.get("nonlinear"):
            txt.insert(1, "磁导率：非线性 B-H(含居里点)" + (f"；导磁体 {r['conc']}，其损耗 {r['conc_loss_kW']:.2f} kW" if r.get("conc") else ""))
        if r["warnings"]:
            txt.append("")
            txt += ["⚠ " + w for w in r["warnings"]]
        self.out.setPlainText("\n".join(txt))
        for k, c in self.cv.items():
            c.fig.clear()
            ax = c.fig.add_subplot(111)
            if k == "glow":
                viz.plot_glow(ax, sim, title="加热结束瞬间（暗处看到的火色）")
            elif k == "temp":
                viz.plot_field(ax, sim, sim.T_end_heat, "inferno", "°C", 25, max(1000, sim.T_end_heat.max()), c.fig)
                ax.set_title("加热结束温度场")
            elif k == "hrc":
                viz.plot_field(ax, sim, sim.HRC, viz.HRC_CMAP, "HRC", 18, 60, c.fig)
                ax.set_title("淬火后硬度（估算）")
            elif k == "curve":
                viz.plot_curves(ax, sim)
            elif k == "line":
                viz.plot_hardness_line(ax, sim)
                ax.set_title("淬火区中部：硬度随深度")
            elif k == "mur":
                from rw.em.viz import plot_mur_map
                plot_mur_map(ax, sim, "加热结束时磁导率分布 (log10 μ_eff)" if hasattr(sim, "_mur_el") else "")
            c.fig.tight_layout()
            c.draw()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"真实的世界 v{VERSION}")
        self.resize(1500, 900)
        import os
        from rw.gui_world import WorldPage
        from rw.gui_struct_demo import StructDemoPage
        from rw.gui_plc_demo import PLCDemoPage
        here = os.path.dirname(os.path.abspath(__file__))
        try:
            from rw.plc.thermal_table import ThermalTable
            th = ThermalTable()
        except Exception:
            th = None
        self.top = QtWidgets.QTabWidget()
        self.ind = InductionPage()
        self.top.addTab(self.ind, "① 感应加热淬火")
        from rw.gui_em import EMPage
        self.em = EMPage()
        self.top.addTab(self.em, "①b 电磁升级")
        st = QtWidgets.QTabWidget(); st.addTab(StructDemoPage(), "机架/桌子 受力")
        pl = QtWidgets.QTabWidget(); self.plc_demo = PLCDemoPage(th); pl.addTab(self.plc_demo, "自己写 PLC 程序（单工位演示）")
        self.plc = None
        self.top.addTab(st, "② 结构受力")
        self.top.addTab(pl, "③ PLC 虚拟调试")
        self.st_tabs, self.plc_tabs = st, pl
        self.world = WorldPage()
        self.top.addTab(self.world, "④ 物理沙盒")
        from rw.gui_materials import MaterialsPage
        self.mat = MaterialsPage()
        self.top.addTab(self.mat, "⑤ 材料库")
        from rw.gui_circuit import CircuitPage
        self.circ = CircuitPage()
        self.top.addTab(self.circ, "⑥ 电路/示波器")
        from rw.gui_mcu import MCUPage

        def _to_scope():
            ct = getattr(self, "circ_tabs", None)
            if ct is not None and self.top.indexOf(ct) >= 0:
                self.top.setCurrentWidget(ct); ct.setCurrentWidget(self.circ)
            else:
                self.top.setCurrentWidget(self.circ)
        self.mcu = MCUPage(self.circ, _to_scope)
        self.top.addTab(self.mcu, "⑦ 单片机/PLC")
        from rw.gui_machine import MachinePage
        self.machine = MachinePage()
        self.top.addTab(self.machine, "⑧ 整机联动")
        from rw.gui_world3d import World3DPage
        self.w3d = World3DPage()
        self.top.addTab(self.w3d, "⑨ 三维装配世界")
        from rw.gui_flow import FlowPage
        self.flow = FlowPage()
        self.top.addTab(self.flow, "⑩ 流体")
        self.setCentralWidget(self.top)
        # 主题：浅色/深色/跟随系统，记住上次选择（v0.17）
        mbar = self.menuBar(); mth = mbar.addMenu("主题")
        grp = QtGui.QActionGroup(self); grp.setExclusive(True)
        cur = theme.current()
        for key, lab in (("system", "跟随系统"), ("light", "浅色"), ("dark", "深色")):
            a = mth.addAction(lab); a.setCheckable(True); a.setChecked(cur == key)
            a.triggered.connect(lambda _=False, k=key: theme.apply(QtWidgets.QApplication.instance(), k))
            grp.addAction(a)


def main():
    app = QtWidgets.QApplication(sys.argv)
    theme.apply(app, theme.current())     # 先应用主题（rcParams 就绪），页面建图即吃到
    w = MainWindow()
    w.show()
    if "--screenshot" in sys.argv:  # 自测：python3 app.py --screenshot out.png [induction|struct|plc]
        i = sys.argv.index("--screenshot")
        out = sys.argv[i + 1]
        which = sys.argv[i + 2] if len(sys.argv) > i + 2 else "induction"
        if which.startswith("em"):
            # em0 B-H/居里点; em1 导磁体对比; em2 三维涡流; em3 ①页 非线性+导磁体完整加热
            sub = int(which[2:]) if which[2:].isdigit() else 0
            if sub == 3:
                m = w.ind
                m.mode.setCurrentIndex(1); m.power.setValue(6.0); m.theat.setValue(1.5); m.tq.setValue(0.3)
                m.freq.setCurrentText("30"); m.nl.setChecked(True); m.conc.setCurrentIndex(m.conc.findData("smc"))
                m.tabs.setCurrentIndex(5)
                m.start()
                poll = QtCore.QTimer(w)

                def check3():
                    if m.btn.isEnabled():
                        poll.stop(); QtCore.QTimer.singleShot(800, lambda: (w.grab().save(out), app.quit()))
                poll.timeout.connect(check3); poll.start(500)
                QtCore.QTimer.singleShot(280000, lambda: (w.grab().save(out), app.quit()))
            else:
                w.top.setCurrentWidget(w.em)
                w.em.tabs.setCurrentIndex(sub)
                if sub == 1:
                    w.em.run_conc()
                    pg = w.em
                    poll = QtCore.QTimer(w)
                    poll.timeout.connect(lambda: (poll.stop(), QtCore.QTimer.singleShot(800, lambda: (w.grab().save(out), app.quit()))) if pg.btn_c.isEnabled() else None)
                    poll.start(500)
                else:
                    QtCore.QTimer.singleShot(1500, lambda: (w.grab().save(out), app.quit()))
        elif which.startswith("flow"):
            # flow0 管网 flow1 水锤 flow2 放空+射流
            sub = int(which[4:]) if which[4:].isdigit() else 0
            w.top.setCurrentWidget(w.flow); w.flow.tabs.setCurrentIndex(sub)
            if sub == 1:
                pg = w.flow.tabs.widget(1); pg.tc.setValue(15); pg.update_()
            QtCore.QTimer.singleShot(1200, lambda: (w.grab().save(out), app.quit()))
        elif which.startswith("w3d"):
            # w3d0 钢板桥(仿真后) w3d1 叠积木倒塌(中途帧) w3d2 水里浮沉
            sub = int(which[3:]) if which[3:].isdigit() else 0
            w.top.setCurrentWidget(w.w3d)
            pg = w.w3d
            pg.preset.setCurrentIndex({0: 0, 1: 2, 2: 6, 3: 3, 4: 5}[sub]); pg.load_preset(pg.preset.currentIndex())
            if sub != 2:
                pg.run()
                if sub in (1, 3):
                    pg.slider.setValue(int(pg.slider.maximum() * 0.45))
                if sub == 4:
                    pg.tabs.setCurrentIndex(1)
            QtCore.QTimer.singleShot(800, lambda: (w.grab().save(out), app.quit()))
        elif which == "struct":
            w.top.setCurrentWidget(w.st_tabs); w.st_tabs.setCurrentIndex(0)
            QtCore.QTimer.singleShot(800, lambda: (w.grab().save(out), app.quit()))
        elif which.startswith("world"):
            w.top.setCurrentWidget(w.world)
            sub = int(which[-1]) if which[-1].isdigit() else 0
            w.world.tabs.setCurrentIndex(sub)
            if sub == 0:
                pg = w.world.tabs.widget(0)
                pg.fluid.setCurrentIndex(0); pg.T.setValue(150); pg.depth.setValue(50)
            QtCore.QTimer.singleShot(1200, lambda: (w.grab().save(out), app.quit()))
        elif which.startswith("mat"):
            w.top.setCurrentWidget(w.mat)
            sub = int(which[-1]) if which[-1].isdigit() else 0
            w.mat.tabs.setCurrentIndex(sub)
            if sub == 0:
                w.mat.pt.prop.setCurrentIndex(1); w.mat.pt.show_el("Ag")
            QtCore.QTimer.singleShot(1200, lambda: (w.grab().save(out), app.quit()))
        elif which.startswith("circ"):
            if getattr(w, "circ_tabs", None) is not None:  # 私有板子页存在时 ⑥ 是子页签
                w.top.setCurrentWidget(w.circ_tabs); w.circ_tabs.setCurrentWidget(w.circ)
            else:
                w.top.setCurrentWidget(w.circ)
            pg = w.circ
            sub = int(which[4:]) if which[4:].isdigit() else 0
            # circ0 半桥四通道+时间光标；circ1 运放振荡器+FFT+电压光标；circ2 导入 KiCad 网表+Auto Set；circ3 整流（差分通道）
            key = {0: "halfbridge", 1: "astable", 3: "rect"}.get(sub)
            for i in range(pg.pick.count()):
                d = pg.pick.itemData(i)
                if (key and d == ("ex", key)) or (sub == 2 and d[0] == "file" and d[1].endswith("npn_switch_kicad.net")) \
                        or (sub == 4 and d[0] == "file" and d[1].endswith("no_source_demo.cir")):  # circ4：v0.11.1 无激励警告
                    pg.pick.setCurrentIndex(i); break
            pg.simulate()
            poll = QtCore.QTimer(w)

            def after():
                if sub == 0:
                    pg.cur_t.setChecked(True); pg.t1.setValue(2.0); pg.t2.setValue(7.0)
                elif sub == 1:
                    pg.fft_on.setChecked(True); pg.win.setCurrentIndex(1); pg.fft_span.setCurrentIndex(1); pg.fmax.setValue(10.0)
                    pg.cur_v.setChecked(True); pg.vch.setCurrentIndex(1); pg.v1.setValue(-1.6); pg.v2.setValue(1.6)
                    pg.ctl.setCurrentIndex(3)
                elif sub == 2:
                    pg.ctl.setCurrentIndex(1)
                QtCore.QTimer.singleShot(800, lambda: (w.grab().save(out), app.quit()))

            def check():
                if pg.btn.isEnabled() and pg.scope.t is not None:
                    poll.stop(); after()
            poll.timeout.connect(check)
            poll.start(300)
            QtCore.QTimer.singleShot(60000, lambda: (w.grab().save(out), app.quit()))
        elif which.startswith("mach"):
            # mach0 正常上电+5 件；mach1 缺水联锁被短接→线圈烧；mach2 长时间运行无换热器；mach3 预充电太短
            w.top.setCurrentWidget(w.machine)
            pg = w.machine
            sub = int(which[4:]) if which[4:].isdigit() else 0
            key = {0: "normal", 1: "water_loss_bypassed", 2: "longrun_no_hex", 3: "precharge_short", 4: "mains_sag"}.get(sub, "normal")
            pg.sel.setCurrentIndex(pg.sel.findData(key))
            pg.simulate()
            poll = QtCore.QTimer(w)

            def check():
                if pg.btn.isEnabled() and pg.m is not None:
                    poll.stop(); QtCore.QTimer.singleShot(1200, lambda: (w.grab().save(out), app.quit()))
            poll.timeout.connect(check)
            poll.start(300)
            QtCore.QTimer.singleShot(90000, lambda: (w.grab().save(out), app.quit()))
        elif which.startswith("mcu"):
            # mcu0 全程时序 + 汇编源码 + MCU 引脚；mcu1 PLC 报警联锁 + 梯形图/IO；mcu2 过流段送进 ⑥ 示波器
            w.top.setCurrentWidget(w.mcu)
            pg = w.mcu
            sub = int(which[3:]) if which[3:].isdigit() else 0
            pg.simulate()
            poll = QtCore.QTimer(w)

            def after():
                if sub == 0:
                    pg.view.setCurrentIndex(0); pg.slider.setValue(95); pg.rt.setCurrentIndex(0)
                elif sub == 1:
                    pg.view.setCurrentIndex(4); pg.slider.setValue(800); pg.rt.setCurrentIndex(1); pg.src_tabs.setCurrentIndex(1)
                else:
                    pg.view.setCurrentIndex(3); pg.send_scope()
                QtCore.QTimer.singleShot(1200, lambda: (w.grab().save(out), app.quit()))

            def check():
                if pg.btn.isEnabled() and pg.cs is not None:
                    poll.stop(); after()
            poll.timeout.connect(check)
            poll.start(300)
            QtCore.QTimer.singleShot(90000, lambda: (w.grab().save(out), app.quit()))
        elif which == "plcdemo":
            w.top.setCurrentWidget(w.plc_tabs); w.plc_tabs.setCurrentWidget(w.plc_demo)
            w.plc_demo.load(); w.plc_demo.spd.setCurrentIndex(2)
            QtCore.QTimer.singleShot(800, lambda: w.plc_demo.sim.press("X000"))
            QtCore.QTimer.singleShot(5200, lambda: (w.grab().save(out), app.quit()))
        elif which == "structdemo":
            w.top.setCurrentWidget(w.st_tabs); w.st_tabs.setCurrentIndex(w.st_tabs.count() - 1)
            QtCore.QTimer.singleShot(1000, lambda: (w.grab().save(out), app.quit()))
        else:
            m = w.ind
            m.mode.setCurrentIndex(1); m.power.setValue(6.0)
            m.start()
            poll = QtCore.QTimer(w)

            def check():
                if m.btn.isEnabled():
                    poll.stop()
                    QtCore.QTimer.singleShot(500, lambda: (w.grab().save(out), app.quit()))
            poll.timeout.connect(check)
            poll.start(300)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
