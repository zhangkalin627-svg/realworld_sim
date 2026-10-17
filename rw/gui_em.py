"""①b 电磁升级页: B-H/居里点、导磁体对比、全三维涡流"""
import os, json, traceback
import numpy as np
from PySide6 import QtCore, QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from .em import viz as ev

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE3D = os.path.join(HERE, "em", "eddy3d_results.json")


class _Canvas(FigureCanvasQTAgg):
    def __init__(self):
        self.fig = Figure(figsize=(8, 5))
        super().__init__(self.fig)


class _Job(QtCore.QObject):
    done = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    @QtCore.Slot()
    def run(self):
        try:
            self.done.emit(self.fn())
        except Exception:
            self.failed.emit(traceback.format_exc())


def _spawn(owner, fn, on_done, on_fail):
    owner._th = QtCore.QThread()
    owner._job = _Job(fn)
    owner._job.moveToThread(owner._th)
    owner._th.started.connect(owner._job.run)
    owner._job.done.connect(on_done)
    owner._job.failed.connect(on_fail)
    owner._job.done.connect(owner._th.quit)
    owner._job.failed.connect(owner._th.quit)
    owner._th.start()


class EMPage(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        lay = QtWidgets.QVBoxLayout(self)
        self.tabs = QtWidgets.QTabWidget()
        lay.addWidget(self.tabs)

        # ---- 0 B-H 曲线
        w0 = QtWidgets.QWidget(); l0 = QtWidgets.QVBoxLayout(w0)
        self.c_bh = _Canvas(); ev.plot_bh(self.c_bh.fig); self.c_bh.draw()
        l0.addWidget(self.c_bh, 1)
        t = QtWidgets.QLabel(
            "模型: Fröhlich–Kennelly 曲线 + Js(T)=Js0·(1−T/Tc)^β (Tc=770°C, β=0.37)。感应加热时表面场强 ~1e4–1e5 A/m，钢早已饱和，\n"
            "有效 μ 只有 10~50，不是手册里的几百；到居里点降为 1。谐波平衡(基波)算法，已用一维时域参考解检验(见 validate §11)。\n"
            "⚠ 真实钢的曲线要用该炉钢实测 B-H (膝点、Hopkinson 峰没建模)。在 ① 页勾选“非线性 B-H”即用此曲线。")
        t.setWordWrap(True); l0.addWidget(t)
        self.tabs.addTab(w0, "B-H 与居里点")

        # ---- 1 导磁体对比
        w1 = QtWidgets.QWidget(); l1 = QtWidgets.QVBoxLayout(w1)
        top = QtWidgets.QHBoxLayout()
        self.f_c = QtWidgets.QComboBox(); self.f_c.setEditable(True); self.f_c.addItems(["10", "20", "30", "50", "100"]); self.f_c.setCurrentText("30")
        self.p_c = QtWidgets.QDoubleSpinBox(); self.p_c.setRange(0.5, 100); self.p_c.setValue(6.0)
        self.btn_c = QtWidgets.QPushButton("▶ 六案对比 (无=木头 / 硅钢 / SMC / 铁氧体 / 铝 / 石墨)")
        self.btn_c.clicked.connect(self.run_conc)
        top.addWidget(QtWidgets.QLabel("频率 kHz")); top.addWidget(self.f_c)
        top.addWidget(QtWidgets.QLabel("吸收功率 kW")); top.addWidget(self.p_c); top.addWidget(self.btn_c); top.addStretch(1)
        l1.addLayout(top)
        self.c_cmp = _Canvas(); l1.addWidget(self.c_cmp, 1)
        self.t_cmp = QtWidgets.QTextEdit(); self.t_cmp.setReadOnly(True); self.t_cmp.setMaximumHeight(230)
        l1.addWidget(self.t_cmp)
        self.tabs.addTab(w1, "导磁体对比")

        # ---- 2 三维涡流
        w2 = QtWidgets.QWidget(); l2 = QtWidgets.QVBoxLayout(w2)
        top2 = QtWidgets.QHBoxLayout()
        self.shape3 = QtWidgets.QComboBox(); self.shape3.addItems(["圆管", "方坯 20×20 (gmsh)"])
        self.ecc = QtWidgets.QDoubleSpinBox(); self.ecc.setRange(0, 4); self.ecc.setValue(3.0); self.ecc.setSingleStep(0.5)
        self.btn_3 = QtWidgets.QPushButton("▶ 重新计算三维 (约 1–3 分钟)")
        self.btn_3.clicked.connect(self.run_3d)
        top2.addWidget(QtWidgets.QLabel("工件")); top2.addWidget(self.shape3)
        top2.addWidget(QtWidgets.QLabel("偏心 mm")); top2.addWidget(self.ecc); top2.addWidget(self.btn_3); top2.addStretch(1)
        l2.addLayout(top2)
        self.c_3d = _Canvas(); l2.addWidget(self.c_3d, 1)
        self.t_3d = QtWidgets.QTextEdit(); self.t_3d.setReadOnly(True); self.t_3d.setMaximumHeight(150)
        l2.addWidget(self.t_3d)
        self.tabs.addTab(w2, "全三维涡流")
        self.show_3d(self._load_cache())

    # ------------------------------------------------------------- 导磁体
    def run_conc(self):
        self.btn_c.setEnabled(False)
        self.t_cmp.setPlainText("计算中 …")
        f = float(self.f_c.currentText()) * 1e3
        P = self.p_c.value()

        def job():
            from .em.check import compare_concentrators
            return compare_concentrators(freq=f, power_kw=P, kinds=(None, "sife", "smc", "ferrite", "alu", "graphite"))
        _spawn(self, job, self._show_conc, self._err_c)

    def _show_conc(self, res):
        self.btn_c.setEnabled(True)
        ev.plot_conc(self.c_cmp.fig, res); self.c_cmp.draw()
        base = res["none"]
        lines = ["【仿真结果】同功率同频率六案对比（粉尘/拉弧指标为量级估计）："]
        for k, v in res.items():
            d = (v["I_rms"] / base["I_rms"] - 1) * 100
            lines.append(f"{v['name']}: 电流 {v['I_rms']:.0f} A ({d:+.0f}%)，电效率 {v['eff']*100:.1f}%，导磁位损耗 {v['conc_loss_kW']*1e3:.0f} W，"
                         f"磁密峰值 {v['Bmax_conc']:.2f} T，绝热升温 {v['dTdt']:.1f} K/s")
            lines.append(f"　　积灰区表面 B {v['B_dust_T']:.2f} T，50µm 铁粉受力约 {v['F_dust_uN']:.1f} µN，单匝电压约 {v['V_turn']:.0f} V")
        lines.append("⚠ 均为仿真结果，非实测。铝/石墨不是导磁体(μr=1)：其涡流由求解器直接解，物理作用是屏蔽/排斥磁场。"
                     "铁粉受力为饱和偶极量级估计 F≈V·Ms·∇B；单匝电压=总功率/电流的单匝等效，未解静电场；"
                     "拉弧只能按 场强+积灰 定性分级，粉尘堆积过程不可解。导磁体按各向同性 μ 处理(叠片实际各向异性)；"
                     "损耗参数是典型量级，请按供应商数据改；线圈铜损估算偏差可达 ±30%。")
        self.t_cmp.setPlainText("\n".join(lines))

    def _err_c(self, tb):
        self.btn_c.setEnabled(True); self.t_cmp.setPlainText(tb)

    # ------------------------------------------------------------- 三维
    def _load_cache(self):
        if os.path.exists(CACHE3D):
            try:
                return json.load(open(CACHE3D))
            except Exception:
                return None
        return None

    def run_3d(self):
        self.btn_3.setEnabled(False)
        self.t_3d.setPlainText("三维求解中 … (棱边单元 + 直接法，内存约 1 GB)")
        e = self.ecc.value()
        billet = self.shape3.currentIndex() == 1

        def job():
            if billet:
                from .em.eddy3d_gmsh import study_billet
                r = study_billet(ecc_mm=e)
                r["_kind"] = "billet"
                return r
            from .em.eddy3d import study_3d
            r = study_3d(ecc_mm=e)
            r["_kind"] = "tube"
            return r
        if billet:
            self.t_3d.setPlainText("方坯三维求解中 … (gmsh 非结构网格 ×2 例，约 30–60 秒)")
        _spawn(self, job, self._done_3d, self._err_3d)

    def _done_3d(self, res):
        self.btn_3.setEnabled(True)
        self.show_3d(res)

    def _err_3d(self, tb):
        self.btn_3.setEnabled(True); self.t_3d.setPlainText(tb)

    def show_3d(self, res):
        if not res:
            self.t_3d.setPlainText("还没有三维结果，点“重新计算三维”。")
            return
        if res.get("_kind") == "billet":
            ev.plot_billet(self.c_3d.fig, res); self.c_3d.draw()
            e0, e1 = res["ecc0"], res["ecc"]
            self.t_3d.setPlainText(
                f"方坯 20×20 (gmsh 非结构网格, {e0['tets']} 四面体): 同心 {e0['P']/1e3:.3f} kW (每 1 kA 安匝)，"
                f"偏心 {e1['ecc_mm']:g} mm {e1['P']/1e3:.3f} kW ({100*(e1['P']/e0['P']-1):+.1f}%)。\n"
                f"表层/心部功率密度 {e0['skin']:.1f}× (集肤效应)，角部/面心 {e0['corner']:.2f} (<1 是 δ 尺度的「冷角」regime: 两侧表面感应电流相消)，"
                f"左右对称 {e0['sym']:.3f}；人为空气电导假损耗 {100*e0['P_air']/e0['P']:.2f}%。\n"
                "⚠ 最低阶棱边单元 + 粗网格(h≈3 mm≈δ)，定量精度 ~5–10%，角部奇异性未分辨；"
                "同几何空心管对照: vs 结构网格 +2.7% / vs 二维轴对称 +0.2% (validate §11g)。")
            return
        ev.plot_3d(self.c_3d.fig, res); self.c_3d.draw()
        e0, e1 = res["ecc0"], res["ecc"]
        sp = np.array(e1["sectors"])
        self.t_3d.setPlainText(
            f"三维: Nédélec 棱边单元，{e0['tets']} 四面体 / {e0['edges']} 棱边；工件热态(μr=1, ρ=1.1e-6)、30 kHz。\n"
            f"同心: 三维 {e0['P_3d']/1e3:.3f} kW vs 二维轴对称 {e0['P_2d']/1e3:.3f} kW (每 1 kA 安匝)，差 {100*(e0['P_3d']/e0['P_2d']-1):+.1f}%；"
            f"同心时各方位功率最大/最小 = {max(e0['sectors'])/min(e0['sectors']):.3f} (轴对称应为 1)。\n"
            f"偏心 {e1['ecc_mm']:g} mm: 管壁各方位功率最大/最小 = {sp.max()/sp.min():.2f}，总功率变化 {100*(e1['P_3d']/e0['P_3d']-1):+.1f}%。\n"
            f"规范用的人为空气电导 σ=1 S/m 引起的假损耗占 {100*e0['P_air']/e0['P_3d']:.2f}%。\n"
            "⚠ 最低阶棱边单元 + 粗网格(管壁 3 层)，估计总功率精度 ~5–10%；非圆截面选上面「方坯」(需 pip install gmsh)。")
