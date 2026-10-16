"""④ 物理沙盒：沉水仪表 / 管路水泵 / 电机带负载 —— 改参数立刻按物理规律给结论。"""
import math
import numpy as np
from PySide6 import QtCore, QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from .world.immersion import PressureGauge, Env, check_gauge
from .world.fluids import FLUIDS, P_ATM
from .world.pipes import example_spray
from .world.motor import InductionMotor, simulate_start, starts_per_hour
from .world import world_viz as WV

ICON = {"OK": "✔", "WARN": "⚠", "FAIL": "✘"}
COLOR = {"OK": "#1a7f37", "WARN": "#b26a00", "FAIL": "#c62828"}


def html(verdicts):
    import html as _h
    return "<br>".join(f'<span style="color:{COLOR[l]}"><b>[{l}]</b> {_h.escape(t)}</span>' for l, t in verdicts)


def spin(lo, hi, val, step=1.0, dec=2):
    s = QtWidgets.QDoubleSpinBox(); s.setRange(lo, hi); s.setDecimals(dec); s.setSingleStep(step); s.setValue(val)
    return s


class _Page(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        lay = QtWidgets.QHBoxLayout(self)
        self.form = QtWidgets.QFormLayout()
        left = QtWidgets.QVBoxLayout()
        g = QtWidgets.QGroupBox("参数（改了立刻重算）"); g.setLayout(self.form); left.addWidget(g)
        self.out = QtWidgets.QTextBrowser(); left.addWidget(self.out, 1)
        lw = QtWidgets.QWidget(); lw.setLayout(left); lw.setFixedWidth(420); lay.addWidget(lw)
        self.fig = Figure(figsize=(8, 5)); self.cv = FigureCanvasQTAgg(self.fig); lay.addWidget(self.cv, 1)

    def add(self, label, w):
        self.form.addRow(label, w)
        for sig in ("valueChanged", "currentIndexChanged", "toggled"):
            if hasattr(w, sig):
                getattr(w, sig).connect(self.update_)
                break
        return w


class ImmersionPage(_Page):
    def __init__(self):
        super().__init__()
        self.fluid = QtWidgets.QComboBox()
        for k, f in FLUIDS.items():
            self.fluid.addItem(f["name"], k)
        self.fluid.addItem("自定义密度", None)
        self.add("液体", self.fluid)
        self.rho = self.add("自定义比重 (水=1)", spin(0.1, 20, 0.7, 0.05))
        self.T = self.add("液体温度 °C", spin(-50, 400, 20, 5, 1))
        self.depth = self.add("放置深度 m", spin(0, 11000, 50, 5, 1))
        self.psurf = self.add("液面上方压力 kPa(绝压)", spin(1, 50000, 101.3, 10, 1))
        self.rng = self.add("压力表量程 MPa", spin(0.06, 100, 1.6, 0.1))
        self.Tmax = self.add("压力表耐温 °C", spin(20, 600, 200, 10, 0))
        self.rdepth = self.add("外壳额定水深 m", spin(0, 11000, 100, 10, 0))
        self.acc = self.add("精度等级 %", spin(0.1, 4, 1.6, 0.1, 1))
        self.mass = self.add("表重 kg", spin(0.05, 50, 0.9, 0.1))
        self.vol = self.add("表体积 L", spin(0.01, 50, 0.35, 0.05))
        self.update_()

    def update_(self, *a):
        key = self.fluid.currentData()
        fl = key if key else self.rho.value()
        env = Env(fl, self.T.value(), self.depth.value(), self.psurf.value() * 1e3, self.psurf.value() < 102)
        g = PressureGauge(range_MPa=self.rng.value(), T_medium_max=self.Tmax.value(), rated_depth_m=self.rdepth.value(),
                          accuracy=self.acc.value(), mass_kg=self.mass.value(), volume_L=self.vol.value())
        v, vals = check_gauge(g, env)
        head = (f"<b>{vals['fluid']}</b>，ρ = {vals['rho']:.0f} kg/m³，{env.depth_m:g} m 处："
                f"绝压 {vals['p_abs']/1e6:.3f} MPa，表压 <b>{vals['p_gauge']/1e6:.3f} MPa</b>；"
                f"这里沸点 {vals['Tb_here']:.0f}°C<br><br>")
        self.out.setHtml(head + html(v))
        WV.plot_immersion(self.fig, g, env, v, vals); self.cv.draw()


class PipePage(_Page):
    def __init__(self):
        super().__init__()
        self.fluid = QtWidgets.QComboBox()
        for k in ("pag10", "water", "oil_quench"):
            self.fluid.addItem(FLUIDS[k]["name"], k)
        self.add("介质", self.fluid)
        self.T = self.add("液温 °C", spin(5, 90, 30, 5, 0))
        self.holes = self.add("喷淋孔数", spin(4, 200, 24, 2, 0))
        self.dh = self.add("孔径 mm", spin(0.5, 5, 1.2, 0.1, 1))
        self.fk = self.add("过滤器阻力系数 K（脏了变大）", spin(1, 500, 8, 5, 0))
        self.sw = self.add("流量开关最小动作 L/min", spin(0, 200, 21, 1, 0))
        self.update_()

    def update_(self, *a):
        try:
            c, op = example_spray(self.fluid.currentData(), self.T.value(), int(self.holes.value()), self.dh.value() * 1e-3, self.fk.value())
        except Exception as e:
            self.out.setPlainText(f"算不出工作点：{e}"); return
        v = []
        q = op["Q_Lmin"]
        if self.sw.value() and q < self.sw.value() * 1.15:
            v.append(("FAIL" if q < self.sw.value() else "WARN",
                      f"实际流量 {q:.1f} L/min，流量开关 HFS-25(1\") 最小动作约 {self.sw.value():g} L/min → X036 "
                      + ("不会闭合，喷淋时会报“淬火液流量不足”" if q < self.sw.value() else "余量不到 15%，过滤器一脏就会误报")))
        hose = [d for d in op["details"] if d[0].startswith("软管")][0]
        if hose[2]["v"] > 3:
            v.append(("WARN", f"软管里流速 {hose[2]['v']:.1f} m/s（>3 m/s），这一段吃掉 {hose[1]/1e3:.0f} kPa，是瓶颈；换内径 10–12 mm 的管"))
        noz = op["details"][-1]
        v.append(("OK", f"每个孔 {q/self.holes.value():.2f} L/min，射流约 {noz[2]['v_jet']:.1f} m/s；泵轴功率 {op['P_shaft']:.0f} W（电机 550 W）"))
        self.out.setHtml(f"<b>工作点：{q:.1f} L/min，扬程 {op['H']:.1f} m</b><br><br>" + html(v))
        WV.plot_circuit(self.fig, c, op, self.sw.value() or None); self.cv.draw()


class MotorPage(_Page):
    def __init__(self):
        super().__init__()
        self.P = self.add("额定功率 kW", spin(0.06, 500, 0.75, 0.05))
        self.poles = QtWidgets.QComboBox(); [self.poles.addItem(str(p), p) for p in (2, 4, 6, 8)]; self.poles.setCurrentIndex(1)
        self.add("极数", self.poles)
        self.n = self.add("额定转速 rpm", spin(100, 3600, 1400, 10, 0))
        self.eta = self.add("效率 %", spin(10, 120, 82.5, 0.5, 1))
        self.pf = self.add("功率因数", spin(0.1, 1.2, 0.76, 0.01))
        self.mass = self.add("重量 kg", spin(0.5, 5000, 13, 1, 1))
        self.Jl = self.add("负载惯量(折到电机轴) kg·m²", spin(0, 10, 0.001, 0.001, 4))
        self.Tl = self.add("负载转矩(折到电机轴) N·m", spin(0, 5000, 0.84, 0.1))
        self.starts = self.add("每小时起停次数", spin(0, 10000, 456, 10, 0))
        self.update_()

    def update_(self, *a):
        p = self.poles.currentData()
        m = InductionMotor(name=f"{self.P.value():g} kW {p}P", P=self.P.value() * 1e3, poles=p, n_rated=self.n.value(),
                           eta=self.eta.value() / 100, pf=self.pf.value(), mass=self.mass.value(),
                           J=0.0021 * (self.P.value() / 0.75) ** 1.1)
        v = m.plausibility()
        st = None
        if all(x[0] != "FAIL" for x in v):
            st = simulate_start(m, self.Jl.value(), self.Tl.value(), t_end=1.5)
            if st["stalled"]:
                v.append(("FAIL", f"负载 {self.Tl.value():g} N·m 大于电机起动转矩 {m.torque(0):.2f} N·m，起不来"))
            else:
                util = self.Tl.value() / m.Tn
                v.append(("OK" if util <= 1 else ("WARN" if util < 1.15 else "FAIL"),
                          f"负载率 {util*100:.0f}%（额定 {m.Tn:.2f} N·m）" + ("，长期超载会烧" if util > 1.15 else "")
                          + ("，电机选大了，可以小一号" if util < 0.3 else "")))
                Z = starts_per_hour(m, self.Jl.value(), self.Tl.value())
                need = self.starts.value()
                v.append(("OK" if need <= Z else "FAIL", f"每小时起停 {need:.0f} 次，允许约 {Z:.0f} 次（样本法，Z0 按 {m.Z0:.0f} 估）"))
        I = m.In
        self.out.setHtml(f"<b>同步转速 {m.n_sync:.0f} rpm，额定转矩 {m.Tn:.2f} N·m，额定电流约 {I:.2f} A</b><br><br>" + html(v))
        WV.plot_motor(self.fig, m, st, self.Tl.value()); self.cv.draw()


class WorldPage(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        lay = QtWidgets.QVBoxLayout(self)
        t = QtWidgets.QTabWidget()
        t.addTab(ImmersionPage(), "沉到液体里的压力表")
        t.addTab(PipePage(), "管路 / 水泵 / 喷淋")
        t.addTab(MotorPage(), "电机 + 负载")
        t.addTab(LongRunPage(), "连续运行 8 小时")
        lay.addWidget(t)
        self.tabs = t


class LongRunPage(_Page):
    def __init__(self):
        super().__init__()
        from .world.longrun import crack_risk_compare
        self.cc = crack_risk_compare()
        self.hours = self.add("连续运行 小时", spin(0.1, 72, 8, 1, 1))
        self.takt = self.add("节拍（淬火 3 s 时）s", spin(3, 120, 8.0, 0.5, 2))
        self.tq = self.add("淬火时间 D201 s", spin(1.0, 3.0, 3.0, 0.1, 1))
        self.V = self.add("水箱容积 L", spin(20, 5000, 300, 50, 0))
        self.UA = self.add("板换能力 UA W/K（0=没有/冷水机停）", spin(0, 5000, 500, 50, 0))
        self.ch = self.add("冷水机制冷量 kW", spin(0, 100, 8, 1, 1))
        self.Tamb = self.add("车间温度 °C", spin(0, 50, 30, 1, 0))
        self.conc = self.add("PAG 浓度 %", spin(0, 30, 10, 1, 0))
        self.update_()

    def update_(self, *a):
        from .world.longrun import run_longrun, verdicts_longrun, crack_verdicts, quench_time_sweep, self_temper_hrc
        tq = self.tq.value()
        takt = self.takt.value() - (3.0 - tq)
        r = run_longrun(hours=self.hours.value(), takt=takt, V_tank_L=self.V.value(), UA=self.UA.value(),
                        chiller_kW=self.ch.value(), T_amb=self.Tamb.value(), conc=self.conc.value() / 100)
        if not hasattr(self, "_sweep"):
            self._sweep = quench_time_sweep()
        sw = self._sweep
        Ta = float(np.interp(tq, [x["t_quench"] for x in sw], [x["T_exit"] for x in sw]))
        Ht = float(np.interp(tq, [x["t_quench"] for x in sw], [x["HRC_temp"] for x in sw]))
        v = verdicts_longrun(r, crack_verdicts(self.cc))
        v.insert(0, ("WARN" if Ta > 60 else "OK", f"工件出淬火区约 {Ta:.0f}°C" + ("，烫手（>60°C），出料皮带若是 PU/PVC（耐温约 80°C）会被烫坏，取放 C 的 V 爪也要耐温" if Ta > 150 else ("，摸上去烫" if Ta > 60 else ""))))
        lv = "OK" if 50 <= Ht <= 55 else "WARN"
        v.insert(1, (lv, f"余热自回火后硬度约 {Ht:.1f} HRC（要求 50–55；淬火态约 58）"))
        self.out.setHtml(f"<b>节拍 {takt:.2f} s，{r['parts']} 件，每件带进水箱约 {r['Q_in_avg']/1e3:.2f} kW</b><br><br>" + html(v))
        f = self.fig; f.clear()
        ax = f.add_subplot(131); h = r["hist"]
        ax.plot(h["t"] / 60, h["T_tank"]); ax.axhspan(20, 40, color="#2ca02c", alpha=0.1)
        ax.set_xlabel("小时"); ax.set_ylabel("淬火液温度 °C"); ax.set_title("水箱温度"); ax.grid(alpha=0.3)
        ax = f.add_subplot(132)
        ax.plot(h["t"] / 60, h["chiller_load"] / 1e3, label="负荷"); ax.plot(h["t"] / 60, h["chiller_cap"] / 1e3, "--", label="能力")
        ax.set_xlabel("小时"); ax.set_ylabel("kW"); ax.set_title("冷水机"); ax.legend(); ax.grid(alpha=0.3)
        ax = f.add_subplot(133)
        x = [s["t_quench"] for s in sw]
        ax.plot(x, [s["T_exit"] for s in sw], color="#d62728"); ax.set_xlabel("淬火时间 s"); ax.set_ylabel("出料表面温度 °C", color="#d62728")
        ax2 = ax.twinx(); ax2.plot(x, [s["HRC_temp"] for s in sw], color="#1f77b4"); ax2.axhspan(50, 55, color="#1f77b4", alpha=0.1)
        ax2.set_ylabel("自回火后 HRC", color="#1f77b4"); ax.axvline(tq, color="k", ls=":")
        ax.set_title("淬火时间 → 出料温度/硬度"); ax.grid(alpha=0.3)
        f.tight_layout(); self.cv.draw()
