"""⑩ 流体：管网（泵/阀/并联/喷淋）· 水锤 · 水箱放空与射流。改参数立刻重算。"""
import math
import numpy as np
from PySide6 import QtWidgets
from matplotlib.figure import Figure
from . import viz  # noqa 字体
from .gui_world import _Page, spin, html
from .world.fluids import FLUIDS, G
from .flow.network import example_two_station, Network, Nozzle
from .flow import transient as TR, jet as JT


class NetPage(_Page):
    def __init__(self):
        super().__init__()
        self.fluid = QtWidgets.QComboBox()
        for k in ("pag10", "water", "oil_quench"):
            self.fluid.addItem(FLUIDS[k]["name"], k)
        self.add("介质", self.fluid)
        self.T = self.add("液温 °C", spin(5, 90, 30, 5, 0))
        self.spd = self.add("泵转速 % 额定", spin(30, 120, 100, 5, 0))
        self.dirty = self.add("吸入过滤器脏污 ×（1=干净）", spin(0.5, 40, 1, 1, 1))
        self.openB = self.add("B 路电磁阀开度 %", spin(0, 100, 100, 10, 0))
        self.holes = self.add("每路喷淋孔数", spin(4, 100, 24, 2, 0))
        self.dh = self.add("孔径 mm", spin(0.5, 4, 1.2, 0.1, 1))
        self.hose = self.add("软管内径 mm", spin(5, 20, 10, 1, 0))
        self.update_()

    def build(self):
        net = example_two_station(self.fluid.currentData(), self.T.value(), self.dirty.value())
        for l in net.links:
            if l.kind == "pump": l.speed = self.spd.value() / 100
            if l.name == "电磁阀B": l.opening = self.openB.value() / 100
            if l.kind == "nozzle": l.n = int(self.holes.value()); l.d = self.dh.value() * 1e-3
            if l.name.startswith("软管"): l.D = self.hose.value() * 1e-3
        return net

    def update_(self, *a):
        net = self.build()
        try:
            res = net.solve()
        except Exception as e:
            self.out.setPlainText(f"算不出：{e}"); return
        chk = net.checks(res)
        L = res["links"]
        qa, qb, qp = L["软管A 内径10"]["Q_Lmin"], L["软管B 内径10"]["Q_Lmin"], L["离心泵"]["Q_Lmin"]
        pm = L["离心泵"]
        txt = (f"<b>泵送 {qp:.1f} L/min（A 路 {qa:.1f}，B 路 {qb:.1f}）</b>，扬程 {pm['head']:.1f} m，"
               f"轴功率 {pm['P_shaft']:.0f} W<br>射流 {L['喷淋孔A']['v_jet']:.1f} m/s，A 环内压 "
               f"{res['nodes']['A环']['p_static']/1e5:.2f} bar<br><br>")
        self.out.setHtml(txt + html(chk))
        self.draw(net, res)

    def draw(self, net, res):
        f = self.fig; f.clear()
        ax = f.add_subplot(121)
        path = ["水箱", "泵吸入", "泵出口", "分叉", "A阀后", "A环"]
        p = [res["nodes"][n]["p_static"] / 1e5 for n in path]
        ax.plot(range(len(path)), p, "o-", color="#1f77b4", lw=2)
        for i, v in enumerate(p):
            ax.annotate(f"{v:.2f}", (i, v), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(range(len(path))); ax.set_xticklabels(path, rotation=30, fontsize=8)
        ax.set_ylabel("静压 bar（表压）"); ax.set_title("沿 A 路的压力变化：水箱 → 泵 → 喷淋环"); ax.grid(alpha=0.3)
        ax2 = f.add_subplot(122)
        pump = [l for l in net.links if l.kind == "pump"][0]
        Q = np.linspace(0, pump.H0 and math.sqrt(pump.H0 / pump.k) * 0.98, 100)
        ax2.plot(Q * 6e4, [pump.H0 - pump.k * q * q for q in Q], color="#999", ls="--", label="泵曲线(额定转速)")
        ax2.plot(Q * 6e4, [pump.gain(q) for q in Q], color="#1f77b4", label=f"泵曲线({pump.speed*100:.0f}% 转速)")
        pr = res["links"]["离心泵"]
        ax2.plot([pr["Q_Lmin"]], [pr["head"]], "o", ms=11, color="#d62728", label="当前工作点")
        # 转速扫描工作点
        sp, qs, hs = np.linspace(0.5, 1.1, 7), [], []
        old = pump.speed
        for s in sp:
            pump.speed = s; r2 = net.solve(); qs.append(r2["links"]["离心泵"]["Q_Lmin"]); hs.append(r2["links"]["离心泵"]["head"])
        pump.speed = old
        ax2.plot(qs, hs, ":", color="#d62728", label="转速 50–110% 的工作点")
        ax2.set_xlabel("流量 L/min"); ax2.set_ylabel("扬程 m"); ax2.set_title("泵曲线与工作点"); ax2.grid(alpha=0.3); ax2.legend(fontsize=8)
        f.tight_layout(); self.cv.draw()


def r_tL(L, D, e, m):
    return 2 * L / TR.wave_speed(1000, 2.2e9, D, e, TR.PIPE_E[m])


class HammerPage(_Page):
    MAT = ["不锈钢", "钢", "铜", "PVC", "PE", "增强软管", "橡胶软管"]

    def __init__(self):
        super().__init__()
        self.mat = QtWidgets.QComboBox(); [self.mat.addItem(m) for m in self.MAT]
        self.add("管材", self.mat)
        self.L = self.add("管长 m（阀到上游稳压点）", spin(0.2, 2000, 3, 1, 1))
        self.D = self.add("内径 mm", spin(3, 500, 26, 1, 1))
        self.e = self.add("壁厚 mm", spin(0.2, 30, 2.0, 0.5, 2))
        self.H = self.add("阀前稳压 bar（表压）", spin(0.1, 40, 3.0, 0.5, 1))
        self.K = self.add("阀全开阻力系数 K（越小流速越高）", spin(0.5, 5000, 20, 5, 1))
        self.tc = self.add("关阀时间 ms", spin(1, 20000, 25, 5, 0))
        self.rate = self.add("管/软管额定压力 bar", spin(1, 400, 10, 1, 1))
        self.update_()

    def update_(self, *a):
        m = self.mat.currentText()
        D, e, L = self.D.value() * 1e-3, self.e.value() * 1e-3, self.L.value()
        Hres = self.H.value() * 1e5 / (1000 * G)
        try:
            r = TR.water_hammer(L=L, D=D, e=e, E=TR.PIPE_E[m], Hres=Hres, K_valve=self.K.value(),
                                t_close=self.tc.value() * 1e-3, N=24, T=25.0, tmax=max(0.05, 12 * r_tL(L, D, e, m), 6 * self.tc.value() * 1e-3))
        except Exception as ex:
            self.out.setPlainText(f"算不出：{ex}"); return
        rho = r["rho"]
        pk, pn, p0 = r["Hmax"] * rho * G / 1e5, r["Hmin"] * rho * G / 1e5, r["H0_valve"] * rho * G / 1e5
        rate = self.rate.value()
        v = [("OK", f"关阀前流速 {r['v0']:.2f} m/s，阀前压力 {p0:.2f} bar；波速 a={r['a']:.0f} m/s（{m}管，Korteweg）")]
        slow = self.tc.value() * 1e-3 > r["tL"]
        frac = (r["Hmax"] - r["H0_valve"]) / r["dH_jou"]
        v.append(("OK" if frac < 0.3 else "WARN", f"2L/a = {r['tL']*1e3:.1f} ms；关阀 {self.tc.value():g} ms，"
                  f"冲击只有瞬关值 ρ·a·v 的 {100*min(frac,1.5):.0f}%" + ("" if slow else "（关得比 2L/a 还快，等于瞬关）")))
        v.append(("OK" if pk <= rate else "FAIL", f"关阀瞬间最高压力约 {pk:.1f} bar（理论上限 阀前+ρav = {p0 + r['dH_jou']*rho*G/1e5:.1f} bar），额定 {rate:g} bar"
                  + ("" if pk <= rate else "：超压！用更长的关阀时间、软管/蓄能器或换更高额定的管")))
        if r["cavitation"]:
            v.append(("FAIL", f"负压波降到 {pn:.2f} bar，低于水的蒸气压 → 水柱拉断、再合拢时会有更大的撞击（本模型不算汽穴，真实峰值更高）"))
        elif pn < 0:
            v.append(("WARN", f"回波最低 {pn:.2f} bar（负压），接头可能吸气"))
        self.out.setHtml(f"<b>水锤：最高 {pk:.1f} bar，最低（线性理论值）{pn:.1f} bar</b><br><br>" + html(v))
        f = self.fig; f.clear(); ax = f.add_subplot(111)
        t = r["t"] * 1e3
        ax.plot(t, r["H_valve"] * rho * G / 1e5, color="#d62728", label="阀前压力")
        ax.plot(t, r["H_up"] * rho * G / 1e5, color="#1f77b4", alpha=0.6, label="上游端")
        ax.axhline(rate, color="#ff7f0e", ls="--", label=f"额定 {rate:g} bar")
        ax.axhline(p0, color="k", lw=0.6, ls=":")
        ax.axvline(self.tc.value(), color="g", ls=":", label="关阀完成")
        ax.set_ylim(bottom=max(pn, -3.0) - 0.5); ax.set_xlabel("时间 ms"); ax.set_ylabel("压力 bar（表压）"); ax.set_title("关阀后阀前压力（特征线法）"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
        f.tight_layout(); self.cv.draw()


class DrainJetPage(_Page):
    def __init__(self):
        super().__init__()
        self.fluid = QtWidgets.QComboBox()
        for k in ("water", "pag10", "oil_quench", "gasoline", "glycerin", "mercury"):
            self.fluid.addItem(FLUIDS[k]["name"], k)
        self.add("液体", self.fluid)
        self.A = self.add("水箱截面积 m²", spin(0.01, 100, 0.5, 0.1))
        self.H0 = self.add("初始液位 m（从底部孔算起）", spin(0.05, 30, 1.0, 0.1))
        self.d = self.add("孔径 mm", spin(1, 100, 12, 1, 1))
        self.zh = self.add("孔离地高度 m", spin(0, 20, 1.0, 0.1))
        self.ang = self.add("孔朝向 仰角 °", spin(-60, 80, 0, 5, 0))
        self.update_()

    def update_(self, *a):
        key = self.fluid.currentData()
        net = Network(key, 25.0)
        z = self.zh.value(); H0 = self.H0.value(); A = self.A.value()
        net.tank("T", z=z, level=H0, area=A); net.reservoir("O", z=z, level=0.0)
        net.add(Nozzle("hole", "T", "O", n=1, d=self.d.value() * 1e-3, Cd=0.62))
        a_hole = math.pi * (self.d.value() * 1e-3) ** 2 / 4
        t_end = JT.drain_time(A, a_hole, H0, 0.0) * 1.0
        hist = net.run_quasi_steady(t_end, dt=max(t_end / 300, 0.05), stop_empty=True)
        rho = net.rho
        q0 = hist["Q"]["hole"][0]; v0 = q0 / (0.62 * a_hole)
        F = JT.jet_force(rho, q0, v0)
        tj = JT.trajectory(v0, self.ang.value(), z)
        r_ = tj["range"]
        txt = (f"<b>{FLUIDS[key]['name']} ρ={rho:.0f} kg/m³</b><br>初始射流 {v0:.2f} m/s，流量 {q0*6e4:.1f} L/min<br>"
               f"放空时间 {t_end/60:.1f} min；落地射程 {r_:.2f} m；垂直打在板上冲击力 {F:.1f} N<br><br>")
        vv = [("OK", "射流速度 v≈√(2gh)，与液体密度无关（密度只影响流量带来的质量、冲击力和孔前压力）；"
                     f"同样 {H0:g} m 液位的底部压力：{rho*G*H0/1e3:.1f} kPa")]
        if abs(self.ang.value()) > 0:
            vv.append(("OK", f"斜向 {self.ang.value():g}° 射出，最高 {tj['apex']:.2f} m"))
        vv.append(("WARN", "射流按无空气阻力的抛物线算；细水柱/雾化射流会被空气明显减速，实际射程更短"))
        self.out.setHtml(txt + html(vv))
        f = self.fig; f.clear()
        ax = f.add_subplot(121)
        ax.plot(hist["t"] / 60, hist["level"]["T"], color="#1f77b4", lw=2, label="网络准稳态")
        tt = np.linspace(0, t_end, 100)
        ax.plot(tt / 60, (np.sqrt(H0) - tt / (A / (0.62 * a_hole) * math.sqrt(2 / G))).clip(0) ** 2, "k--", lw=1, label="Torricelli 解析")
        ax.set_xlabel("分钟"); ax.set_ylabel("液位 m"); ax.set_title("水箱放空"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
        ax2 = f.add_subplot(122)
        for frac, c in ((1.0, "#1f77b4"), (0.5, "#2ca02c"), (0.15, "#ff7f0e")):
            h = H0 * frac; v = 0.97 * math.sqrt(2 * G * h)
            j = JT.trajectory(v, self.ang.value(), z)
            ax2.plot(j["x"], j["z"], color=c, label=f"液位 {h:.2f} m：v={v:.1f} m/s，射程 {j['range']:.2f} m")
        ax2.axhline(0, color="k"); ax2.set_xlabel("m"); ax2.set_ylabel("高度 m"); ax2.set_title("射流轨迹（液位下降时变近）")
        ax2.set_ylim(bottom=-0.1); ax2.grid(alpha=0.3); ax2.legend(fontsize=7)
        f.tight_layout(); self.cv.draw()


class FlowPage(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        lay = QtWidgets.QVBoxLayout(self)
        t = QtWidgets.QTabWidget()
        t.addTab(NetPage(), "管网：泵 / 阀 / 并联喷淋")
        t.addTab(HammerPage(), "水锤（关阀冲击）")
        t.addTab(DrainJetPage(), "水箱放空 + 射流")
        lay.addWidget(t)
        self.tabs = t
