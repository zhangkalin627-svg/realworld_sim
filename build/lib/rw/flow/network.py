"""通用管网稳态求解器（MIT，自写；摩擦系数与局部阻力与 `fluids` 库(MIT) 对照检验）。

节点水头 H = z + p/(ρg) + v²/2g（总水头）。未知量 = 各"接点"的水头；
方程 = 每个接点流量守恒；用 scipy 的 Newton 类求根(hybr)。
连接件(link)：Pipe(管：Darcy-Weisbach + 局部阻力 K)、Valve(阀：K/开度)、Pump(泵：H=H0-kQ²，带止回)、
Nozzle(喷嘴/小孔组：出口到大气)。节点：Reservoir(定压/定液位)、Junction(接点，可有取水量)、Tank(水箱：用 run_quasi_steady 推进液位)。
单位：m, m³/s, Pa, kg/m³；温度 °C。
"""
import math
from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import root

from ..world.fluids import get_fluid, G, P_ATM, water_psat
from ..world.pipes import friction_factor


# ------------------------------------------------------------------ 元件
@dataclass
class Pipe:
    name: str; n1: str; n2: str
    L: float; D: float
    rough: float = 0.045e-3
    K: float = 0.0              # 局部阻力系数合计（弯头、突缩…）
    p_rated: float = None       # 管/软管额定工作压力(表压 Pa)，用于体检
    kind: str = "pipe"

    def dH(self, Q, rho, mu):
        """H1 - H2（米液柱），Q 带符号"""
        A = math.pi * self.D ** 2 / 4
        v = abs(Q) / A
        Re = rho * v * self.D / mu
        f = friction_factor(Re, self.rough / self.D) if Re > 1e-9 else 0.0
        h = (f * self.L / self.D + self.K) * v ** 2 / (2 * G)
        return math.copysign(h, Q)

    def velocity(self, Q):
        return abs(Q) / (math.pi * self.D ** 2 / 4)


@dataclass
class Valve:
    name: str; n1: str; n2: str
    D: float; K_open: float = 1.0
    opening: float = 1.0        # 0..1，阻力 ~ K_open/opening²（简化）
    kind: str = "valve"
    p_rated: float = None

    def dH(self, Q, rho, mu):
        if self.opening <= 1e-6:
            return math.copysign(1e6 * abs(Q) / 1e-3 + 1e4, Q) if Q != 0 else 0.0
        K = self.K_open / self.opening ** 2
        v = abs(Q) / (math.pi * self.D ** 2 / 4)
        return math.copysign(K * v ** 2 / (2 * G), Q)

    def velocity(self, Q):
        return abs(Q) / (math.pi * self.D ** 2 / 4)


@dataclass
class Pump:
    """离心泵 H = H0 − k·Q²（由关死扬程与额定点定出），n1=吸入口，n2=排出口；内置止回(不倒流)。"""
    name: str; n1: str; n2: str
    H0: float; Q_rated: float; H_rated: float
    eta: float = 0.45; NPSHr: float = None; speed: float = 1.0   # speed 相对额定转速（相似定律）
    kind: str = "pump"; p_rated: float = None

    @property
    def k(self):
        return (self.H0 - self.H_rated) / self.Q_rated ** 2

    def gain(self, Q):
        s = self.speed
        return s ** 2 * self.H0 - self.k * Q ** 2 if Q >= 0 else s ** 2 * self.H0 - 1e5 * Q    # 倒流: 被止回

    def dH(self, Q, rho, mu):
        return -self.gain(Q)

    def velocity(self, Q):
        return 0.0


@dataclass
class Nozzle:
    """n 个直径 d 的孔(Cd)，出口为大气(n2 应是定压 0 表压的 Reservoir)。喷射速度 v=Q/(Cd·n·A)。"""
    name: str; n1: str; n2: str
    n: int; d: float; Cd: float = 0.62
    kind: str = "nozzle"; p_rated: float = None

    def dH(self, Q, rho, mu):
        A = self.n * math.pi * self.d ** 2 / 4
        v = abs(Q) / (self.Cd * A)
        return math.copysign(v ** 2 / (2 * G), Q)

    def velocity(self, Q):
        return abs(Q) / (self.Cd * self.n * math.pi * self.d ** 2 / 4)


# ------------------------------------------------------------------ 节点
@dataclass
class Node:
    name: str
    kind: str = "junction"      # junction / reservoir / tank
    z: float = 0.0              # 高程 m（水箱=底部高程）
    demand: float = 0.0         # m³/s 从节点取走(+)/注入(−)
    p_gauge: float = None       # reservoir: 液面/出口处表压 Pa（0=敞口）
    level: float = 0.0          # reservoir/tank：液面高于 z 的高度 m
    area: float = None          # tank 横截面积 m²


class Network:
    def __init__(self, fluid="water", T=25.0):
        self.fl = get_fluid(fluid); self.T = T
        self.rho = float(self.fl["rho"](T)); self.mu = float(self.fl["mu"](T))
        self.nodes = {}; self.links = []

    def reservoir(self, name, z=0.0, level=0.0, p_gauge=0.0):
        self.nodes[name] = Node(name, "reservoir", z, 0.0, p_gauge, level); return self

    def tank(self, name, z, level, area, p_gauge=0.0):
        self.nodes[name] = Node(name, "tank", z, 0.0, p_gauge, level, area); return self

    def junction(self, name, z=0.0, demand=0.0):
        self.nodes[name] = Node(name, "junction", z, demand); return self

    def add(self, link):
        assert link.n1 in self.nodes and link.n2 in self.nodes, f"{link.name}: 节点未定义"
        self.links.append(link); return link

    def head_fixed(self, nd):
        return nd.z + nd.level + (nd.p_gauge or 0.0) / (self.rho * G)

    # --------------------------------------------------------- 求解
    def solve(self):
        names = [n for n, nd in self.nodes.items() if nd.kind == "junction"]
        idx = {n: i for i, n in enumerate(names)}
        fixed = {n: self.head_fixed(nd) for n, nd in self.nodes.items() if nd.kind != "junction"}
        rho, mu = self.rho, self.mu

        def Hof(n, x):
            return x[idx[n]] if n in idx else fixed[n]

        def link_Q(l, x):
            """已知两端水头差 → 流量(对 dH 反解)；用二分/牛顿"""
            dh = Hof(l.n1, x) - Hof(l.n2, x)
            return self._Q_of_dH(l, dh)

        def resid(x):
            r = np.zeros(len(names))
            for n in names:
                r[idx[n]] = -self.nodes[n].demand
            for l in self.links:
                Q = link_Q(l, x)
                if l.n1 in idx: r[idx[l.n1]] -= Q
                if l.n2 in idx: r[idx[l.n2]] += Q
            return r

        if names:
            hmax = max(list(fixed.values()) + [0]) + 200
            # 初值: 各固定水头的平均
            x0 = np.full(len(names), np.mean(list(fixed.values())) if fixed else 0.0)
            sol = root(resid, x0, method="hybr", tol=1e-12)
            if not sol.success or np.max(np.abs(sol.fun)) > 1e-7:
                sol = root(resid, x0, method="lm", tol=1e-13)
            x = sol.x
            self.converged = bool(np.max(np.abs(resid(x))) < 1e-7)
        else:
            x = np.zeros(0); self.converged = True
        H = {n: float(Hof(n, x)) for n in self.nodes}
        Q = {l.name: float(link_Q(l, x)) for l in self.links}
        return self._report(H, Q)

    def _Q_of_dH(self, l, dh):
        """对 dH(Q) 单调求逆。"""
        rho, mu = self.rho, self.mu
        f = lambda Q: l.dH(Q, rho, mu) - dh
        if abs(dh) < 1e-14 and l.kind != "pump":
            return 0.0
        lo, hi = -1e-3, 1e-3
        for _ in range(60):
            if f(lo) > 0 and f(hi) < 0: break
            if f(hi) < 0: hi *= 2
            if f(lo) > 0: lo *= 2
            if f(hi) >= 0 and f(lo) <= 0: break
        if f(lo) * f(hi) > 0:
            return 0.0
        a, b = lo, hi
        fa = f(a)
        for _ in range(80):
            m = 0.5 * (a + b); fm = f(m)
            if fm == 0 or (b - a) < 1e-14: break
            if (fm > 0) == (fa > 0): a, fa = m, fm
            else: b = m
        return 0.5 * (a + b)

    # --------------------------------------------------------- 结果与体检
    def _report(self, H, Q):
        rho, mu = self.rho, self.mu
        vmax_at = {n: 0.0 for n in self.nodes}
        links = {}
        for l in self.links:
            v = l.velocity(Q[l.name]) if l.kind != "pump" else 0.0
            for n in (l.n1, l.n2):
                if l.kind in ("pipe", "valve"):
                    vmax_at[n] = max(vmax_at[n], v)
            links[l.name] = dict(Q=Q[l.name], Q_Lmin=Q[l.name] * 6e4, v=v, kind=l.kind,
                                 dH=H[l.n1] - H[l.n2])
            if l.kind == "pipe":
                A = math.pi * l.D ** 2 / 4; Re = rho * v * l.D / mu
                f = friction_factor(Re, l.rough / l.D) if Re > 1e-9 else 0.0
                links[l.name].update(Re=Re, f=f, dp_loss=(f * l.L / l.D + l.K) * rho * v ** 2 / 2)
            if l.kind == "pump":
                Qp = Q[l.name]; Hp = l.gain(Qp)
                P_hyd = rho * G * Hp * max(Qp, 0)
                links[l.name].update(head=Hp, P_hyd=P_hyd, P_shaft=P_hyd / l.eta, eta=l.eta)
            if l.kind == "nozzle":
                links[l.name].update(v_jet=v, q_each_Lmin=Q[l.name] * 6e4 / l.n)
        nodes = {}
        for n, nd in self.nodes.items():
            dyn = rho * vmax_at[n] ** 2 / 2
            ptot = rho * G * (H[n] - nd.z)
            nodes[n] = dict(H=H[n], z=nd.z, p_total=ptot, p_static=ptot - dyn,
                            p_abs=P_ATM + ptot - dyn)
        return dict(nodes=nodes, links=links, rho=rho, mu=mu, T=self.T)

    def vapor_pressure(self, T=None):
        """介质在 T(°C，默认 self.T) 下的饱和蒸气压 Pa。清水用 Antoine；其余介质由 fluid["tsat"](p)
        （沸点随压力单调增）在 [1 Pa, 22 MPa] 上对数二分反解 tsat(p)=T。没有沸点数据（tsat 恒为大值）
        或 T 超出该压力范围对应的沸点区间时返回 None（下限以下返回 0：几乎不挥发）。"""
        T = self.T if T is None else float(T)
        if self.fl["name"] == "清水":
            return water_psat(T)
        tsat = self.fl.get("tsat")
        if tsat is None:
            return None
        lo, hi = 1.0, 22e6
        try:
            t_lo, t_hi = float(tsat(lo)), float(tsat(hi))
        except Exception:
            return None
        if not (math.isfinite(t_lo) and math.isfinite(t_hi)) or t_hi - t_lo < 1e-6 or t_lo > 1e5:
            return None                      # 无沸点数据（custom_fluid 未给 tsat → 恒 1e9）
        if T <= t_lo:
            return 0.0
        if T >= t_hi:
            return None                      # 超临界/超出范围：不判
        for _ in range(80):
            m = math.sqrt(lo * hi)
            if tsat(m) < T:
                lo = m
            else:
                hi = m
        return math.sqrt(lo * hi)

    def checks(self, res, v_limit=3.0, margin=2.0):
        """体检：汽蚀(泵吸入口、任何点低于饱和蒸气压)、流速、管/软管额定压力、泵 NPSH、泵偏离最佳点。
        返回 [(OK/WARN/FAIL, 文字)]"""
        out = []
        rho = self.rho
        pv = self.vapor_pressure()
        if pv is None:
            out.append(("WARN", f"介质「{self.fl['name']}」无沸点数据，未判汽蚀（NPSHa 按蒸气压 0 计，偏乐观）"))
            pv = 0.0
        for n, r in res["nodes"].items():
            if self.nodes[n].kind != "junction": continue
            if pv > 0 and r["p_abs"] < pv:
                out.append(("FAIL", f"节点 {n} 绝压 {r['p_abs']/1e3:.1f} kPa < 饱和蒸气压 {pv/1e3:.1f} kPa → 汽蚀/汽阻"))
            elif r["p_abs"] < P_ATM * 0.5:
                out.append(("WARN", f"节点 {n} 绝压 {r['p_abs']/1e3:.1f} kPa，负压较大（吸入侧），接头易吸气"))
        for l in self.links:
            r = res["links"][l.name]
            if l.kind == "pump":
                pin = res["nodes"][l.n1]
                vs = 0.0
                for k in self.links:
                    if k.kind in ("pipe", "valve") and l.n1 in (k.n1, k.n2):
                        vs = max(vs, res["links"][k.name]["v"])
                npsha = (P_ATM + pin["p_total"] - pv) / (rho * G)      # 泵吸入口总水头 − 蒸气压头
                r["NPSHa"] = npsha
                if l.NPSHr:
                    ok = npsha >= l.NPSHr + margin * 0.5
                    out.append(("OK" if ok else ("WARN" if npsha >= l.NPSHr else "FAIL"),
                                f"泵 {l.name} 的 NPSHa={npsha:.1f} m，NPSHr={l.NPSHr:g} m" + ("" if ok else "：余量不足，会汽蚀(噪声、流量掉、叶轮蚀)")))
                if r["Q"] > 1.25 * l.Q_rated or (0 < r["Q"] < 0.4 * l.Q_rated):
                    out.append(("WARN", f"泵 {l.name} 工作流量 {r['Q_Lmin']:.1f} L/min 偏离额定点 {l.Q_rated*6e4:.1f} L/min 较多，效率差/易过载或发热"))
            if l.kind in ("pipe", "valve") and r["v"] > v_limit:
                out.append(("WARN", f"{l.name} 流速 {r['v']:.1f} m/s > {v_limit:g} m/s，压降大、冲刷/噪声/水锤风险高"))
            if getattr(l, "p_rated", None):
                pm = max(res["nodes"][l.n1]["p_total"], res["nodes"][l.n2]["p_total"])
                if pm > l.p_rated:
                    out.append(("FAIL", f"{l.name} 最大压力约 {pm/1e5:.2f} bar > 额定 {l.p_rated/1e5:.2f} bar"))
        if not out:
            out.append(("OK", "无汽蚀、流速在限值内、压力在额定内"))
        return out

    # --------------------------------------------------------- 准稳态推进(水箱液位)
    def run_quasi_steady(self, t_end, dt=1.0, stop_empty=True):
        """每步解稳态，再用各水箱净流量更新液位。准稳态：忽略管内惯性(见 transient.py 做水锤)。
        返回 t, 各水箱液位, 各连接件流量 的时间序列。"""
        t = 0.0
        tanks = [n for n, nd in self.nodes.items() if nd.kind == "tank"]
        hist = dict(t=[], level={n: [] for n in tanks}, Q={l.name: [] for l in self.links})
        while t <= t_end + 1e-9:
            res = self.solve()
            hist["t"].append(t)
            for n in tanks: hist["level"][n].append(self.nodes[n].level)
            for l in self.links: hist["Q"][l.name].append(res["links"][l.name]["Q"])
            net = {n: 0.0 for n in tanks}
            for l in self.links:
                q = res["links"][l.name]["Q"]
                if l.n1 in net: net[l.n1] -= q
                if l.n2 in net: net[l.n2] += q
            done = False
            for n in tanks:
                nd = self.nodes[n]
                nd.level = max(nd.level + net[n] * dt / nd.area, 0.0)
                if nd.level <= 1e-4 and net[n] < 0: done = True
            if done and stop_empty:
                break
            t += dt
        for k in ("t",): hist[k] = np.array(hist[k])
        for n in tanks: hist["level"][n] = np.array(hist["level"][n])
        for l in self.links: hist["Q"][l.name] = np.array(hist["Q"][l.name])
        return hist


# ------------------------------------------------------------------ 示例
def example_two_station(fluid="pag10", T=30.0, dirty=1.0):
    """通用示例：水箱 → 泵 → 总管 → 分两路(工位 A、B，各有电磁阀+软管+喷淋环) → 回水箱(开口喷到大气 → 回流)。
    dirty = 吸入过滤器脏污系数(阻力 K 乘数)。"""
    net = Network(fluid, T)
    net.tank("水箱", z=0.0, level=0.5, area=0.6)
    net.reservoir("大气A", z=0.55, level=0.0); net.reservoir("大气B", z=0.55, level=0.0)
    net.junction("泵吸入", z=0.1); net.junction("泵出口", z=0.1); net.junction("分叉", z=0.9)
    net.junction("A阀后", z=1.0); net.junction("B阀后", z=1.0)
    net.junction("A环", z=1.2); net.junction("B环", z=1.2)
    net.add(Pipe("吸入管 DN32", "水箱", "泵吸入", L=0.8, D=0.035, K=0.5 + 4.0 * dirty))
    net.add(Pump("离心泵", "泵吸入", "泵出口", H0=38.0, Q_rated=2.0 / 3600 * 2, H_rated=26.0, eta=0.42, NPSHr=2.5))
    net.add(Pipe("总管 DN25 3 m", "泵出口", "分叉", L=3.0, D=0.026, K=1.2 + 3 * 0.3, p_rated=10e5))
    for s in "AB":
        net.add(Valve(f"电磁阀{s}", "分叉", f"{s}阀后", D=0.015, K_open=6.0))
        net.add(Pipe(f"软管{s} 内径10", f"{s}阀后", f"{s}环", L=1.5, D=0.010, rough=0.0015e-3, K=2.5, p_rated=6e5))
        net.add(Nozzle(f"喷淋孔{s}", f"{s}环", f"大气{s}", n=24, d=1.2e-3))
    return net
