"""管路水力学：Darcy-Weisbach + Colebrook 摩擦系数、局部阻力、喷嘴、泵曲线求工作点。
单位：m, m³/s, Pa, kg/m³。"""
import math
import numpy as np
from scipy.optimize import brentq
from .fluids import get_fluid, G


def friction_factor(Re, rel_rough):
    if Re < 1e-9:
        return 0.0
    if Re < 2300:
        return 64.0 / Re
    # Colebrook（迭代）；过渡区 2300-4000 线性插值
    def colebrook(Re):
        f = 0.02
        for _ in range(50):
            f = (-2 * math.log10(rel_rough / 3.7 + 2.51 / (Re * math.sqrt(f)))) ** -2
        return f
    if Re < 4000:
        a = 64 / 2300; b = colebrook(4000)
        return a + (b - a) * (Re - 2300) / 1700
    return colebrook(Re)


class Pipe:
    def __init__(self, name, L, D, rough=0.045e-3, K=0.0, dz=0.0):
        self.name, self.L, self.D, self.rough, self.K, self.dz = name, L, D, rough, K, dz

    def dp(self, Q, rho, mu):
        A = math.pi * self.D**2 / 4
        v = Q / A
        Re = rho * abs(v) * self.D / mu
        f = friction_factor(Re, self.rough / self.D)
        dp_f = f * self.L / self.D * rho * v**2 / 2
        dp_k = self.K * rho * v**2 / 2
        return dp_f + dp_k + rho * G * self.dz, dict(v=v, Re=Re, f=f, dp_f=dp_f, dp_k=dp_k)


class Nozzles:
    """n 个直径 d 的小孔喷出（流量系数 Cd），出口为大气。"""
    def __init__(self, name, n, d, Cd=0.62):
        self.name, self.n, self.d, self.Cd = name, n, d, Cd

    def dp(self, Q, rho, mu):
        A = self.n * math.pi * self.d**2 / 4
        v = Q / (self.Cd * A)
        return rho * v**2 / 2, dict(v_jet=v * self.Cd / self.Cd, q_each=Q / self.n)


class Pump:
    """二次曲线 H = H0 − k·Q²，由样本两点（关死扬程、额定点）确定。"""
    def __init__(self, name, H0, Q_rated, H_rated, eta=0.45, P_motor=0.55e3):
        self.name, self.H0 = name, H0
        self.k = (H0 - H_rated) / Q_rated**2
        self.Qr, self.Hr, self.eta, self.Pm = Q_rated, H_rated, eta, P_motor

    def H(self, Q):
        return max(self.H0 - self.k * Q**2, 0.0)


class Circuit:
    def __init__(self, fluid="water", T=25.0, elements=None, pump=None):
        self.f = get_fluid(fluid)
        self.T = T
        self.rho = float(self.f["rho"](T)); self.mu = float(self.f["mu"](T))
        self.el = elements or []
        self.pump = pump

    def system_dp(self, Q):
        return sum(e.dp(Q, self.rho, self.mu)[0] for e in self.el)

    def operating_point(self):
        g = lambda Q: self.pump.H(Q) * self.rho * G - self.system_dp(Q)
        Qmax = math.sqrt(self.pump.H0 / self.pump.k) * 0.9999
        Q = brentq(g, 1e-9, Qmax)
        det = [(e.name, *e.dp(Q, self.rho, self.mu)) for e in self.el]
        P_hyd = self.rho * G * self.pump.H(Q) * Q
        return dict(Q=Q, Q_m3h=Q * 3600, Q_Lmin=Q * 60000, H=self.pump.H(Q), P_hyd=P_hyd,
                    P_shaft=P_hyd / self.pump.eta, details=det)


# ---------------------------------------------------------- 示例：感应淬火喷淋回路
def example_spray(fluid="pag10", T=30.0, n_holes=24, d_hole=1.2e-3, filter_K=8.0):
    """CDLF2-40 泵(2 m³/h@30 m, 关死约 38 m) → DN25 管 3 m → Y 型过滤器 → 流量开关 → DN15 电磁阀 →
    软管 Ø10×1(内径8) 1.5 m → 喷淋环 24×Ø1.2 孔。"""
    pump = Pump("CDLF2-40", H0=38.0, Q_rated=2.0 / 3600, H_rated=30.0, eta=0.40, P_motor=0.55e3)
    el = [Pipe("DN25 不锈钢管 3 m + 4 个弯头", 3.0, 0.026, K=4 * 0.3, dz=0.6),
          Pipe("Y 型过滤器 DN25（干净时）", 0.01, 0.026, K=filter_K),
          Pipe("流量开关 HFS-25", 0.01, 0.026, K=2.0),
          Pipe("电磁阀 2KS DN15", 0.01, 0.015, K=6.0),
          Pipe("软管 内径 8 mm × 1.5 m", 1.5, 0.008, rough=0.0015e-3, K=1.5, dz=0.4),
          Pipe("喷淋环 Ø10×1 管（内径 8）", 0.16, 0.008, rough=0.0015e-3, K=1.0),
          Nozzles(f"喷淋孔 {n_holes}×Ø{d_hole*1e3:.1f}", n_holes, d_hole)]
    c = Circuit(fluid, T, el, pump)
    op = c.operating_point()
    return c, op
