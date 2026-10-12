"""水锤：特征线法 (Method of Characteristics, Wylie & Streeter) —— 单根管，上游=水池或泵(准稳态曲线)，下游=按面积线性关闭的阀(通大气)。
自写（教科书算法，MIT）。波速用 Korteweg 公式。恒定摩擦系数(稳态值)；不模拟汽穴溃灭(压力降到蒸气压只报警)。
压力都用相对管轴的表压水头 (m液柱)。"""
import math
import numpy as np
from ..world.fluids import get_fluid, G, P_ATM, water_psat
from ..world.pipes import friction_factor


def wave_speed(rho, K_bulk, D, e, E, nu=0.3, anchored="all"):
    """Korteweg: a = sqrt(K/ρ / (1 + K D c1 /(E e)))；c1: 全约束 1-ν²，一端自由 1-ν/2，伸缩接头 1。"""
    c1 = {"all": 1 - nu ** 2, "upstream": 1 - nu / 2, "free": 1.0}[anchored]
    return math.sqrt(K_bulk / rho / (1 + K_bulk * D * c1 / (E * e)))


PIPE_E = {"钢": 200e9, "不锈钢": 193e9, "铜": 110e9, "PVC": 3.0e9, "PE": 0.8e9, "橡胶软管": 0.05e9, "增强软管": 0.3e9}


def water_hammer(L, D, e, E, Hres=None, pump=None, K_valve=2.0, t_close=0.01, t_hold=None,
                 fluid="water", T=25.0, rough=0.045e-3, K_bulk=2.2e9, N=40, tmax=None, anchored="all", closed_from=0.0,
                 a_override=None, friction=True):
    """返回 dict: t, H_valve, H_up, Q_valve, a, v0, H0_valve, dH_jou, T_period, Hmax, Hmin ...
    上游: Hres(水池水头 m 表压) 或 pump=(H0, k)(H=H0−kQ²，带止回)。阀: 全开阻力 K_valve(以管道流速头计)，
    t_close 为面积线性关到 0 的时间，从 closed_from 秒开始。"""
    fl = get_fluid(fluid); rho = float(fl["rho"](T)); mu = float(fl["mu"](T))
    A = math.pi * D ** 2 / 4
    a = a_override or wave_speed(rho, K_bulk, D, e, E, anchored=anchored)
    dx = L / N; dt = dx / a
    B = a / (G * A)
    # --- 稳态 (用迭代解 f)
    Cvf = A * math.sqrt(2 * G / K_valve)                    # Q = τ Cvf sqrt(H_valve)
    f = 0.02
    for _ in range(30):
        R_tot = (f * L / D / (2 * G * A ** 2)) if friction else 0.0   # H_fric = R_tot Q²
        if Hres is not None:
            Q0 = math.sqrt(Hres / (R_tot + 1 / Cvf ** 2))
        else:
            H0p, kp = pump
            Q0 = math.sqrt(H0p / (R_tot + 1 / Cvf ** 2 + kp))
        v0 = Q0 / A; Re = rho * v0 * D / mu
        f_new = friction_factor(Re, rough / D)
        if abs(f_new - f) < 1e-6: break
        f = f_new
    R = (f * dx / D / (2 * G * A ** 2)) if friction else 0.0  # 每段 H_fric = R Q|Q|
    Hv0 = Q0 ** 2 / Cvf ** 2
    H = np.array([Hv0 + R * N * Q0 ** 2 * (1 - i / N) for i in range(N + 1)])   # 节点 0=上游 N=阀
    Q = np.full(N + 1, Q0)
    Hup0 = H[0]
    Cv2_full = Cvf ** 2
    if tmax is None:
        tmax = max(8 * L / a, 3 * (closed_from + t_close))
    nsteps = int(tmax / dt) + 1
    ts = np.arange(nsteps) * dt
    Hv = np.empty(nsteps); Hu = np.empty(nsteps); Qv = np.empty(nsteps); Hm = np.empty(nsteps)
    Hv[0], Hu[0], Qv[0], Hm[0] = H[-1], H[0], Q[-1], H[N // 2]
    for s in range(1, nsteps):
        t = ts[s]
        Hn = np.empty_like(H); Qn = np.empty_like(Q)
        # 内部点
        Hm1, Hp1 = H[:-2], H[2:]; Qm1, Qp1 = Q[:-2], Q[2:]
        CP = Hm1 + B * Qm1 - R * Qm1 * np.abs(Qm1)
        CM = Hp1 - B * Qp1 + R * Qp1 * np.abs(Qp1)
        Hn[1:-1] = 0.5 * (CP + CM)
        Qn[1:-1] = (CP - CM) / (2 * B)
        # 上游
        CMu = H[1] - B * Q[1] + R * Q[1] * abs(Q[1])
        if Hres is not None:
            Hn[0] = Hres; Qn[0] = (Hres - CMu) / B
        else:
            # H = H0 - k Q² ,  H = CMu + B Q  →  k Q² + B Q + (CMu - H0) = 0
            H0p, kp = pump
            disc = B * B - 4 * kp * (CMu - H0p)
            Qn[0] = (-B + math.sqrt(max(disc, 0))) / (2 * kp)
            if Qn[0] < 0:
                Qn[0] = 0.0
            Hn[0] = CMu + B * Qn[0]
        # 下游阀
        tau = 1.0 if t < closed_from else max(0.0, 1.0 - (t - closed_from) / max(t_close, 1e-9))
        CPv = H[N - 1] + B * Q[N - 1] - R * Q[N - 1] * abs(Q[N - 1])
        if tau <= 0:
            Qn[N] = 0.0; Hn[N] = CPv
        else:
            c = (tau ** 2) * Cv2_full
            Qn[N] = 0.5 * (-c * B + math.sqrt((c * B) ** 2 + 4 * c * max(CPv, 0.0)))
            Hn[N] = CPv - B * Qn[N]
        H, Q = Hn, Qn
        Hv[s], Hu[s], Qv[s], Hm[s] = H[-1], H[0], Q[-1], H[N // 2]
    Hvap = -(P_ATM - water_psat(T)) / (rho * G) if fl["name"] in ("清水", "海水") else None
    peak = Hv.max()
    out = dict(t=ts, H_valve=Hv, H_up=Hu, H_mid=Hm, Q_valve=Qv, a=a, v0=v0, Q0=Q0, f=f, rho=rho,
               H0_valve=Hv0, H0_up=Hup0, dH_jou=a * v0 / G, T_period=4 * L / a, tL=2 * L / a,
               Hmax=peak, Hmin=Hv.min(), Pmax=rho * G * peak, Hvap=Hvap, dt=dt,
               cavitation=(Hvap is not None and Hv.min() < Hvap))
    return out


def michaud(L, v0, t_close):
    """Michaud 缓闭公式 ΔH ≈ 2 L v0 /(g t_c)（t_c > 2L/a，阀门流速线性减小假设下的近似）。"""
    return 2 * L * v0 / (G * t_close)
