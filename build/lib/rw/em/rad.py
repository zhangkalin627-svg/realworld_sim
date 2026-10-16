"""外部热输入 (路线 A 续三): 热工件/线圈对导磁体、辅助体的辐射传热

灰体两面封闭腔网络: Q = σ (Ts⁴ − Tb⁴) / [ (1−ε1)/(ε1 A1) + 1/(A1 F12) + (1−ε2)/(ε2 A2) ]
并给出同轴圆盘的视角系数 (解析式) 与独立的数值积分; conc_heatup 的 q_ext 可传 q(T_body) 可调用对象, 如
    q = lambda T: rad_exchange_W(900.0, T, A1, 0.8, 0.9, F12=0.4)
⚠ 估算/假设项: 发射率、视角系数 F12、工件表面温度 (取均匀) 须按实物核对; 只算辐射, 不含对流/导热。
"""
import numpy as np

SIGMA = 5.670374419e-8      # W/m2K4
K0 = 273.15


def rad_exchange_W(Ts_C, Tb_C, A1, eps1=0.8, eps2=0.9, F12=1.0, A2=None):
    """面 1 (温度 Ts_C, 面积 A1, 发射率 eps1) 到面 2 (Tb_C, A2, eps2) 的净辐射热流 W (正=1→2)。
    F12: 1→2 视角系数。A2=None 时按面 2 远大于面 1 处理 (1−ε2)/(ε2 A2) 项忽略。"""
    if not (0 < eps1 <= 1 and 0 < eps2 <= 1):
        raise ValueError("发射率须在 (0,1]")
    if not (0 < F12 <= 1) or A1 <= 0:
        raise ValueError("F12 须在 (0,1], A1>0")
    R = (1 - eps1) / (eps1 * A1) + 1.0 / (A1 * F12)
    if A2 is not None:
        R += (1 - eps2) / (eps2 * A2)
    return float(SIGMA * ((Ts_C + K0) ** 4 - (Tb_C + K0) ** 4) / R)


def view_factor_coaxial_discs(r1, r2, h):
    """相距 h 的同轴平行圆盘 (半径 r1, r2) 的视角系数 F12 (Hottel 解析式)。"""
    R1, R2 = r1 / h, r2 / h
    S = 1.0 + (1.0 + R2 ** 2) / R1 ** 2
    return float(0.5 * (S - np.sqrt(S * S - 4.0 * (R2 / R1) ** 2)))


def view_factor_numeric(r1, r2, h, n=48):
    """同一视角系数的独立数值计算: F12 = (1/A1)∬ cosθ1 cosθ2/(π s²) dA1 dA2, 三重 Gauss–Legendre 积分 (ρ1, ρ2, 夹角 φ)。"""
    x, w = np.polynomial.legendre.leggauss(n)
    r1_ = 0.5 * r1 * (x + 1); w1 = 0.5 * r1 * w
    r2_ = 0.5 * r2 * (x + 1); w2 = 0.5 * r2 * w
    ph = np.pi * (x + 1); wp = np.pi * w            # φ ∈ [0, 2π]
    P1, P2, PH = np.meshgrid(r1_, r2_, ph, indexing="ij")
    s2 = P1 ** 2 + P2 ** 2 - 2 * P1 * P2 * np.cos(PH) + h * h
    integrand = h * h / (np.pi * s2 * s2) * P1 * P2
    I = np.einsum("i,j,k,ijk->", w1, w2, wp, integrand) * 2.0 * np.pi    # 乘面 1 方位角 2π
    return float(I / (np.pi * r1 ** 2))


def radiation_only_time(T0_C, T1_C, Ts_C, C_J_K, eps_A):
    """解析解: C dT/dt = ε·A·σ (Ts⁴ − T⁴) 的升温时间 (开尔文, 绝热+恒温辐射源, 无其它项)。
    t = C/(4 k Ts³) [ ln((Ts+T)/(Ts−T)) + 2 atan(T/Ts) ] |_{T0}^{T1},  k = ε A σ。 eps_A: ε_eff·A_m2 (含视角/灰体网络的等效值)。"""
    Ts = Ts_C + K0
    F = lambda T: np.log((Ts + T) / (Ts - T)) + 2 * np.arctan(T / Ts)
    k = eps_A * SIGMA
    return float(C_J_K / (4 * k * Ts ** 3) * (F(T1_C + K0) - F(T0_C + K0)))
