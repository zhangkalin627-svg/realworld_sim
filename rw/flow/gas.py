"""气动管路：等温可压缩流（长管近似），给 PLC 气缸的气管选型用。与 `fluids.compressible.isothermal_gas` 对照。"""
import math
from ..world.pipes import friction_factor

R_AIR = 287.05


def isothermal_mass_flow(p1, p2, L, D, T=293.15, rough=1.5e-6, K=0.0):
    """p1、p2 绝压 Pa → 质量流量 kg/s。m² = (p1²−p2²)A² / [R T (fL/D + K + 2 ln(p1/p2))]；f 迭代；超过音速时被限制(壅塞)"""
    A = math.pi * D ** 2 / 4
    mu = 1.8e-5
    f = 0.02
    m = 0.0
    for _ in range(40):
        den = R_AIR * T * (f * L / D + K + 2 * math.log(p1 / p2))
        m = A * math.sqrt((p1 ** 2 - p2 ** 2) / den)
        Re = 4 * m / (math.pi * D * mu)
        f_new = friction_factor(Re, rough / D)
        if abs(f_new - f) < 1e-8: break
        f = f_new
    # 壅塞上限：出口音速 (等温 √(RT))
    rho2 = p2 / (R_AIR * T)
    m_ch = rho2 * math.sqrt(R_AIR * T) * A
    return min(m, m_ch), f


def free_air_Lmin(m):
    """质量流量 → 标准状态(1.013 bar, 20°C)体积流量 L/min (ANR)"""
    return m / (101325 / (R_AIR * 293.15)) * 1000 * 60
