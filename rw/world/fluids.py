"""流体物性库（典型工程值）。温度 °C，压力 Pa(绝对)，密度 kg/m³，粘度 Pa·s。"""
import math
import numpy as np

G = 9.80665
P_ATM = 101325.0


def water_rho(T):
    """纯水密度（0–370°C 饱和液近似, Kell/IAPWS 拟合）"""
    T = np.asarray(T, float)
    Tk = [0, 4, 20, 40, 60, 80, 100, 120, 150, 200, 250, 300, 350, 370]
    rk = [999.8, 1000.0, 998.2, 992.2, 983.2, 971.8, 958.4, 943.1, 917.0, 864.7, 799.0, 712.1, 574.7, 450.0]
    return np.interp(T, Tk, rk)


def water_mu(T):
    """水的动力粘度 (Vogel 公式), Pa·s"""
    T = np.asarray(T, float)
    return 2.414e-5 * 10 ** (247.8 / (T + 273.15 - 140))


def water_psat(T):
    """水的饱和蒸气压 Pa（Antoine，1–374°C 分段）"""
    T = float(T)
    if T <= 100:
        A, B, C = 8.07131, 1730.63, 233.426
    else:
        A, B, C = 8.14019, 1810.94, 244.485
    return 10 ** (A - B / (C + T)) * 133.322


def water_tsat(p):
    """给定绝对压力下水的沸点 °C（反解 Antoine）"""
    lo, hi = 0.01, 373.9
    for _ in range(60):
        m = (lo + hi) / 2
        if water_psat(m) < p:
            lo = m
        else:
            hi = m
    return (lo + hi) / 2


def cc_tsat(Tb, dH_kJmol):
    """克劳修斯-克拉佩龙：常压沸点 Tb(°C)、汽化热 → 任意压力下沸点"""
    R = 8.314

    def f(p):
        inv = 1 / (Tb + 273.15) - R / (dH_kJmol * 1e3) * math.log(max(p, 1.0) / P_ATM)
        return 1 / inv - 273.15
    return f


FLUIDS = {
    "water":    dict(name="清水", rho=water_rho, mu=water_mu, tsat=water_tsat, freeze=0.0),
    "seawater": dict(name="海水", rho=lambda T: water_rho(T) * 1.025, mu=lambda T: water_mu(T) * 1.08,
                     tsat=lambda p: water_tsat(p) + 0.6, freeze=-1.9),
    "pag10":    dict(name="PAG 淬火液 10%", rho=lambda T: water_rho(T) * 1.012, mu=lambda T: water_mu(T) * 2.2,
                     tsat=lambda p: water_tsat(p) + 0.5, freeze=-3.0),
    "gasoline": dict(name="汽油 (0.72)", rho=lambda T: 745 - 0.95 * (T - 0), mu=lambda T: 0.0006 * math.exp(-0.012 * (T - 20)),
                     tsat=cc_tsat(60, 33), freeze=-60),
    "diesel":   dict(name="柴油 (0.84)", rho=lambda T: 850 - 0.7 * (T - 15), mu=lambda T: 0.0035 * math.exp(-0.03 * (T - 20)),
                     tsat=cc_tsat(250, 50), freeze=-10),
    "oil_quench": dict(name="淬火油 (0.87)", rho=lambda T: 880 - 0.65 * (T - 15), mu=lambda T: 0.03 * math.exp(-0.035 * (T - 40)),
                       tsat=cc_tsat(330, 60), freeze=-10),
    "glycerin": dict(name="甘油 (1.26)", rho=lambda T: 1261 - 0.65 * (T - 20), mu=lambda T: 1.41 * math.exp(-0.072 * (T - 20)),
                     tsat=cc_tsat(290, 61), freeze=18),
    "mercury":  dict(name="水银 (13.5)", rho=lambda T: 13546 - 2.45 * (T - 20), mu=lambda T: 0.00155, tsat=cc_tsat(357, 59), freeze=-38.8),
}


def custom_fluid(rho, name=None, mu=0.001, tsat=None):
    return dict(name=name or f"自定义 ρ={rho:g}", rho=lambda T: rho, mu=lambda T: mu,
                tsat=(lambda p: tsat) if tsat else (lambda p: 1e9), freeze=-273)


def get_fluid(key_or_rho):
    if isinstance(key_or_rho, (int, float)):
        r = float(key_or_rho)
        if r < 50:          # 用户写的是比重（0.7）
            r *= 1000
        return custom_fluid(r)
    return FLUIDS[key_or_rho]
