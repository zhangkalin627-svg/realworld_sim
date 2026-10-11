"""材料库 Materials library.

所有数值都是工程手册里的典型值（近似），不是某一炉钢的实测值。
Values are typical handbook values for C45 / 45# steel and copper; treat as engineering estimates.
"""
import numpy as np

# ---------------------------------------------------------------- 45 钢 (C45, ~0.45%C)
STEEL_45 = dict(
    name="45钢 (C45)",
    density=7850.0,          # kg/m3 (温度影响忽略)
    Ac1=725.0, Ac3=780.0,    # °C 平衡相变点；感应快速加热实际偏高
    Ac3_fast=830.0,          # °C 快速加热(几秒)时取的完全奥氏体化温度
    T_curie=770.0,           # °C 居里点，超过后不再导磁
    T_melt=1495.0,           # °C 开始熔化 (固相线)
    Ms=340.0,                # °C 马氏体开始转变
)

_Tk = np.array([20, 200, 400, 600, 700, 750, 800, 900, 1000, 1200, 1500])
_k = np.array([51, 48, 42, 35, 31, 28, 26, 27, 28, 30, 33])          # W/mK 导热系数
_cp = np.array([470, 520, 590, 700, 800, 1100, 900, 650, 650, 660, 680])  # J/kgK 比热(含相变潜热峰)
_rho_e = np.array([0.20, 0.38, 0.62, 0.92, 1.05, 1.10, 1.13, 1.17, 1.20, 1.25, 1.30]) * 1e-6  # Ω·m 电阻率


def steel_k(T):
    return np.interp(T, _Tk, _k)


def steel_cp(T):
    return np.interp(T, _Tk, _cp)


def steel_sigma(T):
    """电导率 S/m"""
    return 1.0 / np.interp(T, _Tk, _rho_e)


def steel_mur(T, mur_cold=20.0):
    """有效相对磁导率。感应加热强磁场下冷态有效值约 10~30（取 20），
    接近居里点平滑降到 1。这是线性化近似，真实钢是非线性 B-H 曲线。"""
    Tc = STEEL_45["T_curie"]
    x = np.clip((Tc - np.asarray(T, float)) / 120.0, 0.0, 1.0)  # 650°C 以下全磁性，770 以上为 1
    return 1.0 + (mur_cold - 1.0) * x**1.5


COPPER_RHO = 1.72e-8   # Ω·m 铜线圈电阻率(常温)

# ---------------------------------------------------------------- 淬火介质 Quenchants
# 换热系数 h(表面温度) W/m2K —— 典型曲线形状：蒸汽膜阶段低 → 沸腾阶段峰值 → 对流阶段
# 喷淋 (spray) 能打破蒸汽膜，所以比浸入高很多。数值来自文献常见范围，需按现场流量校准。
QUENCHANTS = {
    "water_spray": dict(
        name="清水喷淋", T_bath=25.0,
        Ts=[20, 100, 200, 300, 400, 500, 600, 700, 800, 1000, 1200],
        h=[3000, 9000, 25000, 30000, 28000, 22000, 16000, 12000, 9000, 6000, 5000]),
    "water_dip": dict(
        name="清水浸入", T_bath=25.0,
        Ts=[20, 100, 200, 300, 400, 500, 600, 700, 800, 1000, 1200],
        h=[1500, 5000, 12000, 15000, 13000, 9000, 5000, 1500, 800, 600, 500]),
    "polymer_spray": dict(
        name="PAG水溶性淬火液 10% 喷淋", T_bath=30.0,
        Ts=[20, 100, 200, 300, 400, 500, 600, 700, 800, 1000, 1200],
        h=[2500, 5000, 9000, 14000, 16000, 15000, 12000, 8000, 5000, 3000, 2500]),
    "oil_dip": dict(
        name="淬火油浸入", T_bath=60.0,
        Ts=[20, 100, 200, 300, 400, 500, 600, 700, 800, 1000, 1200],
        h=[300, 400, 500, 900, 2000, 3000, 3200, 2000, 700, 400, 350]),
    "air": dict(
        name="空冷", T_bath=25.0,
        Ts=[20, 500, 1000, 1500],
        h=[10, 40, 120, 250]),  # 含辐射等效
}


def quench_h(medium, Ts):
    q = QUENCHANTS[medium]
    return np.interp(Ts, q["Ts"], q["h"])


# ---------------------------------------------------------------- 淬火后硬度估计
def hardness_estimate(Tpeak, t85, Tmin_after=None):
    """根据峰值温度和 800→500°C 冷却时间 t8/5 (秒) 估算 45 钢硬度 HRC。
    规则来自 C45 的 CCT 图（近似）：
      t8/5 ≤ 1 s  → 全马氏体 ≈ 57~60 HRC
      1~4 s       → 马氏体+贝氏体/屈氏体，硬度下降
      ≥ 15 s      → 铁素体+珠光体 ≈ 18~22 HRC（原始调质态取 22）
    未达 Ac1 的区域保持原始硬度；Ac1~Ac3 之间部分淬硬。
    """
    s = STEEL_45
    Tpeak = np.asarray(Tpeak, float)
    t85 = np.asarray(t85, float)
    base = 22.0
    full = 58.0
    # 冷却速度对应的可得硬度
    lt = np.log10(np.clip(t85, 0.05, 100))
    f_cool = np.interp(lt, [np.log10(1.0), np.log10(4.0), np.log10(15.0)], [1.0, 0.35, 0.0])
    # 奥氏体化程度
    f_aus = np.clip((Tpeak - s["Ac1"]) / (s["Ac3_fast"] - s["Ac1"]), 0, 1)
    f_aus = np.where(np.isfinite(t85) | (Tpeak < 800), f_aus, f_aus)
    H = base + (full - base) * f_aus * f_cool
    return H
