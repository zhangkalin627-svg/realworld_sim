"""v0.12 连接件：螺栓组校核 + 轴承（深沟球）与 L10 寿命 + 赫兹接触应力。
螺栓： bearing-type 螺栓组的经典弹性算法（直接剪 + 偏心转动剪 + 轴向拉 + 弯矩撬力），
      螺栓数据用 rw/matlib.py 的 BOLT_AS / BOLT_GRADE（国标公称应力截面、保证强度）。
轴承： 深沟球轴承典型额定值（约 SKF 样本值），L10 = (C/P)^3 × 1e6 转（ISO 281，球轴承指数 3）。
赫兹： 点接触（球-面/球-球）与线接触（圆柱-面）最大接触压力，公式见各函数。"""
import numpy as np
from ..matlib import BOLT_AS, BOLT_GRADE

G = 9.81

# 深沟球轴承（典型样本值，约 SKF）：d mm, C 动额定 N, C0 静额定 N
BEARINGS = {
    "6200": dict(d=10, C=5100,  C0=2400),
    "6202": dict(d=15, C=7650,  C0=3700),
    "6204": dict(d=20, C=12800, C0=6600),
    "6205": dict(d=25, C=14000, C0=7900),
    "6206": dict(d=30, C=19500, C0=11300),
    "6208": dict(d=40, C=29500, C0=18000),
    "6305": dict(d=25, C=22500, C0=11600),
    "6306": dict(d=30, C=26700, C0=15000),
}


def bearing_life(code, P_N, rpm):
    """L10 寿命（小时）。P 当量动载荷（纯径向铰 → P=Fr），球轴承指数 3。返回 (L10h, C, C0, 静安全系数)。"""
    b = BEARINGS[code]
    P = max(abs(P_N), 1e-9)
    l10_rev = (b["C"] / P) ** 3 * 1e6
    l10h = l10_rev / (60 * max(rpm, 1e-9))
    return dict(L10h=l10h, C=b["C"], C0=b["C0"], S0=b["C0"] / P, P=P, rpm=rpm, code=code)


# ----------------------------------------------------------------- 螺栓组
def bolt_group_check(F, M, center, normal, coords, size, grade, e1=None):
    """经典弹性螺栓组分析（bearing-type，忽略撬力放大与被连接件变形；摩擦型另算）。
    F, M: 通过接头传递的总力(N)、对接头中心的总力矩(N·mm)（由静力平衡求得）；
    center: 接头中心 world mm；normal: 接合面外法向（受拉为正方向）；
    coords: 螺栓在接合面内的 (u,v) 坐标 mm（相对 center，u 沿 e1，v 沿 e2）；
    返回每颗螺栓的拉力/剪力与利用率。
    方法：直接剪 V=F_shear/n；面内偏心剪（M·n 转动分量）按 r_i/Σr² 分配，与直接剪矢量叠加；
    轴向 N 均分 + 面内弯矩（绕 e1/e2）按 v_i/Σv²、u_i/Σu² 弹性分配。"""
    n = np.array(normal, float); n /= np.linalg.norm(n)
    if e1 is None:                       # 任取面内正交基
        a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
        e1 = a - n * np.dot(a, n)
    e1 = np.array(e1, float); e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)                  # 右手系: e1 × e2 = n
    F = np.array(F, float)
    N_ax = float(np.dot(F, n))            # >0 = 拉脱
    F_sh = F - N_ax * n                   # 面内剪力
    n_b = len(coords)
    uv = np.array(coords, float)
    As = BOLT_AS[size]
    fy = BOLT_GRADE[grade]["fy"] / 1e6    # MPa = N/mm2
    M1 = float(np.dot(M, e1)); M2 = float(np.dot(M, e2)); Mn = float(np.dot(M, n))
    sum_v2 = float((uv[:, 1] ** 2).sum()); sum_u2 = float((uv[:, 0] ** 2).sum())
    sum_r2 = float((uv ** 2).sum())
    out = []
    for u, v in uv:
        T = N_ax / n_b
        if sum_v2 > 1e-9: T += M1 * v / sum_v2
        if sum_u2 > 1e-9: T -= M2 * u / sum_u2     # 推导: M = Σ t_i (v_i e1 - u_i e2)
        V_direct = F_sh / n_b
        if sum_r2 > 1e-9:                           # 面内扭矩 Mn: 剪力方向 ⊥ r_i, 大小 Mn·r_i/Σr²
            V_rot = Mn / sum_r2 * (-v * e1 + u * e2)   # r_i=(u,v); r×F 沿 n → F ⊥ r 且与 Mn 同向
        else:
            V_rot = np.zeros(3)
        V = V_direct + V_rot
        Vmag = float(np.linalg.norm(V))
        sig = max(T, 0.0) / As                     # 受压不通过螺栓传（被连接件承压），只校核拉
        tau = Vmag / As
        util = (sig / fy) ** 2 + (tau / (0.6 * fy)) ** 2
        out.append(dict(u=float(u), v=float(v), T=float(T), V=Vmag, sig=sig, tau=tau,
                        util=float(util), ok=bool(util <= 1.0)))
    return dict(bolts=out, size=size, grade=grade, As=As, fy=fy,
                N_ax=N_ax, F_shear=float(np.linalg.norm(F_sh)), M_inplane=(M1, M2), M_torsion=Mn,
                worst=float(max(b["util"] for b in out)), ok=bool(all(b["ok"] for b in out)))


# ----------------------------------------------------------------- 赫兹接触
def _estiff(E1, nu1, E2, nu2):
    """等效弹性模量 1/E* = (1-ν1²)/E1 + (1-ν2²)/E2 （Pa）。"""
    return 1.0 / ((1 - nu1 ** 2) / E1 + (1 - nu2 ** 2) / E2)


def hertz_point(F, R1, E1, nu1, E2, nu2, R2=None):
    """点接触（球-平面 R2=None；球-球 R2=另一半径，mm）。返回 接触半径 a mm、最大压力 p0 MPa、平均压力 MPa。
    a = (3 F R_eff / 4 E*)^(1/3)，p0 = (6 F E*² / π³ R_eff²)^(1/3)（F 单位 N，E Pa）。"""
    Es = _estiff(E1, nu1, E2, nu2)
    R_eff = R1 if R2 is None else R1 * R2 / (R1 + R2)
    R_m = R_eff / 1000.0
    a_m = (3 * F * R_m / (4 * Es)) ** (1 / 3)
    p0 = (6 * F * Es ** 2 / (np.pi ** 3 * R_m ** 2)) ** (1 / 3)
    return dict(a=a_m * 1000, p0=p0 / 1e6, pmean=p0 / 1e6 * 2 / 3, R_eff=R_eff)


def hertz_line(F, L, R1, E1, nu1, E2, nu2):
    """线接触（圆柱-平面，F N，接触长 L mm，圆柱半径 R1 mm）。半宽 b、最大压力 p0 MPa。
    b = sqrt(4 F' R / (π E*))，p0 = sqrt(F' E* / (π R))，F' = F/L 单位长度线载荷。"""
    Es = _estiff(E1, nu1, E2, nu2)
    Fl = F / (L / 1000.0)                    # N/m
    R_m = R1 / 1000.0
    b_m = np.sqrt(4 * Fl * R_m / (np.pi * Es))
    p0 = np.sqrt(Fl * Es / (np.pi * R_m))
    return dict(b=b_m * 1000, p0=p0 / 1e6, R_eff=R1)
