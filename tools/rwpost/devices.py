"""
devices.py —— 功率器件损耗估算模型
本工具由 AI 助手（企业微信助手）编写，经我方独立验收（11/12 项，第 12 项为我方额外要求，见 accept_devices.py）后采用；全部为程序运行结果，非实测
只依赖 numpy 和 Python 标准库
"""

import math
import numpy as np


def _check_finite(name, val, allow_negative=False, min_exclusive=None, max_inclusive=None):
    """统一参数合法性检查：有限数、非 NaN、物理范围"""
    if not isinstance(val, (int, float)):
        raise ValueError(f"{name} 必须为数值，当前为 {type(val).__name__}")
    if math.isnan(val) or math.isinf(val):
        raise ValueError(f"{name} 为无效数值（NaN 或无穷），当前为 {val}")
    if not allow_negative and val < 0:
        raise ValueError(f"{name} 不能为负，当前为 {val}")
    if min_exclusive is not None and val <= min_exclusive:
        raise ValueError(f"{name} 必须 > {min_exclusive}，当前为 {val}")
    if max_inclusive is not None and val > max_inclusive:
        raise ValueError(f"{name} 必须 <= {max_inclusive}，当前为 {val}")


def mosfet_loss(i_rms, i_on_peak, i_off_peak, vds, fsw, duty, p, tj=25.0):
    """
    MOSFET 损耗估算（硬开关 CCM）。

    公式出处：
    - 导通损耗：P_cond = I_rms² · Rds(Tj)，Rds(Tj) = Rds_on_25 · (1 + α · (Tj − 25))
      （半导体物理：电阻率随温度线性近似，见 Infineon "MOSFET Power Losses" Application Note AN-6004）
    - 开关损耗：P_sw = 0.5 · Vds · (I_on_peak · tr + I_off_peak · tf) · fsw
      （梯形近似，见 Erickson & Maksimović "Fundamentals of Power Electronics" §4.3）
    - 输出电容损耗：P_oss = 0.5 · Coss · Vds² · fsw
      （每周期对 Coss 充放电一次，能量 ½CV²）
    - 栅极驱动损耗：P_gate = Qg · Vg · fsw
      （栅极电荷每次开关从驱动吸取 Qg·Vg 能量）

    参数：
        i_rms: float — 漏极电流有效值 (A)
        i_on_peak: float — 开通时刻漏极电流峰值 (A)
        i_off_peak: float — 关断时刻漏极电流峰值 (A)
        vds: float — 漏源电压 (V)
        fsw: float — 开关频率 (Hz)，必须 > 0
        duty: float — 占空比，必须在 0~1 之间
        p: dict — 器件参数
        tj: float, default=25.0 — 结温 (°C)

    返回：
        dict: {"p_cond": ..., "p_sw": ..., "p_oss": ..., "p_gate": ..., "p_total": ...}
    """
    i_rms = float(i_rms)
    i_on_peak = float(i_on_peak)
    i_off_peak = float(i_off_peak)
    vds = float(vds)
    fsw = float(fsw)
    duty = float(duty)
    tj = float(tj)

    _check_finite("i_rms", i_rms)
    _check_finite("i_on_peak", i_on_peak)
    _check_finite("i_off_peak", i_off_peak)
    _check_finite("vds", vds)
    _check_finite("fsw", fsw, min_exclusive=0)
    _check_finite("duty", duty)
    if not (0 <= duty <= 1):
        raise ValueError(f"duty 必须在 0~1 之间，当前为 {duty}")
    _check_finite("tj", tj)

    if not isinstance(p, dict):
        raise ValueError("p 必须为字典类型")

    required_keys = ["rds_on_25", "alpha_rds", "qg", "vg", "coss", "tr", "tf"]
    for k in required_keys:
        if k not in p:
            raise ValueError(f"缺少参数 '{k}'")
        v = p[k]
        _check_finite(k, v)
        if v < 0:
            raise ValueError(f"{k} 不能为负，当前为 {v}")

    rds_on_25 = float(p["rds_on_25"])
    alpha_rds = float(p["alpha_rds"])
    qg = float(p["qg"])
    vg = float(p["vg"])
    coss = float(p["coss"])
    tr = float(p["tr"])
    tf = float(p["tf"])

    rds_tj = rds_on_25 * (1.0 + alpha_rds * (tj - 25.0))
    p_cond = i_rms ** 2 * rds_tj
    p_sw = 0.5 * vds * (i_on_peak * tr + i_off_peak * tf) * fsw
    p_oss = 0.5 * coss * vds ** 2 * fsw
    p_gate = qg * vg * fsw
    p_total = p_cond + p_sw + p_oss + p_gate

    return {
        "p_cond": float(p_cond), "p_sw": float(p_sw),
        "p_oss": float(p_oss), "p_gate": float(p_gate),
        "p_total": float(p_total),
    }


def diode_loss(i_avg, i_rms, vf0, rd, qrr, vr, fsw):
    """
    二极管损耗估算（硬开关 CCM）。

    公式出处：
    - 导通损耗：P_cond = Vf0 · I_avg + rd · I_rms²
      （分段线性二极管模型，见 Erickson & Maksimović §3.1）
    - 反向恢复损耗：P_rr = Qrr · Vr · fsw
      （每周期反向恢复能量 Qrr·Vr，见 Infineon "Diode Power Losses" AN）

    参数：
        i_avg: float — 正向平均电流 (A)
        i_rms: float — 正向电流有效值 (A)
        vf0: float — 阈值电压 (V)
        rd: float — 动态电阻 (Ω)
        qrr: float — 反向恢复电荷 (C)
        vr: float — 反向电压 (V)
        fsw: float — 开关频率 (Hz)，必须 > 0

    返回：
        dict: {"p_cond": ..., "p_rr": ..., "p_total": ...}
    """
    i_avg = float(i_avg)
    i_rms = float(i_rms)
    vf0 = float(vf0)
    rd = float(rd)
    qrr = float(qrr)
    vr = float(vr)
    fsw = float(fsw)

    _check_finite("i_avg", i_avg)
    _check_finite("i_rms", i_rms)
    _check_finite("vf0", vf0)
    _check_finite("rd", rd)
    _check_finite("qrr", qrr)
    _check_finite("vr", vr)
    _check_finite("fsw", fsw, min_exclusive=0)

    p_cond = vf0 * i_avg + rd * i_rms ** 2
    p_rr = qrr * vr * fsw
    p_total = p_cond + p_rr

    return {"p_cond": float(p_cond), "p_rr": float(p_rr), "p_total": float(p_total)}


def igbt_loss(i_avg, i_rms, vce0, rce, e_on_ref, e_off_ref, i_ref, vce_ref, vce, fsw, tj=25.0, alpha_vce=0.0):
    """
    IGBT 损耗估算（硬开关 CCM）。

    公式出处：
    - 导通损耗：P_cond = Vce0 · I_avg + rce · I_rms²
      （分段线性 IGBT 模型，见 Infineon "IGBT Power Losses Calculation" AN）
    - 开关损耗：P_sw = (E_on + E_off) · fsw · (Vce / Vce_ref) · (I_avg / I_ref)^K_i
      K_i 通常取 1.0（电流线性缩放），电压线性缩放
      （见 Semikron "Application Manual Power Semiconductors" §5.2.2）
    - 结温缩放：E_on/off(Tj) = E_on/off_ref · (1 + α_vce · (Tj − 25))

    参数：
        i_avg: float — 集电极平均电流 (A)
        i_rms: float — 集电极电流有效值 (A)
        vce0: float — 阈值电压 (V)
        rce: float — 动态电阻 (Ω)
        e_on_ref: float — 参考条件下的开通能量 (J)
        e_off_ref: float — 参考条件下的关断能量 (J)
        i_ref: float — 开关能量的参考电流 (A)，必须 > 0
        vce_ref: float — 开关能量的参考电压 (V)，必须 > 0
        vce: float — 实际集射极电压 (V)
        fsw: float — 开关频率 (Hz)，必须 > 0
        tj: float, default=25.0 — 结温 (°C)
        alpha_vce: float, default=0.0 — 开关能量的温度系数 (1/K)

    返回：
        dict: {"p_cond": ..., "p_sw": ..., "p_total": ...}
    """
    i_avg = float(i_avg)
    i_rms = float(i_rms)
    vce0 = float(vce0)
    rce = float(rce)
    e_on_ref = float(e_on_ref)
    e_off_ref = float(e_off_ref)
    i_ref = float(i_ref)
    vce_ref = float(vce_ref)
    vce = float(vce)
    fsw = float(fsw)
    tj = float(tj)
    alpha_vce = float(alpha_vce)

    _check_finite("i_avg", i_avg)
    _check_finite("i_rms", i_rms)
    _check_finite("vce0", vce0)
    _check_finite("rce", rce)
    _check_finite("e_on_ref", e_on_ref)
    _check_finite("e_off_ref", e_off_ref)
    _check_finite("i_ref", i_ref, min_exclusive=0)
    _check_finite("vce_ref", vce_ref, min_exclusive=0)
    _check_finite("vce", vce)
    _check_finite("fsw", fsw, min_exclusive=0)
    _check_finite("tj", tj)
    _check_finite("alpha_vce", alpha_vce, allow_negative=True)

    p_cond = vce0 * i_avg + rce * i_rms ** 2

    voltage_scale = vce / vce_ref
    current_scale = i_avg / i_ref
    temp_scale = 1.0 + alpha_vce * (tj - 25.0)

    e_on = e_on_ref * voltage_scale * current_scale * temp_scale
    e_off = e_off_ref * voltage_scale * current_scale * temp_scale

    p_sw = (e_on + e_off) * fsw
    p_total = p_cond + p_sw

    return {"p_cond": float(p_cond), "p_sw": float(p_sw), "p_total": float(p_total)}


def solve_tj(loss_fn, rth_ja, tamb, tol=1e-6, max_iter=100):
    """
    迭代求解结温 Tj = Tamb + Rth_JA · P(Tj)。

    方法：定点迭代（fixed-point iteration），适用于损耗依赖结温的情形。
    公式出处：Erickson & Maksimović "Fundamentals of Power Electronics" §15.2

    参数：
        loss_fn: callable — 损耗函数 loss_fn(tj) -> float
        rth_ja: float — 结到环境的热阻 (K/W)，必须 > 0
        tamb: float — 环境温度 (°C)
        tol: float, default=1e-6 — 收敛容差，必须 > 0
        max_iter: int, default=100 — 最大迭代次数，必须 >= 1

    返回：
        dict: {"tj": ..., "p": ..., "iters": ...}

    异常：
        ValueError: 迭代不收敛（热失控）
    """
    rth_ja = float(rth_ja)
    tamb = float(tamb)
    tol = float(tol)
    max_iter = int(max_iter)

    _check_finite("rth_ja", rth_ja, min_exclusive=0)
    _check_finite("tamb", tamb)
    _check_finite("tol", tol, min_exclusive=0)
    if max_iter < 1:
        raise ValueError(f"max_iter 必须 >= 1，当前为 {max_iter}")
    if not callable(loss_fn):
        raise ValueError("loss_fn 必须是可调用函数")

    tj_prev = tamb
    p = loss_fn(tj_prev)

    for it in range(max_iter):
        tj_new = tamb + rth_ja * p

        if not np.isfinite(tj_new):
            raise ValueError(f"迭代发散（Tj={tj_new:.4e}），可能发生热失控")

        if abs(tj_new - tj_prev) < tol * max(1.0, abs(tj_prev)):
            p_final = loss_fn(tj_new)
            return {"tj": float(tj_new), "p": float(p_final), "iters": it + 1}

        tj_prev = tj_new
        p = loss_fn(tj_prev)

    raise ValueError(
        f"定点迭代不收敛（{max_iter} 次迭代后 |ΔTj|={abs(tj_new - tj_prev):.4e}），"
        f"可能发生热失控"
    )
