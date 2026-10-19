"""
rms_stats.py —— 仿真波形统计量计算
本工具由 AI 助手（企业微信助手）编写，经我方独立验收（accept_rwpost.py，45 项）后采用；全部为程序运行结果，非实测
只依赖 numpy 和 Python 标准库
"""

import numpy as np


def rms(x, t=None):
    """
    有效值。给 t 时用梯形法按时间加权：
    sqrt(∫x²dt / (t[-1]-t[0]))；不给 t 则 sqrt(mean(x²))。
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x 必须是一维数组")
    if len(x) < 2:
        raise ValueError("x 的长度必须 >= 2")
    if np.any(np.isnan(x)):
        raise ValueError("x 不能包含 NaN")

    if t is not None:
        t = np.asarray(t, dtype=float)
        if len(t) != len(x):
            raise ValueError(f"t 长度 ({len(t)}) 与 x 长度 ({len(x)}) 不一致")
        if np.any(np.isnan(t)):
            raise ValueError("t 不能包含 NaN")
        if np.any(np.diff(t) <= 0):
            raise ValueError("t 必须严格递增")
        integral = np.trapz(x ** 2, t)
        period = t[-1] - t[0]
        return float(np.sqrt(integral / period))
    else:
        return float(np.sqrt(np.mean(x ** 2)))


def avg(x, t=None):
    """
    平均值。给 t 时用梯形法按时间加权：
    ∫x dt / (t[-1]-t[0])；不给 t 则算术平均。
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x 必须是一维数组")
    if len(x) < 2:
        raise ValueError("x 的长度必须 >= 2")
    if np.any(np.isnan(x)):
        raise ValueError("x 不能包含 NaN")

    if t is not None:
        t = np.asarray(t, dtype=float)
        if len(t) != len(x):
            raise ValueError(f"t 长度 ({len(t)}) 与 x 长度 ({len(x)}) 不一致")
        if np.any(np.isnan(t)):
            raise ValueError("t 不能包含 NaN")
        if np.any(np.diff(t) <= 0):
            raise ValueError("t 必须严格递增")
        integral = np.trapz(x, t)
        period = t[-1] - t[0]
        return float(integral / period)
    else:
        return float(np.mean(x))


def crest_factor(x):
    """峰值（max|x|）/ 有效值。有效值为 0 抛 ValueError。"""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x 必须是一维数组")
    if len(x) < 2:
        raise ValueError("x 的长度必须 >= 2")
    if np.any(np.isnan(x)):
        raise ValueError("x 不能包含 NaN")

    rms_val = rms(x)
    if rms_val == 0.0:
        raise ValueError("有效值为 0，无法计算波峰系数")
    return float(np.max(np.abs(x)) / rms_val)


def power_factor(v, i, t=None):
    """
    功率因数：有功功率 P = avg(v*i) 除以 rms(v)*rms(i)。
    分母为 0 抛 ValueError。
    """
    v = np.asarray(v, dtype=float)
    i = np.asarray(i, dtype=float)
    if v.ndim != 1 or i.ndim != 1:
        raise ValueError("v 和 i 必须是一维数组")
    if len(v) < 2 or len(i) < 2:
        raise ValueError("v 和 i 的长度必须 >= 2")
    if len(v) != len(i):
        raise ValueError(f"v 长度 ({len(v)}) 与 i 长度 ({len(i)}) 不一致")
    if np.any(np.isnan(v)):
        raise ValueError("v 不能包含 NaN")
    if np.any(np.isnan(i)):
        raise ValueError("i 不能包含 NaN")

    if t is not None:
        t = np.asarray(t, dtype=float)
        if len(t) != len(v):
            raise ValueError(f"t 长度 ({len(t)}) 与 v 长度 ({len(v)}) 不一致")
        if np.any(np.isnan(t)):
            raise ValueError("t 不能包含 NaN")
        if np.any(np.diff(t) <= 0):
            raise ValueError("t 必须严格递增")

    p = avg(v * i, t)
    vrms = rms(v, t)
    irms = rms(i, t)

    denom = vrms * irms
    if denom == 0.0:
        raise ValueError("rms(v)*rms(i) 为 0，无法计算功率因数")
    return float(p / denom)
