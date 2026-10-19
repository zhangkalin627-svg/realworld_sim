"""
waveform_metrics.py —— 仿真波形后处理指标计算
本工具由 AI 助手（企业微信助手）编写，经我方独立验收（accept_rwpost.py，45 项）后采用；全部为程序运行结果，非实测
只依赖 numpy 和 Python 标准库
"""

import numpy as np


def zero_crossings(t, x, rising=True):
    """
    返回 x 穿过 0 的时刻数组（线性插值求根）。
    rising=True 只取由负到非负的穿越，False 只取由正到非正。
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)

    if t.ndim != 1 or x.ndim != 1:
        raise ValueError("t 和 x 必须是一维数组")
    if len(t) < 2:
        raise ValueError("t 的长度必须 >= 2")
    if len(t) != len(x):
        raise ValueError(f"t 长度 ({len(t)}) 与 x 长度 ({len(x)}) 不一致")
    if np.any(np.diff(t) <= 0):
        raise ValueError("t 必须严格递增")
    if np.any(np.isnan(t)):
        raise ValueError("t 不能包含 NaN")
    if np.any(np.isnan(x)):
        raise ValueError("x 不能包含 NaN")

    n = len(x)
    crossings = []

    for k in range(n - 1):
        if rising:
            if x[k] < 0 and x[k + 1] >= 0:
                if x[k + 1] == x[k]:
                    t_cross = t[k]
                else:
                    t_cross = t[k] + (t[k + 1] - t[k]) * (-x[k]) / (x[k + 1] - x[k])
                crossings.append(t_cross)
        else:
            if x[k] > 0 and x[k + 1] <= 0:
                if x[k + 1] == x[k]:
                    t_cross = t[k]
                else:
                    t_cross = t[k] + (t[k + 1] - t[k]) * (-x[k]) / (x[k + 1] - x[k])
                crossings.append(t_cross)

    return np.array(crossings, dtype=float)


def frequency(t, x):
    """用上升过零点估计基波频率。"""
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)

    if t.ndim != 1 or x.ndim != 1:
        raise ValueError("t 和 x 必须是一维数组")
    if len(t) < 2:
        raise ValueError("t 的长度必须 >= 2")
    if len(t) != len(x):
        raise ValueError(f"t 长度 ({len(t)}) 与 x 长度 ({len(x)}) 不一致")
    if np.any(np.diff(t) <= 0):
        raise ValueError("t 必须严格递增")
    if np.any(np.isnan(t)):
        raise ValueError("t 不能包含 NaN")
    if np.any(np.isnan(x)):
        raise ValueError("x 不能包含 NaN")

    zc = zero_crossings(t, x, rising=True)
    if len(zc) < 3:
        raise ValueError(
            f"过零点少于 3 个（当前 {len(zc)}），无法估计频率"
        )

    diffs = np.diff(zc)
    median_diff = np.median(diffs)
    return float(1.0 / median_diff)


def _find_integer_periods(t, f0):
    """
    从均匀采样时间向量 t 中找出整数个基波周期。
    返回：(n_trunc, periods)
    - n_trunc: 应截取的采样点数
    - periods: 实际使用的完整周期数
    """
    T0 = 1.0 / f0
    dt_vals = np.diff(t)
    mean_dt = float(np.mean(dt_vals))
    pts_per_period = int(round(T0 / mean_dt))
    n = len(t)

    # 方案1：基于采样点数计算
    n_trunc_by_pts = (n // pts_per_period) * pts_per_period

    # 方案2：基于时间范围计算
    total_time = t[-1] - t[0]
    periods_by_time = int(np.floor(total_time / T0))
    n_trunc_by_time = int(round(periods_by_time * T0 / mean_dt))

    # 取两者中较大的（确保取到尽可能多的完整周期）
    if n_trunc_by_pts > n_trunc_by_time and n_trunc_by_pts <= n:
        return n_trunc_by_pts, n_trunc_by_pts // pts_per_period
    elif n_trunc_by_time > 0 and n_trunc_by_time <= n:
        return n_trunc_by_time, periods_by_time
    else:
        return min(n_trunc_by_pts, n), max(1, n // pts_per_period)


def fft_harmonics(t, x, f0, n_harm=10, periods=None):
    """
    对均匀采样的 x 取整数个基波周期，直接 DFT。
    返回 dict：{"dc": 直流分量, "amp": [基波、2次…n_harm次谐波幅值(峰值)],
                 "phase": [对应相位(rad, 相对cos, 取值(-pi, pi])]}
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)

    if t.ndim != 1 or x.ndim != 1:
        raise ValueError("t 和 x 必须是一维数组")
    if len(t) < 2:
        raise ValueError("t 的长度必须 >= 2")
    if len(t) != len(x):
        raise ValueError(f"t 长度 ({len(t)}) 与 x 长度 ({len(x)}) 不一致")
    if np.any(np.diff(t) <= 0):
        raise ValueError("t 必须严格递增")
    if np.any(np.isnan(t)):
        raise ValueError("t 不能包含 NaN")
    if np.any(np.isnan(x)):
        raise ValueError("x 不能包含 NaN")
    if n_harm < 1:
        raise ValueError(f"n_harm 必须 >= 1，当前为 {n_harm}")
    if f0 <= 0:
        raise ValueError(f"f0 必须 > 0，当前为 {f0}")

    dt = np.diff(t)
    if len(dt) > 0:
        mean_dt = np.mean(dt)
        if np.any(np.abs(dt - mean_dt) / mean_dt > 1e-6):
            raise ValueError("t 必须严格均匀采样（相对误差 1e-6 内）")

    n = len(x)
    T0 = 1.0 / f0
    total_time = t[-1] - t[0]

    if total_time < T0:
        raise ValueError(
            f"数据长度 ({total_time:.6e} s) 不足一个基波周期 ({T0:.6e} s)"
        )

    if periods is not None:
        mean_dt_val = float(np.mean(dt))
        pts_per_period = int(round(T0 / mean_dt_val))
        n_trunc = periods * pts_per_period
        if n_trunc > n:
            n_trunc = n
    else:
        n_trunc, periods = _find_integer_periods(t, f0)

    if n_trunc < 2:
        raise ValueError("取整周期后数据点数不足")

    t_trunc = t[:n_trunc]
    x_trunc = x[:n_trunc]

    X = np.fft.fft(x_trunc) / n_trunc

    dc = float(np.real(X[0]))

    # DFT 频率分辨率
    fs_actual = 1.0 / float(mean_dt)
    df = fs_actual / n_trunc

    amps = []
    phases = []
    for h in range(1, n_harm + 1):
        f_h = h * f0
        k = int(round(f_h / df))
        if k >= len(X):
            break
        amp = 2.0 * np.abs(X[k])
        phase = np.angle(X[k])
        if phase <= -np.pi:
            phase += 2 * np.pi
        amps.append(float(amp))
        phases.append(float(phase))

    return {
        "dc": dc,
        "amp": amps,
        "phase": phases,
    }


def thd(amp):
    """总谐波畸变率：sqrt(sum(amp[1:]**2)) / amp[0]。"""
    amp_arr = np.asarray(amp, dtype=float)
    if amp_arr[0] <= 0:
        raise ValueError(
            f"基波幅值必须 > 0，当前为 {amp_arr[0]}"
        )
    if len(amp_arr) < 2:
        return 0.0
    return float(np.sqrt(np.sum(amp_arr[1:] ** 2)) / amp_arr[0])


def ripple(x, t=None, window=None):
    """纹波：峰峰值。"""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x 必须是一维数组")
    if len(x) < 1:
        raise ValueError("x 长度必须 >= 1")
    if np.any(np.isnan(x)):
        raise ValueError("x 不能包含 NaN")

    if window is not None and t is not None:
        t = np.asarray(t, dtype=float)
        if len(t) != len(x):
            raise ValueError(f"t 长度 ({len(t)}) 与 x 长度 ({len(x)}) 不一致")
        mask = (t >= window[0]) & (t <= window[1])
        if not np.any(mask):
            raise ValueError(
                f"窗口 ({window[0]}, {window[1]}) 内没有采样点"
            )
        x_seg = x[mask]
    else:
        x_seg = x

    return float(np.max(x_seg) - np.min(x_seg))
