# decimate.py
"""
长序列抽取模块：把很长的时间序列压缩到最多 N 个点，但保留每条曲线的峰值、谷值。仅依赖 numpy。

来源标注：外部 AI（Qwen）按我方规格生成，经我方独立评审（tools/test_decimate.py，33 项验收）后采用。
对标清单 N10（日志限量/抽取）。仿真波形的显示与存盘用，非测量数据处理。

已知限制：max_points 小于 2×曲线数+2 时，无法同时保证每条曲线的全局极值和首末点；
此时只保留首点（及 keep_last 时的末点），不保证极值。点数估算偏保守，实际输出会比 max_points 少 1~4 个点。
"""

import numpy as np


def decimate_info(n, max_points):
    """返回抽取信息字典。"""
    return {
        'n': int(n),
        'max_points': int(max_points),
        'ratio': 1.0 if n <= max_points else n / max_points,
    }


def _validate(t, ys, max_points):
    """输入检查，非法则抛出中文 ValueError。"""
    if not isinstance(t, np.ndarray) or t.ndim != 1:
        raise ValueError("t 必须是一维 numpy 数组。")
    if t.size < 2:
        raise ValueError("t 的长度必须 >= 2。")
    if not np.all(np.diff(t) > 0):
        raise ValueError("t 必须严格递增。")

    if not isinstance(ys, dict) or len(ys) == 0:
        raise ValueError("ys 必须是非空字典。")

    n = t.size
    for k, v in ys.items():
        if not isinstance(v, np.ndarray) or v.ndim != 1:
            raise ValueError(f"ys['{k}'] 必须是一维 numpy 数组。")
        if v.size != n:
            raise ValueError(
                f"ys['{k}'] 的长度为 {v.size}，与 t 的长度 {n} 不一致。"
            )

    if not isinstance(max_points, (int, np.integer)):
        raise ValueError("max_points 必须是整数。")
    if max_points < 4:
        raise ValueError(f"max_points 必须 >= 4，当前为 {max_points}。")


def minmax_decimate(t, ys, max_points, keep_last=True):
    """
    把长序列抽取到最多 max_points 个点，保留每条曲线的桶内极值点。

    参数
    ----
    t : 一维 numpy 数组，严格递增时间。
    ys : dict[str, 一维 numpy 数组]，各数组长度等于 len(t)。
    max_points : int，输出点数上限，>= 4。
    keep_last : bool，True 时输出一定包含最后一个点。

    返回
    ----
    (t_out, ys_out, idx)
        t_out  : 抽取后的时间数组（拷贝，严格递增）。
        ys_out : dict，键同 ys，值为对应曲线在 idx 处的拷贝。
        idx    : int64 数组，升序无重复，被保留点在原序列中的下标。
    """
    _validate(t, ys, max_points)
    n = t.size

    # 规则 1：原样返回（拷贝）
    if n <= max_points:
        idx = np.arange(n, dtype=np.int64)
        return t.copy(), {k: v.copy() for k, v in ys.items()}, idx

    # 规则 2：桶内 min/max 抽取
    fixed = {0}
    if keep_last:
        fixed.add(n - 1)
    fixed_budget = len(fixed)  # 2 或 3

    # 每条曲线至少贡献 2 个候选（全局 min、max），桶数至少 1
    min_buckets = 1
    max_buckets = n  # 上界，后续二分会收缩

    def selected_count(nb):
        """给定桶数 nb，估算合并后的候选点数（上界）。"""
        return min(nb * 2 * len(ys) + fixed_budget, n)

    # 若最小桶数都超，说明 max_points 极小，直接退化为只保留固定点
    if selected_count(min_buckets) > max_points:
        idx = np.array(sorted(fixed), dtype=np.int64)
        t_out = t[idx].copy()
        ys_out = {k: v[idx].copy() for k, v in ys.items()}
        return t_out, ys_out, idx

    # 二分查找最大可用桶数
    lo, hi = min_buckets, max_buckets
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if selected_count(mid) <= max_points:
            lo = mid
        else:
            hi = mid - 1
    best_buckets = lo

    # 用 best_buckets 重新计算，若仍超则逐步减桶（保险）
    for nb in range(best_buckets, 0, -1):
        sel = _select_by_buckets(t, ys, nb, fixed)
        if sel.size <= max_points:
            break
    else:
        # 极端情况：只保留固定点
        sel = np.array(sorted(fixed), dtype=np.int64)

    idx = sel
    t_out = t[idx].copy()
    ys_out = {k: v[idx].copy() for k, v in ys.items()}
    return t_out, ys_out, idx


def _select_by_buckets(t, ys, n_buckets, fixed):
    """
    把下标 [0, n) 等分为 n_buckets 个桶，每个桶内对每条曲线取
    非 NaN 的最小值点和最大值点下标，合并后加上 fixed 点，排序去重。
    """
    n = t.size
    base = n // n_buckets
    rem = n % n_buckets  # 前 rem 个桶多 1 个元素

    # 桶边界：bucket_starts[i], bucket_ends[i] 为第 i 桶的 [start, end)
    bucket_starts = np.empty(n_buckets, dtype=np.int64)
    bucket_ends = np.empty(n_buckets, dtype=np.int64)
    pos = 0
    for i in range(n_buckets):
        sz = base + (1 if i < rem else 0)
        bucket_starts[i] = pos
        bucket_ends[i] = pos + sz
        pos += sz

    all_idx = set(fixed)

    for arr in ys.values():
        for i in range(n_buckets):
            s, e = int(bucket_starts[i]), int(bucket_ends[i])
            seg = arr[s:e]
            if seg.size == 0:
                continue
            mask = ~np.isnan(seg)
            if not np.any(mask):
                continue
            valid = seg[mask]
            # 在有效值中找 min / max 对应的局部下标
            local_min = int(np.argmin(valid))
            local_max = int(np.argmax(valid))
            # 映射回 seg 内的下标
            valid_indices = np.flatnonzero(mask)
            idx_min = s + int(valid_indices[local_min])
            idx_max = s + int(valid_indices[local_max])
            all_idx.add(idx_min)
            all_idx.add(idx_max)

    result = np.array(sorted(all_idx), dtype=np.int64)
    return result