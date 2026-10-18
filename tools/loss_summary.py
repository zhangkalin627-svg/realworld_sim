"""
来源标注：外部 AI（Gemini）按我方规格生成；经我方独立评审（tools/test_loss_summary.py，27 项验收，对照独立参考实现）后采用。全部为程序运行结果，非实测。

loss_summary.py
电路仿真波形损耗汇总模块
只依赖 numpy (>=1.24)，Python 3.10+
"""

import numpy as np

def device_power(t, v, i, t0=None, t1=None):
    """
    计算单个器件在统计窗口内的功率、能量及电流有效值。
    
    参数:
        t: 一维严格递增时间数组 (n >= 2)
        v: 同长度一维电压数组
        i: 同长度一维电流数组
        t0, t1: 统计窗口 [t0, t1]，默认整段
        
    返回:
        dict: {"energy": ..., "avg_power": ..., "peak_power": ..., "rms_current": ...}
    """
    t = np.asarray(t, dtype=float)
    v = np.asarray(v, dtype=float)
    i = np.asarray(i, dtype=float)
    
    # 基础维度与长度校验
    if t.ndim != 1 or v.ndim != 1 or i.ndim != 1:
        raise ValueError("时间、电压、电流数组必须是一维的")
    if len(t) < 2:
        raise ValueError("时间数组长度必须至少为 2")
    if len(t) != len(v) or len(t) != len(i):
        raise ValueError("时间、电压、电流数组长度必须一致")
    
    # NaN 校验
    if np.any(np.isnan(t)):
        raise ValueError("时间数组 t 包含 NaN")
    if np.any(np.isnan(v)):
        raise ValueError("电压数组 v 包含 NaN")
    if np.any(np.isnan(i)):
        raise ValueError("电流数组 i 包含 NaN")
        
    # 严格递增校验
    if not np.all(np.diff(t) > 0):
        raise ValueError("时间数组必须严格递增")
        
    t_min = t[0]
    t_max = t[-1]
    
    # 窗口逻辑校验
    if t0 is not None and t1 is not None:
        if t0 >= t1:
            raise ValueError("窗口起点 t0 必须小于终点 t1")
            
    if t0 is not None and t0 >= t_max:
        raise ValueError("统计窗口完全落在数据范围外")
    if t1 is not None and t1 <= t_min:
        raise ValueError("统计窗口完全落在数据范围外")
            
    # 确定有效窗口 [t_start, t_end]（部分超出时截到数据范围内）
    t_start = max(t0, t_min) if t0 is not None else t_min
    t_end = min(t1, t_max) if t1 is not None else t_max
    
    if t_start >= t_end:
        raise ValueError("统计窗口完全落在数据范围外或范围为空")
        
    # 构造子区间及线性插值（向量化）
    mask = (t > t_start) & (t < t_end)
    t_mid = t[mask]
    t_sub = np.concatenate(([t_start], t_mid, [t_end]))
    
    v_sub = np.interp(t_sub, t, v)
    i_sub = np.interp(t_sub, t, i)
    p_sub = v_sub * i_sub
    
    dt = np.diff(t_sub)
    # 梯形法积分计算能量
    energy = np.sum(0.5 * (p_sub[:-1] + p_sub[1:]) * dt)
    duration = t_end - t_start
    avg_power = energy / duration
    peak_power = np.max(np.abs(p_sub))
    
    # 电流有效值计算 (RMS)
    i2_sub = i_sub ** 2
    energy_i2 = np.sum(0.5 * (i2_sub[:-1] + i2_sub[1:]) * dt)
    rms_current = np.sqrt(energy_i2 / duration)
    
    return {
        "energy": float(energy),
        "avg_power": float(avg_power),
        "peak_power": float(peak_power),
        "rms_current": float(rms_current)
    }

def loss_table(devices, t0=None, t1=None, p_in=None, p_out=None):
    """
    汇总多个器件的损耗表、总损耗、效率及功率平衡校验。
    
    参数:
        devices: dict[str, (t, v, i)]
        t0, t1: 统计窗口
        p_in, p_out: 输入功率、输出功率（可选）
        
    返回:
        dict: {"rows": [...], "total_loss": ..., "efficiency": ..., "balance": ...}
    """
    if not isinstance(devices, dict) or len(devices) == 0:
        raise ValueError("devices 必须是非空字典")
        
    rows = []
    total_loss = 0.0
    
    for name, data in devices.items():
        if not isinstance(name, str):
            raise ValueError("器件名必须是字符串")
        if not isinstance(data, (tuple, list)) or len(data) != 3:
            raise ValueError(f"器件 '{name}' 的数据格式必须为 (t, v, i) 元组或列表")
            
        t, v, i = data
        try:
            res = device_power(t, v, i, t0=t0, t1=t1)
        except ValueError as e:
            err_msg = str(e)
            # 丰富 NaN 或其他错误的上下文提示信息
            if "NaN" in err_msg or "数组" in err_msg:
                raise ValueError(f"器件 '{name}' {err_msg}")
            else:
                raise ValueError(f"器件 '{name}': {err_msg}")
                
        row = {
            "name": name,
            "energy": res["energy"],
            "avg_power": res["avg_power"],
            "peak_power": res["peak_power"],
            "rms_current": res["rms_current"],
            "share": 0.0
        }
        rows.append(row)
        total_loss += res["avg_power"]
        
    # 计算 share（总损耗为 0 时所有 share 为 0.0）
    if abs(total_loss) < 1e-15:
        for row in rows:
            row["share"] = 0.0
    else:
        for row in rows:
            row["share"] = float(row["avg_power"] / total_loss)
            
    # 排序：avg_power 从大到小，同值按名字升序
    rows.sort(key=lambda x: (-x["avg_power"], x["name"]))
    
    efficiency = None
    balance = None
    if p_in is not None and p_out is not None:
        p_in_f = float(p_in)
        p_out_f = float(p_out)
        if p_in_f != 0:
            efficiency = float(p_out_f / p_in_f)
        else:
            efficiency = 0.0
        balance = float(p_in_f - p_out_f - total_loss)
        
    return {
        "rows": rows,
        "total_loss": float(total_loss),
        "efficiency": efficiency,
        "balance": balance
    }