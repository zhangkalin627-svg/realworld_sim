"""连续运行：淬火液水箱热平衡、冷水机负荷、PAG 浓度/温度对淬火的影响、开裂倾向、轴承寿命。
时间步 = 每件（节拍），能量守恒逐件累加。"""
import math, json, os
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_longrun(hours=8.0, takt=7.89, P_kW=5.9, t_heat=3.0, frac_to_quench=0.80, V_tank_L=300.0,
                conc=0.10, T0=25.0, T_cw=20.0, UA=500.0, chiller_kW=8.0, T_amb=30.0,
                eff_coil=0.79, P_supply_loss_kW=1.5, chiller_derate_per_K=0.012):
    """返回逐件历史。热量：每件吸收 P·t_heat，其中 frac_to_quench 被喷淋带进水箱（其余随工件带走）。
    换热器：Q = UA·(T_tank − T_cw)（板换，冷水侧流量充足）。冷水机还要带走线圈+电源损耗。"""
    rho_c = 1.02 * 4000.0 * V_tank_L          # J/K（PAG 溶液 cp≈4.0 kJ/kgK）
    n = int(hours * 3600 / takt)
    T = T0
    E_part = P_kW * 1e3 * t_heat
    P_coil_loss = P_kW * 1e3 * (1 / eff_coil - 1) * t_heat / takt   # 平均
    hist = dict(t=[], T_tank=[], Q_hex=[], chiller_load=[], chiller_cap=[])
    for k in range(n):
        q_in = E_part * frac_to_quench
        q_hex = UA * max(T - T_cw, 0) * takt
        q_amb = 8.0 * 2.0 * (T - T_amb) * takt      # 水箱表面 ~2 m², h≈8 W/m²K
        T += (q_in - q_hex - q_amb) / rho_c
        cap = chiller_kW * 1e3 * (1 - chiller_derate_per_K * max(T_amb - 30, 0))
        load = q_hex / takt + P_coil_loss + P_supply_loss_kW * 1e3
        if k % 10 == 0:
            hist["t"].append(k * takt / 60); hist["T_tank"].append(T); hist["Q_hex"].append(q_hex / takt)
            hist["chiller_load"].append(load); hist["chiller_cap"].append(cap)
    T_ss = T_cw + (E_part * frac_to_quench / takt + 16 * T_amb - 16 * 0) / (UA + 16) - 16 * T_cw / (UA + 16) * 0
    h = {k: np.array(v) for k, v in hist.items()}
    t63 = None
    Tend = h["T_tank"][-1]
    for t, TT in zip(h["t"], h["T_tank"]):
        if TT >= T0 + 0.63 * (Tend - T0):
            t63 = t; break
    res = dict(T_end=float(Tend), t63_min=t63, Q_in_avg=E_part * frac_to_quench / takt,
               chiller_load_end=float(h["chiller_load"][-1]), chiller_cap=float(h["chiller_cap"][-1]),
               parts=n, hist=h, conc=conc)
    return res


# PAG 浓度 → 冷却能力（相对清水）与推荐（典型厂商资料趋势，取中值）
PAG_TABLE = [(0.00, 1.00), (0.05, 0.80), (0.08, 0.68), (0.10, 0.62), (0.12, 0.56), (0.15, 0.48), (0.20, 0.40), (0.25, 0.34)]
PAG_CLOUD_C = 74.0   # 逆溶性浊点（PAG 析出温度）


def pag_severity(conc, T_bath):
    s = float(np.interp(conc, *zip(*PAG_TABLE)))
    # 液温升高冷却能力下降（20→40°C 约降 10–15%）
    s *= max(1 - 0.006 * max(T_bath - 25, 0), 0.5)
    return s


def verdicts_longrun(r, crack=None):
    out = []
    T = r["T_end"]
    if T > 45:
        out.append(("FAIL", f"淬火液稳定在 {T:.0f}°C（>45°C），冷却能力明显下降，硬化层变浅、硬度波动"))
    elif T > 38:
        out.append(("WARN", f"淬火液升到 {T:.0f}°C，接近 PAG 推荐上限 40°C"))
    else:
        out.append(("OK", f"淬火液稳定在约 {T:.0f}°C（PAG 推荐 20–40°C）"))
    if r["chiller_load_end"] > r["chiller_cap"]:
        out.append(("FAIL", f"冷水机负荷 {r['chiller_load_end']/1e3:.1f} kW > 能力 {r['chiller_cap']/1e3:.1f} kW，会越跑越热"))
    elif r["chiller_load_end"] > 0.85 * r["chiller_cap"]:
        out.append(("WARN", f"冷水机负荷 {r['chiller_load_end']/1e3:.1f} kW，接近能力 {r['chiller_cap']/1e3:.1f} kW（余量 <15%，夏天要降额）"))
    else:
        out.append(("OK", f"冷水机负荷 {r['chiller_load_end']/1e3:.1f} kW / 能力 {r['chiller_cap']/1e3:.1f} kW"))
    s = pag_severity(r["conc"], T)
    out.append(("OK", f"PAG {r['conc']*100:.0f}% 在 {T:.0f}°C 时冷却烈度约为清水的 {s*100:.0f}%"))
    if crack:
        out += crack
    return out


def crack_risk_compare(cache=os.path.join(HERE, "models", "demo", "crack_compare.json")):
    """用感应加热有限元比较清水喷淋 / PAG 喷淋在马氏体转变区(300→200°C)的冷却速度。
    冷得越猛，组织应力越大，越容易淬裂（45 钢、薄壁、尖角处更危险）。"""
    if os.path.exists(cache):
        return json.load(open(cache))
    import sys; sys.path.insert(0, HERE)
    from rw.induction import InductionQuenchSim
    out = {}
    for med in ("water_spray", "polymer_spray", "oil_dip"):
        s = InductionQuenchSim(freq=30e3, power_kw=5.9, t_heat=3.0, medium=med, t_quench=4.0, t_air=1.0)
        r = s.run()
        t, T = s.hist["t"], s.hist["Tsurf"]
        def tcross(x):
            i = np.nonzero((T[:-1] >= x) & (T[1:] < x) & (t[:-1] > 3.0))[0]
            return float(t[i[0]]) if len(i) else None
        a, b = tcross(300), tcross(200)
        rate = (100 / (b - a)) if (a and b and b > a) else None
        out[med] = dict(rate_300_200=rate, HRC=r["HRC_surface"], case=r["case_depth_mm"])
    json.dump(out, open(cache, "w"), indent=1)
    return out


def crack_verdicts(cc):
    names = {"water_spray": "清水喷淋", "polymer_spray": "PAG 喷淋", "oil_dip": "油浸"}
    v = []
    for k, d in cc.items():
        r = d["rate_300_200"]
        if r is None:
            v.append(("WARN", f"{names[k]}：表面没降到 200°C（冷却不足），硬化层 {d['case']:.1f} mm")); continue
        lv = "FAIL" if r > 500 else ("WARN" if r > 200 else "OK")
        txt = {"FAIL": "淬裂风险高（尤其瓶口端面尖角）", "WARN": "有一定淬裂风险，倒角 R≥0.5 并控制浓度", "OK": "淬裂风险低"}[lv]
        v.append((lv, f"{names[k]}：马氏体区冷速约 {r:.0f}°C/s，硬化层 {d['case']:.1f} mm → {txt}"))
    return v


def bearing_life(C_N=9950, P_N=120, rpm=60, duty=0.4):
    """6004-2RS：L10 = (C/P)^3 百万转。"""
    L10 = (C_N / P_N) ** 3
    hours = L10 * 1e6 / (rpm * 60 * duty)
    return L10, hours


# 45 钢回火 1 h 硬度（典型）
TEMPER_1H = [(20, 58), (150, 57), (200, 55), (250, 52), (300, 50), (350, 47), (400, 45), (450, 42), (500, 38), (550, 33), (600, 28)]


def self_temper_hrc(T_rebound, hold_s=10.0, HRC_q=58.0):
    """淬火后余热回升到 T_rebound 并保持约 hold_s 秒的"自回火"。
    Hollomon-Jaffe：P = T(K)·(20 + log10 t[h])，换算成等效 1 h 回火温度再查表。"""
    if T_rebound < 120:
        return HRC_q, T_rebound
    TK = T_rebound + 273.15
    P = TK * (20 + math.log10(hold_s / 3600))
    T1h = P / 20 - 273.15
    h = float(np.interp(T1h, *zip(*TEMPER_1H)))
    return min(h, HRC_q), T1h


def quench_time_sweep(tt=None, t_heat=3.0):
    from rw.plc.thermal_table import ThermalTable
    tt = tt or ThermalTable()
    rows = []
    for tq in np.linspace(1.0, 3.0, 9):
        r = tt.result(t_heat, tq)
        # 出淬火区后的表面温度（表里算了 3 s 空冷后的值）→ 插值
        Ta = float(np.interp(tq, [1, 2, 3], [x["T_after"] for x in tt.d["rows"] if abs(x["t_heat"] - t_heat) < 1e-6]))
        h, T1h = self_temper_hrc(Ta, 10.0, r["HRC"])
        rows.append(dict(t_quench=float(tq), T_exit=Ta, HRC_q=r["HRC"], HRC_temp=h, case=r["case"]))
    return rows
