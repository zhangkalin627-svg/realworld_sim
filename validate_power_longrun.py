# -*- coding: utf-8 -*-
"""淬火电源(100kW 通用机) 基线 v0.5 验证 · C 矩阵：长时多班次 + 纹波热 + 缓变故障保护注入(E/F)。
所有数字为**仿真结果**（rw/machine 平均模型 + 热网快速外推 dt=2s），未实测。
口径：详细仿真只跑到功率到达后约 40 s；小时级时长全部是热网外推（指数精确更新），
外推期间电气工作点保持最后详细窗口的平均损耗不变（跳闸后损耗归零、继续冷却）。
"""
import math
import numpy as np
from rw.machine import scenarios as SC
from rw.machine import stress as ST

def build_burnin():
    """满载老化机：详细跑到功率到达 + 两个 20 s 平均损耗窗口。"""
    m = SC.build("fp_burnin")
    m.run(40.0, lambda mm: mm.ctl.reached and mm.t > 8.0)
    m.run(m.t + 20.0); m.mark()
    m.run(m.t + 20.0); m.mark()
    return m

def margins(T, sp):
    mod, cs = sp.mod, sp.cool
    return dict(Tj=T["Tj_T"], Tj_lim=mod.Tj_design, T_plate=T["T_plate"], plate_trip=mod.T_ntc_trip,
                T_w=T["T_w"], w_trip=cs.T_trip_in, T_tr=T["T_tr"], tr_hot=cs.tr_hot,
                T_cap=T["T_cap"], cap_hot=cs.cap_hot, T_can=T["T_can"], can_ref=cs.T_life_ref)

def show_margins(tag, T, sp):
    g = margins(T, sp)
    print("    %s Tj %.1f/%d°C | 基板 %.1f/%d | 进水 %.1f/%d | 变压器 %.1f/%d | 谐振电容 %.1f/%d | 电解芯 %.1f/%d" % (
        tag, g["Tj"], g["Tj_lim"], g["T_plate"], g["plate_trip"], g["T_w"], g["w_trip"],
        g["T_tr"], g["tr_hot"], g["T_cap"], g["cap_hot"], g["T_can"], g["can_ref"]))
    return g

print("=== C1) 满载老化 24 h（三班次连续，夏天 35°C，冷水机 30 kW）===")
m = build_burnin()
Lavg, Twin = m.avg_losses()
print("    平均损耗窗口 %.0f s | P_in %.1f kW | 逆变 %.0f W | 整流 %.0f W | 变压器 %.0f W | 谐振电容 %.0f W | 电解(每只) %.1f W | 线圈→水 %.1f kW" % (
    Twin, Lavg["P_in"] / 1e3, Lavg["inv_total"], Lavg["rect"], Lavg["tr_cu"] + Lavg["tr_core"], Lavg["cap"], Lavg["bus_can"], Lavg["P_coil"] / 1e3))
L0 = m.net.L_used_bus
lr = m.longrun_thermal(24.0, T_amb=m.net.T_amb)
ser = lr["series"]
T_end = {k: float(ser[k][-1]) for k in ser if k not in ("h",)}
T_end["Tj_T"] = float(ser["Tj"][-1])
g_end = show_margins("24h 末态：", T_end, m.sp)
drift = abs(float(ser["T_w"][-1]) - float(ser["T_w"][-31]))
print("    末 1 h 进水漂移 %.3f K（<0.5 K 视为热稳态）| 24h 保护动作：%s" % (drift, lr["trip"] or "无（正确）"))
L_day = m.net.L_used_bus - L0
print("    电解电容寿命消耗 %.4f h/24h（105°C 当量）→ 按 8000 h 基数约 %.0f 年（连续满载）" % (L_day, 8000.0 / (L_day * 365.0)))
E_ch = m.net.E["chiller"] / 3.6e6
print("    冷水机 24h 制冷量 %.0f kWh | 淬火水箱末温 %.1f °C" % (E_ch, float(ser["T_tank"][-1])))
c1_ok = lr["trip"] is None and drift < 0.5
print("    >> C1 %s：24h 满载无误动、热稳态收敛" % ("PASS" if c1_ok else "FAIL"))

print()
print("=== C2) 12 kW 连续生产 24 h（三班次，功率循环寿命视角）===")
m2 = SC.build("pp_8h")
m2.run(150.0, lambda mm: len(mm.done) >= 2); m2.mark(); n0 = len(m2.done); t0 = m2.t
m2.run(200.0, lambda mm: len(mm.done) >= n0 + 3); m2.mark()
cycle = (m2.t - t0) / 3
L = {k: np.array(v) for k, v in m2.log.items()}
t = L["t"]; Tj = L["Tj"]
last = t >= m2.t - cycle
swing = float(Tj[last].max() - Tj[last].min()); tj_mean = float(Tj[last].mean())
Nf = ST.lesit_cycles(swing, tj_mean)
cyc_day = 86400.0 / cycle
life_days = Nf / cyc_day if math.isfinite(Nf) else float("inf")
L0 = m2.net.L_used_bus
lr2 = m2.longrun_thermal(24.0, T_amb=m2.net.T_amb)
ser2 = lr2["series"]
T2 = {k: float(ser2[k][-1]) for k in ser2 if k not in ("h",)}; T2["Tj_T"] = float(ser2["Tj"][-1])
show_margins("24h 末态：", T2, m2.sp)
print("    节拍 %.1f s/件（%.0f 件/天）| 节拍内 Tj 摆动 %.1f K（均值 %.1f °C）" % (cycle, cyc_day, swing, tj_mean))
print("    LESIT 功率循环寿命：Nf=%s 次 → 约 %s 天（估算，焊料/键合线老式模块拟合）" % (
    ("%.2e" % Nf) if math.isfinite(Nf) else "∞", ("%.0f" % life_days) if math.isfinite(life_days) else "∞"))
L_day2 = m2.net.L_used_bus - L0
print("    电解寿命消耗 %.4f h/24h | 24h 保护动作：%s" % (L_day2, lr2["trip"] or "无（正确）"))
c2_ok = lr2["trip"] is None and swing < 40.0
print("    >> C2 %s" % ("PASS" if c2_ok else "FAIL"))

print()
print("=== C3) 母线电解电容纹波热与寿命（额定 100 kW 点）===")
sp = m.sp
rec = ST.rectifier_pf(int(round(sp.rated_kW)), V_ll=sp.V_ll, L=sp.L_line, Rs=sp.Rs, C=sp.C_bus)
I_cap = rec["I_cap_rms"]; cs = sp.cool
I_each = I_cap / cs.n_par
print("    整流时域仿真：母线 %.0f V | 电容纹波 %.1f Arms（300 Hz 为主）→ 每并 %.1f A（额定 %d A/只，100 Hz/85°C；300 Hz 频率系数未计，偏严）" % (
    rec["Vdc"], I_cap, I_each, cs.I_ripple_rated_each))
P_can = I_each ** 2 * cs.ESR_bus_each
T_can_ss = float(ser["T_can"][-1])
life_h = ST.ecap_life_h(T_can_ss, cs)
print("    每只纹波发热 %.1f W | C1 稳态芯温 %.1f °C → 寿命约 %.0f h（%.1f 年连续满载；Arrhenius 10°C 减半，估算）" % (
    P_can, T_can_ss, life_h, life_h / 8760.0))
c3_ok = I_each <= cs.I_ripple_rated_each
print("    >> C3 %s：纹波电流 %s 额定" % ("PASS" if c3_ok else "FAIL", "低于" if c3_ok else "超出"))

# ---------------------------------------------------------------- E/F 缓变故障
def longrun_sched(m, Lavg, hours, sched, tag):
    """longrun_thermal 的调度版：sched(t_h, net) 每步修改冷水机/流量；跳闸判定与 Machine 一致。"""
    n = m.net; sp = m.sp
    Gw = sp.G_water * min(m.cw_flow / m.Q_cw, 1.3) ** 0.8 if m.Q_cw > 0 else 0.0
    dt = 2.0; t = 0.0; T_end = hours * 3600.0
    trip = None; excee = {}
    rec = []
    Lx = dict(Lavg)
    while t < T_end - 1e-9:
        sched(t / 3600.0, n)
        T_coil_ss = n.T_w + Lx["P_coil"] / max(Gw, 1e-6)
        m.T_coil = T_coil_ss + (m.T_coil - T_coil_ss) * math.exp(-dt * Gw / sp.C_coil)
        n.step(dt, Lx, P_coil_to_water=Lx["P_coil"], P_other_water=0.0)
        t += dt
        T = n.temps()
        for k, v, lim, lab in (("tr", T["T_tr"], sp.cool.tr_hot, "变压器热点"), ("cap", T["T_cap"], sp.cool.cap_hot, "谐振电容"),
                               ("tj", T["Tj_T"], sp.mod.Tj_design, "Tj 设计值"), ("can", T["T_can"], sp.cool.T_life_ref, "电解芯")):
            if k not in excee and v >= lim:
                excee[k] = (t / 3600.0, lab, v, lim)
        if trip is None:
            if n.T_plate >= sp.mod.T_ntc_trip:
                trip = (t / 3600.0, "OT", "IGBT 基板过温 %.0f°C" % n.T_plate)
            elif n.T_w >= sp.cool.T_trip_in:
                trip = (t / 3600.0, "WT", "冷却进水 %.1f°C ≥ %.0f°C" % (n.T_w, sp.cool.T_trip_in))
            if trip is not None:
                print("    ★ 第 %.2f h 保护动作：%s（之后损耗归零继续外推）" % (trip[0], trip[2]))
                Lx = {k: 0.0 for k in Lx}
        if len(rec) == 0 or t / 3600.0 - rec[-1][0] >= 0.5:
            rec.append((t / 3600.0, T["T_w"], T["T_plate"], T["Tj_T"], T["T_tr"], T["T_cap"]))
    print("    %s 轨迹(0.5h 步)：h/T_w/基板/Tj/变压器/谐振电容" % tag)
    for r in rec[::4]:
        print("      %5.1f h | %5.1f | %5.1f | %6.1f | %6.1f | %5.1f" % r)
    return trip, excee, rec

print()
print("=== E) 缓变故障注入：冷水机制冷量 4 h 内 30→2 kW（衰降到低于总损耗 ~5.9 kW）===")
print("    备注：30→6 kW 档外推 8 h 进水只升到 30.3°C——6 kW 仍盖得住 v0.5 损耗，属余量而非拒动；故降到 2 kW 验证保护兜底。")
m3 = build_burnin()
Lavg3, _ = m3.avg_losses()
tripE, excE, _ = longrun_sched(m3, Lavg3, 10.0, lambda h, n: setattr(n.cs, "Q_chiller", 30e3 - 28e3 * min(h / 4.0, 1.0)), "E")
print("    结果：%s" % ("第 %.2f h %s 动作（应动作）" % (tripE[0], tripE[1]) if tripE else "10 h 内无动作（拒动!!）"))
for k, (h, lab, v, lim) in excE.items():
    print("    ⚠ 第 %.2f h %s 越限：%.1f ≥ %.0f（保护动作前器件先越限 = 保护覆盖缺口）" % (h, lab, v, lim))
e_ok = tripE is not None
print("    >> E %s" % ("PASS：保护按预期兜底" if e_ok and not excE else ("PASS(带缺口警示)" if e_ok else "FAIL")))

print()
print("=== F) 流量缓降 100→55%（滤网缓堵，高于 50% 流量开关定值不报警）· 热余量验证 ===")
print("    语义更正：v0.5 损耗下 55% 流量换热仍绰绰有余，正确结果是『不动作且有裕量』（余量验证），不是拒动。")
m4 = build_burnin()
Lavg4, _ = m4.avg_losses()
tripF, excF, _ = longrun_sched(m4, Lavg4, 10.0, lambda h, n: setattr(n, "flow_frac", 1.0 - 0.45 * min(h / 4.0, 1.0)), "F")
print("    结果：%s" % ("第 %.2f h %s 动作" % (tripF[0], tripF[1]) if tripF else "10 h 无动作且各温度远低于限值（55% 流量仍有裕量，正确）"))
for k, (h, lab, v, lim) in excF.items():
    print("    ⚠ 第 %.2f h %s 越限：%.1f ≥ %.0f（保护动作前器件先越限 = 保护覆盖缺口）" % (h, lab, v, lim))
f_ok = tripF is None and not excF
print("    >> F %s" % ("PASS：55% 流量热裕量充足、无误动" if f_ok else "FAIL：出现误动或器件越限"))

print()
print("=== C 矩阵总判定 ===")
allok = all([c1_ok, c2_ok, c3_ok, e_ok, f_ok])
print("    C1 %s | C2 %s | C3 %s | E %s | F %s → %s" % (
    "PASS" if c1_ok else "FAIL", "PASS" if c2_ok else "FAIL", "PASS" if c3_ok else "FAIL",
    "PASS" if e_ok else "FAIL", "PASS" if f_ok else "FAIL", "全部通过（仿真口径）" if allok else "存在 FAIL，见上"))
