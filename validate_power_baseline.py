# -*- coding: utf-8 -*-
"""淬火电源(100kW 通用机) 基线 v0.5 验证：B 轨锚点复现 + A 电网包线 + B 负载包线。
所有数字为**仿真结果**（基波近似平均模型，rw/machine/power.py），未实测。
锚点来源：kabu499 B 轨 ngspice 开环仿真（2026-10-15，已验收，决策日志 D-001 补充3）。
"""
import dataclasses, math
from rw.machine.power import Spec, Bus, Stage, Controller

M_ANCHOR = 0.949173          # B 轨开环固定相移
F_ANCHOR = 30e3              # B 轨运行频率（由锚点 X=0.526Ω 反推，f/f0≈1.037）

def bus_fixed(sp, sag, P_guess=100e3):
    """母线准静态不动点：Vn=(Vsrc+sqrt(Vsrc²-4·P·Rq))/2"""
    Vsrc = sp.V_src(sag); Rq = sp.Rs + sp.R_comm()
    for _ in range(20):
        disc = Vsrc * Vsrc - 4 * P_guess * Rq
        if disc < 0: return Vsrc / 2, P_guess
        Vn = (Vsrc + math.sqrt(disc)) / 2
        yield Vn
    return Vn

def anchor_row(sp, sag, Vbus_anchor, P_in_est):
    """开环复现：B 轨母线电压直接给定，槽路 I/P/Uc 用模型算"""
    st = Stage(sp)
    Vb, P = Vbus_anchor, P_in_est
    s = st.solve(Vb, F_ANCHOR, 850.0, 1.0, 40.0, M_ANCHOR)
    return s

def grid_bus(sp, sag, P_out, eta=0.985):
    """按给定输出功率求模型母线（含损耗回代迭代）"""
    Vsrc = sp.V_src(sag); Rq = sp.Rs + sp.R_comm()
    P_in = P_out / eta + sp.P_ctrl
    for _ in range(30):
        disc = Vsrc * Vsrc - 4 * P_in * Rq
        Vn = (Vsrc + math.sqrt(max(disc, 0.0))) / 2
        # 用该母线下槽路实际功率回代
        st = Stage(sp)
        s = st.solve(Vn, F_ANCHOR, 850.0, 1.0, 40.0, M_ANCHOR)
        P_in_new = s["P_in"]
        if abs(P_in_new - P_in) < 1.0: return Vn, s
        P_in = P_in_new
    return Vn, s

print("=== 0) B 轨锚点复现（开环 m=%.6f, f=30.0 kHz, 热态槽路）===" % M_ANCHOR)
print("    口径：母线电压用 B 轨原值，槽路 I/P/Uc 用 v0.5 模型独立计算，与已验收锚点对表")
sp = Spec()
for sag, Vb_a, I_a, P_a, Uc_a, note in (
    (0.9, 576.0, 167.5, 81.1e3, 1357.0, "432 Vac(-10%)"),
    (1.1, 702.1, 204.2, 120.5e3, 1653.0, "528 Vac(+10%)"),
):
    s = anchor_row(sp, sag, Vb_a, P_a)
    print("    %s 母线 %.1f V | I1 %.1f A (锚 %.1f, %+.1f%%) | P %.1f kW (锚 %.1f, %+.1f%%) | Uc %.0f V (锚 %.0f, %+.1f%%)" % (
        note, Vb_a, s["I"], I_a, 100*(s["I"]/I_a-1), s["P_out"]/1e3, P_a/1e3, 100*(s["P_out"]/P_a-1),
        s["Vc"], Uc_a, 100*(s["Vc"]/Uc_a-1)))
print("    模型自洽母线（Rs+Rcomm=0.25 Ω 校准后）：")
for sag, Vb_a, note in ((0.9, 576.0, "432 Vac"), (1.1, 702.1, "528 Vac")):
    Vn, s = grid_bus(sp, sag, 81.1e3 if sag < 1 else 120.5e3)
    print("    %s 模型母线 %.1f V vs B 轨 %.1f V (%+.2f%%)" % (note, Vn, Vb_a, 100*(Vn/Vb_a-1)))

def closed_loop(sag, T_wp=850.0, k=1.0, T_coil=40.0, V_ov=None, r_scale=1.0, t_max=4.0, en_t0=0.0):
    """1 kHz 闭环迷你环：Bus+Controller+Stage（PLC/机械不在环）。返回末段稳态与事件。"""
    kw = {}
    if V_ov is not None: kw["V_ov"] = V_ov
    if r_scale != 1.0: kw["R_wp20"] = Spec().R_wp20 * r_scale
    sp = dataclasses.replace(Spec(), **kw)
    bus = Bus(sp); st = Stage(sp); ctl = Controller(sp)
    Vsrc = sp.V_src(sag)
    bus.set_contactors(True, True, Vsrc)
    dt = 1e-3; n = int(t_max / dt)
    stage_fn = lambda f: st.solve(bus.V, f, T_wp, k, T_coil, ctl.m)
    f0_fn = lambda: st.f0(T_wp, k)
    s = None
    for i in range(n):
        t = i * dt
        en = t >= en_t0
        s = ctl.tick(dt, en, bus.V, stage_fn, f0_fn)
        P_in = s["P_in"] if (s is not None and ctl.running) else 0.0
        bus.step(dt, Vsrc, P_in)
    return sp, ctl, bus, s

print()
print("=== A) 电网包线 × 相移闭环（D-001 一期必备，热态额定负载）===")
print("    限值：I1≤186 Arms、Uc≤1540 Vrms（v0.4 硬件约束）；目标 P=100 kW")
for sag, note in ((0.9, "432 Vac(-10%)"), (1.0, "480 Vac(标称)"), (1.1, "528 Vac(+10%)")):
    sp, ctl, bus, s = closed_loop(sag, V_ov=750.0)
    lim = "限内" if (s["I"] <= sp.I_lim + 1e-6 and s["Vc"] <= sp.Uc_lim + 1e-6) else "超限!!"
    sat = " [饱和:m 顶到上限]" if ctl.sat else ""
    print("    %s | Vbus %.0f V | P %.1f kW | I1 %.1f A | Uc %.0f V | m %.3f | f %.2f kHz | %s | %s%s%s" % (
        note, bus.V, s["P_out"]/1e3, s["I"], s["Vc"], ctl.m, s["f"]/1e3, s["mode"], lim,
        "故障:"+ctl.fault if ctl.fault else "", sat))
print()
print("=== A2) +10% 电网 × 现行 V_ov=700 V 阈值（问题复现）===")
sp, ctl, bus, s = closed_loop(1.1)   # 默认 V_ov=700
print("    528 Vac | Vbus %.0f V | 故障锁存: %s %s" % (bus.V, ctl.fault, ctl.fault_log))
print()
print("=== B) 负载包线 × 闭环（标称电网 480 Vac）===")
for T_wp, k, r_scale, note in (
    (850.0, 1.0, 1.0, "热态 满耦合"),
    (850.0, 0.8, 1.0, "热态 耦合0.8"),
    (850.0, 1.0, 0.75, "热态 轻载R×0.75"),
    (850.0, 1.0, 1.25, "热态 重载R×1.25"),
    (25.0, 1.0, 1.0, "冷态 满耦合"),
):
    sp, ctl, bus, s = closed_loop(1.0, T_wp=T_wp, k=k, V_ov=750.0, r_scale=r_scale)
    lim = "限内" if (s["I"] <= sp.I_lim + 1e-6 and s["Vc"] <= sp.Uc_lim + 1e-6) else "超限!!"
    eta = s["P_out"] / s["P_in"] * 100 if s["P_in"] else 0.0
    sat = " [饱和]" if ctl.sat else ""
    print("    %s | P %.1f kW | I1 %.1f A | Uc %.0f V | m %.3f | f %.2f kHz | %s | 逆变输入效率 %.1f%% | %s%s" % (
        note, s["P_out"]/1e3, s["I"], s["Vc"], ctl.m, s["f"]/1e3, s["mode"], eta, lim, sat))
