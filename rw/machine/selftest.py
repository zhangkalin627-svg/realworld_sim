# -*- coding: utf-8 -*-
"""v0.8 整机联动自检（validate.py 第 10 段调用）。"""
import math
import numpy as np

from .power import Spec, Bus, Stage
from . import scenarios as SC
from ..circuit.spice import Circuit
from ..circuit.scope import measure


def pct(a, b):
    return 100.0 * (a / b - 1.0)


def run():
    sp = Spec(); st = Stage(sp)
    # (a) 基波近似 vs ngspice 整桥方波 + 串联 RLC（3 个工况）
    print("[整机] 基波近似 vs ngspice 全桥方波→串联RLC 稳态（含全部谐波）")
    for (f, T, k) in ((30e3, 850, 1.0), (27e3, 700, 1.0), (34e3, 20, 1.0)):
        Vb = 513.0
        L = st.L_of(T, k); Rc, Rw = st.R_parts(T, k, 30.0); R = Rc + Rw
        c = Circuit("full-bridge square -> series RLC")
        per = 1.0 / f; tr = 5e-9
        c.V("s", "a", "0", pulse=(-Vb, Vb, 0, tr, tr, per / 2 - tr, per))
        c.L("1", "a", "b", L); c.C("1", "b", "c", sp.C_res); c.R("1", "c", "0", R)
        w = c.tran(per / 400, 60 * per, tmax=per / 400)
        m = w.x > 40 * per
        i = np.real(w["i(vs)"][m]) if "i(vs)" in w.nets() else None
        if i is None:
            i = np.real(w["c"][m]) / R
        x = w.x[m]
        Irms = math.sqrt(np.trapezoid(i * i, x) / (x[-1] - x[0]))
        P_sp = Irms ** 2 * R
        s = st.solve(Vb, f, T, k, 30.0)
        print("   f=%2.0f kHz T=%3d°C  I_rms 模型 %.2f A / ngspice %.2f A (%+.2f%%)   P 模型 %.0f W / ngspice %.0f W (%+.2f%%)" % (
            f / 1e3, T, s["I"], Irms, pct(s["I"], Irms), s["P_out"], P_sp, pct(s["P_out"], P_sp)))
        assert abs(pct(s["P_out"], P_sp)) < 2.0, "基波近似偏差过大"
    # (b) 合旁路冲击电流 vs ngspice
    print("[整机] 合旁路冲击电流：RK4 vs ngspice（Rs=0.12 Ω, L=0.5 mH, C=2 mF）")
    b = Bus(sp)
    for V0 in (0.0, 300.0, 450.0):
        Ipk, ER, Q, _ = b.inrush(513.0, V0)
        c = Circuit("inrush")
        c.V("s", "s", "0", dc=513.0); c.R("s", "s", "a", sp.Rs); c.L("l", "a", "b", sp.L_line, ic=0.0); c.C("c", "b", "0", sp.C_bus, ic=V0)
        w = c.tran(2e-6, 0.04, tmax=2e-6, uic=True)
        i_sp = np.real(w["i(vs)"]) * -1 if "i(vs)" in w.nets() else np.real((w["s"] - w["a"])) / sp.Rs
        print("   V0=%3.0f V：峰值 RK4 %.1f A / ngspice %.1f A (%+.2f%%)" % (V0, Ipk, i_sp.max(), pct(Ipk, i_sp.max())))
        assert abs(pct(Ipk, i_sp.max())) < 1.5
    # (c) 预充电 RC 指数
    b = Bus(sp); b.set_contactors(True, False, 513.0)
    t = 0.0; dt = 0.001
    while t < 0.2 - 1e-9:
        b.step(dt, 513.0, 0.0); t += dt
    th = 513.0 * (1 - math.exp(-0.2 / (sp.R_pc * sp.C_bus)))
    bleed = 1.0 - 0.0
    print("[整机] 预充电：t=0.2 s 母线 %.1f V，解析 513·(1−e^(−t/RC)) = %.1f V（泄放电阻 %.0f kΩ 影响 %.2f%%）" % (b.V, th, sp.R_bleed / 1e3, 100 * sp.R_pc / sp.R_bleed))
    assert abs(pct(b.V, th)) < 0.5
    # (d) 整机正常场景：能量守恒、与有限元表对比、节拍
    m, S = SC.run("normal")
    bus = m.bus
    lhs = bus.E_src
    rhs = bus.E_cap() + bus.E_R + bus.E_load + bus.E_bleed
    print("[整机] 能量守恒（电网 → 母线电容/电阻/逆变器）：输入 %.1f kJ，分配 %.1f kJ，差 %.3f%%" % (lhs / 1e3, rhs / 1e3, pct(lhs, rhs)))
    assert abs(pct(lhs, rhs)) < 0.5
    ein = bus.E_load
    eparts = m.E_out + m.E_inv
    print("   逆变器输入 %.1f kJ = 输出 %.1f + 逆变器/控制损耗 %.1f kJ (差 %.3f%%)；输出 = 工件 %.1f + 线圈铜损 %.1f kJ" % (
        ein / 1e3, m.E_out / 1e3, m.E_inv / 1e3, pct(ein, eparts), m.E_wp / 1e3, m.E_coil / 1e3))
    assert abs(pct(ein, eparts)) < 1.0
    r = m.done[0]
    Tt = m.th.T_surface(3.0)
    print("   加热 3 s：工件吸收 %.2f kJ（有限元表 5.944 kW×3 s = %.2f kJ），峰值温度 %.0f°C，表(t=3.0 s) %.0f°C；节拍 %.2f s" % (
        r["E_wp_kJ"], 5.944 * 3, r["T_peak"], Tt, m.done[1]["t_done"] - m.done[0]["t_done"]))
    assert abs(r["E_wp_kJ"] / (5.944 * 3) - 1) < 0.02 and abs(r["T_peak"] - Tt) < 15
    Pw = np.array(m.log["P_out"]); Pr = np.array(m.log["P_ref"]); run_ = np.array(m.log["reached"]) > 0
    err = np.abs(Pw[run_] - Pr[run_]) / Pr[run_]
    print("   变频器稳功率：到达设定后输出功率偏差 最大 %.1f%%（设定 %.1f kW）" % (100 * err.max(), sp.P_set_out / 1e3))
    assert err.max() < 0.06
    # (e) 缺水联锁反应时间：流量衰减到 50% 的时间 τ·ln2 + 0.5 s 定时器 + ≤1 扫描
    m2, S2 = SC.run("water_loss")
    pred = 0.3 * math.log(2) + 0.5
    print("[整机] 断水 → 高频关断：仿真 %.0f ms；预计 流量衰减 τ·ln2=%.0f ms + 定时器 500 ms ≈ %.0f ms (+≤2 扫描)" % (S2["t_hf_off"] * 1e3, 0.3 * math.log(2) * 1e3, pred * 1e3))
    assert pred - 0.02 < S2["t_hf_off"] < pred + 0.06
    # (f) 联锁被短接后线圈热平衡
    m4, S4 = SC.run("water_loss_bypassed")
    print("   联锁短接后线圈 %.0f°C（t=%.1f s 起损坏）；最后 1 件峰值温度 %.0f°C（线圈变热 → 铜损占比↑ → 同一输出功率下进工件的少）" % (
        m4.coil_max, m4.coil_damage_t or -1, m4.done[-1]["T_peak"]))
    assert m4.coil_damage_t is not None and m4.done[-1]["T_peak"] < m4.done[0]["T_peak"]
    # (g) 预充电不足的冲击电流判定
    m5, S5 = SC.run("precharge_short")
    print("[整机] 预充电只等 0.1 s：冲击电流 %.0f A（正常预充电 %.1f A）" % (S5["inrush_A"], S["inrush_A"]))
    assert S5["inrush_A"] > 50 * max(S["inrush_A"], 1e-3) and S5["inrush_A"] > 10 * sp.I_dc_rated
    # (h) 电网跌落：40% → 欠压跳闸；15% → 调频补偿
    m6, S6 = SC.run("mains_sag"); m7, S7 = SC.run("mains_sag_mild")
    print("[整机] 电网跌落 40%%：变频器 %s；跌落 15%%：%d 件全合格，进线电流 %.1f → %.1f A（理论 ×%.2f）" % (
        "欠压跳闸" if m6.ctl.fault_log and m6.ctl.fault_log[0][1] == "UV" else "未跳", S7["ok"], S7["I_src_pre"], S7["I_src_post"], 1 / 0.85))
    assert m6.ctl.fault_log and m6.ctl.fault_log[0][1] == "UV" and S7["ng"] == 0
    # (i) 长时间运行：有/无换热器
    m8, S8 = SC.run("longrun_no_hex"); m9, S9 = SC.run("longrun_hex")
    print("[整机] 8 小时后水箱：无换热器 %.1f°C（喷淋等效淬火 %.2f→%.2f s），有换热器 %.1f°C（稳态 %.1f°C）" % (
        S8["ff"]["T1"], m8.done[0]["tq"], m8.done[-1]["tq"], S9["ff"]["T1"], S9["ff"]["T_ss"]))
    assert S8["ff"]["T1"] > 45 > S9["ff"]["T1"]
    print("[整机] 自检通过")
