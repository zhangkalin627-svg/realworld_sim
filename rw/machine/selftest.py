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
    print("[整机] 合旁路冲击电流：RK4 vs ngspice（Rs=%.2f Ω, L=%.2f mH, C=%.0f mF）" % (sp.Rs, sp.L_line * 1e3, sp.C_bus * 1e3))
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
    # (c2) v0.30 issue 1823：预充/泄放改指数精确更新——任意 dt 不过冲；非法参数明确报错
    G = 1.0 / sp.R_pc + 1.0 / sp.R_bleed
    Vinf = (513.0 / sp.R_pc) / G
    th2 = Vinf * (1 - math.exp(-1.0 / (sp.C_bus / G)))
    vs = []
    for dt_big in (0.5, 1.0):
        b2 = Bus(sp); b2.set_contactors(True, False, 513.0)
        for _ in range(int(round(1.0 / dt_big))):
            b2.step(dt_big, 513.0, 0.0)
        assert abs(pct(b2.V, th2)) < 0.1, (dt_big, b2.V, th2)
        vs.append(b2.V)
    print("[整机] 预充大步长 dt=0.5/1 s：母线 %.2f/%.2f V，解析 %.2f V（指数精确不过冲，旧格式曾达 1295 V）" % (vs[0], vs[1], th2))
    for bad in (dict(R_pc=0.0), dict(R_bleed=0.0), dict(C_bus=0.0)):
        try:
            Bus(Spec(**bad)); _ok = False
        except ValueError:
            _ok = True
        assert _ok, bad
    print("[整机] R_pc/R_bleed/C_bus 为零或负：ValueError 明确报错（旧为 ZeroDivisionError）")
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
    tau = r["E_wp_kJ"] / 5.944
    Tt = m.th.T_surface(tau)
    print("   部分功率 %.0f kW 加热 %s：工件吸收 %.2f kJ（v0.8 有限元表 5.944 kW×3 s = %.2f kJ，%+.1f%%），能量等效 τ=%.2f s；峰值温度 %.0f°C，表(τ) %.0f°C；节拍 %.2f s" % (
        m.sp.P_set_out / 1e3, SC.HEAT_PART, r["E_wp_kJ"], 5.944 * 3, pct(r["E_wp_kJ"], 5.944 * 3), tau, r["T_peak"], Tt, m.done[1]["t_done"] - m.done[0]["t_done"]))
    assert abs(r["E_wp_kJ"] / (5.944 * 3) - 1) < 0.08 and abs(r["T_peak"] - Tt) < 15
    Pw = np.array(m.log["P_out"]); Pr = np.array(m.log["P_ref"]); run_ = np.array(m.log["reached"]) > 0
    err = np.abs(Pw[run_] - Pr[run_]) / Pr[run_]
    print("   变频器稳功率：到达设定后输出功率偏差 最大 %.1f%%（设定 %.1f kW）" % (100 * err.max(), m.sp.P_set_out / 1e3))
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
    # (j) 100 kW 满功率（2026-10-12 起整机默认 100 kW 通用机）
    run_100k()
    run_logcap()
    run_sanity()
    print("[整机] 自检通过")


def run_100k():
    """满功率：老化 8 h 不误动、冷水机不足必须动作（不拒动）、温升网络热平衡、负载短路硬件过流保护。"""
    from . import stress as ST
    sp = Spec()
    m, S = SC.run("fp_burnin")
    pr = S["pr"]; lr = S["lr"]; se = lr["series"]
    print("[整机·100kW] 满载老化：输出 %.1f kW，f/f0 %.3f，逆变器效率 %.1f%%，进线 PF %.3f；8 h 外推进水 %.1f°C、IGBT 结温 %.0f°C、变压器 %.0f°C，%s" % (
        pr["P_out"] / 1e3, pr["f"] / pr["f0"], 100 * pr["eta_inv"], pr["PF"], se["T_w"].max(), se["Tj"].max(), se["T_tr"].max(),
        "无保护动作" if lr["trip"] is None else "保护动作 " + lr["trip"][2]))
    assert abs(pr["P_out"] / 100e3 - 1) < 0.03, "满功率没出来"
    assert pr["f"] / pr["f0"] > 1.0, "失去感性负载（ZVS）"
    assert not m.ctl.fault_log and lr["trip"] is None, "满载老化不应保护动作（误动）"
    assert se["Tj"].max() < sp.mod.Tj_design
    # 满功率能量守恒（问题单 20261012-1820 要求）：电网 = 母线电容 + 电阻 + 泄放 + 逆变器输入；逆变器输入 = 输出 + 逆变器/控制损耗
    bus = m.bus
    e1 = pct(bus.E_src, bus.E_cap() + bus.E_R + bus.E_load + bus.E_bleed); e2 = pct(bus.E_load, m.E_out + m.E_inv)
    e3 = pct(m.E_out, m.E_wp + m.E_coil)
    print("   满功率能量守恒（详细仿真 %.0f s）：电网 %.2f MJ，母线侧差 %+.3f%%；逆变器输入 = 输出 + 损耗 差 %+.3f%%；输出 = 工件 %.2f MJ + 线圈侧 %.2f MJ 差 %+.3f%%" % (
        m.t, bus.E_src / 1e6, e1, e2, m.E_wp / 1e6, m.E_coil / 1e6, e3))
    assert abs(e1) < 1.0 and abs(e2) < 1.0 and abs(e3) < 1.0
    # 热平衡：外推末段冷却水回路 得热（器件+线圈+水箱换热）≈ 冷水机/散热带走
    La = lr["Lavg"]
    print("   平均损耗：逆变器 %.2f kW、整流 %.2f kW、变压器 %.2f kW、线圈侧 %.1f kW；冷水机末段 %.1f kW" % (
        La["inv_total"] / 1e3, La["rect"] / 1e3, (La["tr_cu"] + La["tr_core"]) / 1e3, La["P_coil"] / 1e3, se["Q_ch"][-1] / 1e3))
    m2, S2 = SC.run("fp_burnin_hot")
    tr = S2["lr"]["trip"]
    print("[整机·100kW] 40°C + 冷水机 20 kW：%s" % ("第 %.2f h %s" % (tr[0], tr[2]) if tr else "8 h 无保护动作（拒动！）"))
    assert tr is not None, "冷却能力不足时保护拒动"
    m3, S3 = SC.run("fp_short")
    hw = S3["tr"]["hw"]
    print("[整机·100kW] 满功率负载短路 + 硬件过流/电容过压双比较器：电流峰值 %.0f A（I_CRM %.0f A），谐振电容峰值 %.0f V（额定峰值 2828 V，超额定 %.0f%%——快速关断压不住，选型须按 3.6 kV+ 或加吸收回路，issue 1821 如实说明）" % (hw["I_pk_after"], sp.mod.I_crm, hw["Vc_pk"], 100 * (hw["Vc_pk"] / 2828.0 - 1)))
    assert hw["I_pk_after"] < sp.mod.I_crm
    assert hw.get("trip_cause") == "VC", hw.get("trip_cause")  # v0.30：电容过压比较器应先于过流比较器动作


def run_logcap():
    """#2031 回归：日志行数有界（降采样），事件不丢。"""
    from rw.machine import scenarios
    m = scenarios.build("normal"); m.LOG_MAX = 500
    m.run(30.0)
    n = len(m.log["t"])
    assert n <= 500 and len(m.ylog) == n and len(m.xlog) == n and all(len(v) == n for v in m.log.values()), n
    assert m.events, "事件丢失"
    print("[整机] 日志有界：3000 扫描 → %d 行（上限 500），stride=%d，事件 %d 条" % (n, m._log_stride, len(m.events)))
    m2 = scenarios.build("normal")
    for i in range(25000):
        m2.ev("synthetic-%d" % i)
    assert len(m2.events) <= 2 * m2.EV_MAX + 1, len(m2.events)
    assert m2.events[-1][1] == "synthetic-24999", "近期事件应保留"
    print("[整机] 事件有界：25000 次合成事件 → %d 条（上限 2×%d），最新事件保留（issue 2031）" % (len(m2.events), m2.EV_MAX))


def run_sanity():
    """#1823 回归：非法参数构造时报错。"""
    from rw.machine.machine import Machine
    from rw.machine.power import Spec
    import dataclasses
    for kw, name in (({"spec": dataclasses.replace(Spec(), P_set_out=0)}, "P=0"),
                     ({"spec": dataclasses.replace(Spec(), P_set_out=-5e3)}, "P<0"),
                     ({"T_tank0": 200}, "T_tank0=200"), ({"p_air": -1.0}, "p_air<0")):
        try:
            Machine(**kw)
        except ValueError:
            continue
        raise AssertionError(name + " 未被拒绝")
    Machine()
    print("[整机] 参数校验：P_set_out≤0、p_air<0、T_tank0≥100°C 均抛 ValueError")
