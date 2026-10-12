# -*- coding: utf-8 -*-
"""整机联动场景：每个场景 = 机器配置 + 时间表上的操作/故障 + 结束条件 + 结论判定。"""
import numpy as np
from .machine import Machine
from .plc_prog import MACHINE_IL, without_water_interlock, no_wait_for_coil

SCENARIOS = {
    "normal": dict(name="正常：上电 → 连续加工 5 件", desc="冷却水 → 预充电 → 旁路 → 变频器就绪 → 循环加工。", n_parts=5),
    "precharge_short": dict(name="故障：预充电时间设太短（0.1 s）", desc="PLC 预充电只等 0.1 s 就合旁路，母线还没充起来，合闸瞬间有大冲击电流。", n_parts=2),
    "water_loss": dict(name="故障：加热中冷却水中断（有水流联锁）", desc="第 2 件加热时水断流；流量开关 0.5 s 后 PLC 停高频。", n_parts=4),
    "water_loss_bypassed": dict(name="故障：冷却水中断 + 联锁被短接", desc="现场常见的“嫌报警烦就短接”：缺水后继续加工，看线圈几件后坏。", n_parts=9),
    "mains_sag": dict(name="故障：电网跌落 40%", desc="第 2 件加热中电网跌到 60%：母线欠压，变频器跳闸，PLC 报警，工件半途。", n_parts=3),
    "mains_sag_mild": dict(name="电网跌落 15%（变频器能补偿）", desc="母线电压低了，变频器靠调频稳住功率，工件结果不变，但进线电流变大。", n_parts=4),
    "air_low": dict(name="故障：加工中气压降到 0.25 MPa", desc="气压开关（0.35 MPa）动作，PLC 报警停机。", n_parts=3),
    "no_wait_coil": dict(name="PLC 编程错误：线圈下降时就开高频", desc="没等下限位就通高频：看软启动和过流保护能不能兜住，工件多受了多少热。", n_parts=2),
    "longrun_no_hex": dict(name="长时间运行：没有换热器，8 小时后", desc="先跑 4 件，快进 8 小时（水箱升温），再跑 4 件，对比硬度与喷淋。", n_parts=4),
    "longrun_hex": dict(name="长时间运行：有换热器，8 小时后", desc="同上，但冷却水带换热器 UA=500 W/K。", n_parts=4),
}


def build(key, p_air=0.55, sag=1.0, il=None, T_tank0=25.0, UA=500.0, **kw):
    il = il or MACHINE_IL
    sc = SCENARIOS[key]
    m = None
    if key == "precharge_short":
        il = il.replace("MOV K20 D202", "MOV K1 D202")
    elif key == "water_loss_bypassed":
        il = without_water_interlock(il)
    elif key == "no_wait_coil":
        il = no_wait_for_coil(il)
    if key == "longrun_no_hex":
        UA = 0.0
    m = Machine(il=il, p_air=p_air, sag=sag, UA_hex=UA, T_tank0=T_tank0, **kw)
    m.press("X000", 0.5)
    m.scenario = key
    return m


def _stop_n(n, tmax):
    return lambda mm: len(mm.done) >= n or mm.t > tmax


def run(key, **kw):
    """返回 (machine, summary)。"""
    sc = SCENARIOS[key]
    n = sc["n_parts"]
    m = build(key, **kw)
    S = {"key": key}
    if key in ("normal", "precharge_short"):
        m.run(150.0, _stop_n(n, 60.0))
        m.run(m.t + 2.0)
    elif key in ("water_loss", "water_loss_bypassed"):
        # 在第 2 件加热开始后 1.2 s 断水
        m.run(60.0, lambda mm: mm.part is not None and mm.part.id == 2 and mm.part.t_heat0 is not None and mm.t - mm.part.t_heat0 > 1.2)
        t_f = m.t
        m.cw_supply_ok = False; m.ev("★ 注入故障：冷却水断流"); S["t_fault"] = t_f
        m.run(t_f + (60.0 if key == "water_loss" else 140.0), lambda mm: (mm.coil_damage_t is not None and mm.t - mm.coil_damage_t > 3) or
              (key == "water_loss" and mm.t - t_f > 8.0) or len(mm.done) >= n + 2)
    elif key in ("mains_sag", "mains_sag_mild"):
        m.run(60.0, lambda mm: mm.part is not None and mm.part.id == 2 and mm.part.t_heat0 is not None and mm.t - mm.part.t_heat0 > 1.0)
        t_f = m.t; S["t_fault"] = t_f
        m.sag = 0.60 if key == "mains_sag" else 0.85
        m.ev(f"★ 注入故障：电网跌到 {m.sag*100:.0f}%")
        m.run(t_f + 40.0, lambda mm: (key == "mains_sag" and mm.t - t_f > 12.0) or len(mm.done) >= n)
    elif key == "air_low":
        m.run(60.0, lambda mm: mm.part is not None and mm.part.id == 2 and mm.part.t_heat0 is not None and mm.t - mm.part.t_heat0 > 1.0)
        t_f = m.t; S["t_fault"] = t_f
        m.p_air = 0.25; m._cyl_times(); m.ev("★ 注入故障：气压降到 0.25 MPa")
        m.run(t_f + 15.0)
    elif key == "no_wait_coil":
        m.run(30.0, lambda mm: mm.ctl.fault != "" and mm.t > 3)
        m.run(m.t + 3.0)
    elif key in ("longrun_no_hex", "longrun_hex"):
        m.run(150.0, _stop_n(n, 60.0))
        S["ff"] = m.fast_forward(8.0)
        n0 = len(m.done); S["n_before"] = n0
        m.run(m.t + 150.0, lambda mm: len(mm.done) >= n0 + n or mm.t > 1e9)
        m.run(m.t + 1.0)
    return m, summarize(m, S)


def key_is(S, k):
    return S.get('key') == k


def summarize(m, S=None):
    S = S or {}
    L = {k: np.array(v) for k, v in m.log.items()}
    t = L["t"]
    S["t_end"] = float(t[-1])
    ok = sum(1 for r in m.done if r["verdict"] == "OK"); ng = sum(1 for r in m.done if r["verdict"] == "NG"); sc = sum(1 for r in m.done if r["verdict"] == "中断报废")
    S.update(ok=ok, ng=ng, scrap=sc, n_done=len(m.done))
    # 上电时间线
    def first(cond):
        k = np.nonzero(cond)[0]
        return float(t[k[0]]) if len(k) else None
    S["t_bus90"] = first(L["Vbus"] >= m.sp.V_ready)
    S["t_ready"] = first(L["ready"] > 0)
    S["inrush_A"] = max([e["I_peak"] for e in m.bus.inrush_events], default=0.0)
    S["inrush_dV"] = max([e["dV"] for e in m.bus.inrush_events], default=0.0)
    S["coil_max"] = m.coil_max
    S["coil_dmg"] = m.coil_damage_t
    S["I_pk_max"] = float(L["I_pk"].max())
    S["I_src_max"] = float(L["I_src"].max())
    S["T_tank_end"] = float(L["T_tank"][-1])
    S["alarms"] = [e for e in m.events if "报警" in e[1] and "ON" in e[1] and "报警总" not in e[1]]
    S["fault_log"] = list(m.ctl.fault_log)
    tf = S.get("t_fault")
    if tf is not None and (m.ctl.fault_log or S["alarms"]):
        yo = np.array(m.ylog)
        k = np.nonzero((t > tf) & (yo[:, 4] == 0))[0]
        S["t_hf_off"] = float(t[k[0]]) - tf if len(k) else None
    if tf is not None:
        pre = (t > 3.0) & (t < tf); post = t > tf
        S["I_src_pre"] = float(L["I_src"][pre].max()) if pre.any() else 0.0
        S["I_src_post"] = float(L["I_src"][post].max()) if post.any() else 0.0
    # 判定
    v = []
    if S["inrush_A"] > 10 * m.sp.I_dc_rated:
        v.append(("FAIL", f"合旁路瞬间冲击电流 {S['inrush_A']:.0f} A（> 10× 额定 {m.sp.I_dc_rated:.0f} A）：熔断器/接触器触点有熔焊风险"))
    elif S["inrush_A"] > 5 * m.sp.I_dc_rated:
        v.append(("WARN", f"合旁路瞬间冲击电流 {S['inrush_A']:.0f} A（> 5× 额定）"))
    elif m.bus.inrush_events:
        v.append(("OK", f"合旁路冲击电流 {S['inrush_A']:.1f} A（母线已充到位）"))
    if m.coil_damage_t is not None:
        v.append(("FAIL", f"线圈 {m.coil_max:.0f}°C，超过 {m.sp.coil_fail:.0f}°C，t = {m.coil_damage_t:.1f} s 起视为损坏"))
    elif m.coil_max > m.sp.coil_warn:
        v.append(("WARN", f"线圈最高 {m.coil_max:.0f}°C（> {m.sp.coil_warn:.0f}°C，水垢/老化加速）"))
    else:
        v.append(("OK", f"线圈最高 {m.coil_max:.0f}°C"))
    Tk = float(L["T_tank"][-1])
    if "ff" in S:
        if Tk > 45:
            v.append(("FAIL", f"淬火液升到 {Tk:.0f}°C（>45°C）：喷淋等效淬火时间 {m.done[-1]['tq']:.2f} s（原 {m.done[0]['tq']:.2f} s），出口温度 {m.done[-1]['T_exit']:.0f}°C，冷却能力明显下降"))
        elif Tk > 38:
            v.append(("WARN", f"淬火液 {Tk:.0f}°C，冷却能力开始下降"))
        else:
            v.append(("OK", f"快进后淬火液 {Tk:.0f}°C，在控制范围内"))
    if m.ctl.fault_log:
        t0, code, txt = m.ctl.fault_log[0]
        v.append(("WARN", f"变频器在 t = {t0:.2f} s 跳闸：{txt}"))
    if S["alarms"]:
        v.append(("WARN", "PLC 报警：" + "；".join(f"{e[1].replace('PLC ','').replace(' ON','')}（{e[0]:.1f} s）" for e in S["alarms"])))
    if S["ng"]:
        v.append(("FAIL", f"{S['ng']} 件 NG：" + "；".join(f"P{r['id']} {r['reason']}" for r in m.done if r["verdict"] == "NG")))
    if S["scrap"]:
        v.append(("WARN", f"{S['scrap']} 件中断报废"))
    if S["ok"] and not S["ng"] and not S["scrap"]:
        v.append(("OK", f"{S['ok']} 件全部合格（峰值 {m.done[0]['T_peak']:.0f}°C，HRC {m.done[0]['HRC']:.0f}）"))
    if key_is(S, "mains_sag_mild") and S.get("I_src_post"):
        v.append(("OK", f"跌落后进线直流电流 {S['I_src_pre']:.1f} → {S['I_src_post']:.1f} A（功率不变，电压低电流大）"))
    if tf is not None and S.get("t_hf_off") is not None:
        v.append(("OK" if S["t_hf_off"] < 1.0 else "WARN", f"故障到高频关断 {S['t_hf_off']*1e3:.0f} ms"))
    S["verdicts"] = v
    return S
