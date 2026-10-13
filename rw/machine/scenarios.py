# -*- coding: utf-8 -*-
"""整机联动场景：每个场景 = 机器配置 + 时间表上的操作/故障 + 结束条件 + 结论判定。
2026-10-12 起整机电源默认 100 kW（用户定）：
- 原 10 个场景（演示工件）按“部分功率 12 kW”运行（工件只要 10–14 kW），加热时间相应缩短；
- 新增 fp_* 满功率/近满功率场景（满载老化 8 h、夏天冷水机偏小、冷却水流量下降、过居里点过流定值误动、负载短路、电网跌落），
  以及 pp_8h 部分功率 12 kW 连续生产 8 h（效率/功率因数/控制分辨率/功率循环寿命）。"""
import math
import numpy as np
from .machine import Machine
from .power import Spec
from .plc_prog import MACHINE_IL, without_water_interlock, no_wait_for_coil
from . import stress as ST

P_PART = 12e3          # 部分功率工况：输出 12 kW（工件约 10.4 kW）
HEAT_PART = "K17"      # 对应加热时间 1.7 s（能量与 v0.8 的 5.94 kW×3 s 相当）

SCENARIOS = {
    "normal": dict(name="正常：上电 → 连续加工 5 件（部分功率 12 kW）", desc="冷却水 → 预充电 → 旁路 → 变频器就绪 → 循环加工。100 kW 电源带 12 kW 工件。", n_parts=5),
    "precharge_short": dict(name="故障：预充电时间设太短（0.1 s）", desc="PLC 预充电只等 0.1 s 就合旁路，母线还没充起来，合闸瞬间有大冲击电流。", n_parts=2),
    "water_loss": dict(name="故障：加热中冷却水中断（有水流联锁）", desc="第 2 件加热时水断流；流量开关 0.5 s 后 PLC 停高频。", n_parts=4),
    "water_loss_bypassed": dict(name="故障：冷却水中断 + 联锁被短接", desc="现场常见的“嫌报警烦就短接”：缺水后继续加工，看线圈几件后坏（100 kW 机线圈热容大，12 kW 工况要几分钟）。", n_parts=60),
    "mains_sag": dict(name="故障：电网跌落 40%", desc="第 2 件加热中电网跌到 60%：母线欠压，变频器跳闸，PLC 报警，工件半途。", n_parts=3),
    "mains_sag_mild": dict(name="电网跌落 15%（变频器能补偿）", desc="母线电压低了，变频器靠调频稳住功率，工件结果不变，但进线电流变大。", n_parts=4),
    "air_low": dict(name="故障：加工中气压降到 0.25 MPa", desc="气压开关（0.35 MPa）动作，PLC 报警停机。", n_parts=3),
    "no_wait_coil": dict(name="PLC 编程错误：线圈下降时就开高频", desc="没等下限位就通高频：看软启动和过流保护能不能兜住，工件多受了多少热。", n_parts=2),
    "longrun_no_hex": dict(name="长时间运行：淬火水箱没有换热器，8 小时后", desc="先跑 4 件，快进 8 小时（水箱升温），再跑 4 件，对比硬度与喷淋。", n_parts=4),
    "longrun_hex": dict(name="长时间运行：淬火水箱有换热器，8 小时后", desc="同上，但淬火水箱带换热器 UA=500 W/K。", n_parts=4),
    # ---- 100 kW 满功率 / 近满功率
    "fp_burnin": dict(name="【满功率】100 kW 满载老化 8 h（夏天 35°C，冷水机 30 kW）", desc="假负载（热态 850°C 工件，阻抗恒定）连续 100 kW：详细仿真 60 s，再把冷却水/器件温升外推 8 h。看各器件温升、余量、保护会不会误动。", n_parts=0),
    "fp_burnin_hot": dict(name="【满功率】8 h 老化：40°C 高温 + 冷水机只有 20 kW", desc="冷水机选小了/夏天降额：冷却水慢慢升温，看第几小时进水温度或 IGBT 过温保护动作（应该动作，不该拒动）。", n_parts=0),
    "fp_flow60": dict(name="【满功率】冷却水流量降到 60%（滤网堵）", desc="流量开关定在 50%，不会报警：看线圈/IGBT/变压器温升还有没有余量，过温保护能否兜底。", n_parts=0),
    "fp_curie": dict(name="【满功率】大工件冷态起步过居里点（过流定值 480 A）", desc="2 kg 钢件从冷态 100 kW 加热：冷态磁性阻抗大功率上不去，过居里点后阻抗骤降、电流冲高，变频器追功率。", n_parts=0),
    "fp_curie_tight": dict(name="【满功率·误动】同上，但过流定值整定太紧（= 额定峰值，不留余量）", desc="过居里点那一下电流超出定值 → 过流误跳，工件半途。", n_parts=0),
    "fp_short": dict(name="【满功率·拒动】满功率时工件碰线圈（负载短路）", desc="负载电阻突然只剩线圈：微秒级看硬件过流比较器 vs 只靠 1 kHz 软件保护；毫秒级看整机。", n_parts=0),
    "fp_sag40": dict(name="【满功率】满功率时电网跌落 40%", desc="100 kW 时母线电容只够撑几毫秒：欠压跳闸时间、母线最低电压。", n_parts=0),
    "fp_sag15": dict(name="【满功率】满功率时电网跌落 15%", desc="电压低 15 % 时 100 kW 还能不能出：功率降额、频率贴到下限、进线电流。", n_parts=0),
    "pp_8h": dict(name="【部分功率】12 kW 连续生产 8 h（效率/功率因数/分辨率/寿命）", desc="100 kW 电源干 12 kW 的活：效率、进线功率因数、关断电流、频率分辨率对应的功率分辨率；8 h 温升与 IGBT 功率循环寿命。", n_parts=5),
}


def part_spec(**kw):
    d = dict(P_set_out=P_PART); d.update(kw)
    return Spec(**d)


def build(key, p_air=0.55, sag=1.0, il=None, T_tank0=25.0, UA=500.0, **kw):
    il = il or MACHINE_IL
    sc = SCENARIOS[key]
    spec = kw.pop("spec", None)
    T_amb = kw.pop("T_amb", 35.0)
    if key.startswith("fp_"):
        il = il.replace("MOV K30 D200", "MOV K32000 D200")
        if spec is None:
            spec = Spec()
        load = "heavy" if key.startswith("fp_curie") else "burnin"
        if key == "fp_curie_tight":
            # 定值 = 额定工作点峰值电流（热态等效总电阻约 1.51 Ω），不留余量。
            # （第一版取 1.05×：仿真里过居里点只冲高约 2 %，不误动 → 改为不留余量才复现误动。）
            spec.I_trip_pk = round(math.sqrt(2) * math.sqrt(spec.P_set_out / 1.51))
        if key == "fp_burnin_hot":
            spec.cool.Q_chiller = 20e3; T_amb = 40.0
        m = Machine(il=il, spec=spec, p_air=p_air, sag=sag, UA_hex=UA, T_tank0=T_tank0, load=load, T_amb=T_amb, **kw)
        m.press("X000", 0.5); m.scenario = key
        return m
    il = il.replace("MOV K30 D200", "MOV " + HEAT_PART + " D200")
    if key == "precharge_short":
        il = il.replace("MOV K20 D202", "MOV K1 D202")
    elif key == "water_loss_bypassed":
        il = without_water_interlock(il)
    elif key == "no_wait_coil":
        il = no_wait_for_coil(il)
    if key == "longrun_no_hex":
        UA = 0.0
    m = Machine(il=il, spec=spec or part_spec(), p_air=p_air, sag=sag, UA_hex=UA, T_tank0=T_tank0, T_amb=T_amb, **kw)
    m.press("X000", 0.5)
    m.scenario = key
    return m


def _stop_n(n, tmax):
    return lambda mm: len(mm.done) >= n or mm.t > tmax


def _heat_started(mm, pid=2, after=1.0):
    return mm.part is not None and mm.part.id == pid and mm.part.t_heat0 is not None and mm.t - mm.part.t_heat0 > after


def _reached_full(mm):
    return mm.ctl.reached


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
        m.run(60.0, lambda mm: _heat_started(mm, 2, 0.6))
        t_f = m.t
        m.cw_supply_ok = False; m.ev("★ 注入故障：冷却水断流"); S["t_fault"] = t_f
        m.run(t_f + (60.0 if key == "water_loss" else 480.0), lambda mm: (mm.coil_damage_t is not None and mm.t - mm.coil_damage_t > 3) or
              (key == "water_loss" and mm.t - t_f > 8.0) or len(mm.done) >= n + 2)
    elif key in ("mains_sag", "mains_sag_mild"):
        m.run(60.0, lambda mm: _heat_started(mm, 2, 0.6))
        t_f = m.t; S["t_fault"] = t_f
        m.sag = 0.60 if key == "mains_sag" else 0.85
        m.ev(f"★ 注入故障：电网跌到 {m.sag*100:.0f}%")
        m.run(t_f + 40.0, lambda mm: (key == "mains_sag" and mm.t - t_f > 12.0) or len(mm.done) >= n)
    elif key == "air_low":
        m.run(60.0, lambda mm: _heat_started(mm, 2, 0.6))
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
    elif key in ("fp_burnin", "fp_burnin_hot", "fp_flow60"):
        if key == "fp_flow60":
            m.at(1.0, lambda: setattr(m, "Q_cw_supply_frac", 0.6), "★ 冷却水流量降到 60%（滤网堵）")
        m.run(40.0, lambda mm: mm.ctl.reached and mm.t > 8.0)
        m.run(m.t + 20.0); m.mark()
        m.run(m.t + 20.0); m.mark()
        S["detail_end"] = m.t
        S["lr"] = m.longrun_thermal(8.0, T_amb=m.net.T_amb)
    elif key in ("fp_curie", "fp_curie_tight"):
        m.run(45.0, lambda mm: (mm.part.T > 900 and mm.t > 5) or (mm.ctl.fault != "" and mm.t > 3))
        m.run(m.t + 2.0)
    elif key == "fp_short":
        m.run(40.0, lambda mm: mm.ctl.reached and mm.t > 8.0)
        m.run(m.t + 2.0)
        t_f = m.t; S["t_fault"] = t_f
        S["pre"] = dict(m._s_last); S["Vbus_pre"] = m.bus.V
        m.stage.short = True; m.ev("★ 注入故障：工件碰线圈，负载电阻只剩线圈铜损")
        m.run(t_f + 3.0)
        S["tr"] = short_transients(m.sp, S["pre"], S["Vbus_pre"])
    elif key in ("fp_sag40", "fp_sag15"):
        m.run(40.0, lambda mm: mm.ctl.reached and mm.t > 8.0)
        m.run(m.t + 2.0)
        t_f = m.t; S["t_fault"] = t_f
        S["P_pre"] = m._s_last["P_out"]; S["I_src_pre"] = m.bus.I_src
        m.sag = 0.60 if key == "fp_sag40" else 0.85
        m.ev(f"★ 注入故障：电网跌到 {m.sag*100:.0f}%")
        m.run(t_f + 6.0)
        S["P_post"] = m._s_last["P_out"]; S["I_src_post"] = m.bus.I_src; S["f_post"] = m.ctl.f_q
    elif key == "pp_8h":
        m.run(150.0, lambda mm: len(mm.done) >= 2); m.mark(); n0 = len(m.done); t0 = m.t
        m.run(200.0, lambda mm: len(mm.done) >= n0 + 3); m.mark()
        S["cycle"] = (m.t - t0) / 3
        L = m.log; t = np.array(L["t"]); Tj = np.array(L["Tj"])
        last = t >= m.t - S["cycle"]
        S["Tj_swing"] = float(Tj[last].max() - Tj[last].min()); S["Tj_cyc_mean"] = float(Tj[last].mean())
        S["lr"] = m.longrun_thermal(8.0, T_amb=m.net.T_amb)
    return m, summarize(m, S)


def short_transients(sp, s_pre, Vbus):
    """满功率负载短路的微秒级瞬态：硬件比较器 vs 只靠软件环（1 kHz）vs 不保护。"""
    from .power import Stage
    st = Stage(sp)
    T, k = 850.0, 1.0
    L = st.L_of(T, k); Rc, Rw = st.R_parts(T, k, 40.0)
    f = s_pre.get("f", 25e3)
    res = {}
    for prot in ("hw", "sw", "none"):
        # v0.30 issue 1821：硬件路径加谐振电容过压比较器（动作点 = 额定峰值的 95%）。
        # 只靠过流比较器时短路后电容冲到 3157 V > 额定峰值 2828 V；加快压比较器后约 3130 V，
        # 仍超额定 10%——快速关断无法完全压住（关断后电感储能继续灌入电容），残差如实进 verdict
        vc_peak_rated = getattr(getattr(sp, "cool", None), "Vc_rated", 2000.0) * math.sqrt(2)  # v0.30.1：属性名是 cool 不是 cs（Pally 审查发现，原先静默落到默认值）
        res[prot] = ST.fault_transient(L, sp.C_res, Rc + Rw, Vbus, f, L_short=1e-9 + 0.0 * L, R_short=Rw, prot=prot,
                                       I_trip=sp.I_trip_pk, t_end=1.8e-3, dt=25e-9,
                                       V_trip=(0.95 * vc_peak_rated if prot == "hw" else None))
    return res


def key_is(S, k):
    return S.get('key') == k


def power_report(m, S):
    """工作点的效率链、功率因数、器件应力、控制分辨率、杂散磁场（估算）。按最近一个满功率/加热窗口取平均。"""
    sp = m.sp; L = {k: np.array(v, float) for k, v in m.log.items()}
    run = (L["run"] > 0) & (L["reached"] > 0)
    if S.get("t_fault") is not None:          # 故障场景：报告故障前的工作点
        run &= L["t"] < S["t_fault"]
    if not run.any():
        return None
    short0 = m.stage.short; m.stage.short = False
    P_out = float(L["P_out"][run].mean() * 1e3)
    i = np.nonzero(run)[0][-1]
    s = m._s_last if m._s_last.get("I", 0) > 0 else None
    # 用记录的最后一个“到达”点重算工作点
    f = float(L["f"][i] * 1e3)
    T_wp = m.part.T if m.part is not None else 850.0
    k = 1.0
    st = m.stage.solve(float(L["Vbus"][i]), f, T_wp if np.isfinite(T_wp) else 850.0, k, float(L["T_coil"][i]))
    Lz = ST.component_losses(sp, st, float(L["Vbus"][i]), f, st["P_in"], m.k_rip() * st["P_in"] / float(L["Vbus"][i]))
    rec = ST.rectifier_pf(int(round(st["P_in"] / 1e3)), V_ll=sp.V_ll, L=sp.L_line, Rs=sp.Rs, C=sp.C_bus)
    P_grid = rec["P"] + Lz["rect"]
    eta_inv = st["P_out"] / st["P_in"]
    eta_tot = st["P_wp"] / (st["P_in"] + Lz["rect"] + Lz["tr_core"])
    # 频率分辨率 → 功率分辨率
    df = f * f / sp.f_clk if sp.f_clk > 0 else 0.0
    s2 = m.stage.solve(float(L["Vbus"][i]), f + max(df, 1.0), T_wp if np.isfinite(T_wp) else 850.0, k, float(L["T_coil"][i]))
    dP = abs(s2["P_out"] - st["P_out"]) * (df / max(df, 1.0)) if df > 0 else 0.0
    I_coil = st["I"] * sp.cool.n_tr
    sf = ST.stray_field(I_coil, f)
    m.stage.short = short0
    return dict(P_out=st["P_out"], P_wp=st["P_wp"], P_coil=st["P_coil"], P_in=st["P_in"], P_grid=P_grid, eta_inv=eta_inv, eta_tot=eta_tot,
                f=f, f0=st["f0"], phi=Lz["phi_deg"], I=st["I"], I_pk=st["I_pk"], I_off=Lz["I_off"], P_inv=Lz["inv_total"], P_sw=Lz["inv_sw"],
                Vc=Lz["Vc"], Qc=Lz["Qc"], P_cap=Lz["cap"], tr=Lz["tr_cu"] + Lz["tr_core"], rect=Lz["rect"], PF=rec["PF"], THD=rec["THD"], DPF=rec["DPF"],
                I_line=rec["I_rms"], df=df, dP=dP, I_coil=I_coil, stray=sf, mode=Lz["mode"], I_can=Lz["I_can"])


def summarize(m, S=None):
    S = S or {}
    sp = m.sp
    L = {k: np.array(v) for k, v in m.log.items()}
    t = L["t"]
    S["t_end"] = float(t[-1])
    ok = sum(1 for r in m.done if r["verdict"] == "OK"); ng = sum(1 for r in m.done if r["verdict"] == "NG"); sc = sum(1 for r in m.done if r["verdict"] == "中断报废")
    S.update(ok=ok, ng=ng, scrap=sc, n_done=len(m.done))
    def first(cond):
        k = np.nonzero(cond)[0]
        return float(t[k[0]]) if len(k) else None
    S["t_bus90"] = first(L["Vbus"] >= sp.V_ready)
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
    S["Vbus_min_run"] = float(L["Vbus"][t > 3].min()) if (t > 3).any() else 0.0
    tf = S.get("t_fault")
    if tf is not None and (m.ctl.fault_log or S["alarms"]):
        yo = np.array(m.ylog)
        k = np.nonzero((t > tf) & ((yo[:, 4] == 0) | (L["run"] == 0)))[0]
        S["t_hf_off"] = float(t[k[0]]) - tf if len(k) else None
    if tf is not None:
        pre = (t > 3.0) & (t < tf); post = t > tf
        S["I_src_pre"] = S.get("I_src_pre") or (float(L["I_src"][pre].max()) if pre.any() else 0.0)
        S["I_src_post_max"] = float(L["I_src"][post].max()) if post.any() else 0.0
        S.setdefault("I_src_post", S["I_src_post_max"])
    if m.net is not None:
        S["net_max"] = dict(m.net.max)
    S["pr"] = power_report(m, S) if sp.use_device_model else None
    # 判定
    v = []
    if S["inrush_A"] > 10 * sp.I_dc_rated:
        v.append(("FAIL", f"合旁路瞬间冲击电流 {S['inrush_A']:.0f} A（> 10× 额定 {sp.I_dc_rated:.0f} A）：熔断器/接触器触点有熔焊风险"))
    elif S["inrush_A"] > 5 * sp.I_dc_rated:
        v.append(("WARN", f"合旁路瞬间冲击电流 {S['inrush_A']:.0f} A（> 5× 额定）"))
    elif m.bus.inrush_events:
        v.append(("OK", f"合旁路冲击电流 {S['inrush_A']:.1f} A（母线已充到位）"))
    if m.coil_damage_t is not None:
        v.append(("FAIL", f"线圈 {m.coil_max:.0f}°C，超过 {sp.coil_fail:.0f}°C，t = {m.coil_damage_t:.1f} s 起视为损坏"))
    elif m.coil_max > sp.coil_warn:
        v.append(("WARN", f"线圈最高 {m.coil_max:.0f}°C（> {sp.coil_warn:.0f}°C，水垢/老化加速）"))
    else:
        v.append(("OK", f"线圈最高 {m.coil_max:.0f}°C（详细仿真段）"))
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
    v += _verdicts_100k(m, S)
    S["verdicts"] = v
    return S


def _verdicts_100k(m, S):
    sp = m.sp; v = []
    if not sp.use_device_model:
        return v
    mod, cs = sp.mod, sp.cool
    pr = S.get("pr")
    key = S.get("key", "")
    if pr:
        tag = "满功率" if pr["P_out"] > 0.8 * sp.rated_kW * 1e3 else "部分功率"
        v.append(("OK", f"[{tag}工作点] 输出 {pr['P_out']/1e3:.1f} kW，f {pr['f']/1e3:.2f} kHz（f/f0 {pr['f']/pr['f0']:.3f}），相角 {pr['phi']:.0f}°，"
                        f"电流 {pr['I']:.0f} A 有效 / {pr['I_pk']:.0f} A 峰值，关断电流 {pr['I_off']:.0f} A（{pr['mode']}）"))
        lv = "OK" if pr["eta_tot"] > 0.75 else "WARN"
        v.append((lv, f"效率：逆变器 {100*pr['eta_inv']:.1f} %（器件损耗 {pr['P_inv']/1e3:.2f} kW，其中关断 {pr['P_sw']/1e3:.2f} kW）；"
                      f"电网→工件 {100*pr['eta_tot']:.1f} %（线圈/引线/变压器铜损 {pr['P_coil']/1e3:.1f} kW）"))
        lv = "OK" if pr["PF"] >= 0.90 else "WARN"
        v.append((lv, f"进线功率因数 {pr['PF']:.3f}（位移 {pr['DPF']:.3f}，电流 THD {100*pr['THD']:.0f} %），进线电流 {pr['I_line']:.0f} A 有效"))
        if pr["df"] > 0:
            rel = pr["dP"] / max(pr["P_out"], 1)
            lv = "OK" if rel < 0.01 else "WARN"
            v.append((lv, f"控制分辨率：计数时钟 {sp.f_clk/1e6:.0f} MHz → 频率步长 {pr['df']:.1f} Hz → 功率步长 {pr['dP']:.0f} W（{100*rel:.2f} % 设定）"))
        # 应力
        r_i = pr["I_pk"] / sp.I_trip_pk
        v.append(("OK" if r_i < 0.9 else "WARN", f"过流余量：峰值 {pr['I_pk']:.0f} A / 定值 {sp.I_trip_pk:.0f} A = {100*r_i:.0f} %（模块 I_CRM {mod.I_crm:.0f} A）"))
        lv = "OK" if pr["Vc"] < cs.Vc_rated and pr["Qc"] < cs.Qc_rated else "FAIL"
        v.append((lv, f"谐振电容：{pr['Vc']:.0f} V 有效（额定 {cs.Vc_rated:.0f}）、{pr['Qc']/1e3:.0f} kvar（额定 {cs.Qc_rated/1e3:.0f}），介损 {pr['P_cap']:.0f} W"))
        sf = pr["stray"]
        worst = max(sf["rows"], key=lambda r: r["dT"] / max(r["T_lim"], 1))
        v.append(("OK" if worst["dT"] + 35 < worst["T_lim"] else "WARN",
                  f"杂散磁场（线圈电流 {pr['I_coil']/1e3:.2f} kA，匝比 {cs.n_tr:g} 假设）：{worst['name']} {worst['P']:.0f} W，稳态温升 {worst['dT']:.0f} K；"
                  f"人员 80 A/m（ICNIRP 职业）距离 {sf['d_occ']:.2f} m，21 A/m（公众）{sf['d_pub']:.2f} m"))
    nm = S.get("net_max")
    if nm:
        Tj = nm["Tj_T"]
        lv = "OK" if Tj < mod.Tj_design else ("WARN" if Tj < mod.Tj_max else "FAIL")
        v.append((lv, f"详细仿真段最高：IGBT 结温 {Tj:.0f}°C（设计 ≤{mod.Tj_design:.0f}），基板 {nm['T_plate']:.0f}°C（NTC 跳 {mod.T_ntc_trip:.0f}），"
                      f"进水 {nm['T_w']:.1f}°C，变压器 {nm['T_tr']:.0f}°C，谐振电容 {nm['T_cap']:.0f}°C"))
    lr = S.get("lr")
    if lr:
        se = lr["series"]
        tr = lr["trip"]
        msg = (f"8 h 外推：进水 {se['T_w'].max():.1f}°C，IGBT 结温最高 {se['Tj'].max() + S.get('Tj_swing', 0)/2:.0f}°C，变压器 {se['T_tr'].max():.0f}°C，"
               f"谐振电容 {se['T_cap'].max():.0f}°C，电解电容芯 {se['T_can'].max():.0f}°C，柜内 {se['T_cab'].max():.0f}°C，线圈 {se['T_coil'].max():.0f}°C")
        if tr:
            lv = "WARN" if key == "fp_burnin_hot" else "FAIL"
            v.append((lv, msg + f"；第 {tr[0]:.2f} h 保护动作：{tr[2]}"))
        else:
            hot = se['Tj'].max() > mod.Tj_design or se['T_tr'].max() > cs.tr_hot or se['T_cap'].max() > cs.cap_hot
            v.append(("WARN" if hot else "OK", msg + "；无保护动作"))
            if key == "fp_burnin_hot":
                v.append(("FAIL", "冷水机不够但 8 h 内没有任何保护动作：检查进水温度保护定值（拒动风险）"))
        life = ST.ecap_life_h(se["T_can"].max(), sp.cool)
        v.append(("OK" if life > 40000 else "WARN", f"母线电解电容寿命（芯温 {se['T_can'].max():.0f}°C，10 °C 减半，估算）≈ {life/8760:.1f} 年连续运行"))
    if key == "pp_8h" and "Tj_swing" in S:
        cyc = S["cycle"]; N = ST.lesit_cycles(S["Tj_swing"], S["Tj_cyc_mean"])
        yrs = N * cyc / (3600 * 16 * 300)
        v.append(("OK" if yrs > 10 else "WARN", f"IGBT 功率循环：每件 ΔTj {S['Tj_swing']:.1f} K（Tj 均值 {S['Tj_cyc_mean']:.0f}°C），节拍 {cyc:.1f} s → LESIT 寿命 {N:.2e} 次 ≈ {yrs:.0f} 年（两班 16 h×300 天，估算）"))
    if key in ("fp_curie", "fp_curie_tight"):
        p = m.part
        if m.ctl.fault_log:
            v.append(("FAIL" if key == "fp_curie" else "WARN", f"过居里点时变频器跳闸（工件 {p.T:.0f}°C）：定值 {sp.I_trip_pk:.0f} A，过程中电流峰值 {S['I_pk_max']:.0f} A — {'误动' if S['I_pk_max'] < mod.I_crm * 0.5 else ''}"))
        else:
            v.append(("OK", f"冷态 → {p.T:.0f}°C 全程不跳闸，电流峰值 {S['I_pk_max']:.0f} A（定值 {sp.I_trip_pk:.0f} A）"))
    if key == "fp_short" and S.get("tr"):
        tr = S["tr"]; hw, sw, no = tr["hw"], tr["sw"], tr["none"]
        lim_v = cs.Vc_rated * math.sqrt(2)
        def lvl(r):
            if r["I_pk_after"] >= mod.I_crm or r["Vc_pk"] >= 1.5 * lim_v:
                return "FAIL"
            return "OK" if r["Vc_pk"] < lim_v else "WARN"   # v0.30 issue 1821：超额定峰值即 WARN，不再被 1.5× 掩盖
        over = hw["Vc_pk"] / lim_v
        note = "" if over < 1 else f"；电容峰值超额定 {100*(over-1):.0f}%（额定峰值 {lim_v:.0f} V）：快速关断压不住（电感储能续灌），选型须按 {hw['Vc_pk']*1.15/1000:.1f} kV+ 或加吸收回路"
        cause_zh = {"OC": "过流", "VC": "电容过压", None: "未"}.get(hw.get("trip_cause"), "未")
        v.append((lvl(hw), f"负载短路 + 硬件比较器（过流 {sp.I_trip_pk:.0f} A / 电容过压 {0.95*lim_v:.0f} V，2 µs）：{cause_zh}比较器 {(hw['t_off'] - 0.4e-3)*1e6:.0f} µs 后全关，电流峰值 {hw['I_pk_after']:.0f} A，电容峰值 {hw['Vc_pk']:.0f} V{note}"))
        v.append((lvl(sw), f"负载短路 + 只有 1 kHz 软件保护：{(sw['t_off'] - 0.4e-3)*1e3 if sw['t_off'] else float('nan'):.2f} ms 后才关，电流峰值 {sw['I_pk_after']:.0f} A（I_CRM {mod.I_crm:.0f}），"
                             f"谐振电容峰值 {sw['Vc_pk']:.0f} V（额定峰值 {lim_v:.0f} V）"))
        v.append(("WARN", f"不保护时：电流向 {no['I_pk_after']:.0f} A 峰值发展、电容 {no['Vc_pk']:.0f} V 峰值（短路前 {no['I_pk_before']:.0f} A / {no['Vc_pk_before']:.0f} V）"))
    if key in ("fp_sag40", "fp_sag15"):
        if key == "fp_sag40":
            v.append(("OK" if m.ctl.fault_log and m.ctl.fault_log[0][1] == "UV" else "FAIL",
                      f"满功率跌落 40 %：母线最低 {S['Vbus_min_run']:.0f} V，{'欠压跳闸' if m.ctl.fault_log else '未跳闸（拒动）'}"))
        else:
            v.append(("WARN" if S["P_post"] < 0.97 * S["P_pre"] else "OK",
                      f"满功率跌落 15 %：输出 {S['P_pre']/1e3:.1f} → {S['P_post']/1e3:.1f} kW（频率贴下限 {S['f_post']/1e3:.2f} kHz），"
                      f"进线直流电流 {S['I_src_pre']:.0f} → {S['I_src_post']:.0f} A"))
    return v
