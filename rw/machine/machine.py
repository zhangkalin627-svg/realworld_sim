# -*- coding: utf-8 -*-
"""整机：PLC(10 ms 扫描) + 变频器固件(1 kHz) + 主回路 + 气缸 + 冷却水 + 喷淋 + 工件加热 + 水箱/线圈温度。
链条：上电 → PLC 顺序 → 预充电/旁路 → 变频器就绪 → 下降线圈 → 高频使能 → 谐振功率 → 工件升温(红外测温) → 喷淋淬火 → 提升 → 下一件。
工件加热：用 v0.1 有限元算出的“加热时间→温度/硬度”表(models/demo/thermal_table.json，恒定吸收功率 5.94 kW)，
按“能量等效时间” τ = ∫P_wp dt / 5.94 kW 查表。这是近似：功率变化时的温度场分布(表面/心部)不同于恒功率，已在汇报里标明。
"""
import math, os, re
import numpy as np

from ..plc.fx3u import FX3U
from ..plc.pneumatics import Cylinder
from ..plc.thermal_table import ThermalTable
from ..world.pipes import example_spray
from ..world.longrun import pag_severity
from .power import Spec, Bus, Stage, Controller
from .plc_prog import MACHINE_IL

CP_PAG = 4000.0 * 1.02   # J/(L·K)·1000 → 4.08 kJ/(L·K)


class Part:
    def __init__(self, pid, t_in):
        self.id = pid; self.t_in = t_in
        self.tau = 0.0; self.tq = 0.0; self.E_wp = 0.0; self.E_to_tank = 0.0
        self.T = 25.0; self.T_peak = 25.0; self.touched = False
        self.t_heat0 = None; self.t_heat1 = None; self.pyro = False
        self.res = None


class Machine:
    DT = 0.001
    SCAN = 0.010

    def __init__(self, il=MACHINE_IL, spec=None, p_air=0.55, T_tank0=25.0, V_tank_L=300.0, UA_hex=500.0,
                 T_cw=20.0, T_amb=30.0, sag=1.0, Q_cw=8.0, load_time=1.5, thermal=None):
        self.sp = spec or Spec()
        self.il = il
        self.plc = FX3U(il.splitlines(), scan_ms=int(self.SCAN * 1000))
        self.bus = Bus(self.sp); self.stage = Stage(self.sp); self.ctl = Controller(self.sp)
        self.th = thermal or ThermalTable()
        self.p_air = p_air
        self.cyl = Cylinder("线圈升降 TCL50×80", 50, 20, 80, 31.8, "down", v_set=250)
        self._cyl_times()
        self.sag = sag
        self.t = 0.0
        # 机械
        self.pos = 0.0
        # 接触器 / 阀 / 泵（带动作延迟）
        self.k1 = self.k2 = False
        self._pend = []                  # (t_eff, name, value)
        self.cw_on = False; self.cw_flow = 0.0; self.cw_supply_ok = True; self.Q_cw = Q_cw
        self.sp_on = False; self.sp_flow = 0.0
        self.valve_coil = False
        # 热
        self.T_coil = 25.0; self.T_tank = T_tank0; self.T_cw = T_cw; self.T_amb = T_amb
        self.V_tank = V_tank_L; self.UA = UA_hex
        self.E_tank_total = 0.0
        # 工件
        self.part = None; self.n_part = 0; self.t_empty = 0.5 - load_time; self.load_time = load_time
        self.done = []; self.scrapped = 0
        # 操作 / 故障
        self.sched = []                  # (t, fn)
        self.ext = {"X000": False, "X001": True, "X002": True, "X003": False}
        self._pulses = {}
        self.events = []
        # 日志
        self.log = {k: [] for k in ("t", "Vbus", "I_src", "f", "P_out", "P_wp", "P_in", "I_pk", "T_wp", "T_coil", "T_tank",
                                    "cw", "spray", "pos", "step", "ready", "fault", "run", "reached", "P_ref")}
        self.ylog = []; self.xlog = []
        self.step_names = {}
        # 能量账
        self.E_wp = 0.0; self.E_coil = 0.0; self.E_inv = 0.0; self.E_out = 0.0
        self.tq_ref = None
        self._Q_ref = None
        self.coil_max = 0.0; self.i_peak_max = 0.0
        self.cycle_marks = []
        self.Y = {}; self.X = {}
        self._spray_cache = {}
        self.cyl_stuck_note = None
        self.coil_damage_t = None
        self.hex_ok = True
        self.segments = [0.0]            # 快进标记 (sim time, skipped hours) 在 events 里

    # ----------------------------------------------------------- 工具
    def _cyl_times(self):
        te, oke, _ = self.cyl.travel_time(self.p_air, True)
        tr, okr, _ = self.cyl.travel_time(self.p_air, False)
        self.te, self.tr = te, tr

    def ev(self, s):
        self.events.append((round(self.t, 3), s))

    def at(self, t, fn, note=None):
        self.sched.append((t, fn, note))

    def press(self, x, t, dur=0.2):
        nc = x in ("X001", "X002")
        self.at(t, lambda: self._set_ext(x, not nc), f"按下 {x}")
        self.at(t + dur, lambda: self._set_ext(x, nc))

    def _set_ext(self, x, v):
        self.ext[x] = v

    def spray_Q(self):
        key = round(self.T_tank * 2) / 2
        if key not in self._spray_cache:
            c, op = example_spray("pag10", T=key)
            self._spray_cache[key] = op["Q_Lmin"]
        return self._spray_cache[key]

    def spray_Q_ref(self):
        if self._Q_ref is None:
            self._Q_ref = self.spray_Q_at(25.0)
        return self._Q_ref

    def spray_Q_at(self, T):
        c, op = example_spray("pag10", T=T)
        return op["Q_Lmin"]

    def k_coupling(self):
        return float(np.clip((self.pos - 0.55) / 0.40, 0.0, 1.0)) if self.part is not None else 0.0

    def T_wp_for_R(self):
        return self.part.T if self.part is not None else 25.0

    # ----------------------------------------------------------- 主循环
    def run(self, t_end, stop_if=None):
        while self.t < t_end - 1e-9:
            if stop_if and stop_if(self):
                break
            self._scan_step()

    def _scan_step(self):
        t = self.t
        for item in list(self.sched):
            if item[0] <= t + 1e-9:
                self.sched.remove(item); item[1]()
                if item[2]:
                    self.ev(item[2])
        # --- PLC 扫描（输入映像 = 扫描开始时的现场状态）
        X = self._inputs()
        self.plc.scan(X)
        Y = {f"Y{i:03o}": bool(self.plc.bits.get(f"Y{i:03o}", False)) for i in range(8)}
        self._apply_outputs(Y)
        self.X, self.Y = X, Y
        # --- 对象推进：运行高频时 1 ms 小步，其余 10 ms 一步
        running_before = self.ctl.running or Y["Y004"]
        n = 10 if (running_before or self.bus.k1 and self.bus.V < self.sp.V_src(self.sag) * 0.98) else 1
        dt = self.SCAN / n
        I_pk_scan = 0.0; sm = None
        for _ in range(n):
            sm = self._tick(dt, Y)
            I_pk_scan = max(I_pk_scan, sm["I_pk"])
        self.t = round(self.t + self.SCAN, 6)
        self._logrow(sm, I_pk_scan, Y, X)
        self._detect_events(X, Y)

    def _inputs(self):
        sp = self.sp
        X = dict(self.ext)
        X["X004"] = self.cw_flow > 0.5 * self.Q_cw
        X["X005"] = self.p_air >= 0.35
        X["X010"] = self.pos <= 0.0
        X["X011"] = self.pos >= 1.0
        X["X012"] = bool(self.ctl.ready)
        X["X013"] = bool(self.ctl.fault)
        X["X014"] = bool(self.ctl.reached)
        X["X020"] = self.part is not None
        p = self.part
        X["X040"] = bool(p is not None and self.ctl.running and self.pos >= 1.0 and p.T >= 850.0)
        return X

    def _apply_outputs(self, Y):
        t = self.t
        # 接触器 30 ms，电磁阀 20 ms
        for name, y, delay in (("k1", "Y000", 0.030), ("k2", "Y001", 0.030), ("cw", "Y002", 0.020), ("valve", "Y003", 0.020), ("sp", "Y005", 0.020)):
            cur = {"k1": self.k1, "k2": self.k2, "cw": self.cw_on, "valve": self.valve_coil, "sp": self.sp_on}[name]
            tgt = Y[y]
            pend = [q for q in self._pend if q[1] == name]
            last = pend[-1][2] if pend else cur
            if tgt != last:
                self._pend.append((t + delay, name, tgt))
        for q in list(self._pend):
            if q[0] <= t + 1e-9:
                self._pend.remove(q)
                setattr(self, {"k1": "k1", "k2": "k2", "cw": "cw_on", "valve": "valve_coil", "sp": "sp_on"}[q[1]], q[2])

    # ----------------------------------------------------------- 物理小步
    def _tick(self, dt, Y):
        sp = self.sp
        Vsrc = sp.V_src(self.sag)
        # 机械：气缸
        if self.valve_coil:
            if self.te < 1e6:
                self.pos = min(1.0, self.pos + dt / self.te)
        elif self.tr < 1e6:
            self.pos = max(0.0, self.pos - dt / self.tr)
        # 冷却水 / 喷淋
        tgt = self.Q_cw if (self.cw_on and self.cw_supply_ok) else 0.0
        self.cw_flow += (tgt - self.cw_flow) * min(1.0, dt / 0.3)
        tgt = self.spray_Q() if self.sp_on else 0.0
        self.sp_flow += (tgt - self.sp_flow) * min(1.0, dt / 0.4)
        # 母线与变频器
        self.bus.set_contactors(self.k1, self.k2, Vsrc)
        p = self.part
        T_wp = p.T if p is not None else 25.0
        k = self.k_coupling()
        en = Y["Y004"]
        Vb = self.bus.V
        stage_fn = lambda f: self.stage.solve(Vb, f, T_wp, k, self.T_coil)
        f0_fn = lambda: self.stage.f0(T_wp, k)
        s = self.ctl.tick(dt, en, Vb, stage_fn, f0_fn)
        if s is not None and self.ctl.running:
            P_in = s["P_in"]
        else:
            P_in = sp.P_ctrl * 0.0
            s = dict(I=0.0, I_pk=0.0, P_out=0.0, P_wp=0.0, P_coil=0.0, P_inv=0.0, P_in=0.0)
            if self.ctl.fault and s["I_pk"] == 0.0:
                pass
        self.bus.step(dt, Vsrc, P_in)
        self.i_peak_max = max(self.i_peak_max, s["I_pk"])
        # 能量账
        self.E_wp += s["P_wp"] * dt; self.E_coil += s["P_coil"] * dt; self.E_inv += s["P_inv"] * dt; self.E_out += s["P_out"] * dt
        # 线圈温度：铜损 + 工件辐射 − 冷却水(+空气)
        P_rad = 0.0
        if p is not None and self.pos >= 0.9 and p.T > 300:
            P_rad = 0.5 * 5.67e-8 * 0.0017 * ((p.T + 273.15) ** 4 - (self.T_coil + 273.15) ** 4) * 1.0
        Gw = sp.G_water * min(self.cw_flow / self.Q_cw, 1.3) ** 0.8 if self.Q_cw > 0 else 0.0
        P_air = 2.0 * (self.T_coil - self.T_amb)
        self.T_coil += dt * (s["P_coil"] + P_rad - Gw * (self.T_coil - self.T_cw) - P_air) / sp.C_coil
        if s["P_coil"] > 0:
            self._hf_seen = True
        if getattr(self, "_hf_seen", False):
            self.coil_max = max(self.coil_max, self.T_coil)
        if self.T_coil >= sp.coil_fail and self.coil_damage_t is None:
            self.coil_damage_t = self.t
            self.ev(f"线圈温度 {self.T_coil:.0f}°C 超过 {sp.coil_fail:.0f}°C：焊点/绝缘失效，视为损坏")
        # 工件
        if p is not None:
            if s["P_wp"] > 0 and self.pos >= 0.55:
                if p.t_heat0 is None:
                    p.t_heat0 = self.t
                p.touched = True
                dE = s["P_wp"] * dt
                p.E_wp += dE; p.tau += dE / (self.th.P * 1e3)
                p.T = self.th.T_surface(p.tau); p.T_peak = max(p.T_peak, p.T); p.t_heat1 = self.t
            elif self.sp_on and self.sp_flow > 0.3 and self.pos >= 0.95:
                Qr = self.spray_Q_ref()
                sev = pag_severity(0.10, self.T_tank) / pag_severity(0.10, 25.0)
                dq = dt * (self.sp_flow / Qr) * sev
                p.tq += dq; p.touched = True
                Ehot = p.E_wp * 0.8
                dEt = Ehot * dt / max(3.0, 1e-9) * 1.0
                dEt = min(dEt, Ehot - p.E_to_tank)
                p.E_to_tank += dEt
                self.E_tank_total += dEt
                self._tank_heat(dEt, dt)
                Tafter = 204.0
                p.T = max(p.T - (p.T_peak - Tafter) * dq / 3.0, 25.0) if p.T > 25 else p.T
        # 水箱（每步都算：换热器 + 环境）
        self._tank_step(dt)
        return s

    def _tank_heat(self, dE, dt):
        self.T_tank += dE / (CP_PAG * self.V_tank * 1.0)

    def _tank_step(self, dt):
        UA = self.UA * (min(self.cw_flow / self.Q_cw, 1.3) ** 0.6 if self.Q_cw > 0 else 0.0)
        q = UA * (self.T_tank - self.T_cw) + 16.0 * (self.T_tank - self.T_amb)
        self.T_tank -= q * dt / (CP_PAG * self.V_tank)

    # ----------------------------------------------------------- 工件流转与事件
    def _detect_events(self, X, Y):
        p = self.part
        m0 = self.plc.bits.get("M0", False)
        if p is not None:
            if self.pos <= 0.0 and p.tq > 0.5 and p.res is None:
                self._finish(p)
            elif self.pos <= 0.0 and (not m0) and p.touched and p.res is None and self.t - getattr(p, "t_idle", self.t) > 4.0:
                p.res = dict(verdict="中断报废", reason="加工中途停机，工件被取走报废")
                self._finish(p, scrap=True)
            if p is not None and (not m0) and p.touched:
                if not hasattr(p, "t_idle"):
                    p.t_idle = self.t
        else:
            if self.t - self.t_empty >= self.load_time:
                self.n_part += 1; self.part = Part(self.n_part, self.t)

    def _finish(self, p, scrap=False):
        r = self.th.result(max(p.tau, 1e-3), max(p.tq, 1e-3))
        r["T_exit"] = self.th.T_exit(max(p.tau, 1e-3), max(p.tq, 1e-3))
        if scrap:
            res = p.res
        else:
            T = p.T_peak
            ok = 850.0 <= T <= 1050.0 and r["HRC"] >= 50.0
            if T < 850:
                v, why = "NG", f"加热不足（最高 {T:.0f}°C < 850°C）"
            elif T > 1050:
                v, why = "NG", f"过热（{T:.0f}°C），晶粒粗大/开裂风险"
            elif r["HRC"] < 50:
                v, why = "NG", f"硬度偏低 HRC {r['HRC']:.0f}（淬火不足：等效淬火 {p.tq:.2f} s）"
            else:
                v, why = "OK", ""
            res = dict(verdict=v, reason=why)
        res.update(T_exit=r['T_exit'], id=p.id, t_done=self.t, tau=p.tau, T_peak=p.T_peak, HRC=r["HRC"], case=r["case"], tq=p.tq,
                   E_wp_kJ=p.E_wp / 1e3, t_heat=(p.t_heat1 - p.t_heat0) if p.t_heat0 is not None and p.t_heat1 is not None else 0.0,
                   pyro_NG=bool(self.plc.bits.get("M30", False) is False and False))
        p.res = res
        self.done.append(res)
        self.ev(f"工件 P{p.id} 下料：{res['verdict']} {res['reason']}  T峰值 {p.T_peak:.0f}°C HRC {r['HRC']:.0f}")
        self.part = None; self.t_empty = self.t
        if scrap:
            self.scrapped += 1

    def _logrow(self, s, ipk, Y, X):
        L = self.log
        L["t"].append(self.t); L["Vbus"].append(self.bus.V); L["I_src"].append(self.bus.I_src)
        L["f"].append(self.ctl.f / 1e3 if self.ctl.running else 0.0)
        L["P_out"].append(s["P_out"] / 1e3); L["P_wp"].append(s["P_wp"] / 1e3); L["P_in"].append(s["P_in"] / 1e3)
        L["I_pk"].append(ipk); L["T_wp"].append(self.part.T if self.part is not None else np.nan)
        L["T_coil"].append(self.T_coil); L["T_tank"].append(self.T_tank)
        L["cw"].append(self.cw_flow); L["spray"].append(self.sp_flow); L["pos"].append(self.pos)
        st = [int(d[1:]) for d in self.plc.bits if d.startswith("S") and self.plc.bits[d]]
        L["step"].append(max(st) if st else 0)
        L["ready"].append(int(self.ctl.ready)); L["fault"].append(int(bool(self.ctl.fault))); L["run"].append(int(self.ctl.running))
        L["reached"].append(int(self.ctl.reached)); L["P_ref"].append(self.ctl.p_ref / 1e3)
        self.ylog.append([int(Y[f"Y{i:03o}"]) for i in range(8)])
        self.xlog.append([int(X.get(x, False)) for x in ("X004", "X005", "X010", "X011", "X012", "X013", "X020", "X040")])
        m = self.plc.bits
        for k in ("M0", "M9", "M300", "M301", "M302", "M303", "M304"):
            v = bool(m.get(k, False)); key = "_prev_" + k
            if v != getattr(self, key, False):
                setattr(self, key, v)
                names = {"M0": "运行", "M9": "报警总", "M300": "报警：冷却水流量中断", "M301": "报警：变频器故障", "M302": "报警：气压低",
                         "M303": "报警：线圈下降超时", "M304": "报警：变频器就绪超时"}
                self.ev(f"PLC {names[k]} {'ON' if v else 'OFF'}")
        for key, txt in (("ready", "变频器就绪"), ("fault", "变频器故障")):
            v = bool(L[key][-1]); kk = "_p2_" + key
            if v != getattr(self, kk, False):
                setattr(self, kk, v); self.ev(f"变频器 {txt} {'ON' if v else 'OFF'}" + (f"（{self.ctl.fault_log[-1][2]}）" if key == "fault" and v and self.ctl.fault_log else ""))
        st = L["step"][-1]
        if st != getattr(self, "_prev_step", -1):
            self._prev_step = st
            nm = {0: "待机", 10: "S10 开冷却水并等流量", 11: "S11 母线预充电", 12: "S12 合旁路等变频器就绪", 13: "S13 待料", 20: "S20 线圈下降",
                  21: "S21 加热", 22: "S22 喷淋淬火", 23: "S23 线圈上升"}.get(st, f"S{st}")
            self.ev("PLC 步进 → " + nm)

    # ----------------------------------------------------------- 慢变量快进（长时间运行）
    def fast_forward(self, hours, takt=None):
        """把水箱温度按“每件平均进水箱热量 / 节拍 + 换热器 + 散热”的一阶解析解推进 hours 小时。
        其它（工件、线圈、PLC）不动。takt 不给就用已完成工件的平均间隔。"""
        if takt is None:
            ts = [r["t_done"] for r in self.done if r["verdict"] != "中断报废"]
            if len(ts) >= 2:
                takt = (ts[-1] - ts[0]) / (len(ts) - 1)
            else:
                takt = 12.0
        n_ok = max(len(self.done), 1)
        q_part = self.E_tank_total / n_ok                          # J/件
        q_avg = q_part / takt
        UA = self.UA * (1.0 if self.cw_on and self.cw_supply_ok else 0.0) + 16.0
        T_ss = (q_avg + self.UA * (1.0 if (self.cw_on and self.cw_supply_ok) else 0.0) * self.T_cw + 16.0 * self.T_amb) / UA
        tau = CP_PAG * self.V_tank / UA
        T0 = self.T_tank
        self.T_tank = T_ss + (T0 - T_ss) * math.exp(-hours * 3600 / tau)
        self.ev(f"【快进 {hours:g} 小时】水箱 {T0:.1f}°C → {self.T_tank:.1f}°C（稳态 {T_ss:.1f}°C，时间常数 {tau/3600:.2f} h，节拍 {takt:.1f} s）")
        return dict(T0=T0, T1=self.T_tank, T_ss=T_ss, tau_h=tau / 3600, takt=takt, q_avg=q_avg)
