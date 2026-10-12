# -*- coding: utf-8 -*-
"""主回路：三相整流 + 母线电容(含预充电) + 全桥逆变 + 串联谐振负载（基波近似） + 变频器控制固件（行为级）。
全部参数是“通用 10 kW 级 30 kHz 感应加热电源”的假设值，不是任何具体产品。

模型层次（为什么这样分）：
- 微秒级的开关波形由 v0.6/v0.7 的 ngspice 与指令级 MCU 负责；整机要跑几十秒到几小时，
  所以这里用“基波近似”平均模型：全桥方波基波有效值 V1 = (2√2/π)·Vbus ≈ 0.9·Vbus，串联 RLC 的稳态阻抗算电流和功率。
  这个近似用 ngspice 整桥方波瞬态逐点核对过（validate.py 第 10 段）。
- 母线预充电、合旁路瞬间的冲击电流：RLC 阶跃响应数值积分（和 ngspice 核对）。
"""
import math
import numpy as np
from dataclasses import dataclass, field

MU = None


def _steel_s(T):
    """45 钢表面电阻 ∝ sqrt(ρ·μr)（薄壁/集肤效应），返回相对 20°C 的比值。"""
    from ..materials import steel_sigma, steel_mur
    def s(t):
        return math.sqrt(float(steel_mur(t)) / float(steel_sigma(t)))
    return s(T) / s(20.0)


@dataclass
class Spec:
    # 电网与母线
    V_ll: float = 380.0          # 线电压 V
    Rs: float = 0.12             # 电网+变压器+整流等效串联电阻 Ω
    L_line: float = 0.5e-3       # 进线电抗 H
    C_bus: float = 2.0e-3        # 母线电容 F
    R_pc: float = 100.0          # 预充电阻 Ω
    R_bleed: float = 100e3       # 泄放电阻 Ω
    I_dc_rated: float = 25.0     # 直流侧额定电流 A（冲击电流判定基准）
    # 谐振负载（折算到逆变器输出侧）
    L_wp_cold: float = 130e-6    # 有工件(冷、有磁性)时线圈电感 H
    L_wp_hot: float = 138e-6     # 有工件(过居里点)时
    L_empty: float = 150e-6      # 空线圈
    C_res: float = 0.328e-6      # 谐振电容 F
    R_coil: float = 0.8          # 线圈铜损等效电阻（20°C）Ω
    R_wp20: float = 6.4          # 工件等效电阻 20°C Ω（随 sqrt(ρ·μr) 变化）
    alpha_cu: float = 0.0039
    # 逆变器
    R_sw: float = 0.05           # 开关管导通等效电阻 Ω
    E_sw: float = 0.4e-3         # 100 A 时每周期开关能量 J（关断损耗，ZVS 之上工作）
    P_ctrl: float = 40.0         # 驱动+控制耗电 W
    # 保护
    I_trip_pk: float = 110.0     # 过流跳闸 A(峰值)
    V_uv: float = 330.0          # 母线欠压跳闸 V
    V_ov: float = 700.0
    V_ready: float = 410.0       # 就绪所需母线电压（80% 额定）
    f_max: float = 60e3
    f_zvs: float = 1.04          # 最低开关频率/谐振频率（保持感性负载，ZVS）
    # 控制
    P_set_out: float = 7.3e3     # 输出功率设定 W（线圈输出，含线圈损耗）
    ramp: float = 25e3           # 软启动 W/s
    Ki: float = 22.0             # kHz/(s·kW)
    boot_s: float = 0.30
    # 冷却与线圈
    C_coil: float = 120.0        # 线圈(铜管+水)热容 J/K
    G_water: float = 530.0       # 水流满流量时线圈到水的热导 W/K
    coil_warn: float = 60.0
    coil_fail: float = 150.0

    def V_src(self, sag=1.0):
        return 1.35 * self.V_ll * sag


class Bus:
    """母线电容 + 预充电/旁路接触器。时间步 dt（≤ 几 ms）。"""

    def __init__(self, sp: Spec):
        self.sp = sp; self.V = 0.0
        self.inrush_peak = 0.0; self.inrush_events = []
        self.E_src = 0.0; self.E_R = 0.0; self.E_bleed = 0.0; self.E_load = 0.0
        self.k1 = False; self.k2 = False
        self.I_src = 0.0

    def inrush(self, Vsrc, V0, R=None):
        """K2 闭合瞬间：Vsrc —(Rs, L)— C，C 初始电压 V0。RK4 积分，返回(峰值电流, 电阻耗能, 源输出电荷)。"""
        sp = self.sp; R = sp.Rs if R is None else R
        L, C = sp.L_line, sp.C_bus
        i = 0.0; v = V0; dt = 2e-6; Ipk = 0.0; ER = 0.0; Q = 0.0
        def f(i, v):
            return ((Vsrc - R * i - v) / L, i / C)
        for n in range(int(0.04 / dt)):
            k1 = f(i, v); k2 = f(i + dt / 2 * k1[0], v + dt / 2 * k1[1])
            k3 = f(i + dt / 2 * k2[0], v + dt / 2 * k2[1]); k4 = f(i + dt * k3[0], v + dt * k3[1])
            i += dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]); v += dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
            Ipk = max(Ipk, i); ER += i * i * R * dt; Q += i * dt
        return Ipk, ER, Q, v

    def set_contactors(self, k1, k2, Vsrc):
        if k2 and not self.k2:
            dv = Vsrc - self.V
            if dv > 0.05:
                Ipk, ER, Q, vend = self.inrush(Vsrc, self.V)
                self.inrush_events.append(dict(dV=dv, I_peak=Ipk, E_R=ER))
                self.inrush_peak = max(self.inrush_peak, Ipk)
                self.E_src += Vsrc * Q; self.E_R += ER
                self.V = Vsrc - 0.0   # 之后由准静态公式接管
        self.k1, self.k2 = k1, k2

    def step(self, dt, Vsrc, P_load):
        sp = self.sp; V = max(self.V, 1e-3)
        Vold = self.V
        if self.k2:
            disc = Vsrc * Vsrc - 4 * P_load * sp.Rs
            Vn = (Vsrc + math.sqrt(disc)) / 2 if disc >= 0 else Vsrc / 2
            Isrc = P_load / max(Vn, 1.0) + sp.C_bus * (Vn - Vold) / dt
            self.E_R += Isrc * Isrc * sp.Rs * dt
        elif self.k1:
            # C dV/dt = (Vsrc − V)/Rpc − P/V − V/Rbleed（二极管单向，V>Vsrc 时不倒灌）
            Ic = max((Vsrc - V) / sp.R_pc, 0.0)
            Il = P_load / max(V, 20.0) if P_load > 0 else 0.0
            Ib = V / sp.R_bleed
            Vn = V + dt * (Ic - Il - Ib) / sp.C_bus
            Isrc = Ic
            self.E_R += Ic * Ic * sp.R_pc * dt
            P_load = Il * V
        else:
            Il = P_load / max(V, 20.0) if P_load > 0 else 0.0
            Vn = max(V + dt * (-Il - V / sp.R_bleed) / sp.C_bus, 0.0); Isrc = 0.0
            P_load = Il * V
        self.V = max(Vn, 0.0)
        self.I_src = Isrc
        self.E_src += Vsrc * Isrc * dt
        self.E_load += P_load * dt
        self.E_bleed += (self.V ** 2 / sp.R_bleed) * dt
        return Isrc

    def E_cap(self):
        return 0.5 * self.sp.C_bus * self.V ** 2


class Stage:
    """全桥 + 串联谐振负载（基波近似）。"""

    def __init__(self, sp: Spec):
        self.sp = sp

    def L_of(self, T_wp, k):
        sp = self.sp
        hot = 1.0 - float(np.clip((770.0 - T_wp) / 120.0, 0.0, 1.0))
        Lw = sp.L_wp_cold + (sp.L_wp_hot - sp.L_wp_cold) * hot
        return sp.L_empty - (sp.L_empty - Lw) * k

    def R_parts(self, T_wp, k, T_coil):
        sp = self.sp
        Rc = sp.R_coil * (1 + sp.alpha_cu * (T_coil - 20.0))
        Rw = sp.R_wp20 * _steel_s(T_wp) * k * k
        return Rc, Rw

    def f0(self, T_wp, k):
        return 1.0 / (2 * math.pi * math.sqrt(self.L_of(T_wp, k) * self.sp.C_res))

    def solve(self, Vbus, f, T_wp, k, T_coil):
        sp = self.sp
        V1 = 0.9003 * Vbus                        # (4/π)/√2
        w = 2 * math.pi * f
        L = self.L_of(T_wp, k)
        X = w * L - 1.0 / (w * sp.C_res)
        Rc, Rw = self.R_parts(T_wp, k, T_coil)
        R = Rc + Rw
        Z = math.sqrt(R * R + X * X)
        I = V1 / Z
        P_out = I * I * R
        P_loss = sp.R_sw * I * I + sp.E_sw * f * (I / 70.0) + sp.P_ctrl
        return dict(I=I, I_pk=I * math.sqrt(2), P_out=P_out, P_wp=I * I * Rw, P_coil=I * I * Rc, P_inv=P_loss,
                    P_in=P_out + P_loss, X=X, Z=Z, V1=V1, cosphi=R / Z, f0=self.f0(T_wp, k))


class Controller:
    """变频器控制固件（行为级，1 kHz 控制周期）。输入：EN(来自 PLC)、母线电压、功率级测量；输出：READY/FAULT/到达 + 开关频率。
    固件逻辑与 v0.7 的汇编固件一致的思路：软启动 → 积分调频稳功率(f 往谐振频率靠近则功率升) → 过流/欠压锁存。"""

    def __init__(self, sp: Spec):
        self.sp = sp
        self.t = 0.0; self.mode = "boot"; self.f = sp.f_max * 0.75
        self.p_ref = 0.0; self.p_meas = 0.0; self.fault = ""; self.ready = False; self.reached = False
        self.running = False; self.t_en_low = 0.0; self.t_sat = 0.0; self.sat = False
        self.fault_log = []

    def tick(self, dt, en, Vbus, stage_fn, f0_fn):
        """stage_fn(f) → 功率级测量字典；f0_fn() → 当前谐振频率估计(锁相环给出)。返回 stage dict 或 None(未开管)。"""
        sp = self.sp
        self.t += dt
        if self.t < sp.boot_s:
            self.running = False; return None
        if self.fault:
            self.running = False; self.ready = False
            if not en:
                self.t_en_low += dt
                if self.t_en_low > 0.1 and sp.V_uv < Vbus < sp.V_ov:
                    self.fault = ""; self.mode = "wait"
            else:
                self.t_en_low = 0.0
            return None
        self.t_en_low = 0.0
        # 保护（母线）
        if en and Vbus < sp.V_uv:
            return self._trip("UV", "母线欠压"); 
        if Vbus > sp.V_ov:
            return self._trip("OV", "母线过压")
        if not en:
            self.running = False; self.reached = False; self.p_ref = 0.0; self.p_meas = 0.0
            self.f = sp.f_max * 0.75
            self.ready = Vbus >= sp.V_ready
            self.mode = "ready" if self.ready else "wait"
            return None
        if Vbus < sp.V_ready * 0.95 and not self.running:
            self.running = False; return None      # 母线没建立好，不开管
        # 运行
        self.running = True; self.mode = "run"; self.ready = True
        self.p_ref = min(sp.P_set_out, self.p_ref + sp.ramp * dt)
        fmin = f0_fn() * sp.f_zvs
        s = stage_fn(self.f)
        self.p_meas += (s["P_out"] - self.p_meas) * min(1.0, dt / 2e-3)
        e = (self.p_ref - self.p_meas) / 1e3
        self.f -= sp.Ki * e * dt * 1e3
        self.f = float(np.clip(self.f, fmin, sp.f_max))
        s = stage_fn(self.f)
        if s["I_pk"] > sp.I_trip_pk:
            return self._trip("OC", f"过流 {s['I_pk']:.0f} A 峰值", s)
        self.sat = self.f <= fmin + 1.0 and self.p_meas < 0.95 * self.p_ref
        self.t_sat = self.t_sat + dt if self.sat else 0.0
        self.reached = abs(s["P_out"] - self.p_ref) < 0.05 * max(self.p_ref, 1e3) and self.p_ref >= sp.P_set_out - 1
        return s

    def _trip(self, code, text, s=None):
        self.fault = code; self.running = False; self.ready = False; self.mode = "fault"
        self.fault_log.append((self.t, code, text))
        return s
