# -*- coding: utf-8 -*-
"""主回路：三相整流 + 母线电容(含预充电) + 全桥逆变 + 串联谐振负载（基波近似） + 变频器控制固件（行为级）。
默认参数是“通用 100 kW 级 20–40 kHz 感应加热电源”的假设值（用户 2026-10-12 定整机按 100 kW 通用机），不是任何具体产品；
旧版 10 kW 示例机保留为 Spec.preset("10kW")。负载参数是“折算到逆变器输出侧”的等效值（经匹配变压器，匝比 cool.n_tr）。
器件损耗/温升/保护瞬态见 stress.py。

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
    rated_kW: float = 100.0
    # 电网与母线
    V_ll: float = 480.0          # 线电压 V（D-001：380→480 Vac 升压变压器直驱，v0.5）
    Rs: float = 0.205            # 电网+变压器+整流等效串联电阻 Ω（v0.5：与换相电阻合计 0.25，按 B 轨 ±10% 负载母线锚点 576.0/702.1 V 校准）
    L_line: float = 0.15e-3      # 进线电抗 H（≈2 %，以 100 kW/480 V 为基准）
    C_bus: float = 10.0e-3       # 母线电容 F（v0.5：B 轨仿真口径 10 mF）
    R_pc: float = 33.0           # 预充电阻 Ω（τ = 0.40 s）
    R_bleed: float = 20e3        # 泄放电阻 Ω
    I_dc_rated: float = 210.0    # 直流侧额定电流 A（冲击电流判定基准）
    # 谐振负载（折算到逆变器输出侧；= 旧 10 kW 示例按阻抗 ×0.352 缩放：谐振频率、Q 值不变，满功率在热态 f/f0≈1.07）
    L_wp_cold: float = 47.09e-6  # 有工件(冷、有磁性)时线圈电感 H（v0.5：卡布复算 5 匝+引线，5.5:1 折算到原边；f0≈28.4 kHz）
    L_wp_hot: float = 45.41e-6   # 有工件(过居里点)时（v0.5：居里后 μr→1，L 下降，f0≈28.9 kHz——修正旧版冷热顺序颠倒）
    L_empty: float = 44.0e-6     # 空线圈（估算：无工件 μ 最低）
    C_res: float = 0.667e-6      # 谐振电容 F（v0.5：现行硬件 2P+1S 0.667 µF，原边侧串联）
    R_coil: float = 0.11         # 线圈+引线+变压器绕组铜损等效电阻（20°C）Ω（v0.5：卡布复算 3.37-3.94 mΩ×30.25 折算原边）
    R_wp20: float = 5.17         # 工件等效电阻 20°C Ω（v0.5：按 B 轨锚点热态总 R=2.89 Ω 反推校准）
    alpha_cu: float = 0.0039
    frac_tr: float = 0.10        # R_coil 里匹配变压器绕组占比（假设）
    frac_lead: float = 0.15      # R_coil 里高频引线占比（假设），其余是线圈本体
    # 逆变器（器件模型见 stress.Module；旧版 R_sw/E_sw 仅 10kW 预设用）
    R_sw: float = 0.0
    E_sw: float = 0.0
    P_ctrl: float = 200.0        # 驱动+控制耗电 W
    use_device_model: bool = True
    # 保护
    I_trip_pk: float = 480.0     # 过流跳闸 A(峰值)：硬件比较器逐周期 + 软件环 1 kHz
    V_uv: float = 330.0          # 母线欠压跳闸 V
    V_ov: float = 700.0
    V_ready: float = 410.0       # 就绪所需母线电压（80% 额定）
    f_max: float = 60e3
    f_zvs: float = 1.04          # 最低开关频率/谐振频率（保持感性负载，ZVS）
    f_clk: float = 100e6         # 频率发生计数时钟 Hz（周期 = 整数个时钟 → Δf ≈ f²/f_clk），假设
    # 闭环限值与相移调制（v0.5，D-001：闭环一期必备；限值来源 v0.4 输出箱）
    I_lim: float = 186.0         # I1 限值 Arms（v0.4 硬件约束）
    Uc_lim: float = 1540.0       # 谐振电容电压限值 Vrms（v0.4）
    ctl_mode: str = "pm"         # "pm"=相移调制闭环（D-001 一期必备）；"fm"=旧调频对照
    track_mode: str = "ideal"    # "ideal"=频率直接用模型谐振点（基线复现）；"pll"=相位锁定环：只用实测相位 φ=atan2(X,R) 跟踪谐振
    phi_ref: float = 0.22        # PLL 目标相位 rad（感性 ZVS 裕度；Q≈2.85 时 ≈ f/f0·1.04 ≈ 12.7°）
    pll_K: float = 150e3         # VCO 积分增益 Hz/s 每 rad 相位误差
    pll_sweep: float = 150e3     # 捕获扫频速率 Hz/s（自高位向下扫到 φ 进捕获窗）
    pll_capture: float = 0.10    # 捕获窗 rad
    pll_loss_dwell: float = 0.05 # φ<0（容性，ZVS 丢失）持续此秒数 → LOSS 锁存跳闸
    m_min: float = 0.30          # 相移调制比下限
    Km_rel: float = 1.2          # 调制比积分增益 1/s 每单位相对功率误差
    # 控制
    P_set_out: float = 100e3     # 输出功率设定 W（线圈输出，含线圈损耗）
    ramp: float = 250e3          # 软启动 W/s
    Ki: float = 0.0              # 旧参数 kHz/(s·kW)（10kW 预设用）；>0 时用旧的绝对误差积分
    Ki_rel: float = 160.6        # kHz/s 每单位相对功率误差（= 旧 22 kHz/(s·kW) × 7.3 kW，动态与旧版一致）
    boot_s: float = 0.30
    # 冷却与线圈
    C_coil: float = 400.0        # 线圈(铜管+水)热容 J/K
    G_water: float = 1500.0      # 水流满流量时线圈到水的热导 W/K
    Q_cw_Lmin: float = 30.0      # 线圈冷却水额定流量 L/min
    coil_warn: float = 60.0
    coil_fail: float = 150.0
    use_thermal_net: bool = True
    mod: object = None
    cool: object = None

    def __post_init__(self):
        from .stress import Module, CoolingSpec
        if self.mod is None:
            self.mod = Module()
        if self.cool is None:
            self.cool = CoolingSpec()

    def V_src(self, sag=1.0):
        return math.sqrt(2.0) * self.V_ll * sag   # v0.5：电容输入滤波母线贴近峰值（B 轨锚点 480→679 V）

    def R_comm(self):
        """二极管整流换相压降的等效电阻 3ωL/π（整流平均模型用）。"""
        return 3 * 2 * math.pi * 50.0 * self.L_line / math.pi

    @classmethod
    def preset(cls, name="100kW", **kw):
        if name == "10kW":
            d = dict(rated_kW=10.0, Rs=0.12, L_line=0.5e-3, C_bus=2.0e-3, R_pc=100.0, R_bleed=100e3, I_dc_rated=25.0,
                     L_wp_cold=130e-6, L_wp_hot=138e-6, L_empty=150e-6, C_res=0.328e-6, R_coil=0.8, R_wp20=6.4,
                     R_sw=0.05, E_sw=0.4e-3, P_ctrl=40.0, use_device_model=False, I_trip_pk=110.0, P_set_out=7.3e3,
                     ramp=25e3, Ki=22.0, C_coil=120.0, G_water=530.0, Q_cw_Lmin=8.0, use_thermal_net=False, f_clk=0.0)
            d.update(kw)
            return cls(**d)
        return cls(**kw)


class Bus:
    """母线电容 + 预充电/旁路接触器。时间步 dt（≤ 几 ms）。"""

    def __init__(self, sp: Spec):
        for f in ("R_pc", "R_bleed", "C_bus"):
            v = getattr(sp, f)
            if isinstance(v, bool) or not (isinstance(v, (int, float)) and math.isfinite(v) and v > 0):
                raise ValueError(f"母线参数 {f} 必须为正的有限值，得到 {v!r}（issue 1823）")
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
        if isinstance(dt, bool) or not (math.isfinite(dt) and dt > 0):
            raise ValueError(f"母线步长 dt 必须为正的有限值，得到 {dt!r}（issue 1823）")
        sp = self.sp; V = max(self.V, 1e-3)
        Vold = self.V
        if self.k2:
            Rq = sp.Rs + (sp.R_comm() if sp.use_device_model else 0.0)
            disc = Vsrc * Vsrc - 4 * P_load * Rq
            Vn = (Vsrc + math.sqrt(disc)) / 2 if disc >= 0 else Vsrc / 2
            Isrc = P_load / max(Vn, 1.0) + sp.C_bus * (Vn - Vold) / dt
            self.E_R += Isrc * Isrc * Rq * dt
        elif self.k1:
            # C dV/dt = (Vsrc − V)/Rpc − P/V − V/Rbleed（二极管单向，V>Vsrc 时不倒灌）
            # v0.30：指数精确 RC 更新（对任意 dt 稳定），替代显式 Euler——dt>τ 时旧格式
            # 会过冲到非物理电压（dt=1 s 时 1295 V > 源 513 V，issue 1823）。负载电流按步
            # 初电压冻结（一阶），能量项取中点电流（二阶）。
            Il = P_load / max(V, 20.0) if P_load > 0 else 0.0
            if Vsrc > V:
                G = 1.0 / sp.R_pc + 1.0 / sp.R_bleed
                tau = sp.C_bus / G
                Vinf = (Vsrc / sp.R_pc - Il) / G
            else:
                tau = sp.C_bus * sp.R_bleed
                Vinf = -Il * sp.R_bleed
            Vn = Vinf + (V - Vinf) * math.exp(-dt / tau)
            Vmid = 0.5 * (V + Vn)
            Ic = max((Vsrc - Vmid) / sp.R_pc, 0.0)
            Isrc = Ic
            self.E_R += Ic * Ic * sp.R_pc * dt
            P_load = Il * V
        else:
            Il = P_load / max(V, 20.0) if P_load > 0 else 0.0
            tau = sp.C_bus * sp.R_bleed
            Vinf = -Il * sp.R_bleed
            Vn = max(Vinf + (V - Vinf) * math.exp(-dt / tau), 0.0); Isrc = 0.0
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
        self.short = False            # 故障注入：工件碰线圈，工件等效电阻被短接

    def L_of(self, T_wp, k):
        sp = self.sp
        hot = 1.0 - float(np.clip((770.0 - T_wp) / 120.0, 0.0, 1.0))
        Lw = sp.L_wp_cold + (sp.L_wp_hot - sp.L_wp_cold) * hot
        return sp.L_empty - (sp.L_empty - Lw) * k

    def R_parts(self, T_wp, k, T_coil):
        sp = self.sp
        Rc = sp.R_coil * (1 + sp.alpha_cu * (T_coil - 20.0))
        Rw = 0.0 if self.short else sp.R_wp20 * _steel_s(T_wp) * k * k
        return Rc, Rw

    def f0(self, T_wp, k):
        return 1.0 / (2 * math.pi * math.sqrt(self.L_of(T_wp, k) * self.sp.C_res))

    def solve(self, Vbus, f, T_wp, k, T_coil, m=1.0):
        sp = self.sp
        V1 = m * 0.9003 * Vbus                    # (4/π)/√2 × 相移调制比 m（基波近似，v0.5）
        w = 2 * math.pi * f
        L = self.L_of(T_wp, k)
        X = w * L - 1.0 / (w * sp.C_res)
        Rc, Rw = self.R_parts(T_wp, k, T_coil)
        R = Rc + Rw
        Z = math.sqrt(R * R + X * X)
        I = V1 / Z
        P_out = I * I * R
        if sp.use_device_model:
            from .stress import inverter_losses
            inv = inverter_losses(I * math.sqrt(2), math.atan2(X, R), f, Vbus, sp.mod)
            P_loss = inv["P_total"] + sp.P_ctrl
            mode = inv["mode"]; I_off = inv["I_off"]
        else:
            P_loss = sp.R_sw * I * I + sp.E_sw * f * (I / 70.0) + sp.P_ctrl
            mode = "ZVS" if X >= 0 else "容性"; I_off = I * math.sqrt(2) * X / Z
        return dict(I=I, I_pk=I * math.sqrt(2), P_out=P_out, P_wp=I * I * Rw, P_coil=I * I * Rc, P_inv=P_loss,
                    P_in=P_out + P_loss, X=X, Z=Z, R=R, V1=V1, cosphi=R / Z, f0=self.f0(T_wp, k), mode=mode, I_off=I_off,
                    Vc=I / (w * sp.C_res), f=f, m=m)


class Controller:
    """变频器控制固件（行为级，1 kHz 控制周期）。输入：EN(来自 PLC)、母线电压、功率级测量；输出：READY/FAULT/到达 + 开关频率。
    固件逻辑与 v0.7 的汇编固件一致的思路：软启动 → 积分调频稳功率(f 往谐振频率靠近则功率升) → 过流/欠压锁存。"""

    def __init__(self, sp: Spec):
        self.sp = sp
        self.t = 0.0; self.mode = "boot"; self.f = sp.f_max * 0.75; self.m = 1.0
        self.p_ref = 0.0; self.p_meas = 0.0; self.fault = ""; self.ready = False; self.reached = False
        self.running = False; self.t_en_low = 0.0; self.t_sat = 0.0; self.sat = False
        self.fault_log = []
        self.ext_trip = None          # (code, text)：过温等外部保护由 Machine 置位
        self.f_q = self.f
        self.pll = "sweep"       # sweep → locked（track_mode="pll" 时用）
        self.t_cap = 0.0         # 容性持续时间

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
        if sp.ctl_mode == "pm":
            # v0.5 相移调制闭环（D-001 一期必备）：频率固定于谐振点上侧（ZVS），功率经调制比 m 调节，
            # m 上限由 I1≤186 Arms 与 Uc≤1.54 kVrms 双限值实时折算（v0.4 硬件约束）。
            if sp.track_mode == "pll":
                # 相位锁定环：频率只由实测相位驱动，不知谐振点（v0.31；ideal 为基线复现通道）
                self.f_q = sp.f_clk / round(sp.f_clk / self.f) if sp.f_clk > 0 else self.f
                s_probe = stage_fn(self.f_q)
                phi = math.atan2(s_probe["X"], s_probe["R"])
                self.t_cap = self.t_cap + dt if s_probe["X"] < 0 else 0.0
                if self.t_cap > sp.pll_loss_dwell:
                    return self._trip("LOSS", "失锁：负载进入容性（ZVS 丢失）", s_probe)
                if self.pll == "sweep":
                    self.f -= sp.pll_sweep * dt
                    if phi >= sp.phi_ref - sp.pll_capture:
                        self.pll = "locked"
                else:
                    self.f += sp.pll_K * (sp.phi_ref - phi) * dt
                self.f = float(np.clip(self.f, 1e3, sp.f_max))
                self.phi = phi
            else:
                self.f = float(np.clip(fmin, 1e3, sp.f_max))
            self.f_q = sp.f_clk / round(sp.f_clk / self.f) if sp.f_clk > 0 else self.f
            s = stage_fn(self.f_q)
            self.p_meas += (s["P_out"] - self.p_meas) * min(1.0, dt / 2e-3)
            w = 2 * math.pi * self.f_q
            i_eff = min(sp.I_lim, sp.Uc_lim * w * sp.C_res)
            self.m_cap = min(1.0, max(sp.m_min, i_eff * s["Z"] / max(0.9003 * Vbus, 1.0)))
            e = (self.p_ref - self.p_meas) / max(sp.P_set_out, 1.0)
            self.m = float(np.clip(self.m + sp.Km_rel * e * dt, sp.m_min, self.m_cap))
            s = stage_fn(self.f_q)
            self.sat = self.m >= self.m_cap - 1e-9 and self.p_meas < 0.95 * self.p_ref
        else:
            s = stage_fn(self.f)
            self.p_meas += (s["P_out"] - self.p_meas) * min(1.0, dt / 2e-3)
            if sp.Ki > 0:
                e = (self.p_ref - self.p_meas) / 1e3
                self.f -= sp.Ki * e * dt * 1e3
            else:
                e = (self.p_ref - self.p_meas) / max(sp.P_set_out, 1.0)
                self.f -= sp.Ki_rel * e * dt * 1e3
            self.f = float(np.clip(self.f, fmin, sp.f_max))
            self.f_q = sp.f_clk / round(sp.f_clk / self.f) if sp.f_clk > 0 else self.f   # 周期计数量化
            s = stage_fn(self.f_q)
            self.sat = self.f <= fmin + 1.0 and self.p_meas < 0.95 * self.p_ref
        if self.ext_trip:
            code, text = self.ext_trip; self.ext_trip = None
            return self._trip(code, text, s)
        if s["I_pk"] > sp.I_trip_pk:
            return self._trip("OC", f"过流 {s['I_pk']:.0f} A 峰值", s)
        self.t_sat = self.t_sat + dt if self.sat else 0.0
        self.reached = abs(s["P_out"] - self.p_ref) < 0.05 * max(self.p_ref, 1e3) and self.p_ref >= sp.P_set_out - 1
        return s

    def _trip(self, code, text, s=None):
        self.fault = code; self.running = False; self.ready = False; self.mode = "fault"
        self.fault_log.append((self.t, code, text))
        return s
