# -*- coding: utf-8 -*-
"""100 kW 级感应加热电源的器件应力、损耗、温升与保护瞬态（通用模型，参数全是假设值，不是任何具体产品）。

内容
- inverter_losses()  全桥 IGBT/二极管 导通 + 关断损耗（串联谐振、方波电压、正弦电流，相角 φ）。
  ZVS（感性，φ>0）：开通零电压、二极管自然换流 → 只有关断损耗，关断电流 = I_pk·sinφ。
  容性（φ<0）：硬开通 + 二极管反向恢复（E_on_hard/E_rr 量级假设）。
- rectifier_pf()     三相二极管整流 + 进线电抗 + 母线电容 + 恒功率负载：自写理想二极管时域仿真
                     → 进线电流有效值、基波、THD、功率因数、直流侧 300 Hz 纹波电流。用 ngspice 核对（selftest）。
- ThermalNet         冷却水回路(冷水机) + IGBT 模块(Foster) + 整流桥 + 谐振电容 + 匹配变压器 + 母线电解电容 + 柜内空气。
                     每个节点用指数精确更新（任意步长稳定），长时间运行可以 1 s 一步跑 8 h。
- fault_transient()  满功率时负载短路（工件碰线圈/打火）的微秒级瞬态：串联 RLC + 方波，矩阵指数精确离散，
                     保护：硬件比较器逐周期（延时 t_d 后全部关断，电流经二极管回馈母线）或只靠 1 kHz 软件环。
- stray_field()      线圈电流 → 圆环线圈毕奥-萨伐尔磁场（椭圆积分）→ 附近金属件涡流损耗/温升、人员暴露距离（ICNIRP 2010）。
- 寿命估算：IGBT 功率循环（LESIT 公式）、电解电容（10 °C 减半）。都是“估算”，不是验证。
"""
import math
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

CP_W = 4180.0     # 水 J/(kg·K)
RHO_W = 997.0


# ===================================================================== 器件参数（通用量级）
@dataclass
class Module:
    """1200 V / 600 A IGBT 半桥模块（通用参数，量级参照常见 62 mm 模块数据手册，125 °C；不是具体型号）。"""
    name: str = "通用 1200 V/600 A IGBT 半桥模块 ×2（全桥）"
    V_ces: float = 1200.0
    I_nom: float = 600.0
    I_crm: float = 1200.0          # 可重复峰值电流 2×Ic
    Vce0: float = 0.80; rce: float = 1.6e-3       # IGBT 导通 Vce = Vce0 + rce·i
    Vf0: float = 0.85; rf: float = 1.3e-3         # 二极管
    E_off_ref: float = 60e-3       # 600 V / 600 A 硬关断能量 J
    E_on_ref: float = 45e-3        # 硬开通（只在容性区出现）
    E_rr_ref: float = 35e-3        # 二极管反向恢复（只在容性区）
    V_ref: float = 600.0; I_ref: float = 600.0; kv: float = 1.3
    k_snub: float = 0.5            # 关断缓冲电容把关断能量降到的比例（假设）
    Rth_jc_T: float = 0.045        # K/W 每个 IGBT
    Rth_jc_D: float = 0.080        # K/W 每个二极管
    Rth_ch: float = 0.018          # K/W 每个开关位置（壳→冷板，含导热硅脂）
    tau_jc: float = 0.08           # s 结→壳时间常数
    Rth_hw: float = 0.012          # K/W 每个模块的冷板→水
    C_plate: float = 2500.0        # J/K 每个模块的冷板（铝/铜板+水）
    Tj_max: float = 150.0
    Tj_design: float = 125.0
    T_ntc_trip: float = 80.0       # 基板 NTC 过温跳闸 °C


def _esw(Eref, I, V, m: Module):
    return Eref * (max(I, 0.0) / m.I_ref) * (max(V, 0.0) / m.V_ref) ** m.kv


def inverter_losses(I_pk, phi, f, Vbus, m: Module = None):
    """全桥每个开关位置（IGBT+反并二极管）的损耗。I_pk 输出电流峰值 A；phi 负载相角 rad（>0 感性）；f 开关频率 Hz。
    返回 dict：每个位置 P_T/P_D/P_sw，全桥合计 P_total，关断电流 I_off，模式 ZVS/容性。
    推导（电压方波在 θ∈[0,π] 为正，电流 i = I_pk·sin(θ−φ)）：
      φ>0：θ∈[0,φ] 二极管导通，θ∈[φ,π] IGBT 导通，θ=π 关断电流 I_pk·sinφ；
      I_avg,T = I_pk(1+cosφ)/2π，I²_rms,T = I_pk²/2π·[(π−φ)/2 + sin2φ/4]；二极管 (1−cosφ)、[φ/2 − sin2φ/4]。"""
    m = m or Module()
    p = abs(phi)
    if phi >= 0:
        aT = I_pk * (1 + math.cos(p)) / (2 * math.pi)
        rT2 = I_pk ** 2 / (2 * math.pi) * ((math.pi - p) / 2 + math.sin(2 * p) / 4)
        aD = I_pk * (1 - math.cos(p)) / (2 * math.pi)
        rD2 = I_pk ** 2 / (2 * math.pi) * (p / 2 - math.sin(2 * p) / 4)
        I_off = I_pk * math.sin(p)
        Esw_T = m.k_snub * _esw(m.E_off_ref, I_off, Vbus, m)
        Esw_D = 0.0
        mode = "ZVS"
    else:
        # 容性：开通时电流已为正（硬开通，对管二极管反向恢复），关断时电流已过零（二极管接续，零电流关断）
        aT = I_pk * (1 + math.cos(p)) / (2 * math.pi)
        rT2 = I_pk ** 2 / (2 * math.pi) * ((math.pi - p) / 2 + math.sin(2 * p) / 4)
        aD = I_pk * (1 - math.cos(p)) / (2 * math.pi)
        rD2 = I_pk ** 2 / (2 * math.pi) * (p / 2 - math.sin(2 * p) / 4)
        I_off = -I_pk * math.sin(p)
        Ion = I_pk * math.sin(p)
        Esw_T = _esw(m.E_on_ref, Ion, Vbus, m)
        Esw_D = _esw(m.E_rr_ref, Ion, Vbus, m)
        mode = "容性(硬开通)"
    P_T = m.Vce0 * aT + m.rce * rT2 + Esw_T * f
    P_D = m.Vf0 * aD + m.rf * rD2 + Esw_D * f
    return dict(P_T=P_T, P_D=P_D, P_pos=P_T + P_D, P_sw=(Esw_T + Esw_D) * f, P_cond=P_T + P_D - (Esw_T + Esw_D) * f,
                P_total=4 * (P_T + P_D), I_off=I_off, mode=mode, I_avg_T=aT, I_rms_T=math.sqrt(max(rT2, 0)), I_avg_D=aD,
                I_rms_D=math.sqrt(max(rD2, 0)))


# ===================================================================== 三相二极管整流（自写时域仿真）
def rectifier_sim(P=None, R_load=None, V_ll=380.0, f=50.0, L=0.15e-3, Rs=0.02, C=12e-3, cycles=12, dt=5e-6, ret_wave=False):
    """理想二极管三相桥 + 每相 Rs、L + 母线电容 C + 负载（恒功率 P 或电阻 R_load）。
    状态：三相线电流 i_k，母线电压 v。导通集：i_k>0 接正母线，i_k<0 接负母线，=0 悬空。
    正母线电位 V+ 由“导通相 di/dt 之和为 0”（三线制 Σi=0）解出。断流：电流过零即关断；悬空相电位越过母线即导通。
    最后 2 个工频周期统计：线电流有效值、基波（DFT）、THD、PF（= P_avg/(√3·V_ll·I_rms)）、电容纹波电流。"""
    w = 2 * math.pi * f
    Vph = V_ll / math.sqrt(3) * math.sqrt(2)
    ph = np.array([0.0, -2 * math.pi / 3, 2 * math.pi / 3])
    i = np.zeros(3)
    v = 1.35 * V_ll * 0.97
    n = int(round(cycles / f / dt))
    keep = int(round(2 / f / dt))
    rec_i = np.zeros((keep, 3)); rec_v = np.zeros(keep); rec_ic = np.zeros(keep); rec_p = np.zeros(keep)
    state = np.zeros(3, int)            # +1 / -1 / 0
    for s in range(n):
        t = s * dt
        e = Vph * np.cos(w * t + ph)
        # 确定导通集（最多迭代 4 次）
        for _ in range(4):
            on = state != 0
            if on.sum() >= 2:
                npl = (state == 1).sum(); nmi = (state == -1).sum()
                if npl == 0 or nmi == 0:
                    state[:] = 0; continue
                Vp = (np.sum((e - Rs * i)[on]) + nmi * v) / on.sum()
            else:
                Vp = None
            changed = False
            if Vp is None:
                # 没有导通路径：线电压最大的两相是否超过母线电压
                kmax, kmin = int(np.argmax(e)), int(np.argmin(e))
                if e[kmax] - e[kmin] > v:
                    state[:] = 0; state[kmax] = 1; state[kmin] = -1; changed = True
            else:
                Vm = Vp - v
                for k in range(3):
                    if state[k] == 0:
                        if e[k] > Vp + 1e-9:
                            state[k] = 1; changed = True
                        elif e[k] < Vm - 1e-9:
                            state[k] = -1; changed = True
            if not changed:
                break
        on = state != 0
        if on.sum() >= 2 and (state == 1).any() and (state == -1).any():
            nmi = (state == -1).sum()
            Vp = (np.sum((e - Rs * i)[on]) + nmi * v) / on.sum()
            Vn = np.where(state == 1, Vp, Vp - v)
            di = np.where(on, (e - Rs * i - Vn) / L, 0.0)
            inew = i + di * dt
            # 过零关断
            for k in range(3):
                if state[k] == 1 and inew[k] <= 0:
                    inew[k] = 0.0; state[k] = 0
                elif state[k] == -1 and inew[k] >= 0:
                    inew[k] = 0.0; state[k] = 0
            if (state != 0).sum() < 2:
                inew[:] = 0.0; state[:] = 0
            else:
                inew[state != 0] -= inew.sum() / (state != 0).sum() if abs(inew.sum()) > 0 else 0.0
            i = inew
        else:
            i[:] = 0.0
        idc = float(np.sum(i[state == 1]))
        iload = (P / max(v, 50.0)) if P is not None else v / R_load
        ic = idc - iload
        v += ic / C * dt
        if s >= n - keep:
            j = s - (n - keep)
            rec_i[j] = i; rec_v[j] = v; rec_ic[j] = ic; rec_p[j] = v * iload
    tt = np.arange(keep) * dt
    Irms = float(np.sqrt(np.mean(rec_i ** 2, axis=0)).mean())
    a = rec_i[:, 0]
    c1 = 2 * np.mean(a * np.cos(w * tt)); s1 = 2 * np.mean(a * np.sin(w * tt))
    I1 = math.hypot(c1, s1) / math.sqrt(2)
    THD = math.sqrt(max(Irms ** 2 - I1 ** 2, 0)) / I1 if I1 > 0 else float("nan")
    Pavg = float(rec_p.mean())
    S = math.sqrt(3) * V_ll * Irms
    out = dict(P=Pavg, Vdc=float(rec_v.mean()), Vdc_pp=float(rec_v.max() - rec_v.min()), I_rms=Irms, I1=I1, THD=THD,
               PF=Pavg / S if S > 0 else 0.0, DPF=(Pavg / (math.sqrt(3) * V_ll * I1)) if I1 > 0 else 0.0,
               I_cap_rms=float(np.sqrt(np.mean(rec_ic ** 2))), I_pk=float(np.abs(rec_i).max()))
    if ret_wave:
        out["wave"] = dict(t=tt, i=rec_i, v=rec_v)
    return out


@lru_cache(maxsize=64)
def rectifier_pf(P_kW_rounded, V_ll=380.0, L=0.15e-3, Rs=0.02, C=12e-3):
    """查表用（功率按 1 kW 取整缓存）。"""
    P = max(float(P_kW_rounded), 0.5) * 1e3
    return rectifier_sim(P=P, V_ll=V_ll, L=L, Rs=Rs, C=C)


# ===================================================================== 冷却与温升网络
@dataclass
class CoolingSpec:
    V_loop_L: float = 400.0        # 冷却水箱+管路水量 L
    Q_chiller: float = 30e3        # 冷水机制冷量 W（T_amb 35 °C 名义值）
    T_set: float = 25.0            # 冷水机设定 °C
    band: float = 2.0              # 比例带 K（T_set 上方 band K 内线性加到满负荷）
    UA_loop_amb: float = 15.0      # 水箱/管路对环境 W/K
    T_warn_in: float = 35.0        # 进水温度报警
    T_trip_in: float = 40.0        # 进水温度跳闸
    Q_inv_Lmin: float = 12.0       # 逆变/整流冷板支路流量 L/min
    Q_cap_Lmin: float = 6.0        # 谐振电容支路
    Q_tr_Lmin: float = 8.0         # 匹配变压器支路
    # 匹配变压器（水冷）：
    n_tr: float = 10.0             # 匝比（线圈电流 = n·逆变器电流），假设
    P_core_ref: float = 350.0      # 额定电压、25 kHz 时铁芯损耗 W（铁氧体/非晶，量级假设）
    f_core_ref: float = 25e3
    V1_ref: float = 462.0
    G_tr: float = 80.0; C_tr: float = 30e3
    tr_hot: float = 130.0          # 绕组/磁芯热点报警 °C（F 级 155 留裕量）
    # 谐振电容（水冷薄膜）
    tan_d: float = 3e-4
    Vc_rated: float = 2000.0       # 额定电压 V 有效值
    Qc_rated: float = 600e3        # 额定无功 var
    G_cap: float = 25.0; C_cap: float = 4e3
    cap_hot: float = 70.0
    # 母线电解电容（柜内风冷）
    C_bus_desc: str = "12 mF（2 串 5 并 ×4700 µF/400 V，示意）"
    ESR_bus_each: float = 0.030; n_series: int = 2; n_par: int = 5
    I_ripple_rated_each: float = 18.0   # A（100 Hz，85 °C）
    Rth_can: float = 2.5           # K/W 芯→柜内空气（每只）
    L0_h: float = 8000.0; T_life_ref: float = 105.0
    # 柜内空气
    P_cab_misc: float = 300.0      # 控制电源、接触器线圈、风机等 W
    G_cab_hex: float = 120.0       # 柜内空水换热器 W/K
    G_cab_wall: float = 25.0
    C_cab: float = 60e3
    # 整流桥（冷板与 IGBT 同一路水）
    Vf_rect: float = 1.0           # 每只二极管压降 V（一次导通两只）
    Rth_rect: float = 0.03


class ThermalNet:
    """所有节点：T_w(冷却水箱/进水) T_plate[2] Tj(IGBT/二极管, Foster 一阶) T_rect T_tr T_cap T_bus_can T_cab。
    线圈温度仍在 Machine 里算（沿用 v0.8 模型），只是进水温度来自这里。"""

    def __init__(self, cs: CoolingSpec = None, mod: Module = None, T0=25.0, T_amb=35.0):
        self.cs = cs or CoolingSpec(); self.m = mod or Module()
        self.T_amb = T_amb
        self.T_w = T0; self.T_plate = T0; self.dTjc_T = 0.0; self.dTjc_D = 0.0
        self.T_rect = T0; self.T_tr = T0; self.T_cap = T0; self.T_can = T0; self.T_cab = T0
        self.Q_ch = 0.0
        self.flow_frac = 1.0
        self.E = dict(chiller=0.0, amb=0.0)
        self.max = {}
        self.L_used_bus = 0.0           # 电解电容寿命消耗（小时，按 105 °C 当量）

    @staticmethod
    def _relax(T, Tss, tau, dt):
        return Tss + (T - Tss) * math.exp(-dt / max(tau, 1e-9))

    def step(self, dt, L, P_coil_to_water=0.0, P_other_water=0.0):
        """L：损耗字典（W）pos_T,pos_D（每个开关位置的 IGBT/二极管）, rect, tr_cu, tr_core, cap, bus_can(每只), cab。
        P_coil_to_water：线圈传给水的热（Machine 算）；P_other_water：其它进水热（淬火水箱换热器）。"""
        cs, m = self.cs, self.m
        ff = max(self.flow_frac, 1e-3)
        mcp_inv = cs.Q_inv_Lmin / 60.0 * RHO_W / 1e3 * CP_W * ff
        # IGBT 模块：每模块 2 个开关位置
        P_mod = 2.0 * (L.get("pos_T", 0) + L.get("pos_D", 0))
        Rhw = m.Rth_hw / max(ff, 0.05) ** 0.8          # 流量下降 → 对流换热下降（Dittus–Boelter 0.8 次方）
        T_win = self.T_w + 0.5 * (2 * P_mod + L.get("rect", 0)) / max(mcp_inv, 1e-6)   # 支路平均水温
        Tp_ss = T_win + P_mod * Rhw
        self.T_plate = self._relax(self.T_plate, Tp_ss, m.C_plate * Rhw, dt)
        dT_T = L.get("pos_T", 0) * (m.Rth_jc_T + m.Rth_ch)
        dT_D = L.get("pos_D", 0) * (m.Rth_jc_D + m.Rth_ch)
        self.dTjc_T = self._relax(self.dTjc_T, dT_T, m.tau_jc, dt)
        self.dTjc_D = self._relax(self.dTjc_D, dT_D, m.tau_jc, dt)
        self.T_rect = self._relax(self.T_rect, T_win + L.get("rect", 0) * cs.Rth_rect, 30.0, dt)
        # 变压器、谐振电容
        P_tr = L.get("tr_cu", 0) + L.get("tr_core", 0)
        G = cs.G_tr * max(ff, 0.05) ** 0.8
        self.T_tr = self._relax(self.T_tr, self.T_w + P_tr / G, cs.C_tr / G, dt)
        G = cs.G_cap * max(ff, 0.05) ** 0.8
        self.T_cap = self._relax(self.T_cap, self.T_w + L.get("cap", 0) / G, cs.C_cap / G, dt)
        # 柜内空气 + 母线电解电容
        n_can = cs.n_series * cs.n_par
        P_cab = cs.P_cab_misc + n_can * L.get("bus_can", 0)
        Gc = cs.G_cab_hex * ff ** 0.5 + cs.G_cab_wall
        Tcab_ss = (P_cab + cs.G_cab_hex * ff ** 0.5 * self.T_w + cs.G_cab_wall * self.T_amb) / Gc
        self.T_cab = self._relax(self.T_cab, Tcab_ss, cs.C_cab / Gc, dt)
        self.T_can = self._relax(self.T_can, self.T_cab + L.get("bus_can", 0) * cs.Rth_can, 600.0, dt)
        # 冷却水箱：所有水冷损耗 + 线圈 + 柜内换热 + 淬火换热器 − 冷水机 − 环境
        Q_in = 4 * (L.get("pos_T", 0) + L.get("pos_D", 0)) + L.get("rect", 0) + P_tr + L.get("cap", 0) + P_coil_to_water + P_other_water \
            + cs.G_cab_hex * ff ** 0.5 * (self.T_cab - self.T_w)
        x = (self.T_w - cs.T_set) / cs.band
        self.Q_ch = cs.Q_chiller * float(np.clip(x, 0.0, 1.0)) * self._chiller_derate()
        Q_amb = cs.UA_loop_amb * (self.T_w - self.T_amb)
        Cw = cs.V_loop_L * RHO_W / 1e3 * CP_W
        self.T_w += dt * (Q_in - self.Q_ch - Q_amb) / Cw
        self.E["chiller"] += self.Q_ch * dt; self.E["amb"] += Q_amb * dt
        # 电解电容寿命消耗（Arrhenius 10 °C 减半）
        self.L_used_bus += dt / 3600.0 * 2.0 ** ((self.T_can - cs.T_life_ref) / 10.0)
        for k, v in self.temps().items():
            self.max[k] = max(self.max.get(k, -1e9), v)

    def _chiller_derate(self):
        # 环境每高于 35 °C 1 K，制冷量下降 2.5 %（风冷冷水机典型量级，假设）
        return max(0.3, 1.0 - 0.025 * max(self.T_amb - 35.0, 0.0))

    def temps(self):
        return dict(T_w=self.T_w, T_plate=self.T_plate, Tj_T=self.T_plate + self.dTjc_T, Tj_D=self.T_plate + self.dTjc_D,
                    T_rect=self.T_rect, T_tr=self.T_tr, T_cap=self.T_cap, T_can=self.T_can, T_cab=self.T_cab)


def component_losses(sp, s, Vbus, f, P_in_dc, ripple300=None):
    """从功率级稳态解 s（Stage.solve 的结果）算各器件损耗。返回 dict（W）。"""
    from .power import Spec  # noqa
    cs = sp.cool; m = sp.mod
    I = s["I"]; Ipk = I * math.sqrt(2)
    phi = math.atan2(s["X"], s["R"]) if I > 0 else 0.0
    inv = inverter_losses(Ipk, phi, f, Vbus, m) if I > 0 else dict(P_T=0, P_D=0, P_total=0, I_off=0, P_sw=0, mode="-")
    # 匹配变压器：铜损是 R_coil 的一部分（sp.frac_tr），铁损 ∝ f^1.4·B^2.5，B ∝ V1/f
    if I > 0:
        Bn = (s["V1"] / cs.V1_ref) * (cs.f_core_ref / f)
        P_core = cs.P_core_ref * (f / cs.f_core_ref) ** 1.4 * Bn ** 2.5
    else:
        P_core = 0.0
    Vc = I / (2 * math.pi * f * sp.C_res) if I > 0 else 0.0
    Qc = I * Vc
    P_cap = Qc * cs.tan_d
    Id = P_in_dc / max(Vbus, 1.0)
    P_rect = 2 * cs.Vf_rect * Id
    # 母线电解电容：300 Hz 纹波（来自整流仿真）；高频纹波由薄膜吸收电容承担（不计入电解）
    I300 = ripple300 if ripple300 is not None else 0.0
    n_can = cs.n_series * cs.n_par
    I_each = I300 / cs.n_par
    P_can = I_each ** 2 * cs.ESR_bus_each
    return dict(pos_T=inv["P_T"], pos_D=inv["P_D"], inv_total=inv["P_total"], inv_sw=4 * inv["P_sw"], I_off=inv["I_off"], mode=inv["mode"],
                phi_deg=math.degrees(phi), tr_cu=s.get("P_coil", 0.0) * sp.frac_tr, tr_core=P_core, cap=P_cap, Vc=Vc, Qc=Qc,
                rect=P_rect, bus_can=P_can, I_can=I_each, Id=Id)


# ===================================================================== 寿命估算
def lesit_cycles(dTj, Tj_mean_C):
    """LESIT 功率循环寿命 Nf = 640·ΔTj^−5·exp(Ea/(R·Tm))，Ea = 7.8e4 J/mol（文献常用拟合，焊料/键合线老式模块；估算）。"""
    if dTj <= 0.5:
        return float("inf")
    return 640.0 * dTj ** -5.0 * math.exp(7.8e4 / (8.314 * (Tj_mean_C + 273.15)))


def ecap_life_h(T_core, cs: CoolingSpec):
    return cs.L0_h * 2.0 ** ((cs.T_life_ref - T_core) / 10.0)


# ===================================================================== 满功率负载短路瞬态
def _expm(A):
    from scipy.linalg import expm
    return expm(A)


def fault_transient(L, C, R, Vbus, f, L_short, R_short, t_fault=0.4e-3, t_end=1.6e-3, prot="hw", I_trip=480.0, t_d=2e-6,
                    t_sw_period=1e-3, dt=20e-9, R_switch=1e-4):
    """串联 RLC（L、C、R 总值）由 ±Vbus 方波（频率 f，固定）驱动；t_fault 时刻串联元件中的 (L_short, R_short) 一段被短路开关并上
    （工件碰线圈/对地打火：工件等效电阻和一部分电感被短接）。拓扑：主回路 L1=L−L_short、R1=R−R_short、C；短接段自成回路
    L_short–R_short–开关（R_switch）。保护：
      prot="hw"  ：硬件比较器 |i|>I_trip，延时 t_d 后四管全关，电流经二极管回馈母线（逆变器输出 = −sign(i)·Vbus）直到过零；
      prot="sw"  ：只靠软件环：每 t_sw_period 检查一次（用上一周期电流峰值），超过 I_trip 才关断（最坏情况在故障后约 1 ms）；
      prot="none"：不保护。
    返回 t, i(主回路), vC, 关断时刻, 峰值。离散化：x' = A x + B u，u 分段常数 → 矩阵指数精确。"""
    L1 = L - L_short; R1 = R - R_short
    # 状态 x = [i1, vC, i2]
    def mats(after):
        if not after:
            A = np.array([[-R / L, -1 / L, 0.0], [1 / C, 0, 0], [0, 0, 0]])
            B = np.array([1 / L, 0, 0])
        else:
            # 短路后：主回路 L1 di1 = u − R1 i1 − vC − R_switch(i1 − i2)；短接段 L2 di2 = −R_short i2 + R_switch(i1 − i2)
            A = np.array([[-(R1 + R_switch) / L1, -1 / L1, R_switch / L1], [1 / C, 0, 0],
                          [R_switch / L_short, 0, -(R_short + R_switch) / L_short]])
            B = np.array([1 / L1, 0, 0])
        M = np.zeros((4, 4)); M[:3, :3] = A; M[:3, 3] = B
        E = _expm(M * dt)
        return E[:3, :3], E[:3, 3]
    Pa, Qa = mats(False); Pb, Qb = mats(True)
    n = int(round(t_end / dt)); per = 1.0 / f
    x = np.zeros(3)
    out_t = np.empty(n); out_i = np.empty(n); out_v = np.empty(n)
    t_off = None; tripped = False; trip_seen = None; stopped = False
    pk_win = 0.0; next_chk = t_fault + t_sw_period
    for k in range(n):
        t = k * dt
        after = t >= t_fault
        if after and not getattr(fault_transient, "_dummy", False):
            pass
        P, Q = (Pb, Qb) if after else (Pa, Qa)
        if after and k > 0 and (k - 1) * dt < t_fault:
            x[2] = x[0]                         # 短接瞬间短接段电流 = 主回路电流
        if not tripped:
            u = Vbus if (t % per) < per / 2 else -Vbus
        else:
            if stopped:
                u = None
            else:
                u = -math.copysign(Vbus, x[0]) if abs(x[0]) > 1e-6 else 0.0
        if u is None:
            # 四管关断且电流为零：主回路断开，只剩短接段自己衰减
            x[0] = 0.0
            x[2] *= math.exp(-dt * (R_short + R_switch) / L_short) if after else 1.0
        else:
            x0 = x[0]
            x = P @ x + Q * u
            if tripped and x0 != 0 and np.sign(x[0]) != np.sign(x0):
                x[0] = 0.0; stopped = True
        out_t[k] = t; out_i[k] = x[0]; out_v[k] = x[1]
        # 保护
        if not tripped:
            if prot == "hw":
                if trip_seen is None and abs(x[0]) > I_trip:
                    trip_seen = t
                if trip_seen is not None and t >= trip_seen + t_d:
                    tripped = True; t_off = t
            elif prot == "sw":
                pk_win = max(pk_win, abs(x[0]))
                if t >= next_chk:
                    if pk_win > I_trip:
                        tripped = True; t_off = t
                    pk_win = 0.0; next_chk += t_sw_period
    m_after = out_t >= t_fault
    return dict(t=out_t, i=out_i, vC=out_v, t_off=t_off, I_pk_after=float(np.abs(out_i[m_after]).max()),
                I_pk_before=float(np.abs(out_i[(out_t > t_fault - 3 * per) & (out_t < t_fault)]).max()),
                Vc_pk=float(np.abs(out_v).max()), Vc_pk_before=float(np.abs(out_v[(out_t > t_fault - 3 * per) & (out_t < t_fault)]).max()),
                i2t=float(np.sum(out_i[m_after] ** 2) * dt))


# ===================================================================== 杂散磁场
MU0 = 4e-7 * math.pi


def loop_H(I, a, r, z):
    """单匝圆环（半径 a，电流 I）在 (r,z) 处的磁场 (Hr, Hz)，A/m。椭圆积分精确式。"""
    from scipy.special import ellipk, ellipe
    r = max(r, 1e-9)
    Q = (a + r) ** 2 + z ** 2
    k2 = 4 * a * r / Q
    K = ellipk(k2); E = ellipe(k2)
    D = (a - r) ** 2 + z ** 2
    Hz = I / (2 * math.pi * math.sqrt(Q)) * (K + (a * a - r * r - z * z) / D * E)
    Hr = I * z / (2 * math.pi * r * math.sqrt(Q)) * (-K + (a * a + r * r + z * z) / D * E)
    return Hr, Hz


def coil_H(I_coil, r, z, a=0.05, turns=1, pitch=0.0):
    Hr = Hz = 0.0
    for j in range(turns):
        zj = (j - (turns - 1) / 2) * pitch
        hr, hz = loop_H(I_coil, a, r, z - zj)
        Hr += hr; Hz += hz
    return math.hypot(Hr, Hz)


STRAY_PARTS = [
    # 名称, 径向距离 r(m), 轴向 z(m), 受场面积 m², 材料(ρ Ω·m, μr), 质量 kg, 散热面积 m², 允许温度 °C
    ("喷淋环 SUS304 Ø10 管", 0.075, -0.03, 0.0047, (7.2e-7, 1.0), 0.20, 0.0094, 200.0),
    ("线圈下方托板 Q235 10 mm（有磁性）", 0.12, -0.06, 0.020, (1.6e-7, 100.0), 3.0, 0.045, 100.0),
    ("机架立柱 Q235 40×40×3", 0.25, 0.00, 0.030, (1.6e-7, 100.0), 4.0, 0.10, 60.0),
    ("铝防护罩 2 mm", 0.30, 0.00, 0.20, (2.8e-8, 1.0), 1.1, 0.40, 60.0),
    ("电控柜门 Q235 1.5 mm", 1.0, 0.00, 0.80, (1.6e-7, 100.0), 12.0, 1.6, 60.0),
]


def stray_field(I_coil, f, a=0.05, parts=STRAY_PARTS, h=12.0):
    """附近金属件：表面损耗 P = ½·Hpk²·Rs·A（厚度 ≫ 透入深度，场按线圈单匝圆环在零件中心处的值，不计零件本身的屏蔽/聚磁）。
    稳态温升 ΔT = P/(h·A_散热)，h 自然对流+辐射 12 W/(m²K)（假设）。人员暴露：沿线圈平面找 H 降到 ICNIRP 2010 参考水平的距离
    （3 kHz–10 MHz：职业 80 A/m、公众 21 A/m，有效值）。"""
    w = 2 * math.pi * f
    rows = []
    for (nm, r, z, A, (rho, mur), mass, A_cool, Tlim) in parts:
        H = coil_H(I_coil, r, z, a)                      # 有效值
        Rs = math.sqrt(w * MU0 * mur * rho / 2)
        P = 0.5 * (H * math.sqrt(2)) ** 2 * Rs * A
        dT = P / (h * A_cool)
        delta = math.sqrt(2 * rho / (w * MU0 * mur))
        rows.append(dict(name=nm, r=r, H=H, P=P, dT=dT, T_lim=Tlim, delta_mm=delta * 1e3))
    def dist(lim):
        lo, hi = a * 1.2, 20.0
        for _ in range(60):
            mid = math.sqrt(lo * hi)
            if coil_H(I_coil, mid, 0.0, a) > lim:
                lo = mid
            else:
                hi = mid
        return hi
    return dict(rows=rows, d_occ=dist(80.0), d_pub=dist(21.0), H_05=coil_H(I_coil, 0.5, 0.0, a), H_10=coil_H(I_coil, 1.0, 0.0, a))
