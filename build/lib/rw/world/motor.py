"""电机：三相异步电机模型（铭牌 → 机械特性），带负载起动仿真，频繁起停发热校核，
以及"参数合不合物理/现有工业水平"的检查。单位 SI。"""
import math
from dataclasses import dataclass
import numpy as np

OK, WARN, FAIL = "OK", "WARN", "FAIL"

# IE3 4 极 50 Hz 最低效率 (GB 18613 / IEC 60034-30-1) 与典型重量 (铸铁机座, 不带刹车)
IE3 = [(0.12, 0.643, 4.0), (0.18, 0.69, 5.0), (0.25, 0.73, 5.5), (0.37, 0.77, 7.0), (0.55, 0.808, 9.0),
       (0.75, 0.825, 11.0), (1.1, 0.841, 14.0), (1.5, 0.853, 17.0), (2.2, 0.867, 24.0), (3.0, 0.877, 30.0),
       (4.0, 0.886, 38.0), (5.5, 0.896, 50.0), (7.5, 0.904, 62.0), (11, 0.914, 95.0), (15, 0.92, 120.0),
       (22, 0.927, 180.0), (37, 0.936, 270.0), (55, 0.941, 380.0), (90, 0.948, 600.0), (160, 0.955, 1050.0)]
IE4_GAIN = 0.03  # IE4 比 IE3 约高 2–4 个点，最高超效率参考


def _ie3(P_kW):
    Ps = [r[0] for r in IE3]
    return float(np.interp(math.log(P_kW), np.log(Ps), [r[1] for r in IE3])), \
        float(np.exp(np.interp(math.log(P_kW), np.log(Ps), np.log([r[2] for r in IE3]))))


@dataclass
class InductionMotor:
    name: str = "YE3 0.75kW 4P"
    P: float = 750.0          # 额定输出 W
    V: float = 380.0
    f: float = 50.0
    poles: int = 4
    n_rated: float = 1400.0   # rpm
    eta: float = 0.825
    pf: float = 0.76
    Tst_ratio: float = 2.3    # 起动转矩/额定
    Tmax_ratio: float = 2.3   # 最大转矩/额定
    Ist_ratio: float = 6.0
    J: float = 0.0021         # 转子惯量 kg·m²（含刹车）
    mass: float = 13.0        # kg
    Z0: float = 3000.0        # 空载允许起动次数/小时（样本值，未给时按经验）

    @property
    def n_sync(self):
        return 120 * self.f / self.poles

    @property
    def w_sync(self):
        return 2 * math.pi * self.n_sync / 60

    @property
    def Tn(self):
        return self.P / (2 * math.pi * self.n_rated / 60)

    @property
    def In(self):
        return self.P / (math.sqrt(3) * self.V * self.eta * self.pf)

    def sk(self):
        sn = 1 - self.n_rated / self.n_sync
        r = self.Tmax_ratio
        return sn * (r + math.sqrt(r * r - 1))

    def torque(self, w):
        s = max(1 - w / self.w_sync, 1e-6)
        sk = self.sk()
        Tm = self.Tmax_ratio * self.Tn
        T = 2 * Tm / (s / sk + sk / s)
        if s > sk:  # 高转差段向样本起动转矩修正
            T1 = 2 * Tm / (1 / sk + sk)
            frac = (s - sk) / (1 - sk)
            T = T + frac * (self.Tst_ratio * self.Tn - T1)
        return T

    def current(self, w):
        s = max(1 - w / self.w_sync, 0)
        sn = 1 - self.n_rated / self.n_sync
        I0 = self.In * 0.55
        Ist = self.Ist_ratio * self.In
        # 负载电流随转差近似：s<=sn 线性到 In，s>sn 平滑到 Ist
        if s <= sn:
            return I0 + (self.In - I0) * s / sn
        return self.In + (Ist - self.In) * min((s - sn) / (1 - sn), 1) ** 0.6

    def plausibility(self):
        out = []
        add = lambda lv, t: out.append((lv, t))
        P_kW = self.P / 1e3
        eta3, m3 = _ie3(P_kW)
        if self.n_rated >= self.n_sync:
            add(FAIL, f"异步电机额定转速 {self.n_rated:g} rpm ≥ 同步转速 {self.n_sync:g} rpm，不可能（没有转差就没有转矩）")
        if self.eta >= 1:
            add(FAIL, "效率 ≥ 100%，违反能量守恒")
        elif self.eta > eta3 + IE4_GAIN + 0.01:
            add(FAIL, f"效率 {self.eta*100:.1f}% 高于目前 IE4/IE5 水平（同功率 IE3 {eta3*100:.1f}%），现实做不出来")
        elif self.eta < eta3 - 0.05:
            add(WARN, f"效率 {self.eta*100:.1f}% 低于 IE3 {eta3*100:.1f}%，国内已不允许销售新的低效电机")
        if not 0.3 < self.pf <= 1:
            add(FAIL, f"功率因数 {self.pf} 不合理（0.3~1）")
        if self.mass < 0.5 * m3:
            add(FAIL, f"{P_kW:g} kW 电机只有 {self.mass:g} kg，比同功率 IE3 典型 {m3:.0f} kg 轻一半以上，铜铁装不下，会过热")
        elif self.mass < 0.75 * m3:
            add(WARN, f"重量 {self.mass:g} kg 偏轻（典型 {m3:.0f} kg），需要铝壳+强制风冷或高等级绝缘")
        if self.Tmax_ratio < 1.6:
            add(WARN, "最大转矩倍数 <1.6，国标要求 ≥1.6~2.0")
        if not out:
            add(OK, f"参数符合物理和现有工业水平（同功率 IE3：效率 ≥{eta3*100:.1f}%，约 {m3:.0f} kg）")
        return out


def simulate_start(m: InductionMotor, J_load_motor_side, T_load_motor_side, t_end=2.0, dt=1e-4, V_ratio=1.0):
    """直接起动：J dω/dt = T_m(ω)·V² − T_L(ω)。返回时间、转速、转矩、电流。"""
    J = m.J + J_load_motor_side
    w, t = 0.0, 0.0
    hist = dict(t=[], n=[], T=[], I=[])
    t90 = None
    E_rotor = 0.0
    while t < t_end:
        Tm = m.torque(w) * V_ratio**2
        TL = T_load_motor_side(w) if callable(T_load_motor_side) else T_load_motor_side
        a = (Tm - TL) / J
        s = 1 - w / m.w_sync
        E_rotor += Tm * s * m.w_sync * dt        # 转子铜耗 = 气隙功率 × s
        w = max(w + a * dt, 0.0)
        t += dt
        if t90 is None and w >= 0.9 * (m.w_sync * (m.n_rated / m.n_sync)):
            t90 = t
        if len(hist["t"]) == 0 or t - hist["t"][-1] >= 2e-3:
            hist["t"].append(t); hist["n"].append(w * 60 / (2 * math.pi)); hist["T"].append(Tm); hist["I"].append(m.current(w) * V_ratio)
    return dict(t_accel=t90, n_end=w * 60 / (2 * math.pi), E_rotor=E_rotor, stalled=(w < 0.3 * m.w_sync),
                hist={k: np.array(v) for k, v in hist.items()})


def starts_per_hour(m: InductionMotor, J_ext, T_load, cdf=0.4):
    """SEW 样本方法：Z = Z0 · KJ · KM · KP"""
    KJ = 1 / (1 + J_ext / m.J)
    Ma = m.Tst_ratio * m.Tn
    KM = max(1 - T_load / Ma, 0)
    KP = max(1 - 0.6 * cdf, 0.1)
    return m.Z0 * KJ * KM * KP


# ---------------------------------------------------------------- 示例：凸轮分割器转盘分度传动
def example_indexer_drive(J_table=2.45, T_out_peak=35.9, t_index=0.8, cycle=7.89, motor=None):
    m = motor or InductionMotor()
    i_worm, eta_worm = 50.0, 0.75
    eta_cam = 0.9
    w_in = 2 * math.pi * (m.n_rated / i_worm) / 60          # 分割器输入轴
    theta = 2 * math.pi / 6
    Vm = 1.76                                               # 变形正弦曲线最大无因次速度
    w_out_pk = Vm * theta / t_index
    ratio = w_out_pk / w_in                                 # 输出/输入角速度比（峰值）
    T_in_pk = T_out_peak * ratio / eta_cam
    T_motor_pk = T_in_pk / (i_worm * eta_worm)
    J_refl = J_table * (w_out_pk / (m.n_rated * 2 * math.pi / 60)) ** 2 + 0.0004  # +蜗轮减速机自身
    st = simulate_start(m, J_refl, 0.15 * T_motor_pk, t_end=0.3)
    starts = 3600 / cycle
    Zperm = starts_per_hour(m, J_refl, T_motor_pk * 0.5, cdf=t_index / cycle)
    w_drive_angle = 120 / 360 * 60 / (m.n_rated / i_worm)  # 120°驱动角对应时间
    out = dict(T_motor_peak=T_motor_pk, T_motor_rated=m.Tn, util=T_motor_pk / m.Tn, J_refl=J_refl,
               t_accel=st["t_accel"], index_time=w_drive_angle, starts_needed=starts, starts_perm=Zperm,
               P_needed=T_motor_pk * m.n_rated * 2 * math.pi / 60, start=st, motor=m)
    return out
