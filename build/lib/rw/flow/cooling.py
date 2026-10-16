"""100 kW 感应加热电源的冷却水系统（MIT，自写；换热关联式/ε-NTU 与 `ht`(MIT) 对照，水物性/露点与 CoolProp(MIT) 对照）。

系统：内循环(去离子水) 水箱 → 泵 → 分水器 → 并联支路(整流、逆变器、谐振电容、匹配变压器、感应器线圈)
      → 集水器 → 板式换热器(热侧) → 回水箱；板换冷侧接冷水机。
每条支路：流量(由管网求解 rw.flow.network 得到，支路平衡阀 K 可调)、出水温度、
          冷却通道壁温(Gnielinski 换热系数 + 热流密度峰值系数)、壁温与当地饱和温度的余量、结垢温度。
系统：板换 ε-NTU(逆流) 求内循环温度；冷水机选型(按环境温度/出水温度修正，带余量)；
      露点(凝露)、流量开关、温度报警；冷水机跳闸后的升温时间(集总热容)。
单位 SI（温度 °C 显示），所有默认损耗/几何/报警值是**典型假设**，不是某台实机的测量值。
"""
import math
from dataclasses import dataclass, field, replace
import numpy as np

from ..world.fluids import water_rho, water_mu, water_psat, water_tsat, G, P_ATM
from .network import Network, Pipe, Valve, Pump

# ------------------------------------------------------------------ 水物性（IAPWS-95，0.1 MPa 饱和液线附近表值）
_T = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100.0])
_CP = np.array([4219.9, 4195.5, 4184.4, 4180.1, 4179.6, 4181.5, 4185.1, 4190.2, 4196.9, 4205.3, 4215.7])
_K = np.array([0.5610, 0.5800, 0.5984, 0.6154, 0.6305, 0.6435, 0.6543, 0.6631, 0.6700, 0.6753, 0.6791])


def water_cp(T):
    """比热 J/(kg·K)，T °C（0–100）"""
    return float(np.interp(T, _T, _CP))


def water_k(T):
    """导热系数 W/(m·K)，T °C（0–100）"""
    return float(np.interp(T, _T, _K))


def water_pr(T):
    return water_cp(T) * float(water_mu(T)) / water_k(T)


def dew_point(T_air, RH):
    """露点 °C（Magnus，Sonntag 系数 a=17.62, b=243.12 °C；−45~60 °C 误差约 0.35 K）。RH 0..1"""
    a, b = 17.62, 243.12
    g = math.log(max(RH, 1e-6)) + a * T_air / (b + T_air)
    return b * g / (a - g)


# ------------------------------------------------------------------ 管内换热
def nu_gnielinski(Re, Pr, f_darcy=None):
    """Gnielinski（3000<Re<5e6, 0.5<Pr<2000）；f 用 Petukhov 光滑管 f=(0.790 lnRe − 1.64)^−2（同 ht 的默认）。
    Re<2300 取充分发展层流等热流 Nu=4.36；2300–3000 线性过渡（工程处理）。"""
    if Re < 2300:
        return 4.36
    if Re < 3000:
        n_hi = nu_gnielinski(3000.0, Pr)
        return 4.36 + (n_hi - 4.36) * (Re - 2300) / 700.0
    f = f_darcy if f_darcy else (0.790 * math.log(Re) - 1.64) ** -2
    return (f / 8) * (Re - 1000) * Pr / (1 + 12.7 * math.sqrt(f / 8) * (Pr ** (2 / 3) - 1))


def htc_channel(Q_m3s, Dh, n_pass, T):
    """每条并联通道流量 Q/n_pass，返回 (h W/m²K, Re, v m/s)。方截面按水力直径近似。"""
    A = Dh ** 2                                      # 方孔面积 = Dh²（方孔 Dh=边长）
    v = Q_m3s / n_pass / A
    rho, mu = float(water_rho(T)), float(water_mu(T))
    Re = rho * v * Dh / mu
    h = nu_gnielinski(Re, water_pr(T)) * water_k(T) / Dh
    return h, Re, v


# ------------------------------------------------------------------ 板式换热器 ε-NTU
def eps_counterflow(NTU, Cr):
    """逆流 ε(NTU, Cr)；Cr=Cmin/Cmax"""
    if Cr < 1e-9:
        return 1 - math.exp(-NTU)
    if abs(1 - Cr) < 1e-9:
        return NTU / (1 + NTU)
    e = math.exp(-NTU * (1 - Cr))
    return (1 - e) / (1 - Cr * e)


def lmtd(Th_in, Th_out, Tc_in, Tc_out):
    d1, d2 = Th_in - Tc_out, Th_out - Tc_in
    if abs(d1 - d2) < 1e-9:
        return d1
    return (d1 - d2) / math.log(d1 / d2)


@dataclass
class PlateHX:
    UA: float = 9000.0          # W/K（典型：去离子水/冷冻水板换 U≈3000 W/m²K × 3 m²；假设）
    dp_rated: float = 50e3      # 额定流量下热侧压降 Pa（K 由此折算；假设）
    Q_rated: float = 120 / 6e4  # m³/s 额定流量
    D_port: float = 0.032       # 接口直径 m

    def K_equiv(self):
        A = math.pi * self.D_port ** 2 / 4
        v = self.Q_rated / A
        return self.dp_rated / (1000 * v ** 2 / 2)

    def solve(self, Q_heat, m_h, Th_in_guess, Tc_in, m_c, cp_h=4180.0, cp_c=4180.0):
        """已知热负荷 Q(W)、两侧质量流量、冷侧进水 → 热侧进/出、冷侧出（稳态，逆流）。
        内循环稳态：板换带走的热 = 负荷 → Th_in = Tc_in + Q/(ε·Cmin)。"""
        Ch, Cc = m_h * cp_h, m_c * cp_c
        Cmin, Cmax = min(Ch, Cc), max(Ch, Cc)
        NTU = self.UA / Cmin
        eps = eps_counterflow(NTU, Cmin / Cmax)
        Th_in = Tc_in + Q_heat / (eps * Cmin)
        Th_out = Th_in - Q_heat / Ch
        Tc_out = Tc_in + Q_heat / Cc
        return dict(eps=eps, NTU=NTU, Cr=Cmin / Cmax, Th_in=Th_in, Th_out=Th_out, Tc_in=Tc_in, Tc_out=Tc_out,
                    LMTD=lmtd(Th_in, Th_out, Tc_in, Tc_out))


# ------------------------------------------------------------------ 冷水机
# 风冷工业冷水机常见档（名义制冷量 kW @ 出水 7 °C/环境 35 °C；各厂不同，这是典型档位，假设）
CHILLER_SIZES = [(5, 15.0), (8, 23.0), (10, 29.0), (12, 34.0), (15, 43.0), (20, 58.0), (25, 72.0), (30, 86.0), (40, 116.0)]


def chiller_capacity(Q_nom, T_out=20.0, T_amb=35.0):
    """制冷量随出水温度升高而增大（约 +3 %/K），随环境温度升高而下降（约 −1.2 %/K）——典型趋势，估算。"""
    return Q_nom * (1 + 0.03 * (T_out - 7.0)) * (1 - 0.012 * (T_amb - 35.0))


def chiller_select(Q_need, T_out=25.0, T_amb=40.0, margin=1.15):
    for hp, qn in CHILLER_SIZES:
        cap = chiller_capacity(qn * 1e3, T_out, T_amb)
        if cap >= Q_need * margin:
            return dict(hp=hp, Q_nom=qn * 1e3, Q_avail=cap, margin=cap / Q_need, P_elec=cap / 2.8)
    hp, qn = CHILLER_SIZES[-1]
    cap = chiller_capacity(qn * 1e3, T_out, T_amb)
    return dict(hp=hp, Q_nom=qn * 1e3, Q_avail=cap, margin=cap / Q_need, P_elec=cap / 2.8, too_small=True)


# ------------------------------------------------------------------ 支路
@dataclass
class Branch:
    name: str
    loss_frac: float            # 损耗 = loss_frac × P_out（+ loss_fixed）
    loss_fixed: float = 0.0     # W，与功率无关的部分（铁损/驱动/辐射等）
    Dh: float = 0.008           # 冷却通道水力直径 m（方孔边长）
    n_pass: int = 1             # 并联水路数
    L_pass: float = 1.0         # 每路长度 m
    heated_frac: float = 1.0    # 通道周长中受热的比例（线圈电流挤在内侧 → 受热面集中）
    peak: float = 1.5           # 热流密度峰值/平均（假设）
    K_extra: float = 8.0        # 弯头/接头/快插 局部阻力合计
    K_bal: float = 0.0          # 平衡阀附加阻力
    hose_L: float = 2.0; hose_D: float = 0.013   # 进出水软管（各一根合计长度）
    T_out_max: float = 50.0     # 出水温度报警 °C
    Q_min_Lmin: float = 3.0     # 流量开关设定 L/min
    T_wall_scale: float = 60.0  # 壁温超过此值时自来水/硬水易结垢（去离子水不结垢但仍提示）


def default_branches(kind="sic_inverter"):
    """100 kW 串联谐振感应加热电源的典型损耗分配（P_out = 线圈端输出功率；全部为假设值，按 IGBT/SiC 30 kHz 量级）。"""
    return [
        Branch("整流桥+母线", 0.010, 100, Dh=0.008, n_pass=1, L_pass=1.2, peak=1.5, T_out_max=45, Q_min_Lmin=2.0),
        Branch("逆变器模块", 0.020, 150, Dh=0.008, n_pass=2, L_pass=1.0, peak=2.0, T_out_max=40, Q_min_Lmin=4.0),
        Branch("谐振电容组", 0.004, 0, Dh=0.006, n_pass=2, L_pass=1.5, peak=1.5, T_out_max=45, Q_min_Lmin=2.0),
        Branch("匹配变压器", 0.010, 200, Dh=0.008, n_pass=2, L_pass=3.0, peak=1.5, T_out_max=50, Q_min_Lmin=4.0),
        Branch("感应器线圈", 0.170, 1500, Dh=0.008, n_pass=2, L_pass=2.5, heated_frac=0.25, peak=2.0,
               K_extra=14.0, T_out_max=55, Q_min_Lmin=12.0, hose_L=4.0, hose_D=0.016),
    ]


@dataclass
class CoolingSystem:
    P_out: float = 100e3                      # 电源输出功率 W（默认 100 kW 满功率）
    branches: list = field(default_factory=default_branches)
    T_chw: float = 25.0                       # 冷水机出水(板换冷侧进水) °C
    m_chw: float = 1.2                        # 冷侧质量流量 kg/s（≈72 L/min，假设）
    hx: PlateHX = field(default_factory=PlateHX)
    pump_H0: float = 75.0; pump_Q: float = 110 / 6e4; pump_H: float = 55.0; pump_eta: float = 0.55
    tank_L: float = 120.0                     # 水箱+管路水量 L
    T_amb: float = 35.0; RH: float = 0.70     # 柜内空气（凝露判断）
    T_amb_chiller: float = 40.0               # 冷水机所在环境温度（选型）
    p_return: float = 0.0                     # 回水箱表压 Pa（敞口 0）

    # ---------------- 管网
    def network(self, T=30.0):
        net = Network("water", T)
        net.reservoir("水箱出", z=0.0, level=0.5); net.reservoir("水箱回", z=0.0, level=0.5, p_gauge=self.p_return)
        net.junction("泵入", z=0.0); net.junction("分水器", z=1.0); net.junction("集水器", z=1.0); net.junction("板换出", z=0.5)
        net.add(Pipe("吸入管", "水箱出", "泵入", L=0.6, D=0.040, K=1.5))
        net.add(Pump("循环泵", "泵入", "分水器", H0=self.pump_H0, Q_rated=self.pump_Q, H_rated=self.pump_H, eta=self.pump_eta, NPSHr=3.0))
        for b in self.branches:
            a, c = f"{b.name}进", f"{b.name}出"
            net.junction(a, z=1.0); net.junction(c, z=1.0)
            net.add(Pipe(f"{b.name}软管", "分水器", a, L=b.hose_L / 2, D=b.hose_D, rough=0.0015e-3, K=1.0 + b.K_bal))
            # n_pass 并联方孔：等效成一根“面积×n、同 Dh 摩阻”的管 → 用当量圆管直径 D_eq 使流速和 Re 不变不成立，
            # 故用 Pipe 的方式：D = Dh，流量分摊（network 只认单管），这里把 n 路合成：同 dH 下 Q 是单路的 n 倍。
            net.add(_MultiPass(f"{b.name}水道", a, c, Dh=b.Dh, n=b.n_pass, L=b.L_pass, K=b.K_extra))
            net.add(Pipe(f"{b.name}回软管", c, "集水器", L=b.hose_L / 2, D=b.hose_D, rough=0.0015e-3, K=1.0))
        hx = self.hx
        net.add(Pipe("板换(热侧)", "集水器", "板换出", L=0.01, D=hx.D_port, K=hx.K_equiv()))
        net.add(Pipe("回水管", "板换出", "水箱回", L=2.0, D=0.040, K=2.0))
        return net

    def losses(self):
        return {b.name: b.loss_frac * self.P_out + b.loss_fixed * (1.0 if self.P_out > 0 else 0.0) for b in self.branches}

    # ---------------- 稳态
    def solve(self, T_guess=32.0, iters=6):
        Tm = T_guess
        for _ in range(iters):
            net = self.network(Tm); res = net.solve()
            loss = self.losses()
            pump = res["links"]["循环泵"]
            Q_tot = sum(loss.values()) + pump["P_shaft"] * 0.9          # 泵轴功率大部分变成水的热（假设 90%）
            m_h = pump["Q"] * float(water_rho(Tm))
            cp = water_cp(Tm)
            hx = self.hx.solve(Q_tot, m_h, Tm, self.T_chw, self.m_chw, cp, water_cp(self.T_chw))
            T_sup = hx["Th_out"]                                       # 板换出口 = 回水箱 = 供水（水箱混合，忽略水箱散热）
            Tm_new = T_sup + 0.5 * Q_tot / (m_h * cp)
            if abs(Tm_new - Tm) < 0.01: Tm = Tm_new; break
            Tm = Tm_new
        br = []
        for b in self.branches:
            lk = res["links"][f"{b.name}水道"]; Q = lk["Q"]; m = Q * float(water_rho(T_sup))
            P = loss[b.name]
            dT = P / (m * cp) if m > 0 else float("inf")
            T_out = T_sup + dT
            Tf = 0.5 * (T_sup + T_out)
            h, Re, v = htc_channel(Q, b.Dh, b.n_pass, Tf)
            A_heat = b.n_pass * 4 * b.Dh * b.L_pass * b.heated_frac
            q_avg = P / A_heat; q_pk = q_avg * b.peak
            T_wall = T_out + q_pk / h                                 # 最热点：出口处 + 峰值热流
            p_abs = res["nodes"][f"{b.name}出"]["p_abs"]
            Tsat = water_tsat(p_abs)
            br.append(dict(name=b.name, P=P, Q_Lmin=Q * 6e4, v=v, Re=Re, h=h, dT=dT, T_out=T_out, q_avg=q_avg, q_pk=q_pk,
                           T_wall=T_wall, p_out_abs=p_abs, Tsat=Tsat, dp=(res["nodes"]["分水器"]["p_total"] - res["nodes"]["集水器"]["p_total"]),
                           b=b))
        chil = chiller_select(Q_tot, self.T_chw, self.T_amb_chiller)
        out = dict(T_sup=T_sup, T_ret=hx["Th_in"], Q_tot=Q_tot, loss=loss, pump=pump, m_h=m_h, hx=hx, branches=br,
                   chiller=chil, net=net, res=res, dew=dew_point(self.T_amb, self.RH), T_mean=Tm,
                   eff=self.P_out / (self.P_out + sum(loss.values()) - loss.get("感应器线圈", 0)) if self.P_out > 0 else 0)
        out["alarms"] = self.alarms(out)
        return out

    # ---------------- 报警/体检
    def alarms(self, r):
        a = []
        if r["T_sup"] < r["dew"] + 2:
            a.append(("FAIL" if r["T_sup"] < r["dew"] else "WARN",
                      f"供水 {r['T_sup']:.1f}°C 接近/低于柜内露点 {r['dew']:.1f}°C（{self.T_amb:.0f}°C/{self.RH*100:.0f}%RH）→ 逆变器/电容表面凝露，可能打火"))
        if r["T_sup"] > 35:
            a.append(("WARN", f"供水 {r['T_sup']:.1f}°C > 35°C，功率模块结温余量变小"))
        for x in r["branches"]:
            b = x["b"]
            if x["Q_Lmin"] < b.Q_min_Lmin:
                a.append(("FAIL", f"{b.name} 流量 {x['Q_Lmin']:.1f} L/min < 流量开关 {b.Q_min_Lmin:g} L/min → 停机"))
            elif x["Q_Lmin"] < 1.3 * b.Q_min_Lmin:
                a.append(("WARN", f"{b.name} 流量 {x['Q_Lmin']:.1f} L/min 离流量开关 {b.Q_min_Lmin:g} L/min 不到 30%，滤网一脏就误停"))
            if x["T_out"] > b.T_out_max:
                a.append(("FAIL", f"{b.name} 出水 {x['T_out']:.1f}°C > 报警值 {b.T_out_max:g}°C"))
            margin = x["Tsat"] - x["T_wall"]
            if margin < 10:
                a.append(("FAIL", f"{b.name} 最热点壁温 {x['T_wall']:.0f}°C 离当地沸点 {x['Tsat']:.0f}°C 只有 {margin:.0f} K → 过冷沸腾/汽膜，铜管烧穿风险"))
            elif margin < 30:
                a.append(("WARN", f"{b.name} 最热点壁温 {x['T_wall']:.0f}°C 离当地沸点 {x['Tsat']:.0f}°C 余量 {margin:.0f} K（<30 K）"))
            if x["T_wall"] > b.T_wall_scale:
                a.append(("WARN", f"{b.name} 最热点壁温 {x['T_wall']:.0f}°C > {b.T_wall_scale:g}°C：非去离子水会结垢，水垢 0.5 mm 可使壁温再升数十 K"))
            if x["v"] > 4.0:
                a.append(("WARN", f"{b.name} 水道流速 {x['v']:.1f} m/s > 4 m/s，铜管冲刷腐蚀"))
        c = r["chiller"]
        if c.get("too_small"):
            a.append(("FAIL", f"冷水机最大档也不够：需 {r['Q_tot']/1e3:.1f} kW×1.15"))
        if not r["net"].converged:
            a.append(("FAIL", "管网未收敛"))
        if not a:
            a.append(("OK", "流量、温度、壁温、凝露全部在限值内"))
        return a

    # ---------------- 冷水机跳闸后升温（集总热容）
    def trip_transient(self, r, t_end=600.0, dt=1.0, chiller_on=False):
        """冷水机跳闸(冷侧断)：dT/dt = Q/(M·cp)。chiller_on=True 时从 T0 偏离后回到稳态：
        dT/dt = (Q − εCmin(T_hin − Tc))/(M cp)，返回 t, T_sup(t)，以及各支路首个报警时刻(按支路温升叠加)。"""
        M = self.tank_L * float(water_rho(r["T_sup"])) / 1000.0
        cp = water_cp(r["T_sup"])
        t = np.arange(0, t_end + dt, dt); T = np.empty_like(t); T[0] = r["T_sup"]
        Q = r["Q_tot"]
        k = 0.0
        if chiller_on:
            hx = r["hx"]; Cmin = min(r["m_h"] * cp, self.m_chw * water_cp(self.T_chw)); k = hx["eps"] * Cmin
        for i in range(1, len(t)):   # RK4
            f = lambda Ts: (Q - k * (Ts + Q / (r["m_h"] * cp) - self.T_chw)) / (M * cp)
            y = T[i - 1]
            k1 = f(y); k2 = f(y + dt / 2 * k1); k3 = f(y + dt / 2 * k2); k4 = f(y + dt * k3)
            T[i] = y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        first = {}
        for x in r["branches"]:
            Tout = T + x["dT"]
            idx = np.nonzero(Tout > x["b"].T_out_max)[0]
            first[x["name"]] = float(t[idx[0]]) if len(idx) else None
        return dict(t=t, T_sup=T, first_alarm=first, M=M)


class _MultiPass:
    """n 路相同方孔水道并联（Dh 方孔，长 L，局部阻力 K 每路）。dH 按单路 Q/n 计算。"""
    kind = "pipe"; p_rated = None

    def __init__(self, name, n1, n2, Dh, n, L, K, rough=0.0015e-3):
        self.name, self.n1, self.n2, self.Dh, self.n, self.L, self.K, self.rough = name, n1, n2, Dh, n, L, K, rough
        self.D = Dh

    def dH(self, Q, rho, mu):
        from ..world.pipes import friction_factor
        q = abs(Q) / self.n; A = self.Dh ** 2
        v = q / A; Re = rho * v * self.Dh / mu
        f = friction_factor(Re, self.rough / self.Dh) if Re > 1e-9 else 0.0
        return math.copysign((f * self.L / self.Dh + self.K) * v ** 2 / (2 * G), Q)

    def velocity(self, Q):
        return abs(Q) / self.n / self.Dh ** 2


def report_rows(r):
    rows = []
    for x in r["branches"]:
        rows.append([x["name"], f"{x['P']/1e3:.2f}", f"{x['Q_Lmin']:.1f}", f"{x['v']:.2f}", f"{x['dT']:.1f}", f"{x['T_out']:.1f}",
                     f"{x['h']/1e3:.1f}", f"{x['q_pk']/1e3:.0f}", f"{x['T_wall']:.0f}", f"{x['Tsat']:.0f}"])
    return rows


ROW_HEAD = ["支路", "损耗 kW", "流量 L/min", "流速 m/s", "温升 K", "出水 °C", "h kW/m²K", "峰值热流 kW/m²", "最热壁温 °C", "当地沸点 °C"]
