"""100 kW 满功率下的电磁/热专项分析 (路线 A)

四块，每块都有独立检验 (见 validate_em.py [11h]):
  1) 线圈区域的空气磁场: 有限元 vs 圆环电流 Biot–Savart (椭圆积分) 求和          -> field_check
  2) 杂散场加热周边金属 (辅助体 AuxBody): 有限元体积功率 vs 表面阻抗公式 ½·Rs·H_t² -> stray_check
  3) 线圈铜损: 把线圈当真导体(电压驱动, 趋肤/邻近效应)解出, 复功率恒等式自检; 与旧的
     "表面电流薄层"估算比较                                                       -> coil_loss_fem
  4) 导磁体自发热: 集总热模型, 与解析指数解比较; 居里点/温度上限下的存活时间     -> conc_thermal
⚠ 估算/假设项: 导磁体温度上限、冷却条件、铜电阻率随温度、线圈水冷换热能力，需按实物/供应商核对。
"""
import numpy as np
import scipy.sparse.linalg as spla
from scipy.special import ellipk, ellipe
from skfem import asm, condense

from .. import induction as I
from .. import materials as M
from .bh import MU0

RHO_CU = M.COPPER_RHO


# ------------------------------------------------------------------ 1) 空气磁场 vs Biot-Savart
def loop_field(R, zl, r, z):
    """半径 R(m)、位于 z=zl 的单位电流圆环在 (r,z) 的 (Br, Bz) (T/A)。r 可为数组。"""
    r = np.asarray(r, float); z = np.asarray(z, float) - zl
    r_ = np.maximum(r, 1e-9)
    a2 = (R + r_) ** 2 + z**2
    b2 = (R - r_) ** 2 + z**2
    m = np.clip(4 * R * r_ / a2, 0, 1 - 1e-14)
    K, E = ellipk(m), ellipe(m)
    c = MU0 / (2 * np.pi)
    Bz = c / np.sqrt(a2) * (K + (R**2 - r_**2 - z**2) / b2 * E)
    Br = c * z / (r_ * np.sqrt(a2)) * (-K + (R**2 + r_**2 + z**2) / b2 * E)
    Br = np.where(r < 1e-7, 0.0, Br)
    return Br, Bz


def coil_field(g, r, z, n=24):
    """矩形截面均匀电流线圈(单位总电流)的 (Br, Bz)，截面分 n×n 根细丝求和。长度 m。"""
    rs = (np.arange(n) + 0.5) / n
    R = (g.coil_r0 + (g.coil_r1 - g.coil_r0) * rs) * 1e-3
    Z = (g.coil_z0 + (g.coil_z1 - g.coil_z0) * rs) * 1e-3
    Br = np.zeros_like(np.asarray(r, float)); Bz = np.zeros_like(Br)
    for Ri in R:
        for Zi in Z:
            b_r, b_z = loop_field(Ri, Zi, r, z)
            Br = Br + b_r; Bz = Bz + b_z
    return Br / n**2, Bz / n**2


def air_solution(sim):
    """去掉工件，只有线圈的空气场: 返回 A (每 1 A 峰值电流)"""
    K = asm(I._k_em, sim.basis, mu=sim.b0.interpolate(np.full(sim.mesh.t.shape[1], MU0)))
    x = np.zeros(K.shape[0])
    Kc, bc, xc, Ii = condense(K.tocsc(), sim.f_unit, x=x, D=sim.em_D)
    x[Ii] = spla.spsolve(Kc.tocsc(), bc)
    return x


def _Bcomp(sim, x):
    Gr, Gz, Av, rc = sim._grad_ops()
    return -(Gz @ x), Av @ x / rc + Gr @ x      # Br, Bz (每单元)


def field_check(sim, box_r=45.0, dz=30.0, dist=3.0):
    """空气场: 线圈外 dist mm 以外、r<box_r、|z−线圈中面|<线圈半高+dz 的所有单元，
    有限元 |B| vs Biot–Savart。返回 dict(p50, p95, p99 相对 |B| 峰值的误差, 单元数)"""
    g = sim.g
    x = air_solution(sim)
    Br, Bz = _Bcomp(sim, x)
    c = sim.mesh_mm.p[:, sim.mesh_mm.t].mean(axis=1)
    zm = 0.5 * (g.coil_z0 + g.coil_z1)
    half = 0.5 * (g.coil_z1 - g.coil_z0)
    dr = np.maximum(np.maximum(g.coil_r0 - c[0], c[0] - g.coil_r1), 0)
    dzz = np.maximum(np.abs(c[1] - zm) - half, 0)
    far = np.hypot(dr, dzz) >= dist
    sel = far & (c[0] < box_r) & (np.abs(c[1] - zm) < half + dz) & (c[0] > 2.0)   # 近轴 P1 单元 Bz=A/r+dA/dr 精度差，另报
    bsr, bsz = coil_field(g, c[0, sel] * 1e-3, c[1, sel] * 1e-3)
    bs = np.hypot(bsr, bsz)
    fe = np.hypot(Br[sel], Bz[sel])
    err = np.abs(fe - bs) / bs.max()
    ax = sel & (c[0] < 3.0)
    bsa = np.hypot(*coil_field(g, c[0, ax] * 1e-3, c[1, ax] * 1e-3))
    fea = np.hypot(Br[ax], Bz[ax])
    return dict(n=int(sel.sum()), p50=float(np.percentile(err, 50)), p95=float(np.percentile(err, 95)),
                p99=float(np.percentile(err, 99)), Bmax_mT_per_kA=float(bs.max() * 1e6),
                axis_err=float(np.max(np.abs(fea - bsa) / bsa)) if ax.any() else float("nan"))


# ------------------------------------------------------------------ 2) 杂散场加热辅助体
def surface_resistance(rho, mur, f):
    return np.sqrt(np.pi * f * MU0 * mur * rho)


def stray_check(sim, T=25.0):
    """sim 需带 aux。有限元体积功率 vs 沿辅助体各外表面积分 ½·Rs·H_t² (H_t=紧邻空气侧单元的切向磁场)。
    返回每个辅助体的 dict。适用 厚度≫趋肤深度、曲率半径≫趋肤深度。"""
    q, Ic = sim.solve_em(np.full(sim.tb.N, T))
    Br, Bz = _Bcomp(sim, sim._x_unit)
    Br, Bz = np.abs(Br) * Ic, np.abs(Bz) * Ic
    c = sim.mesh_mm.p[:, sim.mesh_mm.t].mean(axis=1)
    from scipy.spatial import cKDTree
    air = ~(sim.steel_el | sim.coil_el | sim.conc_el | sim.aux_el)
    tree = cKDTree(c[:, air].T)
    ia = np.nonzero(air)[0]
    out = []
    for a, P_fem in zip(sim.aux, sim.aux_loss_each_W):
        d = a.skin_mm(sim.f)
        eps = 0.1 * d
        Rs = surface_resistance(a.rho, a.mur, sim.f)
        P_est = 0.0
        parts = {}
        n = 600
        for name, fixed, lo, hi, normal, comp in (
                ("内壁", a.r0, a.z0, a.z1, (-1, 0), Bz), ("外壁", a.r1, a.z0, a.z1, (1, 0), Bz),
                ("下端", a.z0, a.r0, a.r1, (0, -1), Br), ("上端", a.z1, a.r0, a.r1, (0, 1), Br)):
            s = lo + (np.arange(n) + 0.5) / n * (hi - lo)
            if name in ("内壁", "外壁"):
                pts = np.column_stack([np.full(n, fixed) + normal[0] * eps, s]); rr = np.full(n, fixed)
            else:
                pts = np.column_stack([s, np.full(n, fixed) + normal[1] * eps]); rr = s
            _, k = tree.query(pts)
            Ht = comp[ia[k]] / (MU0)          # A/m, 空气侧
            dA = 2 * np.pi * rr * 1e-3 * (hi - lo) * 1e-3 / n
            Pf = float(np.sum(0.5 * Rs * Ht**2 * dA))
            parts[name] = Pf
            P_est += Pf
        out.append(dict(name=a.name, material=a.material, skin_mm=d, P_fem_W=P_fem, P_est_W=P_est, parts=parts,
                        ratio=P_fem / P_est, dTdt_K_s=P_fem / (a.rho_cp * a.volume_m3())))
    return out


# ------------------------------------------------------------------ 3) 线圈铜损 (真导体)
def coil_refine_lines(g, f, rho=RHO_CU, n=(0.1, 0.25, 0.5, 1.0)):
    """线圈四个面向内加密到铜趋肤深度 δ 的若干分之一 (mm)，作为 mesh_extra"""
    d = 1e3 * np.sqrt(rho / (np.pi * f * MU0))
    rl, zl = [], []
    for k in n:
        o = k * d
        rl += [g.coil_r0 + o, g.coil_r1 - o, g.coil_r0 - 0.5 * o, g.coil_r1 + 0.5 * o]
        zl += [g.coil_z0 + o, g.coil_z1 - o, g.coil_z0 - 0.5 * o, g.coil_z1 + 0.5 * o]
    return rl, zl


def coil_loss_fem(sim, T=25.0, rho_cu=RHO_CU):
    """线圈按铜导体(σ=1/rho_cu)，单匝，电压驱动解出电流分布(趋肤+邻近效应)，缩放到工件吸收功率=sim.P。
    sim 宜带 mesh_extra=coil_refine_lines(...)。返回 dict。"""
    q, I0 = sim.solve_em(np.full(sim.tb.N, T))
    mur = sim._mur_el
    sig = sim._sig_el.copy()
    sig_c = np.where(sim.coil_el, 1.0 / rho_cu, 0.0)
    w = 2 * np.pi * sim.f
    Gr, Gz, Av, rc = sim._grad_ops()
    K = asm(I._k_em, sim.basis, mu=sim.b0.interpolate(MU0 * mur))
    Mm = asm(I._m_em, sim.basis, sig=sim.b0.interpolate(sig + sig_c))
    f = asm(I._src, sim.basis, J=sim.b0.interpolate(sig_c / (2 * np.pi * rc)))
    A = (K + 1j * w * Mm).tocsc()
    x = np.zeros(A.shape[0], complex)
    Ac, bc, xc, Ii = condense(A, f.astype(complex), x=x, D=sim.em_D)
    x[Ii] = spla.spsolve(Ac.tocsc(), bc)
    ar = sim._el_area_r()               # 面积×r
    a2d = ar / rc                       # 截面积 m2
    Ael = x[sim.mesh.t].mean(axis=0)
    J = sig_c * (1.0 / (2 * np.pi * rc) - 1j * w * Ael)
    I_tot = np.sum(J * a2d)
    dV = ar * 2 * np.pi
    P_c = float(np.sum(0.5 * np.where(sim.coil_el, np.abs(J) ** 2 * rho_cu, 0.0) * dV))
    q_el = 0.5 * sig * w**2 * (np.abs(x[sim.mesh.t]) ** 2).mean(axis=0)
    P_w = float(np.sum(np.where(sim.steel_el, q_el, 0.0) * dV))
    P_aux = float(np.sum(np.where(sim.aux_el, q_el, 0.0) * dV))
    P_in = 0.5 * float(np.real(np.conj(I_tot)))        # ½ Re(V I*), V=1
    s2 = sim.P / P_w                                    # 缩放: 工件功率 -> sim.P
    out = dict(
        identity_err=abs(P_in - (P_c + P_w + P_aux)) / P_in,
        coil_loss_kW=P_c * s2 / 1e3, aux_kW=P_aux * s2 / 1e3,
        I_rms_A=abs(I_tot) * np.sqrt(s2) / np.sqrt(2),
        I_rms_fixedJ_A=I0 / np.sqrt(2),
        eff=sim.P / (sim.P + P_c * s2),
        coil_loss_est_kW=sim.coil_loss_estimate(abs(I_tot) * np.sqrt(s2)) / 1e3,
    )
    out["ratio_vs_est"] = out["coil_loss_kW"] / out["coil_loss_est_kW"]
    # 线圈内的平均/最大损耗密度 (W/m3)
    pv = 0.5 * np.abs(J) ** 2 * rho_cu * s2
    out["coil_pv_max_GWm3"] = float(pv[sim.coil_el].max() / 1e9)
    g = sim.g
    perim = 2 * ((g.coil_r1 - g.coil_r0) + (g.coil_z1 - g.coil_z0)) * 1e-3
    area_cool = perim * 2 * np.pi * 0.5 * (g.coil_r0 + g.coil_r1) * 1e-3
    out["cool_area_m2"] = float(area_cool)
    out["heat_flux_MWm2"] = float(P_c * s2 / area_cool / 1e6)
    out["water_Lmin_for_10K"] = float(P_c * s2 / (4180.0 * 10.0) * 60.0)   # kg/s ≈ L/s
    return out


# ------------------------------------------------------------------ 4) 导磁体自发热
# ⚠ 假设: 各导磁体可连续工作的温度上限 (°C)。铁氧体即居里点附近; SMC/硅钢取绝缘/粘结剂耐温, 需按供应商数据改
CONC_T_LIMIT = {"smc": 150.0, "sife": 180.0, "ferrite": 200.0}
CONC_RHO_CP = {"smc": 3.6e6, "sife": 3.7e6, "ferrite": 3.2e6}     # J/m3K 典型值


def conc_thermal(P_loss_W, volume_m3, kind, T0=30.0, T_water=30.0, hA_W_K=0.0, t_end=None):
    """集总热模型: C dT/dt = P − hA (T − T_w)。解析解 (hA>0):
        T(t) = T_inf + (T0 − T_inf) exp(−t/τ),  T_inf = T_w + P/hA, τ = C/hA ;  hA=0: 线性升温。
    返回 dict(dTdt_K_s, T_inf, t_limit_s (到温度上限，inf=永不到), T_limit)"""
    C = CONC_RHO_CP[kind] * volume_m3
    Tl = CONC_T_LIMIT[kind]
    out = dict(T_limit=Tl, dTdt_K_s=P_loss_W / C)
    if hA_W_K <= 0:
        out["T_inf"] = float("inf")
        out["t_limit_s"] = (Tl - T0) * C / P_loss_W if P_loss_W > 0 else float("inf")
    else:
        Tinf = T_water + P_loss_W / hA_W_K
        tau = C / hA_W_K
        out["T_inf"], out["tau_s"] = Tinf, tau
        out["t_limit_s"] = (-tau * np.log((Tl - Tinf) / (T0 - Tinf)) if Tinf > Tl else float("inf"))
    if t_end is not None:
        t = float(t_end)
        out["T_end"] = (T0 + P_loss_W / C * t) if hA_W_K <= 0 else out["T_inf"] + (T0 - out["T_inf"]) * np.exp(-t / out["tau_s"])
    out["hA_needed_W_K"] = P_loss_W / max(Tl - T_water, 1.0)       # 稳态不超限所需的 hA
    return out


def conc_volume_m3(sim):
    return float(np.sum(sim.conc_el * sim._el_area_r() * 2 * np.pi))


# ------------------------------------------------------------------ 4b) 导磁体温度反馈: 居里点 → μ、损耗、线圈电流随温度变化
def conc_loss_vs_T(sim, T_grid, T=25.0):
    """同一个 sim (带导磁体) 在导磁体均匀温度 T_grid 下各解一次 EM: 返回 dict(T, loss_W, I_rms_A, Bmax_T)。
    sim.conc_T 临时改写, 结束恢复。⚠ 假设导磁体温度均匀 (集总)。"""
    if sim.conc is None:
        raise ValueError("sim 没有导磁体")
    T_grid = np.asarray(sorted(float(t) for t in T_grid))
    old = sim.conc_T
    P, Ir, Bm = [], [], []
    try:
        for t in T_grid:
            sim.conc_T = float(t)
            q, Ic = sim.solve_em(np.full(sim.tb.N, T))
            P.append(sim.conc_loss_W); Ir.append(Ic / np.sqrt(2)); Bm.append(float(sim._B1_el[sim.conc_el].max()))
    finally:
        sim.conc_T = old
    return dict(T=T_grid, loss_W=np.array(P), I_rms_A=np.array(Ir), Bmax_T=np.array(Bm))


def conc_heatup(T_grid, P_grid_W, volume_m3, kind, T0=30.0, T_water=30.0, hA_W_K=0.0, q_ext_W=0.0,
                t_end=600.0, T_stop=None):
    """导磁体集总升温 C dT/dt = P(T) + q_ext − hA (T − T_w), P(T) 由有限元网格点 (T_grid, P_grid_W) 的保形插值给出。
    T_stop: 到该温度停止 (默认材料温度上限 CONC_T_LIMIT[kind])。返回 dict(t, T, P_W, t_stop_s(inf=未到), T_end)。"""
    from scipy.integrate import solve_ivp
    from scipy.interpolate import PchipInterpolator
    Tg = np.asarray(T_grid, float); Pg = np.asarray(P_grid_W, float)
    f = PchipInterpolator(Tg, Pg, extrapolate=True)
    P_of_T = lambda T: float(f(min(max(T, Tg[0]), Tg[-1])))
    C = CONC_RHO_CP[kind] * volume_m3
    Ts = CONC_T_LIMIT[kind] if T_stop is None else float(T_stop)
    qx = q_ext_W if callable(q_ext_W) else (lambda T: q_ext_W)      # q_ext_W 可为常数或 q(T_body[°C]) (如 rad.rad_exchange_W 的闭包)
    rhs = lambda t, y: [(P_of_T(y[0]) + qx(y[0]) - hA_W_K * (y[0] - T_water)) / C]
    ev = lambda t, y: y[0] - Ts
    ev.terminal, ev.direction = True, 1
    sol = solve_ivp(rhs, (0.0, t_end), [T0], events=ev, rtol=1e-9, atol=1e-9, dense_output=True)
    t = np.linspace(0, sol.t[-1], 200)
    Tt = sol.sol(t)[0]
    hit = len(sol.t_events[0]) > 0
    return dict(t=t, T=Tt, P_W=np.array([P_of_T(x) for x in Tt]), t_stop_s=float(sol.t_events[0][0]) if hit else float("inf"),
                T_end=float(Tt[-1]), T_stop=Ts)


# ------------------------------------------------------------------ 5) 线圈铜电阻率随温度 + 水冷温升耦合
ALPHA_CU = 0.00393          # 1/K，退火铜 20 °C 附近电阻温度系数 (手册值)
T_REF_CU = 20.0             # COPPER_RHO 对应的温度 (°C)


def rho_cu_at(T):
    """铜电阻率 ρ(T)=ρ20·(1+α(T−20))，T 在 0–200 °C 内线性够用"""
    return RHO_CU * (1.0 + ALPHA_CU * (T - T_REF_CU))


def coil_loss_hot(sim, T_cu, T=25.0):
    """线圈铜均匀取 T_cu(°C) 时的铜损 (coil_loss_fem 的 ρ 换成 ρ(T_cu))。
    注意: sim 的 mesh_extra 趋肤层加密按 20 °C 的 δ 设置, 热态 δ 略大, 网格仍足够细。"""
    return coil_loss_fem(sim, T, rho_cu=rho_cu_at(T_cu))


def coil_thermal_coupled(sim, T_water=30.0, h_w=15e3, t_wall_mm=1.5, cool_frac=0.6, T=25.0,
                         tol=0.05, max_iter=12, verbose=False):
    """线圈铜损 ↔ 铜温度 自洽迭代:  T_cu = T_water + q_in·(1/h_w + t_wall/k_cu),
    q_in = P_c(ρ(T_cu)) / (cool_frac·A_外周长)。每轮一次 FEM (约十几秒)。
    ⚠ 假设项: h_w (水侧换热系数, 单相湍流水冷铜管典型 1–2e4 W/m²K)、t_wall (损耗层到水的铜壁厚度)、
      cool_frac (内冷却流道周长/线圈外周长)。T_cu>100 °C 时水侧会沸腾, h_w 取值不再成立 → 返回 boiling=True。
    返回 dict(T_cu, P_c_kW, eff, q_in_MWm2, iters, converged, boiling, hist)"""
    K_CU = 390.0                                         # W/mK
    T_cu = float(T_water)
    hist = []
    for it in range(max_iter):
        r = coil_loss_hot(sim, T_cu, T)
        q = r["coil_loss_kW"] * 1e3 / (cool_frac * r["cool_area_m2"])
        T_new = T_water + q * (1.0 / h_w + t_wall_mm * 1e-3 / K_CU)
        hist.append((T_cu, r["coil_loss_kW"]))
        if verbose:
            print(f"  it{it}: T_cu={T_cu:.1f} P_c={r['coil_loss_kW']:.2f} kW q={q/1e6:.2f} MW/m2 -> T={T_new:.1f}", flush=True)
        if abs(T_new - T_cu) < tol or T_new > 400.0:
            T_cu = T_new
            break
        T_cu = 0.6 * T_new + 0.4 * T_cu                   # 欠松弛
    else:
        it = max_iter
    ok = abs(T_new - hist[-1][0]) < max(tol, 0.5) and T_new < 400.0
    return dict(T_cu=float(T_new), P_c_kW=r["coil_loss_kW"], eff=r["eff"], q_in_MWm2=q / 1e6,
                I_rms_A=r["I_rms_A"], iters=it + 1, converged=bool(ok), boiling=bool(T_new > 100.0),
                hist=hist, rho_ratio=rho_cu_at(T_new) / RHO_CU)


# ------------------------------------------------------------------ 汇总报告
def full_power_report(power_kw=100.0, freq=30e3, geom=None, conc_kinds=("smc", "sife", "ferrite"),
                      aux=None, T=25.0, hA_W_K=0.0, coil_refine=(0.1, 0.25, 0.5, 1.0), verbose=False):
    """冷态工件满功率下的电磁/热清单。每项是 dict；'已验证'与'估算'见各字段说明。
      coil : 线圈铜损(真导体 FEM, 已验证: 复功率恒等式 + 网格收敛), 水冷热流密度, 10K 温升所需水量
      conc : 各导磁体损耗(FEM 磁密 + 手册损耗模型, 估算), 集总升温/到温度上限时间(解析, 已验证; 温度上限为假设)
      aux  : 杂散场加热的辅助体 (FEM 体积功率, 与表面阻抗公式比较)
    """
    g = geom or I.BottleNeckGeometry()
    out = dict(power_kw=power_kw, freq_kHz=freq / 1e3)
    kw = dict(geom=g, freq=freq, power_kw=power_kw, nonlinear=True, t_heat=1.0, t_quench=0.0, t_air=0.0)
    ex = coil_refine_lines(g, freq, n=coil_refine)
    s = I.InductionQuenchSim(mesh_extra=ex, aux=aux, **kw)
    out["coil"] = coil_loss_fem(s, T)
    if aux:
        out["aux"] = stray_check(s, T)
    out["conc"] = {}
    from .conc import Concentrator
    for k in conc_kinds:
        sc = I.InductionQuenchSim(conc=Concentrator(k), **kw)
        q, Ic = sc.solve_em(np.full(sc.tb.N, T))
        V = conc_volume_m3(sc)
        th = conc_thermal(sc.conc_loss_W, V, k, hA_W_K=hA_W_K)
        out["conc"][k] = dict(name=sc.conc.mat.name, loss_kW=sc.conc_loss_W / 1e3, volume_cm3=V * 1e6,
                              Bmax_T=float(sc._B1_el[sc.conc_el].max()), I_rms_A=Ic / np.sqrt(2), **th)
        if verbose:
            print(k, out["conc"][k], flush=True)
    return out
