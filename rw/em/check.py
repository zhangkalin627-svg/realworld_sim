"""v0.9 电磁升级的检验与对比工具"""
import numpy as np
from ..induction import InductionQuenchSim, BottleNeckGeometry
from .bh import MU0
from . import slab1d
from .. import materials as M


def _rows(sim):
    c = sim.mesh_mm.p[:, sim.mesh_mm.t].mean(axis=1)
    return c


def local_power_check(sim, T=25.0):
    """用有限元解出沿淬火区中部一条径向线的功率面密度，和一维半无限平板(时域真值)比。
    表面磁场 H0 取壁外侧第一层空气单元的 |B|/mu0。返回 dict."""
    g = sim.g
    T_th = np.full(sim.tb.N, T)
    q, I = sim.solve_em(T_th)
    c = _rows(sim)
    st = sim.steel_idx
    zmid = g.L - g.zone / 2
    zs = c[1, st]
    zrow = zs[np.argmin(np.abs(zs - zmid))]
    sel = np.nonzero(np.abs(zs - zrow) < 1e-6)[0]
    # 单元宽度 (mm→m): 从 mesh 取
    p = sim.mesh_mm.p
    t = sim.mesh_mm.t[:, st[sel]]
    rr = p[0, t]
    dr = (rr.max(axis=0) - rr.min(axis=0)) * 1e-3
    rc = rr.mean(axis=0)
    # 圆柱面功率: 每单位外表面积 = ∫ q (r/ro) dr
    P_fem = float(np.sum(q[sel] * dr * rc / g.ro))
    # 空气侧
    air = np.nonzero((np.abs(c[1] - zrow) < 1e-6) & (c[0] > g.ro) & (~sim.coil_el))[0]
    j = air[np.argmin(c[0, air])]
    B1 = sim._B1(sim._x_unit)[j] * I if (sim.nonlinear or sim.conc is not None) else sim._B1(sim._x_unit)[j] * I
    H0 = B1 / MU0
    rho = 1.0 / M.steel_sigma(T)
    out = dict(H0_kAm=H0 / 1e3, P_fem_MWm2=P_fem / 1e6)
    if sim.nonlinear:
        out["P_1d_time_MWm2"] = slab1d.time_domain(sim.bh, T, float(rho), sim.f, H0) / 1e6
        out["P_1d_harm_MWm2"] = slab1d.harmonic(sim.bh, T, float(rho), sim.f, H0)[0] / 1e6
    else:
        out["P_1d_lin_MWm2"] = slab1d.linear_power(H0, sim.f, M.steel_mur(T, sim.mur_cold), float(rho)) / 1e6
    return out


def axial_profile(sim, q):
    """外表面最外一层钢单元的焦耳热密度沿轴向的分布 (每个网格层把上下两个三角形取平均): 返回 z(mm), q(W/m3)"""
    c = _rows(sim)
    st = sim.steel_idx
    g = sim.g
    zl = np.unique(np.round(sim.mesh_mm.p[1], 6))
    zc = c[1, st]
    m = (zc >= g.L - 40) & (zc <= g.L) & (np.abs(c[0, st] - g.ro) < 0.08)
    bi = np.searchsorted(zl, zc[m]) - 1
    qq = q[m]
    zz = 0.5 * (zl[bi] + zl[bi + 1])
    ub = np.unique(bi)
    z = np.array([0.5 * (zl[i] + zl[i + 1]) for i in ub])
    qv = np.array([qq[bi == i].mean() for i in ub])
    return z, qv


def _dust_zone(s):
    """积灰区 = 导磁体"腿"贴近工件一侧（内 1/3 半径、超出线圈端面之外）。返回单元布尔掩码。"""
    g = s.g
    c = s.mesh.p[:, s.mesh.t].mean(axis=1)          # (2, nel) 单元中心
    leg = ((c[1] < g.coil_z0) | (c[1] > g.coil_z1))
    inner = c[0] <= g.coil_r0 + 0.3 * (g.coil_r1 - g.coil_r0)
    return s.conc_el & leg & inner


def _gradB_el(s, B1):
    """单元 |∇B| (T/m)：单元值→节点平均→三角形线性梯度。量级后处理，非精确场解。"""
    p, t = s.mesh.p, s.mesh.t
    bn = np.zeros(p.shape[1]); cnt = np.zeros(p.shape[1])
    np.add.at(bn, t, np.broadcast_to(B1, t.shape))
    np.add.at(cnt, t, 1)
    bn /= np.maximum(cnt, 1)
    x = p[0, t] * 1e-3; y = p[1, t] * 1e-3          # mm → m
    v = bn[t]
    den = (y[1] - y[2]) * (x[0] - x[2]) + (x[2] - x[1]) * (y[0] - y[2])
    den = np.where(np.abs(den) < 1e-30, 1e-30, den)
    dvdx = ((y[1] - y[2]) * (v[0] - v[2]) + (y[2] - y[0]) * (v[1] - v[2])) / den
    dvdy = ((x[2] - x[1]) * (v[0] - v[2]) + (x[0] - x[2]) * (v[1] - v[2])) / den
    return np.hypot(dvdx, dvdy)


def _nonmag_mesh_extra(conc, g, f):
    """非磁导体的趋肤层网格加密线（铝 30 kHz δ≈1.5 mm）。"""
    rho = 1.0 / conc.sigma
    d = 1e3 * np.sqrt(rho / (np.pi * f * 4e-7 * np.pi))   # δ mm
    off = np.array([0.25, 0.5, 1.0, 2.0]) * d
    rl, zl = [], []
    for o in off:
        for e in (g.coil_r0 + o, g.coil_r1 - o, g.coil_r1 + o, conc.r_edges(g) - o):
            if g.coil_r0 <= e <= conc.r_edges(g):
                rl.append(e)
        for e in (g.coil_z0 - o, g.coil_z0 - conc.leg_t + o, g.coil_z1 + o, g.coil_z1 + conc.leg_t - o):
            if g.coil_z0 - conc.leg_t <= e <= g.coil_z1 + conc.leg_t:
                zl.append(e)
    return (rl, zl)


def compare_concentrators(freq=30e3, power_kw=6.0, T=25.0,
                          kinds=(None, "sife", "smc", "ferrite", "alu", "graphite"), geom=None, **kw):
    """同一功率、同一频率、同一冷态工件，比较各种导磁位材料（含非磁导体铝/石墨，木头=无导磁体基线）:
    线圈电流、效率、导磁位损耗、积灰区表面磁密、铁粉受力量级、单匝电压估计。均为仿真结果。"""
    from .conc import Concentrator
    M_S_FE = 1.7e6                                   # 铁饱和磁化强度 A/m（偶极受力量级估计用）
    V_P = 4 / 3 * np.pi * (25e-6) ** 3               # 50 µm 铁粉颗粒体积
    out = {}
    for k in kinds:
        conc = Concentrator(k) if k else None
        g = geom or BottleNeckGeometry()
        extra = _nonmag_mesh_extra(conc, g, freq) if (conc and not conc.is_magnetic) else kw.pop("mesh_extra", None)
        s = InductionQuenchSim(geom=g, freq=freq, power_kw=power_kw, nonlinear=True,
                               conc=conc, t_heat=1.0, t_quench=0.0, t_air=0.0,
                               mesh_extra=extra, **kw)
        q, I = s.solve_em(np.full(s.tb.N, T))
        Pc = s.coil_loss_estimate(I)
        z, qz = axial_profile(s, q)
        I_rms = I / np.sqrt(2)
        rec = dict(
            I_rms=I_rms, coil_loss_kW=Pc / 1e3, conc_loss_kW=s.conc_loss_W / 1e3,
            eff=power_kw * 1e3 / (power_kw * 1e3 + Pc + s.conc_loss_W), z=z, q=qz, n_iter=s.n_iter_em,
            name=(conc.name if conc else "无（木头基线）"),
            dTdt=(s.conc_loss_W / max(float(np.sum(s.conc_el * s._el_area_r() * 2 * np.pi)), 1e-12) / 3.5e6) if conc else 0.0,
            Bmax_conc=(float(s._B1_el[s.conc_el].max()) if conc else 0.0))
        # 粉尘/拉弧指标（量级，均为仿真结果）
        if conc:
            dz = _dust_zone(s)
            B_surf = float(s._B1_el[dz].max()) if dz.any() else 0.0
            gB = _gradB_el(s, s._B1_el)
            gB_dz = float(np.percentile(gB[dz], 90)) if dz.any() else 0.0
            rec["B_dust_T"] = B_surf
            rec["F_dust_uN"] = V_P * M_S_FE * gB_dz * 1e6
            rec["V_turn"] = (power_kw * 1e3 + Pc + s.conc_loss_W) / max(I_rms, 1e-9)
        else:
            rec["B_dust_T"] = 0.0; rec["F_dust_uN"] = 0.0
            rec["V_turn"] = (power_kw * 1e3 + Pc) / max(I_rms, 1e-9)
        out[k or "none"] = rec
    return out
