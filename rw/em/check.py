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


def compare_concentrators(freq=30e3, power_kw=6.0, T=25.0, kinds=(None, "smc", "sife", "ferrite"), geom=None, **kw):
    """同一功率、同一频率、同一冷态工件，比较各种导磁体: 线圈电流、效率、导磁体损耗、轴向热分布。"""
    from .conc import Concentrator
    out = {}
    for k in kinds:
        conc = Concentrator(k) if k else None
        s = InductionQuenchSim(geom=geom or BottleNeckGeometry(), freq=freq, power_kw=power_kw, nonlinear=True,
                               conc=conc, t_heat=1.0, t_quench=0.0, t_air=0.0, **kw)
        q, I = s.solve_em(np.full(s.tb.N, T))
        Pc = s.coil_loss_estimate(I)
        z, qz = axial_profile(s, q)
        out[k or "none"] = dict(
            I_rms=I / np.sqrt(2), coil_loss_kW=Pc / 1e3, conc_loss_kW=s.conc_loss_W / 1e3,
            eff=power_kw * 1e3 / (power_kw * 1e3 + Pc + s.conc_loss_W), z=z, q=qz, n_iter=s.n_iter_em,
            name=(conc.mat.name if conc else "无导磁体"),
            dTdt=(s.conc_loss_W / max(float(np.sum(s.conc_el * s._el_area_r() * 2 * np.pi)), 1e-12) / 3.5e6) if conc else 0.0,
            Bmax_conc=(float(s._B1_el[s.conc_el].max()) if conc else 0.0))
    return out
