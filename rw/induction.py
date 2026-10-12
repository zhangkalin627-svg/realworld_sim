"""感应加热 + 淬火 求解器 (二维轴对称有限元)

物理:
  1) 电磁: 轴对称时谐涡流方程，未知量 A_phi(r,z)
        -div( 1/mu grad A ) + A/(mu r^2) + j w sigma A = J_coil
     工件 sigma(T)、mu(T) 随温度变化，每 0.05 s 重算一次。
     电源按"恒功率"控制：自动调线圈电流，使工件吸收功率 = 设定值。
  2) 传热: rho cp(T) dT/dt = div(k(T) grad T) + q_joule
     表面: 加热时空冷，淬火时用介质换热系数 h(T_surface) 曲线。
  3) 组织/硬度: 记录每点峰值温度和 800→500°C 冷却时间 t8/5，按 45 钢 CCT 规律估算 HRC。

近似（第一版有意简化，后续可逐项换成更精确模型）:
  - 磁导率用随温度变化的"有效 mu_r"，不是完整非线性 B-H 曲线
  - 线圈按均匀电流截面处理，线圈自身铜损用表面电阻公式估算
  - 工件不旋转的二维轴对称（自转使周向均匀，正好符合轴对称假设）
"""
import numpy as np
import scipy.sparse.linalg as spla
from skfem import MeshTri, Basis, FacetBasis, ElementTriP1, ElementTriP0, BilinearForm, LinearForm, asm, condense
from skfem.helpers import dot, grad

from . import materials as M

MU0 = 4e-7 * np.pi


# ------------------------------------------------------------------ 几何
class BottleNeckGeometry:
    """瓶形件（单位 mm）。默认 = 你那台 6 工位淬火机的工件：
    瓶口外径 22、内径 16（壁厚 3），瓶颈长 50，淬火区为顶部 20 mm；
    下面接 30 mm 肩部过渡到瓶身 Ø45（模型在肩部下端截断，截面视为绝热）。
    """

    def __init__(self, neck_od=22.0, wall=3.0, neck_len=50.0, shoulder_len=30.0, body_od=45.0,
                 coil_id=30.0, coil_radial=6.0, coil_h=25.0, coil_z0=27.0, zone=20.0):
        self.ro = neck_od / 2
        self.ri = self.ro - wall
        self.wall = wall
        self.L = neck_len
        self.Ls = shoulder_len
        self.Rb = body_od / 2
        self.coil_r0 = coil_id / 2
        self.coil_r1 = coil_id / 2 + coil_radial
        self.coil_z0 = coil_z0
        self.coil_z1 = coil_z0 + coil_h
        self.zone = zone

    def outer_r(self, z):
        z = np.asarray(z, float)
        s = np.clip(-z / self.Ls, 0, 1)
        return self.ro + (self.Rb - self.ro) * s

    def inner_r(self, z):
        return self.outer_r(z) - self.wall

    def is_steel(self, r, z):
        return (z <= self.L) & (z >= -self.Ls) & (r >= self.inner_r(z)) & (r <= self.outer_r(z))

    def is_coil(self, r, z):
        return (r >= self.coil_r0) & (r <= self.coil_r1) & (z >= self.coil_z0) & (z <= self.coil_z1)


def _graded(a, b, h_a, h_b, growth=1.15):
    """从 a 到 b 的一维网格，两端尺寸 h_a / h_b，中间逐渐变粗。"""
    pts_l = [a]
    pts_r = [b]
    ha, hb = h_a, h_b
    while pts_l[-1] + ha < pts_r[-1] - hb:
        if ha <= hb:
            pts_l.append(pts_l[-1] + ha)
            ha *= growth
        else:
            pts_r.append(pts_r[-1] - hb)
            hb *= growth
    return np.unique(np.concatenate([pts_l, pts_r[::-1]]))


def build_mesh(g: BottleNeckGeometry, h_skin=0.04, conc=None):
    ri, ro = g.ri, g.ro
    if conc is None:
        r_out = [_graded(g.coil_r1, g.Rb, 0.6, 0.5), _graded(g.Rb, 90.0, 0.5, 8.0, 1.25)]
        z_mid = [_graded(-g.Ls, 15.0, 0.5, 1.0), _graded(15.0, g.L, 1.0, h_skin * 3, 1.12)]
        z_top = [_graded(g.L, g.coil_z1, h_skin * 3, 0.4), _graded(g.coil_z1, 150.0, 0.4, 8.0, 1.25)]
    else:   # 导磁体边界要落在网格线上
        a, b = sorted([g.Rb, conc.r_edges(g)])
        r_out = [_graded(g.coil_r1, a, 0.6, 0.5), _graded(a, b, 0.5, 0.5), _graded(b, 90.0, 0.5, 8.0, 1.25)]
        zl = g.coil_z0 - conc.leg_t
        z_mid = [_graded(-g.Ls, 15.0, 0.5, 1.0)]
        if conc.leg_t > 0:
            z_mid += [_graded(15.0, zl, 1.0, 0.5), _graded(zl, g.coil_z0, 0.5, 0.5), _graded(g.coil_z0, g.L, 0.5, h_skin * 3, 1.12)]
        else:
            z_mid += [_graded(15.0, g.coil_z0, 1.0, 0.5), _graded(g.coil_z0, g.L, 0.5, h_skin * 3, 1.12)]
        zt = g.coil_z1 + conc.leg_t
        z_top = [_graded(g.L, g.coil_z1, h_skin * 3, 0.4)]
        if conc.leg_t > 0:
            z_top += [_graded(g.coil_z1, zt, 0.4, 0.5), _graded(zt, 150.0, 0.5, 8.0, 1.25)]
        else:
            z_top += [_graded(g.coil_z1, 150.0, 0.4, 8.0, 1.25)]
    r = np.concatenate([
        _graded(0, ri, 1.5, h_skin * 2),
        _graded(ri, ro, h_skin, h_skin, 1.12),
        _graded(ro, g.coil_r0, h_skin * 2, 0.3),
        _graded(g.coil_r0, g.coil_r1, 0.3, 0.6),
        *r_out,
    ])
    z = np.concatenate([
        _graded(-80.0, -g.Ls, 8.0, 0.5, 1.25),
        *z_mid, *z_top,
    ])
    r = np.unique(np.round(r, 6))
    z = np.unique(np.round(z, 6))
    m = MeshTri.init_tensor(r, z)
    return m


# ------------------------------------------------------------------ 双线性形式
@BilinearForm
def _k_em(u, v, w):
    r = w.x[0]
    return (1.0 / w["mu"]) * ((u.grad[0] + u / r) * (v.grad[0] + v / r) + u.grad[1] * v.grad[1]) * r


@BilinearForm
def _m_em(u, v, w):
    return w["sig"] * u * v * w.x[0]


@LinearForm
def _src(v, w):
    return w["J"] * v * w.x[0]


@BilinearForm
def _k_th(u, v, w):
    return w["k"] * dot(grad(u), grad(v)) * w.x[0]


@BilinearForm
def _c_th(u, v, w):
    return w["rc"] * u * v * w.x[0]


@LinearForm
def _q_th(v, w):
    return w["q"] * v * w.x[0]


@BilinearForm
def _h_bil(u, v, w):
    return w["h"] * u * v * w.x[0]


@LinearForm
def _h_lin(v, w):
    return w["h"] * w["Tinf"] * v * w.x[0]


# ------------------------------------------------------------------ 主求解器
class InductionQuenchSim:
    def __init__(self, geom=None, freq=30e3, power_kw=9.0, t_heat=3.0, t_delay=0.0,
                 medium="water_spray", t_quench=3.0, t_air=6.0, inner_quench=0.0,
                 mur_cold=20.0, T0=25.0, dt=0.01, em_every=0.05, verbose=False,
                 nonlinear=False, bh=None, conc=None):
        self.g = geom or BottleNeckGeometry()
        self.f = freq
        self.P = power_kw * 1e3
        self.t_heat, self.t_delay, self.t_quench, self.t_air = t_heat, t_delay, t_quench, t_air
        self.medium = medium
        self.inner_quench = inner_quench
        self.mur_cold = mur_cold
        self.T0 = T0
        self.dt = dt
        self.em_every = em_every
        self.verbose = verbose
        self.nonlinear = nonlinear          # True: 45 钢用非线性 B-H(含居里点)，False: 旧的线性有效 μr(T)
        from .em.bh import STEEL45_BH
        self.bh = bh or STEEL45_BH
        self.conc = conc                    # rw.em.conc.Concentrator 或 None
        self._mur_state = None
        self.conc_loss_W = 0.0
        self.em_tol = 0.012
        if nonlinear or conc is not None:
            self.em_every = max(em_every, 0.1)   # 温度变化慢，非线性迭代贵
        h_skin = min(0.04, 0.25 * self.skin_depth_mm(25.0))
        self._setup(h_skin)

    def skin_depth_mm(self, T):
        rho = 1 / M.steel_sigma(T)
        mu = M.steel_mur(T, self.mur_cold)
        return 1e3 * np.sqrt(rho / (np.pi * self.f * MU0 * mu))

    # -------------------------------------------------------------- setup
    def _setup(self, h_skin):
        g = self.g
        mm = build_mesh(g, h_skin, self.conc)
        self.mesh_mm = mm
        m = MeshTri(mm.p * 1e-3, mm.t)          # 计算用 SI 单位 (m)
        self.mesh = m
        c = mm.p[:, mm.t].mean(axis=1)           # 单元中心 (mm)
        self.steel_el = g.is_steel(c[0], c[1])
        self.coil_el = g.is_coil(c[0], c[1])
        self.conc_el = self.conc.mask(g, c[0], c[1]) if self.conc is not None else np.zeros_like(self.coil_el)
        self.basis = Basis(m, ElementTriP1())
        self.b0 = self.basis.with_element(ElementTriP0())
        # Dirichlet: 轴线 r=0 和外边界 A=0
        self.em_D = m.boundary_nodes()
        # 线圈电流密度 (每 1 A 总电流)
        area = (g.coil_r1 - g.coil_r0) * (g.coil_z1 - g.coil_z0) * 1e-6
        Jel = np.where(self.coil_el, 1.0 / area, 0.0)
        self.f_unit = asm(_src, self.basis, J=self.b0.interpolate(Jel))

        # 热学子网格（只含钢）
        st = np.nonzero(self.steel_el)[0]
        tt = mm.t[:, st]
        nodes = np.unique(tt)
        remap = -np.ones(mm.p.shape[1], int)
        remap[nodes] = np.arange(len(nodes))
        self.th_nodes = nodes
        self.th_mesh_mm = MeshTri(mm.p[:, nodes], remap[tt])
        tm = MeshTri(mm.p[:, nodes] * 1e-3, remap[tt])
        self.th_mesh = tm
        self.tb = Basis(tm, ElementTriP1())
        self.tb0 = self.tb.with_element(ElementTriP0())
        self.steel_idx = st

        # 表面分类：外表面(喷淋) / 内孔 / 底部截面(绝热)
        bf = tm.boundary_facets()
        p = tm.p * 1e3
        f = tm.facets[:, bf]
        mid = 0.5 * (p[:, f[0]] + p[:, f[1]])
        tvec = p[:, f[1]] - p[:, f[0]]
        n = np.vstack([tvec[1], -tvec[0]])
        n /= np.linalg.norm(n, axis=0)
        el = tm.f2t[0, bf]
        cen = p[:, tm.t[:, el]].mean(axis=1)
        sgn = np.sign(np.sum(n * (mid - cen), axis=0))
        n *= sgn
        probe = mid + n * 0.01
        bottom = probe[1] < -g.Ls + 1e-6
        inner = (~bottom) & (probe[0] < g.inner_r(probe[1])) & (probe[1] < g.L)
        outer = ~bottom & ~inner
        self.fb_outer = FacetBasis(tm, ElementTriP1(), facets=bf[outer])
        self.fb_inner = FacetBasis(tm, ElementTriP1(), facets=bf[inner])

    # -------------------------------------------------------------- 电磁
    def solve_em(self, T_th):
        """给定钢的温度场，返回每个钢单元的焦耳热密度 (W/m3)、所需线圈电流 (A)。
        nonlinear=False 且无导磁体: 线性(有效 μr(T))，一次求解。
        否则: 磁导率逐单元迭代 (描述函数法/基波)，每步同时按功率设定重算线圈电流，直到自洽。"""
        T_el = np.full(self.mesh.t.shape[1], self.T0)
        T_el[self.steel_idx] = T_th[self.th_mesh.t].mean(axis=0)
        sig = np.where(self.steel_el, M.steel_sigma(T_el), 0.0)
        w = 2 * np.pi * self.f
        areas = self._el_area_r()
        if not self.nonlinear and self.conc is None:
            mur = np.where(self.steel_el, M.steel_mur(T_el, self.mur_cold), 1.0)
            x, I_coil, P1 = self._em_solve(mur, sig, w)
            self.n_iter_em = 1
        else:
            mur = self._init_mur(T_el)
            act = self.steel_el | self.conc_el
            xs, fs = [], []                       # Anderson 加速的历史 (log μ)
            for it in range(40):
                x, I_coil, P1 = self._em_solve(mur, sig, w)
                B1 = self._B1(x) * I_coil
                new = mur.copy()
                if self.nonlinear:
                    m = self.steel_el
                    new[m] = self.bh.mur_from_B1(B1[m], T_el[m])
                if self.conc is not None:
                    m = self.conc_el
                    new[m] = self.conc.mat.curve.mur_from_B1(B1[m], 20.0)
                chg = np.abs(np.log(new / mur))
                chg = chg[(B1 > 0.02) & act]
                dmax = float(chg.max()) if chg.size else 0.0
                if getattr(self, "verbose", False): print("  em it", it, "dmax", round(dmax, 4), "I", round(I_coil))
                if dmax < self.em_tol:        # 收敛: 这次求解用的 μ 与更新值一致, x、I 与 μ 自洽
                    break
                # 不动点 y = log new；Anderson(m=5) + 阻尼 0.6
                xk = np.log(mur[act]); fk = np.log(new[act]) - xk
                xs.append(xk); fs.append(fk)
                xs, fs = xs[-6:], fs[-6:]
                beta = 0.6
                if len(xs) >= 2:
                    dF = np.stack([fs[i + 1] - fs[i] for i in range(len(fs) - 1)], axis=1)
                    dX = np.stack([xs[i + 1] - xs[i] for i in range(len(xs) - 1)], axis=1)
                    gam = np.linalg.lstsq(dF, fk, rcond=1e-8)[0]
                    xn = xk + beta * fk - (dX + beta * dF) @ gam
                else:
                    xn = xk + beta * fk
                mur = mur.copy()
                mur[act] = np.exp(np.clip(xn, 0.0, np.log(8000.0)))
            self._mur_state = mur
            self.n_iter_em = it + 1
            self._B1_el = B1
            self._mur_el = mur
            if self.conc is not None:
                pv, _, _ = self.conc.mat.loss_density(B1, self.f)
                self.conc_pv = np.where(self.conc_el, pv, 0.0)
                self.conc_loss_W = float(np.sum(self.conc_pv * areas * 2 * np.pi))
        a_el = np.abs(x[self.mesh.t]) ** 2
        q_el = 0.5 * sig * w**2 * a_el.mean(axis=0)
        q = q_el[self.steel_idx] * I_coil**2
        self._last_A = x * I_coil
        return q, I_coil

    def _init_mur(self, T_el):
        if self._mur_state is not None:
            mur = self._mur_state.copy()
        else:
            mur = np.ones(self.mesh.t.shape[1])
            if self.nonlinear:
                mur[self.steel_el] = 30.0
            else:
                mur = np.where(self.steel_el, M.steel_mur(T_el, self.mur_cold), 1.0)
            if self.conc is not None:
                mur[self.conc_el] = self.conc.mat.curve.mu_i * 0.2
        if self.nonlinear:   # 超过居里点必为 1
            hot = self.steel_el & (T_el >= self.bh.Tc)
            mur[hot] = 1.0
        return mur

    def _em_solve(self, mur, sig, w):
        """线性求解(给定 μ)，单位电流；再按功率设定缩放。返回 (A 复振幅(已缩放), I 峰值, 每1A^2功率)"""
        K = asm(_k_em, self.basis, mu=self.b0.interpolate(MU0 * mur))
        Mm = asm(_m_em, self.basis, sig=self.b0.interpolate(sig))
        A = (K + 1j * w * Mm).tocsc()
        x = np.zeros(A.shape[0], complex)
        Ac, bc, xc, I = condense(A, self.f_unit.astype(complex), x=x, D=self.em_D)
        x[I] = spla.spsolve(Ac.tocsc(), bc)
        a_el = np.abs(x[self.mesh.t]) ** 2
        q_el = 0.5 * sig * w**2 * a_el.mean(axis=0)
        P1 = np.sum(q_el * self._el_area_r() * 2 * np.pi)
        I_coil = np.sqrt(self.P / P1)
        self._x_unit = x
        return x, I_coil, P1

    def _grad_ops(self):
        if not hasattr(self, "_Gr"):
            import scipy.sparse as sp
            p, t = self.mesh.p, self.mesh.t
            r, z = p[0, t], p[1, t]             # (3, nel)
            A2 = (r[1] - r[0]) * (z[2] - z[0]) - (r[2] - r[0]) * (z[1] - z[0])
            # P1 形函数梯度: dN_i/dr = (z_j - z_k)/(2A) ,  dN_i/dz = (r_k - r_j)/(2A)  (i,j,k 循环)
            nel = t.shape[1]
            rows = np.tile(np.arange(nel), 3)
            cols = t.reshape(-1)
            dr = np.concatenate([(z[1] - z[2]) / A2, (z[2] - z[0]) / A2, (z[0] - z[1]) / A2])
            dz = np.concatenate([(r[2] - r[1]) / A2, (r[0] - r[2]) / A2, (r[1] - r[0]) / A2])
            n = p.shape[1]
            self._Gr = sp.csr_matrix((dr, (rows, cols)), shape=(nel, n))
            self._Gz = sp.csr_matrix((dz, (rows, cols)), shape=(nel, n))
            self._Av = sp.csr_matrix((np.full(3 * nel, 1 / 3), (rows, cols)), shape=(nel, n))
            self._rc = r.mean(axis=0)
        return self._Gr, self._Gz, self._Av, self._rc

    def _B1(self, x):
        """单元磁感应强度幅值 |B| (T，每 1 A 峰值电流)。B_z = A/r + dA/dr,  B_r = -dA/dz"""
        Gr, Gz, Av, rc = self._grad_ops()
        Bz = Av @ x / rc + Gr @ x
        Br = -(Gz @ x)
        return np.sqrt(np.abs(Bz) ** 2 + np.abs(Br) ** 2)

    def _el_area_r(self):
        if not hasattr(self, "_ar"):
            p = self.mesh.p
            t = self.mesh.t
            a = 0.5 * np.abs((p[0, t[1]] - p[0, t[0]]) * (p[1, t[2]] - p[1, t[0]]) -
                             (p[0, t[2]] - p[0, t[0]]) * (p[1, t[1]] - p[1, t[0]]))
            rc = p[0, t].mean(axis=0)
            self._ar = a * rc
        return self._ar

    def coil_loss_estimate(self, I):
        """单匝铜线圈交流损耗估算：电流集中在线圈内表面一个铜透入深度内。"""
        g = self.g
        d_cu = np.sqrt(M.COPPER_RHO / (np.pi * self.f * MU0))
        R = M.COPPER_RHO * 2 * np.pi * g.coil_r0 * 1e-3 / ((g.coil_z1 - g.coil_z0) * 1e-3 * d_cu)
        return 0.5 * I**2 * R  # I 为峰值, P = I_rms^2 R

    # -------------------------------------------------------------- 传热一步
    def _thermal_step(self, T, q, phase):
        tb = self.tb
        Tq = tb.interpolate(T)
        k = M.steel_k(Tq.value)
        rc = M.STEEL_45["density"] * M.steel_cp(Tq.value)
        K = asm(_k_th, tb, k=k)
        C = asm(_c_th, tb, rc=rc)
        Q = asm(_q_th, tb, q=self.tb0.interpolate(q)) if q is not None else 0.0
        A = C / self.dt + K
        b = C @ T / self.dt + Q
        for fb, frac in ((self.fb_outer, 1.0), (self.fb_inner, self.inner_quench)):
            Ts = fb.interpolate(T).value
            h_air = M.quench_h("air", Ts)
            if phase == "quench":
                hq = M.quench_h(self.medium, Ts)
                h = frac * hq + (1 - frac) * h_air
                Tinf = M.QUENCHANTS[self.medium]["T_bath"] * frac + 25.0 * (1 - frac)
            else:
                h, Tinf = h_air, 25.0
            A = A + asm(_h_bil, fb, h=h)
            b = b + asm(_h_lin, fb, h=h, Tinf=Tinf)
        return spla.spsolve(A.tocsc(), b)

    # -------------------------------------------------------------- 运行
    def _snapshot(self, T, Tpeak, t800, t500, hist, t, I):
        """断点续算状态（纯 JSON 可序列化）。恢复后与原连续运行的差异仅在：
        恢复点立刻重解一次电磁（等效于 EM 求解时刻轻微移动），终态偏差实测远小于 1%。"""
        def cx(a):
            return None if a is None else {"re": np.real(a).tolist(), "im": np.imag(a).tolist()}
        return {"T": T.tolist(), "Tpeak": Tpeak.tolist(), "t800": t800.tolist(), "t500": t500.tolist(),
                "hist": {k: list(v) for k, v in hist.items()}, "t": float(t), "I": float(I),
                "I_hist": [float(x) for x in self.I_hist],
                "T_end_heat": None if self.T_end_heat is None else self.T_end_heat.tolist(),
                "A_end_heat": cx(self.A_end_heat), "I_end": self.I_end,
                "mur_state": None if self._mur_state is None else self._mur_state.tolist()}

    def run(self, progress=None, checkpoint=None, resume=None, checkpoint_every=200):
        nT = self.tb.N
        T = np.full(nT, self.T0)
        Tpeak = T.copy()
        t800 = np.full(nT, np.nan)
        t500 = np.full(nT, np.nan)
        hist = dict(t=[], Tsurf=[], Tcore=[], Tin=[], I=[])
        # 监测点：淬火区中部 z = L - zone/2
        g = self.g
        p = self.th_mesh_mm.p
        zmid = g.L - g.zone / 2
        iz = np.argmin(np.abs(p[1] - zmid) + 0 * p[0])
        zline = p[1, iz]
        on_line = np.nonzero(np.isclose(p[1], zline))[0]
        on_line = on_line[np.argsort(p[0, on_line])]
        self.line_nodes = on_line
        i_out, i_in = on_line[-1], on_line[0]
        i_mid = on_line[np.argmin(np.abs(p[0, on_line] - 0.5 * (g.ri + g.ro)))]

        t = 0.0
        t_end = self.t_heat + self.t_delay + self.t_quench + self.t_air
        q, I = self.solve_em(T)
        t_lastem = 0.0
        self.T_end_heat = None
        self.A_end_heat = None
        self.I_end = None
        self.I_hist = []
        steps = int(round(t_end / self.dt))
        s0 = 0
        if resume:  # 断点续算：从 checkpoint 恢复全部携带状态（_thermal_step 本身无状态）
            T = np.asarray(resume["T"], float)
            Tpeak = np.asarray(resume["Tpeak"], float)
            t800 = np.asarray(resume["t800"], float)
            t500 = np.asarray(resume["t500"], float)
            hist = {k: list(v) for k, v in resume["hist"].items()}
            t = float(resume["t"])
            I = float(resume["I"])
            t_lastem = -1e18          # 强制下一步重解电磁（q 与当前 T 一致）
            q = None
            self.I_hist = [float(x) for x in resume["I_hist"]]
            te = resume.get("T_end_heat")
            self.T_end_heat = np.asarray(te, float) if te is not None else None
            ae = resume.get("A_end_heat")
            self.A_end_heat = (np.asarray(ae["re"]) + 1j * np.asarray(ae["im"])) if ae is not None else None
            self.I_end = resume.get("I_end")
            ms = resume.get("mur_state")
            self._mur_state = np.asarray(ms, float) if ms is not None else None
            s0 = int(round(t / self.dt))
        for s in range(s0, steps):
            if t < self.t_heat - 1e-9:
                phase = "heat"
                if t - t_lastem >= self.em_every - 1e-9:
                    q, I = self.solve_em(T)
                    t_lastem = t
                    self.I_hist.append(I)
                Tn = self._thermal_step(T, q, phase)
            else:
                if self.T_end_heat is None:
                    self.T_end_heat = T.copy()
                    self.A_end_heat = self._last_A
                    self.I_end = I
                tq = t - self.t_heat
                if tq < self.t_delay:
                    phase = "delay"
                elif tq < self.t_delay + self.t_quench:
                    phase = "quench"
                else:
                    phase = "air"
                Tn = self._thermal_step(T, None, phase)
            tn = t + self.dt
            # 记录 800/500 下穿时间（只在峰值之后、下降段）
            Tpeak = np.maximum(Tpeak, Tn)
            cooling = Tn < T
            for thr, arr in ((800.0, t800), (500.0, t500)):
                cross = cooling & (T >= thr) & (Tn < thr) & np.isnan(arr)
                if np.any(cross):
                    fr = (T[cross] - thr) / (T[cross] - Tn[cross])
                    arr[cross] = t + fr * self.dt
            T, t = Tn, tn
            hist["t"].append(t)
            hist["Tsurf"].append(T[i_out])
            hist["Tcore"].append(T[i_mid])
            hist["Tin"].append(T[i_in])
            if progress and s % 20 == 0:
                progress(s / steps)
            if checkpoint and checkpoint_every and s % checkpoint_every == 0:
                checkpoint(self._snapshot(T, Tpeak, t800, t500, hist, t, I))
        if self.T_end_heat is None:
            self.T_end_heat = T.copy()
            self.A_end_heat = self._last_A
            self.I_end = I
        # 未到 800 的点用峰值开始下降时间近似（基本不会淬硬，影响不大）
        t85 = t500 - t800
        t85 = np.where(np.isnan(t800) & (Tpeak >= 800), np.inf, t85)
        t85 = np.where(Tpeak < 800, np.where(np.isnan(t500), np.inf, 2.0), t85)
        t85 = np.where(np.isnan(t85), np.inf, t85)
        self.T = T
        self.Tpeak = Tpeak
        self.t85 = t85
        self.HRC = M.hardness_estimate(Tpeak, t85)
        self.hist = {k: np.array(v) for k, v in hist.items()}
        return self.summary()

    # -------------------------------------------------------------- 结果
    def summary(self):
        g = self.g
        p = self.th_mesh_mm.p
        ln = self.line_nodes
        r = p[0, ln]
        H = self.HRC[ln]
        depth_from_out = g.ro - r
        # 有效硬化层深度：HRC ≥ 50 从外表面往里
        ok = H >= 50
        if ok[-1]:
            j = len(ok) - 1
            while j > 0 and ok[j - 1]:
                j -= 1
            case = depth_from_out[j]
        else:
            case = 0.0
        Tm = self.T_end_heat
        inzone = p[1] >= g.L - g.zone
        Pc = self.coil_loss_estimate(self.I_end)
        Pcn = self.conc_loss_W
        res = dict(
            freq_kHz=self.f / 1e3,
            power_kW=self.P / 1e3,
            medium=M.QUENCHANTS[self.medium]["name"],
            skin_cold_mm=self.skin_depth_mm(25),
            skin_hot_mm=self.skin_depth_mm(900),
            T_surf_end_heat=float(Tm[ln[-1]]),
            T_inner_end_heat=float(Tm[ln[0]]),
            T_max=float(self.Tpeak.max()),
            T_zone_min=float(Tm[inzone].min()),
            coil_current_A=float(self.I_end / np.sqrt(2)),  # 有效值
            coil_loss_kW=float(Pc / 1e3),
            efficiency=float(self.P / (self.P + Pc + Pcn)),
            conc_loss_kW=float(Pcn / 1e3),
            nonlinear=bool(self.nonlinear),
            conc=(self.conc.mat.name if self.conc is not None else None),
            case_depth_mm=float(case),
            through_hardened=bool(case >= g.wall - 0.05),
            HRC_surface=float(H[-1]),
            HRC_inner=float(H[0]),
            t85_surface=float(self.t85[ln[-1]]),
            t85_inner=float(self.t85[ln[0]]),
            melted=bool(self.Tpeak.max() >= M.STEEL_45["T_melt"]),
            overheated=bool(self.Tpeak.max() >= 1100),
        )
        warn = []
        if res["melted"]:
            warn.append("表面熔化！功率太大或时间太长")
        elif res["overheated"]:
            warn.append(f"最高 {res['T_max']:.0f}°C 过烧/晶粒粗大风险(>1100°C)")
        if res["T_zone_min"] < M.STEEL_45["Ac3_fast"]:
            warn.append(f"淬火区有部位只到 {res['T_zone_min']:.0f}°C，未完全奥氏体化")
        if res["efficiency"] < 0.6:
            warn.append(f"电效率低 ({res['efficiency']*100:.0f}%)，频率与壁厚不匹配")
        if self.conc is not None and self.conc_loss_W > 0:
            vol = float(np.sum(self.conc_el * self._el_area_r() * 2 * np.pi))
            pv = self.conc_loss_W / vol           # 平均损耗密度 W/m3
            res["conc_dTdt"] = pv / 3.5e6         # 绝热升温速率 K/s (ρcp≈3.5e6 J/m3K)
            if res["conc_dTdt"] > 20:
                warn.append(f"导磁体({self.conc.mat.name})自身发热 {self.conc_loss_W/1e3:.2f} kW，"
                            f"绝热升温约 {res['conc_dTdt']:.0f} K/s，必须水冷或会过热")
        res["warnings"] = warn
        return res


def find_power(target_C=900.0, which="surface", tol=10.0, p0=5.0, **kw):
    """自动找功率：加热结束时淬火区中部（外表面 / 内孔）温度 = 目标值。
    对应实际设备里"红外测温闭环"或调功率试烧的过程。返回 kW。"""
    if target_C > M.STEEL_45["T_melt"]:
        raise ValueError(f"目标温度 {target_C:.0f}°C 超过 45 钢固相线 {M.STEEL_45['T_melt']:.0f}°C：物理上不可达，请检查目标温度")
    kw = dict(kw)
    kw.update(t_quench=0.0, t_air=0.0, t_delay=0.0)
    key = "T_surf_end_heat" if which == "surface" else "T_inner_end_heat"

    def T_of(P):
        s = InductionQuenchSim(power_kw=P, **kw)
        s.run()
        return s.summary()[key]

    P1 = p0
    T1 = T_of(P1)
    for _ in range(8):
        # 温升近似正比于功率
        P2 = P1 * (target_C - 25.0) / max(T1 - 25.0, 1.0)
        P2 = float(np.clip(P2, 0.2, 200.0))
        T2 = T_of(P2)
        if abs(T2 - target_C) < tol:
            return P2
        P1, T1 = P2, T2
    raise ValueError(f"自动调功率 8 轮未收敛：最近 {P1:.2f} kW 只能到 {T1:.0f}°C（目标 {target_C:.0f}°C）。"
                     "该工件/频率组合可能达不到目标温度，或加热时间太短")
