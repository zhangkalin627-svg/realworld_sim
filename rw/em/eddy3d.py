"""全三维时谐涡流 (A 公式, Nédélec 棱边单元) + 轴对称参照解

开源方案调研结果 (2026-10-12):
  - gmsh: PyPI 没有 aarch64 轮子 (x86_64 有)；Debian 有 deb 但依赖 OpenCASCADE 很重 → 本版不用。
  - Elmer FEM (GPL): Debian 有 elmerfem-csc deb，依赖 MPI 全家桶；GPL 只能当外部程序调用 → 留作以后。
  - scikit-fem (BSD-3) 自带 ElementTetN0 (最低阶 Nédélec 棱边单元)，pip 在 arm64/x86 都能装 → 本版采用。
  - 网格: 自写"极坐标结构网格"(圆柱/圆管/线圈环天然贴合)，每个六面体拆 6 个四面体(Kuhn 剖分，面一致)，
    轴线上的退化六面体自动退成 3 个四面体。非圆截面(方钢)以后再用 gmsh。

方程 (峰值相量, e^{jωt}):   curl( ν curl A ) + jωσ A = J_s     ,  E = -jωA
  - 空气里的 σ 取一个很小的人为值(默认 1 S/m)来固定规范 (A-only 公式零空间)；对结果的影响用 σ_air 扫描检验。
  - 边界: 外盒 n×A = 0。
  - 适用: 工件是一个整体、电荷不在表面累积 (圆管/圆棒/方坯放在线圈里都满足)；有开缝、需要 V 的情形 (电流必须绕缝) 不在此公式内。

功率: P = ½ ∫ σ |E|² dV 。线圈电流为每匝总安匝 I (峰值)，均匀分布在线圈截面。
"""
import numpy as np
import scipy.sparse.linalg as spla
from skfem import MeshTet, Basis, ElementTetN0, BilinearForm, LinearForm, asm, MeshTri, ElementTriP1, ElementTriP0, condense
from skfem.helpers import dot, curl

MU0 = 4e-7 * np.pi


# ------------------------------------------------------------------ 极坐标结构四面体网格
def polar_tet_mesh(r_lines, nth, z_lines):
    """r_lines[0] 必须是 0 (轴)；单位 m。返回 MeshTet 与 每个单元的 (i,j,k) 索引(六面体编号)。"""
    r = np.asarray(r_lines, float)
    z = np.asarray(z_lines, float)
    assert r[0] == 0.0 and nth >= 3
    nr, nz = len(r), len(z)
    th = 2 * np.pi * np.arange(nth) / nth
    # 正 nth 边形的面积比圆小；把半径放大 sc 使每一圈的面积等于圆面积 (线圈/管壁的体积、电流面积才对)
    sc = np.sqrt((2 * np.pi / nth) / np.sin(2 * np.pi / nth))
    # 节点编号: 轴上每个 z 一个节点；其余 (i>=1) 按 (k, i-1, j)
    def nid(i, j, k):
        j = j % nth
        if i == 0:
            return k * (1 + (nr - 1) * nth)
        return k * (1 + (nr - 1) * nth) + 1 + (i - 1) * nth + j
    P = np.zeros((3, nz * (1 + (nr - 1) * nth)))
    for k in range(nz):
        base = k * (1 + (nr - 1) * nth)
        P[:, base] = (0, 0, z[k])
        for i in range(1, nr):
            idx = base + 1 + (i - 1) * nth + np.arange(nth)
            P[0, idx] = r[i] * sc * np.cos(th)
            P[1, idx] = r[i] * sc * np.sin(th)
            P[2, idx] = z[k]
    # 六面体 → 6 四面体 (Kuhn)
    perms = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
    tets, hexid = [], []
    I, J, K = np.meshgrid(np.arange(nr - 1), np.arange(nth), np.arange(nz - 1), indexing="ij")
    I, J, K = I.ravel(), J.ravel(), K.ravel()
    vec = np.vectorize(nid)
    for pm in perms:
        v = [np.zeros(3, int)]
        for a in pm:
            e = v[-1].copy(); e[a] += 1; v.append(e)
        cols = [vec(I + c[0], J + c[1], K + c[2]) for c in v]
        T = np.array(cols)          # (4, nhex)
        # 去掉退化(有重复节点)的四面体
        ok = np.ones(T.shape[1], bool)
        for a in range(4):
            for b in range(a + 1, 4):
                ok &= T[a] != T[b]
        tets.append(T[:, ok]); hexid.append(np.nonzero(ok)[0])
    t = np.hstack(tets)
    hx = np.concatenate(hexid)
    # 方向: 体积为正
    p0 = P[:, t[0]]
    a, b, c = P[:, t[1]] - p0, P[:, t[2]] - p0, P[:, t[3]] - p0
    det = np.einsum("in,in->n", a, np.cross(b.T, c.T).T)
    flip = det < 0
    tt = t.copy()
    tt[2, flip], tt[3, flip] = t[3, flip], t[2, flip]
    m = MeshTet(P, tt)
    return m, (I[hx], J[hx], K[hx])


@BilinearForm
def _cc(u, v, w):
    return w["nu"] * dot(curl(u), curl(v))


@BilinearForm
def _mm(u, v, w):
    return w["sig"] * dot(u, v)


@LinearForm
def _src3(v, w):
    return w["Jx"] * v.value[0] + w["Jy"] * v.value[1] + w["Jz"] * v.value[2]


# ------------------------------------------------------------------ 问题定义
class TubeInCoil3D:
    """圆管工件 + 圆环线圈 (单位 mm)。管子轴 = 极坐标网格轴 = z 轴；线圈轴可偏心 (ecc_mm, 沿 x)。
    工件材料非磁性或常 μr (热态钢 μr=1)，电阻率 rho (Ω·m)。"""

    def __init__(self, ri=8.0, ro=11.0, z_tube=(-20.0, 50.0), coil_r0=15.0, coil_r1=21.0, coil_z=(27.0, 52.0),
                 rho=1.1e-6, mur=1.0, freq=30e3, ecc_mm=0.0, nth=24, box_r=50.0, box_z=(-45.0, 100.0),
                 sigma_air=1.0, fine=1.0, I_peak=1000.0):
        self.__dict__.update(locals())
        del self.__dict__["self"]
        self.omega = 2 * np.pi * freq
        self._lines()

    def _lines(self):
        ri, ro, r0, r1, R = self.ri, self.ro, self.coil_r0, self.coil_r1, self.box_r
        f = self.fine

        def seg(a, b, h):
            n = max(1, int(round((b - a) / (h * f))))
            return np.linspace(a, b, n + 1)[:-1]
        r = np.concatenate([seg(0, ri, 2.7), seg(ri, ro, 1.0), seg(ro, r0, 2.0), seg(r0, r1, 2.0), seg(r1, 33.0, 4.0),
                            seg(33.0, R, 8.5), [R]])
        z0, z1 = self.box_z
        zt0, zt1 = self.z_tube
        c0, c1 = self.coil_z
        zl = np.concatenate([seg(z0, zt0, 12.0), seg(zt0, 15.0, 5.0), seg(15.0, c0, 4.0), seg(c0, c1, 2.5),
                             seg(c1, 62.0, 3.0), seg(62.0, 80.0, 6.0), seg(80.0, z1, 10.0), [z1]])
        zl = np.unique(np.round(np.concatenate([zl, [zt1]]), 6))
        self.r_lines, self.z_lines = np.unique(np.round(r, 6)), zl

    def build(self):
        r, z = self.r_lines * 1e-3, self.z_lines * 1e-3
        self.mesh, (ii, jj, kk) = polar_tet_mesh(r, self.nth, z)
        m = self.mesh
        c = m.p[:, m.t].mean(axis=1)                       # 单元中心 (m)
        self.cen = c
        self.sc = np.sqrt((2 * np.pi / self.nth) / np.sin(2 * np.pi / self.nth))
        rl, zl_ = self.r_lines, self.z_lines                # 材料按六面体编号分类(精确, 与网格线对齐)
        rho_c = 0.5 * (rl[ii] + rl[ii + 1])
        zc = 0.5 * (zl_[kk] + zl_[kk + 1])
        self.tube = (rho_c >= self.ri) & (rho_c <= self.ro) & (zc >= self.z_tube[0]) & (zc <= self.z_tube[1])
        sig = np.where(self.tube, 1.0 / self.rho, self.sigma_air)
        self.sig = sig
        self.nu = np.where(self.tube, 1.0 / (MU0 * self.mur), 1.0 / MU0)
        self.basis = Basis(m, ElementTetN0())
        # 线圈电流密度 (方向: 绕线圈轴的切向), 在积分点上判断位置
        X = self.basis.global_coordinates().value * 1e3      # mm (3,nel,nqp)
        dx = X[0] / self.sc - (-self.ecc_mm)                   # 线圈轴在 x=-ecc (工件相对线圈向 +x 偏)
        dy = X[1] / self.sc
        rr = np.hypot(dx, dy)
        incoil = (rr >= self.coil_r0) & (rr <= self.coil_r1) & (X[2] >= self.coil_z[0]) & (X[2] <= self.coil_z[1])
        # 实际离散线圈截面积 = Σ(incoil*dx_vol)/(2π r̄) ; 用它归一化，使总安匝=I_peak
        vol = float(np.sum(incoil * self.basis.dx))
        rbar = float(np.sum(incoil * rr * 1e-3 * self.basis.dx) / max(vol, 1e-30))
        area = vol / (2 * np.pi * rbar)
        J0 = self.I_peak / area
        rs = np.maximum(rr, 1e-9)
        self.Jx = np.where(incoil, -J0 * dy / rs, 0.0)
        self.Jy = np.where(incoil, J0 * dx / rs, 0.0)
        self.Jz = np.zeros_like(self.Jx)
        self.coil_area_mm2 = area * 1e6
        self.nedges = self.basis.N
        return self

    def solve(self):
        b = self.basis
        K = asm(_cc, b, nu=self._el_to_qp(self.nu))
        Mm = asm(_mm, b, sig=self._el_to_qp(self.sig))
        f = asm(_src3, b, Jx=self.Jx, Jy=self.Jy, Jz=self.Jz)
        f = self._divergence_clean(f)
        A = (K + 1j * self.omega * Mm).tocsc()
        D = b.get_dofs().all()
        x = np.zeros(A.shape[0], complex)
        Ac, bc, xc, Iin = condense(A, f.astype(complex), x=x, D=D)
        x[Iin] = spla.splu(Ac.tocsc(), permc_spec='NATURAL').solve(bc)   # 结构网格自然序 = 带状，比 COLAMD/MMD 快很多
        self.x = x
        # 功率密度 (按单元)
        Ar = b.interpolate(x.real).value
        Ai = b.interpolate(x.imag).value
        a2 = (Ar**2 + Ai**2).sum(axis=0)                      # (nel,nqp)
        pe = 0.5 * self.omega**2 * self.sig[:, None] * a2     # W/m3
        self.pq = (pe * b.dx).sum(axis=1)                      # 每单元功率 W
        self.P_tube = float(self.pq[self.tube].sum())
        self.P_air_reg = float(self.pq[~self.tube].sum())      # 规范用的人为 σ 在空气里的"假损耗"，应 ≪ P_tube
        return self

    def _incidence(self):
        """棱边-节点关联矩阵 G (nedges × nnodes): 梯度场 grad(φ) 的 N0 系数 = G φ。
        棱边方向取 skfem 的约定 (mesh.edges 的两个端点, 小编号指向大编号)。"""
        import scipy.sparse as sp
        m, b = self.mesh, self.basis
        ed = m.edges
        dof = b.dofs.edge_dofs[0]
        n = ed.shape[1]
        rows = np.concatenate([dof, dof])
        cols = np.concatenate([ed[1], ed[0]])
        val = np.concatenate([np.ones(n), -np.ones(n)])
        return sp.csr_matrix((val, (rows, cols)), shape=(b.N, m.p.shape[1]))

    def _divergence_clean(self, f):
        """线圈电流在积分点上按"位置是否在线圈里"取值，当线圈边界与网格不对齐(偏心)时离散电流有散度，
        A-only 公式会把它当成电荷堆积，在空气人为电导里产生假电流。这里把源投影到无散子空间: f' = f - G φ, GᵀGφ = Gᵀf。"""
        import scipy.sparse as sp
        G = self._incidence()
        L = (G.T @ G).tocsc() + 1e-9 * sp.identity(G.shape[1], format="csc")
        phi = spla.spsolve(L, G.T @ f)
        self.src_div_before = float(np.linalg.norm(G.T @ f))
        f2 = f - G @ phi
        self.src_div_after = float(np.linalg.norm(G.T @ f2))
        self.src_norm = float(np.linalg.norm(f))
        return f2

    def _el_to_qp(self, v):
        return np.repeat(v[:, None], self.basis.dx.shape[1], axis=1)

    def power_by_sector(self, nsec=12):
        c = self.cen
        ang = (np.degrees(np.arctan2(c[1], c[0])) + 360) % 360
        s = np.minimum((ang / (360 / nsec)).astype(int), nsec - 1)
        out = np.zeros(nsec)
        np.add.at(out, s[self.tube], self.pq[self.tube])
        return out


# ------------------------------------------------------------------ 二维轴对称参照解 (同一几何同一网格线)
def axi_reference(prob: TubeInCoil3D):
    from ..induction import _k_em, _m_em, _src
    from skfem import Basis
    r = prob.r_lines * 1e-3
    z = prob.z_lines * 1e-3
    m = MeshTri.init_tensor(r, z)
    basis = Basis(m, ElementTriP1())
    b0 = basis.with_element(ElementTriP0())
    c = m.p[:, m.t].mean(axis=1) * 1e3
    tube = (c[0] >= prob.ri) & (c[0] <= prob.ro) & (c[1] >= prob.z_tube[0]) & (c[1] <= prob.z_tube[1])
    coil = (c[0] >= prob.coil_r0) & (c[0] <= prob.coil_r1) & (c[1] >= prob.coil_z[0]) & (c[1] <= prob.coil_z[1])
    sig = np.where(tube, 1.0 / prob.rho, 0.0)
    mu = np.where(tube, MU0 * prob.mur, MU0)
    area = (prob.coil_r1 - prob.coil_r0) * (prob.coil_z[1] - prob.coil_z[0]) * 1e-6
    J = np.where(coil, prob.I_peak / area, 0.0)
    K = asm(_k_em, basis, mu=b0.interpolate(mu))
    Mm = asm(_m_em, basis, sig=b0.interpolate(sig))
    f = asm(_src, basis, J=b0.interpolate(J))
    A = (K + 1j * prob.omega * Mm).tocsc()
    x = np.zeros(A.shape[0], complex)
    Ac, bc, xc, I = condense(A, f.astype(complex), x=x, D=m.boundary_nodes())
    x[I] = spla.spsolve(Ac.tocsc(), bc)
    a2 = (np.abs(x[m.t]) ** 2).mean(axis=0)
    p = m.p; t = m.t
    ar = 0.5 * np.abs((p[0, t[1]] - p[0, t[0]]) * (p[1, t[2]] - p[1, t[0]]) - (p[0, t[2]] - p[0, t[0]]) * (p[1, t[1]] - p[1, t[0]]))
    rc = p[0, t].mean(axis=0)
    q = 0.5 * sig * prob.omega**2 * a2
    return float(np.sum(q * ar * rc * 2 * np.pi))


def study_3d(ecc_mm=3.0, nth=16, fine=1.5, nsec=8, sweep_sigma=False, **kw):
    """标准算例: 热态钢管(μr=1, ρ=1.1e-6)、30 kHz。同心(与二维轴对称比)+偏心。返回可存 JSON 的 dict。"""
    out = {}
    p0 = TubeInCoil3D(nth=nth, fine=fine, ecc_mm=0.0, **kw).build().solve()
    P2 = axi_reference(p0)
    out["ecc0"] = dict(P_3d=p0.P_tube, P_2d=P2, P_air=p0.P_air_reg, sectors=list(map(float, p0.power_by_sector(nsec))),
                       tets=int(p0.mesh.t.shape[1]), edges=int(p0.nedges), ecc_mm=0.0, I_peak=p0.I_peak)
    if ecc_mm > 0:
        p1 = TubeInCoil3D(nth=nth, fine=fine, ecc_mm=ecc_mm, **kw).build().solve()
        out["ecc"] = dict(P_3d=p1.P_tube, P_air=p1.P_air_reg, sectors=list(map(float, p1.power_by_sector(nsec))), ecc_mm=ecc_mm)
    else:
        out["ecc"] = out["ecc0"]
    if sweep_sigma:
        out["sigma_sweep"] = {}
        for sa in (0.1, 10.0):
            ps = TubeInCoil3D(nth=nth, fine=fine, ecc_mm=0.0, sigma_air=sa, **kw).build().solve()
            out["sigma_sweep"][str(sa)] = ps.P_tube
    return out
