"""非圆截面涡流（v0.14）：方坯/任意矩形截面工件 + 圆环线圈，gmsh 非结构四面体网格。
gmsh 是可选依赖（pip 有 x86_64/aarch64 轮子）；不装也能用其它全部功能，装了才解锁本模块。
网格用 gmsh OCC 布尔 fragment（空气盒、工件、线圈环共形），物理分组靠"体心落在哪个几何域"分类；
求解完全复用 eddy3d.py 的 A 公式 Nédélec 棱边单元（子类只换 build()）。
集肤层靠"工件表面附近加密"（h_skin 默认 3.0 mm ≈ 30 kHz 热态钢 δ≈3 mm；内存够可降到 1.5-2.0 更准）。"""
import numpy as np
from .eddy3d import TubeInCoil3D, MU0


def _gmsh():
    try:
        import gmsh
        return gmsh
    except ImportError:
        raise ImportError("非圆截面涡流需要 gmsh：pip install gmsh（可选依赖，不装不影响其它功能）")


class BilletInCoil3D(TubeInCoil3D):
    """方坯（a×a 方截面，z 向长棒）置于圆环线圈中。coil 轴 = z 轴，工件轴可偏心 ecc_mm（沿 x）。
    billet=None 退化为圆柱工件（半径 ro，用于与结构网格/轴对称解对照）。"""

    def __init__(self, a=20.0, h_skin=3.0, h_coil=5.0, h_air=25.0, **kw):
        kw.setdefault("ri", 0.0)              # 方坯是实心的
        kw.setdefault("box_r", 40.0)          # 比结构网格小的空气盒：省内存(低内存机器可跑)，截断影响已含在 §11g 对照误差内
        kw.setdefault("box_z", (-35.0, 80.0))
        super().__init__(**kw)
        self.a, self.h_skin, self.h_coil, self.h_air = a, h_skin, h_coil, h_air

    def build(self):
        gmsh = _gmsh()
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)   # 静默；异常仍会抛出
        try:
            gmsh.model.add("billet")
            occ = gmsh.model.occ
            z0, z1 = self.box_z
            R = self.box_r
            air = occ.addBox(-R, -R, z0, 2 * R, 2 * R, z1 - z0)
            if self.a:                                   # 方坯
                b = occ.addBox(self.ecc_mm - self.a / 2, -self.a / 2, self.z_tube[0], self.a, self.a,
                               self.z_tube[1] - self.z_tube[0])
            else:                                        # 圆柱工件（对照用；ri>0 时为空心管，与结构网格同几何）
                b = occ.addCylinder(self.ecc_mm, 0, self.z_tube[0], 0, 0, self.z_tube[1] - self.z_tube[0], self.ro)
                if self.ri > 0:
                    ib = occ.addCylinder(self.ecc_mm, 0, self.z_tube[0] - 1, 0, 0, self.z_tube[1] - self.z_tube[0] + 2, self.ri)
                    b = occ.cut([(3, b)], [(3, ib)], removeObject=True, removeTool=True)[0][0][1]
            oc = occ.addCylinder(0, 0, self.coil_z[0], 0, 0, self.coil_z[1] - self.coil_z[0], self.coil_r1)
            ic = occ.addCylinder(0, 0, self.coil_z[0] - 1, 0, 0, self.coil_z[1] - self.coil_z[0] + 2, self.coil_r0)
            coil = occ.cut([(3, oc)], [(3, ic)], removeObject=True, removeTool=True)[0][0][1]
            frag, fmap = occ.fragment([(3, air)], [(3, b), (3, coil)], removeObject=True, removeTool=True)
            occ.synchronize()
            # 体分类：fragment 映射表——fmap[1]=工件工具体，fmap[2]=线圈工具体，其余为空气（确定性，不靠体心猜）
            wp_vol = [t for d, t in fmap[1] if d == 3]
            coil_vol = [t for d, t in fmap[2] if d == 3]
            tagged = set(wp_vol) | set(coil_vol)
            air_vol = [t for d, t in gmsh.model.getEntities(3) if t not in tagged]
            assert wp_vol and coil_vol, "工件/线圈体分类失败"
            # 网格尺寸：工件与线圈表面附近的点加密
            fine_pts, coil_pts = set(), set()
            for tag in wp_vol:
                for d, t in gmsh.model.getBoundary([(3, tag)], recursive=True):
                    if d == 0: fine_pts.add(t)
            for tag in coil_vol:
                for d, t in gmsh.model.getBoundary([(3, tag)], recursive=True):
                    if d == 0: coil_pts.add(t)
            gmsh.model.mesh.setSize([(0, t) for t in fine_pts], self.h_skin)
            gmsh.model.mesh.setSize([(0, t) for t in coil_pts - fine_pts], self.h_coil)
            gmsh.option.setNumber("Mesh.MeshSizeMax", self.h_air)
            gmsh.model.mesh.generate(3)
            # 取节点与四面体（按实体分组建 tag）
            ntags, P, _ = gmsh.model.mesh.getNodes()
            P = P.reshape(-1, 3).T * 1e-3                  # mm → m
            idx = {int(t): i for i, t in enumerate(ntags)}
            tets, etag = [], []
            for grp, tags in ((0, air_vol), (1, wp_vol), (2, coil_vol)):
                for tag in tags:
                    try:
                        _, nids = gmsh.model.mesh.getElementsByType(4, tag)
                    except Exception:
                        continue
                    if not len(nids): continue
                    T = np.array(nids, dtype=np.int64).reshape(-1, 4)
                    tets.append(np.vectorize(idx.get)(T).T)
                    etag += [grp] * len(T)
            t = np.hstack(tets); etag = np.array(etag)
            # 体积为正
            p0 = P[:, t[0]]
            a1, b1, c1 = P[:, t[1]] - p0, P[:, t[2]] - p0, P[:, t[3]] - p0
            det = np.einsum("in,in->n", a1, np.cross(b1.T, c1.T).T)
            flip = det < 0
            tt = t.copy(); tt[2, flip], tt[3, flip] = t[3, flip], t[2, flip]
            from skfem import MeshTet, Basis, ElementTetN0
            self.mesh = MeshTet(P, tt)
            self.etag = etag
        finally:
            gmsh.finalize()
        m = self.mesh
        self.cen = m.p[:, m.t].mean(axis=1)
        self.tube = self.etag == 1
        self.sig = np.where(self.tube, 1.0 / self.rho, self.sigma_air)
        self.nu = np.where(self.tube, 1.0 / (MU0 * self.mur), 1.0 / MU0)
        self.basis = Basis(m, ElementTetN0())
        # 线圈电流（绕 z 轴切向；线圈轴在 (0,0)，工件偏心 ecc 不影响源的位置）
        X = self.basis.global_coordinates().value * 1e3    # mm
        rr = np.hypot(X[0], X[1])
        incoil = (self.etag[:, None] == 2) & np.ones_like(rr, bool)
        # 实际离散线圈体积归一化总安匝
        vol = float(np.sum(incoil * self.basis.dx))
        rbar = float(np.sum(incoil * rr * 1e-3 * self.basis.dx) / max(vol, 1e-30))
        area = vol / (2 * np.pi * rbar)
        J0 = self.I_peak / area
        rs = np.maximum(rr, 1e-9)
        self.Jx = np.where(incoil, -J0 * X[1] / rs, 0.0)
        self.Jy = np.where(incoil, J0 * X[0] / rs, 0.0)
        self.Jz = np.zeros_like(self.Jx)
        self.coil_area_mm2 = area * 1e6
        self.nedges = self.basis.N
        return self

    def solve(self):
        return super().solve(permc="COLAMD")     # 非结构网格：COLAMD 排序防 LU fill-in（NATURAL 会内存爆炸）

def _el_vol(p):
    """每个四面体体积 (m³，网格为 SI)。"""
    t, pts = p.mesh.t, p.mesh.p
    cr = np.cross((pts[:, t[1]] - pts[:, t[0]]).T, (pts[:, t[2]] - pts[:, t[0]]).T)
    return np.abs(np.sum(cr * (pts[:, t[3]] - pts[:, t[0]]).T, axis=1)) / 6.0


def _billet_stats(p, a):
    """方坯结果统计: 总功率、表层/心部集肤比、角部/面心比、左右对称度、空气假损耗 (线圈窗口 z 内)。"""
    vol = _el_vol(p); m = p.tube
    x, y, z = (p.cen[i][m] * 1000 for i in range(3))      # m → mm
    pq, vv = p.pq[m], vol[m]
    zw = (z >= p.coil_z[0]) & (z <= p.coil_z[1])
    xc = x - p.ecc_mm                                     # 相对坯料中心
    a2 = a / 2
    depth = a2 - np.maximum(np.abs(xc), np.abs(y))
    sf, co = zw & (depth < 1.5), zw & (depth > 4.5)
    skin = float((pq[sf].sum() / vv[sf].sum()) / (pq[co].sum() / vv[co].sum()))
    mn = np.minimum(np.abs(xc), np.abs(y))
    cor, fac = sf & (mn > 0.6 * a2), sf & (mn < 0.3 * a2)
    corner = float((pq[cor].sum() / vv[cor].sum()) / (pq[fac].sum() / vv[fac].sum()))
    sym = float(pq[zw & (y > 0)].sum() / max(pq[zw & (y < 0)].sum(), 1e-30))
    return dict(P=p.P_tube, P_air=p.P_air_reg, tets=int(p.mesh.t.shape[1]), ecc_mm=float(p.ecc_mm),
                skin=skin, corner=corner, sym=sym)


def _billet_map(p, a, nmax=4000):
    """线圈窗口内工件单元的功率密度 (x,y mm + W/m³)，供 GUI 画截面热图；子采样到 nmax 点。"""
    vol = _el_vol(p)
    z = p.cen[2] * 1000
    idx = np.where(p.tube & (z >= p.coil_z[0]) & (z <= p.coil_z[1]))[0]
    den, x, y = p.pq[idx] / vol[idx], p.cen[0][idx] * 1000, p.cen[1][idx] * 1000
    if len(idx) > nmax:
        sel = np.linspace(0, len(idx) - 1, nmax).astype(int)
        x, y, den = x[sel], y[sel], den[sel]
    return dict(x=list(map(float, x)), y=list(map(float, y)), d=list(map(float, den)), a=a)


def study_billet(a=20.0, ecc_mm=3.0, with_map=True, **kw):
    """方坯标准算例: 同心 + 偏心各一发粗网格 (约 15 s/例，2 GB 内存可跑)。返回可 JSON 化的统计与截面热图数据。"""
    import gc
    out = {}
    p0 = BilletInCoil3D(a=a, ecc_mm=0.0, **kw).build().solve()
    out["ecc0"] = _billet_stats(p0, a)
    if with_map:
        out["ecc0"]["map"] = _billet_map(p0, a)
    del p0; gc.collect()
    if ecc_mm > 0:
        p1 = BilletInCoil3D(a=a, ecc_mm=ecc_mm, **kw).build().solve()
        out["ecc"] = _billet_stats(p1, a)
        if with_map:
            out["ecc"]["map"] = _billet_map(p1, a)
        del p1; gc.collect()
    else:
        out["ecc"] = out["ecc0"]
    return out

