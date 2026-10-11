"""薄板弯曲有限元（Kirchhoff 板, Morley 单元），用于转盘、台板等平板零件。单位 N, mm, MPa。"""
import numpy as np
from skfem import MeshTri, Basis, ElementTriMorley, BilinearForm, LinearForm, asm, condense, solve
from skfem.helpers import dd, ddot, trace, eye


def annulus_mesh(r_in, r_out, nr=24, nt=120):
    r = r_in + (r_out - r_in) * (np.linspace(0, 1, nr + 1) ** 1.0)
    th = np.linspace(0, 2 * np.pi, nt, endpoint=False)
    R, T = np.meshgrid(r, th, indexing="ij")
    p = np.vstack([(R * np.cos(T)).ravel(), (R * np.sin(T)).ravel()])
    idx = np.arange(p.shape[1]).reshape(nr + 1, nt)
    tris = []
    for i in range(nr):
        for j in range(nt):
            a, b = idx[i, j], idx[i, (j + 1) % nt]
            c, d = idx[i + 1, j], idx[i + 1, (j + 1) % nt]
            tris += [[a, b, d], [a, d, c]]
    return MeshTri(p, np.array(tris).T)


class Plate:
    def __init__(self, mesh, h, E, nu, rho):
        self.m, self.h, self.E, self.nu, self.rho = mesh, h, E / 1e6, nu, rho * 1e-9
        self.basis = Basis(mesh, ElementTriMorley())
        D = self.E * h**3 / (12 * (1 - nu**2))
        self.D = D
        nu_ = nu

        def C(T):
            return (1 - nu_) * T + nu_ * eye(trace(T), 2)

        @BilinearForm
        def a(u, v, w):
            return D * ddot(C(dd(u)), dd(v))

        self.K = asm(a, self.basis)
        self.f = np.zeros(self.basis.N)

    def self_weight(self, g=9.81):
        q = self.rho * self.h * g  # N/mm2 向下

        @LinearForm
        def l(v, w):
            return -q * v

        self.f += asm(l, self.basis)

    def point_load(self, xy, Fz):
        """在最近的顶点加集中力 Fz (N, 向上为正)。"""
        p = self.m.p
        i = int(np.argmin((p[0] - xy[0]) ** 2 + (p[1] - xy[1]) ** 2))
        dof = self.basis.nodal_dofs[0, i]
        self.f[dof] += Fz
        return i

    def clamp_inner(self, r):
        rr = np.hypot(*self.m.p)
        facets = self.m.boundary_facets()
        mid = self.m.p[:, self.m.facets[:, facets]].mean(axis=1)
        fin = facets[np.hypot(*mid) <= r + 1e-6]
        self.D_dofs = self.basis.get_dofs(facets=fin).all()

    def solve(self):
        self.w = solve(*condense(self.K, self.f, D=self.D_dofs))
        return self.w

    def deflection_at(self, xy):
        p = self.m.p
        i = int(np.argmin((p[0] - xy[0]) ** 2 + (p[1] - xy[1]) ** 2))
        return self.w[self.basis.nodal_dofs[0, i]]

    def max_stress(self):
        W = self.basis.interpolate(self.w)
        H = W.hess
        nu = self.nu
        kx, ky, kxy = H[0][0], H[1][1], H[0][1]
        Mx = -self.D * (kx + nu * ky)
        My = -self.D * (ky + nu * kx)
        Mxy = -self.D * (1 - nu) * kxy
        # 主弯矩 → 表面应力 6M/h²
        Mavg = 0.5 * (Mx + My)
        Rm = np.sqrt((0.5 * (Mx - My)) ** 2 + Mxy**2)
        M1, M2 = Mavg + Rm, Mavg - Rm
        s1, s2 = 6 * M1 / self.h**2, 6 * M2 / self.h**2
        vm = np.sqrt(s1**2 - s1 * s2 + s2**2)
        return float(vm.max())
