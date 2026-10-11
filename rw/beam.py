"""三维梁/框架有限元（每节点 6 自由度，Euler-Bernoulli 梁）。单位：N, mm, MPa。"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


class Section:
    def __init__(self, A, Iy, Iz, J, cy, cz, name=""):
        self.A, self.Iy, self.Iz, self.J, self.cy, self.cz, self.name = A, Iy, Iz, J, cy, cz, name

    @staticmethod
    def rhs(b, h, t):
        """矩形空心方管 b×h×t (mm)"""
        bi, hi = b - 2 * t, h - 2 * t
        A = b * h - bi * hi
        Iy = (b * h**3 - bi * hi**3) / 12
        Iz = (h * b**3 - hi * bi**3) / 12
        J = 2 * t**2 * (b - t) ** 2 * (h - t) ** 2 / (b * t + h * t - 2 * t**2)
        return Section(A, Iy, Iz, J, h / 2, b / 2, f"□{b:g}×{h:g}×{t:g}")

    @staticmethod
    def rect(b, h):
        A = b * h
        a, c = max(b, h), min(b, h)
        J = a * c**3 * (1 / 3 - 0.21 * c / a * (1 - c**4 / (12 * a**4)))
        return Section(A, b * h**3 / 12, h * b**3 / 12, J, h / 2, b / 2, f"{b:g}×{h:g}")


class Frame:
    def __init__(self):
        self.nodes = []      # (x,y,z) mm
        self.elems = []      # dict(n1,n2,sec,E,G,mat,name)
        self.fix = {}        # node -> 6 bools
        self.loads = {}      # node -> 6 vector (N, N·mm)

    def node(self, xyz, tol=0.5):
        xyz = np.asarray(xyz, float)
        for i, p in enumerate(self.nodes):
            if np.linalg.norm(p - xyz) < tol:
                return i
        self.nodes.append(xyz)
        return len(self.nodes) - 1

    def beam(self, a, b, sec, mat, name="", nseg=1):
        pa, pb = np.asarray(a, float), np.asarray(b, float)
        ids = [self.node(pa + (pb - pa) * k / nseg) for k in range(nseg + 1)]
        for i in range(nseg):
            self.elems.append(dict(n1=ids[i], n2=ids[i + 1], sec=sec, E=mat["E"] / 1e6,
                                   G=mat["E"] / 1e6 / (2 * (1 + mat["nu"])), mat=mat, name=name,
                                   rho=mat["rho"] * 1e-9))
        return ids

    def add_load(self, n, F=(0, 0, 0), M=(0, 0, 0)):
        v = self.loads.setdefault(n, np.zeros(6))
        v[:3] += F
        v[3:] += M

    def nearest(self, xyz):
        P = np.array(self.nodes)
        return int(np.argmin(np.linalg.norm(P - np.asarray(xyz), axis=1)))

    # --------------------------------------------------------------
    @staticmethod
    def _T(p1, p2):
        ex = (p2 - p1) / np.linalg.norm(p2 - p1)
        ref = np.array([0, 0, 1.0]) if abs(ex[2]) < 0.9 else np.array([1.0, 0, 0])
        ey = np.cross(ref, ex); ey /= np.linalg.norm(ey)
        ez = np.cross(ex, ey)
        R = np.vstack([ex, ey, ez])
        T = np.zeros((12, 12))
        for k in range(4):
            T[3 * k:3 * k + 3, 3 * k:3 * k + 3] = R
        return T

    @staticmethod
    def _k_local(E, G, A, Iy, Iz, J, L):
        k = np.zeros((12, 12))
        a = E * A / L
        k[0, 0] = k[6, 6] = a; k[0, 6] = k[6, 0] = -a
        t = G * J / L
        k[3, 3] = k[9, 9] = t; k[3, 9] = k[9, 3] = -t
        # 绕 z 弯曲 (位移 v=1, 转角 θz=5)
        c = E * Iz / L**3
        for (i, j, v) in [(1, 1, 12), (1, 5, 6 * L), (1, 7, -12), (1, 11, 6 * L), (5, 5, 4 * L * L), (5, 7, -6 * L),
                          (5, 11, 2 * L * L), (7, 7, 12), (7, 11, -6 * L), (11, 11, 4 * L * L)]:
            k[i, j] = k[j, i] = c * v
        c = E * Iy / L**3
        for (i, j, v) in [(2, 2, 12), (2, 4, -6 * L), (2, 8, -12), (2, 10, -6 * L), (4, 4, 4 * L * L), (4, 8, 6 * L),
                          (4, 10, 2 * L * L), (8, 8, 12), (8, 10, 6 * L), (10, 10, 4 * L * L)]:
            k[i, j] = k[j, i] = c * v
        return k

    def solve(self, self_weight=True, g=(0, 0, -9.81e3)):
        n = len(self.nodes)
        N = 6 * n
        P = np.array(self.nodes)
        rows, cols, vals = [], [], []
        Fv = np.zeros(N)
        self._Tl = []
        for e in self.elems:
            p1, p2 = P[e["n1"]], P[e["n2"]]
            L = np.linalg.norm(p2 - p1)
            s = e["sec"]
            kl = self._k_local(e["E"], e["G"], s.A, s.Iy, s.Iz, s.J, L)
            T = self._T(p1, p2)
            kg = T.T @ kl @ T
            dof = np.r_[6 * e["n1"]:6 * e["n1"] + 6, 6 * e["n2"]:6 * e["n2"] + 6]
            rows.append(np.repeat(dof, 12)); cols.append(np.tile(dof, 12)); vals.append(kg.ravel())
            e["L"] = L
            self._Tl.append((T, kl))
            if self_weight:
                w = e["rho"] * s.A * L * np.asarray(g) * 1e-3  # N (g in mm/s2 → *1e-3 m)
                Fv[6 * e["n1"]:6 * e["n1"] + 3] += w / 2
                Fv[6 * e["n2"]:6 * e["n2"] + 3] += w / 2
        K = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N))
        for nd, v in self.loads.items():
            Fv[6 * nd:6 * nd + 6] += v
        fixed = np.zeros(N, bool)
        for nd, f in self.fix.items():
            fixed[6 * nd:6 * nd + 6] = f
        free = ~fixed
        Kff = K[free][:, free].tocsc()
        # 机构/悬空检测：对角太小或奇异
        u = np.zeros(N)
        try:
            lu = spla.splu(Kff)
            u[free] = lu.solve(Fv[free])
            diag = np.abs(lu.U.diagonal())
            singular = diag.min() < 1e-9 * diag.max()
        except RuntimeError:
            singular = True
        self.u = u.reshape(-1, 6)
        self.singular = singular or not np.all(np.isfinite(u))
        R = K @ u - Fv
        self.reactions = {nd: R[6 * nd:6 * nd + 6] for nd in self.fix}
        self._stress()
        return self

    def _stress(self):
        U = self.u.ravel()
        for e, (T, kl) in zip(self.elems, self._Tl):
            dof = np.r_[6 * e["n1"]:6 * e["n1"] + 6, 6 * e["n2"]:6 * e["n2"] + 6]
            f = kl @ (T @ U[dof])
            s = e["sec"]
            Nf = max(abs(f[0]), abs(f[6]))
            My = max(abs(f[4]), abs(f[10]))
            Mz = max(abs(f[5]), abs(f[11]))
            Tq = max(abs(f[3]), abs(f[9]))
            sig = abs(f[6]) / s.A + My * s.cy / s.Iy + Mz * s.cz / s.Iz
            e["N"] = f[6]    # 正=拉, 负=压
            e["sigma"] = sig
            e["force"] = f

    def buckling_sf(self, e, K=1.0):
        """单根杆欧拉屈曲安全系数（受压时）。"""
        if e.get("N", 0) >= 0:
            return np.inf
        s = e["sec"]
        I = min(s.Iy, s.Iz)
        Pcr = np.pi**2 * e["E"] * I / (K * e["Lmember"]) ** 2
        return Pcr / abs(e["N"])
