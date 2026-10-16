"""装配体通用检查：把每个零件的三维网格拆成实体，算质量/重心，
找出哪些实体互相接触，判断有没有"悬空"（没有传力路径到地面）的东西。

这是"真实世界"的第一条规律：任何东西都要有东西托着，否则会掉下去。
"""
import numpy as np
import trimesh
import networkx as nx

G = 9.81


class Body:
    def __init__(self, part, idx, mesh, material, mass_override=None):
        self.part = part              # 零件名
        self.idx = idx                # 零件内第几个实体
        self.mesh = mesh
        self.material = material      # dict
        self.volume = abs(mesh.volume) * 1e-9  # m3 (网格单位 mm)
        self.mass = mass_override if mass_override is not None else self.volume * material["rho"]
        self.cog = mesh.center_mass if mesh.is_volume else mesh.bounds.mean(axis=0)
        self.bounds = mesh.bounds

    @property
    def name(self):
        return self.part if self.idx == 0 else f"{self.part}#{self.idx}"


def split_bodies(V, F):
    """网格拆成实体：正体积的外壳是实体，负体积的壳是空腔（方管内壁），并到包住它的实体里。"""
    m = trimesh.Trimesh(V, F, process=True)
    shells = m.split(only_watertight=False)
    outer, inner = [], []
    for s in shells:
        (outer if s.volume >= 0 else inner).append(s)
    if not outer:
        return [m]
    # 空腔并入：找 AABB 包含它的最小外壳
    merged = [[o] for o in outer]
    for c in inner:
        best, bvol = None, 1e30
        for i, o in enumerate(outer):
            if np.all(o.bounds[0] <= c.bounds[0] + 1e-3) and np.all(o.bounds[1] >= c.bounds[1] - 1e-3):
                v = np.prod(o.bounds[1] - o.bounds[0])
                if v < bvol:
                    best, bvol = i, v
        if best is not None:
            merged[best].append(c)
    return [trimesh.util.concatenate(g) if len(g) > 1 else g[0] for g in merged]


class Assembly:
    def __init__(self):
        self.bodies = []

    def add_part(self, name, V, F, material, mass_override=None, split=True):
        meshes = split_bodies(V, F) if split else [trimesh.Trimesh(V, F, process=True)]
        vols = np.array([abs(m.volume) for m in meshes])
        for i, m in enumerate(meshes):
            mo = None
            if mass_override is not None:
                mo = mass_override * vols[i] / vols.sum()
            self.bodies.append(Body(name, i, m, material, mo))

    # ------------------------------------------------------------ 接触图
    def contact_graph(self, tol=0.6, n_samples=1500):
        g = nx.Graph()
        B = self.bodies
        for i in range(len(B)):
            g.add_node(i)
        lo = np.array([b.bounds[0] for b in B]) - tol
        hi = np.array([b.bounds[1] for b in B]) + tol
        for i in range(len(B)):
            for j in range(i + 1, len(B)):
                if np.any(lo[i] > hi[j]) or np.any(lo[j] > hi[i]):
                    continue
                if self._touch(B[i], B[j], tol, n_samples):
                    g.add_edge(i, j)
        self.graph = g
        return g

    @staticmethod
    def _touch(a, b, tol, n):
        # 在 a 的表面取点（只取落在 b 包围盒附近的），看离 b 表面多近；反过来再查一次
        for p, q in ((a, b), (b, a)):
            pts = np.vstack([p.mesh.vertices, p.mesh.sample(n)])
            m = np.all((pts >= q.bounds[0] - tol) & (pts <= q.bounds[1] + tol), axis=1)
            pts = pts[m]
            if len(pts) == 0:
                continue
            _, d, _ = trimesh.proximity.closest_point(q.mesh, pts)
            if np.min(d) <= tol:
                return True
        return False

    def grounded(self, floor_z=0.0, tol=0.6):
        g = self.graph
        ground = [i for i, b in enumerate(self.bodies) if b.bounds[0][2] <= floor_z + tol]
        reach = set()
        for c in nx.connected_components(g):
            if any(i in c for i in ground):
                reach |= c
        floating = [i for i in range(len(self.bodies)) if i not in reach]
        groups = []
        sub = g.subgraph(floating)
        for c in nx.connected_components(sub):
            c = sorted(c)
            groups.append(dict(bodies=c, names=[self.bodies[i].name for i in c],
                               mass=sum(self.bodies[i].mass for i in c)))
        return ground, groups

    def total_mass(self, idx=None):
        idx = range(len(self.bodies)) if idx is None else idx
        m = np.array([self.bodies[i].mass for i in idx])
        c = np.array([self.bodies[i].cog for i in idx])
        return m.sum(), (m[:, None] * c).sum(axis=0) / m.sum()
