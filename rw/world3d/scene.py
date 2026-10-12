"""场景 = 一堆参数化零件（箱、圆柱、管、球），每个有材料、位置(mm)、旋转(度, xyz 外旋)、
是否固定（固定=焊死在大地上，如地基、台虎钳）、组（同组=刚性连接成一个物体，如焊接件）。
几何来自 trimesh（MIT）。质量=体积×材料密度，所以"钢块比木块重"是真的算出来的。"""
import json
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R
from ..matlib import MAT
from . import profile as _prof

# 补充一些常见的非金属/建材（典型值）。fy 为屈服或抗压强度，Pa
EXTRA = {
    "wood":   dict(name="松木(顺纹)",  E=10e9,  nu=0.35, rho=500,  fy=40e6, fu=4e+07,  Tmax=120, magnetic=False, resist=None, mur=1),
    "concrete": dict(name="混凝土 C30", E=30e9, nu=0.2, rho=2400, fy=20e6, fu=3e+07,  Tmax=300, magnetic=False, resist=None, mur=1),
    "glass":  dict(name="玻璃",        E=70e9,  nu=0.22, rho=2500, fy=40e6, fu=4e+07,  Tmax=300, magnetic=False, resist=None, mur=1),
    "rubber": dict(name="橡胶",        E=0.01e9, nu=0.49, rho=1100, fy=15e6, fu=1.5e+07, Tmax=80, magnetic=False, resist=None, mur=1),
    "ice":    dict(name="冰",          E=9e9,   nu=0.33, rho=917,  fy=5e6, fu=2e+06,   Tmax=0,   magnetic=False, resist=None, mur=1),
}
MATS = {**MAT, **EXTRA}
# 滑动摩擦系数（材料对的粗略取值：干燥；这里按"较低者"粗估，标注为估算）
MU = {"Q235": 0.5, "45": 0.5, "6061": 0.45, "6063": 0.45, "SUS304": 0.45, "2Cr13": 0.5, "Cu": 0.4, "Brass": 0.4,
      "PP": 0.3, "POM": 0.2, "PU": 0.8, "PC": 0.35, "MIX": 0.5, "wood": 0.45, "concrete": 0.7, "glass": 0.4,
      "rubber": 1.0, "ice": 0.05}

KINDS = {
    "box":    dict(zh="方块/板/梁", dims=["长X", "宽Y", "高Z"]),
    "cyl":    dict(zh="圆柱",       dims=["直径", "高"]),
    "tube":   dict(zh="圆管",       dims=["外径", "内径", "长"]),
    "sphere": dict(zh="球",         dims=["直径"]),
}
# v0.12 型材（局部坐标: 长度沿 X，截面在 Y-Z 平面，H 沿 Z、B 沿 Y）
for k, v in _prof.PROFILE_KINDS.items():
    KINDS[k] = dict(zh=v["zh"], dims=v["dims"])


class Part:
    def __init__(self, name, kind, dims, mat="Q235", pos=(0, 0, 0), rot=(0, 0, 0), fixed=False, group="", color=None):
        self.name, self.kind, self.dims = name, kind, [float(d) for d in dims]
        self.mat, self.pos, self.rot = mat, [float(p) for p in pos], [float(r) for r in rot]
        self.fixed, self.group, self.color = fixed, group, color

    def to_dict(self):
        return dict(name=self.name, kind=self.kind, dims=self.dims, mat=self.mat, pos=self.pos, rot=self.rot,
                    fixed=self.fixed, group=self.group, color=self.color)

    @property
    def material(self):
        return MATS[self.mat]

    @property
    def R(self):
        return R.from_euler("xyz", self.rot, degrees=True)

    def volume_mm3(self):
        d = self.dims
        if self.kind == "box": return d[0] * d[1] * d[2]
        if self.kind == "cyl": return np.pi * (d[0] / 2) ** 2 * d[1]
        if self.kind == "tube": return np.pi * ((d[0] / 2) ** 2 - (d[1] / 2) ** 2) * d[2]
        if self.kind == "sphere": return 4 / 3 * np.pi * (d[0] / 2) ** 3
        if self.kind in _prof.PROFILE_KINDS: return _prof.area_mm2(self.kind, d) * d[-1]
        raise ValueError(self.kind)

    def primitives(self):
        """物理用分解体：[(kind, dims, offset_mm)]，offset 在零件局部坐标内。非型材 = 自身。"""
        if self.kind in _prof.PROFILE_KINDS:
            return [("box", list(d), list(o)) for d, o in _prof.primitives(self.kind, self.dims)]
        return [(self.kind, list(self.dims), [0.0, 0.0, 0.0])]

    @property
    def mass(self):
        return self.volume_mm3() * 1e-9 * self.material["rho"]

    def local_mesh(self, seg=32):
        d = self.dims
        if self.kind in _prof.PROFILE_KINDS:
            ms = []
            for pd, po in _prof.primitives(self.kind, d):
                m = trimesh.creation.box(extents=pd); m.apply_translation(po); ms.append(m)
            return trimesh.util.concatenate(ms)
        if self.kind == "box": return trimesh.creation.box(extents=d)
        if self.kind == "cyl": return trimesh.creation.cylinder(radius=d[0] / 2, height=d[1], sections=seg)
        if self.kind == "tube": return trimesh.creation.annulus(r_min=d[1] / 2, r_max=d[0] / 2, height=d[2], sections=seg)
        if self.kind == "sphere": return trimesh.creation.icosphere(subdivisions=2, radius=d[0] / 2)

    def mesh(self, seg=32):
        m = self.local_mesh(seg).copy()
        T = np.eye(4); T[:3, :3] = self.R.as_matrix(); T[:3, 3] = self.pos
        m.apply_transform(T)
        return m

    def bbox_half_height(self):
        m = self.mesh(); return (m.bounds[1][2] - m.bounds[0][2]) / 2


class Scene:
    def __init__(self, name="场景"):
        self.name = name
        self.parts = []
        self.fluid = None      # (key 或 密度, 液面高度 mm)
        self.floor_mat = "concrete"
        self.joints = []       # v0.12 螺栓连接校核: dict(base, part, side=[..], size, grade, coords, center, normal)
        self.hinges = []       # v0.12 轴承/铰链: dict(part, anchor=[x,y,z] mm, axis=[..] 局部轴, bearing="6204", damp=0, spin_rpm=0)

    def add_joint(self, **kw):
        self.joints.append(kw); return kw

    def add_hinge(self, **kw):
        self.hinges.append(kw); return kw

    def add(self, *a, **k):
        p = Part(*a, **k); self.parts.append(p); return p

    def get(self, name):
        for p in self.parts:
            if p.name == name: return p
        raise KeyError(name)

    def total_mass(self):
        m = np.array([p.mass for p in self.parts])
        c = np.array([p.pos for p in self.parts])
        return m.sum(), (m[:, None] * c).sum(axis=0) / m.sum()

    def save(self, path):
        json.dump(dict(name=self.name, fluid=self.fluid, floor_mat=self.floor_mat,
                       joints=self.joints, hinges=self.hinges,
                       parts=[p.to_dict() for p in self.parts]), open(path, "w"), ensure_ascii=False, indent=1)

    @staticmethod
    def load(path):
        d = json.load(open(path)); s = Scene(d["name"]); s.fluid = d.get("fluid"); s.floor_mat = d.get("floor_mat", "concrete")
        s.joints = d.get("joints", []); s.hinges = d.get("hinges", [])
        for p in d["parts"]: s.add(**p)
        return s

    def copy(self):
        s = Scene(self.name); s.fluid, s.floor_mat = self.fluid, self.floor_mat
        s.joints = [dict(j) for j in self.joints]; s.hinges = [dict(h) for h in self.hinges]
        for p in self.parts: s.add(**p.to_dict())
        return s
