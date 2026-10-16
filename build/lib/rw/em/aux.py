"""辅助金属体: 夹具、屏蔽套、托盘等被线圈杂散场感应加热的非工件金属 (轴对称矩形截面)。

电磁上按常数 μr 和 ρ 处理(线性)，不参与热网格；其发热功率由有限元体积积分得到，
温升用绝热集总估算(不含散热，是上限)。⚠ 未建模其温度对 ρ、μ 的影响。
"""
from dataclasses import dataclass
import numpy as np

MU0 = 4e-7 * np.pi

# 典型电阻率 (Ω·m, 20°C) 和常数 μr —— 工程手册典型值
AUX_PRESETS = {
    "al":     dict(rho=2.8e-8, mur=1.0, rho_cp=2.43e6, note="铝 6061 (典型值)"),
    "cu":     dict(rho=1.72e-8, mur=1.0, rho_cp=3.45e6, note="紫铜"),
    "ss304":  dict(rho=7.4e-7, mur=1.0, rho_cp=3.9e6, note="304 不锈钢(非磁性)"),
    "steel":  dict(rho=2.0e-7, mur=20.0, rho_cp=3.7e6, note="碳钢, 强场下有效 μr≈20 (线性化假设)"),
}


@dataclass
class AuxBody:
    r0: float          # mm
    r1: float
    z0: float
    z1: float
    material: str = "al"
    name: str = "aux"
    rho: float = None  # 覆盖预设 (Ω·m)
    mur: float = None
    layers: tuple = (0.25, 0.5, 1.0, 2.0, 3.5)   # 趋肤层加密线位置 (δ 的倍数)

    def __post_init__(self):
        p = AUX_PRESETS[self.material]
        if self.rho is None:
            self.rho = p["rho"]
        if self.mur is None:
            self.mur = p["mur"]
        if not (self.r1 > self.r0 >= 0 and self.z1 > self.z0):
            raise ValueError("辅助体需要 r1>r0>=0 且 z1>z0 (mm)")
        if self.rho <= 0 or self.mur < 1:
            raise ValueError("辅助体 ρ>0、μr≥1")

    @property
    def sigma(self):
        return 1.0 / self.rho

    @property
    def rho_cp(self):
        return AUX_PRESETS[self.material]["rho_cp"]

    def skin_mm(self, f):
        return 1e3 * np.sqrt(self.rho / (np.pi * f * MU0 * self.mur))

    def mask(self, r, z):
        return (r >= self.r0) & (r <= self.r1) & (z >= self.z0) & (z <= self.z1)

    def mesh_lines(self, f):
        """辅助体边界 + 趋肤层内加密线 (mm)。加密到 δ/4 级别，最多 6 层。"""
        d = self.skin_mm(f)
        off = np.array((0.0,) + tuple(self.layers)) * d
        lr = [self.r0, self.r1]
        for o in off:
            for edge, sgn in ((self.r0, 1), (self.r1, -1)):
                v = edge + sgn * o
                if self.r0 < v < self.r1:
                    lr.append(v)
                if o > 0:                       # 空气侧也加几层
                    v2 = edge - sgn * o
                    if v2 > 0:
                        lr.append(v2)
        lz = [self.z0, self.z1]
        return lr, lz

    def volume_m3(self):
        return np.pi * (self.r1**2 - self.r0**2) * (self.z1 - self.z0) * 1e-9
