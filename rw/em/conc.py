"""导磁体(磁集中器)几何: 线圈背面 + 上下两个"腿"，形成开口朝向工件的 U 形截面 (轴对称)。"""
from dataclasses import dataclass
import numpy as np
from .bh import FLUX_MATERIALS

# 非磁导电"导磁位"材料 (用户 2026-10-17 六案对比): 物理上不是导磁体，
# 整块导电→涡流反而排斥/屏蔽磁场。μr=1，电磁靠求解器 σ 项直接解，不走 B-H 曲线。
NONMAG_CONC = {
    "alu":     dict(name="铝 (μr=1 导电体)", rho=2.8e-8,  rho_cp=2.43e6),
    "graphite": dict(name="石墨 (μr=1 导电体)", rho=1.2e-5, rho_cp=1.9e6),
}


@dataclass
class Concentrator:
    material: str = "smc"      # FLUX_MATERIALS 或 NONMAG_CONC 的键
    back_t: float = 4.0        # 线圈外侧(背面)厚度 mm
    leg_t: float = 3.0         # 线圈上下端面"腿"的厚度 mm (0 = 只有背面)

    @property
    def is_magnetic(self):
        return self.material in FLUX_MATERIALS

    @property
    def mat(self):
        if not self.is_magnetic:
            raise AttributeError(f"{self.material} 是非磁导电材料，没有 B-H 曲线（用 sigma/is_magnetic 分支）")
        return FLUX_MATERIALS[self.material]

    @property
    def name(self):
        return self.mat.name if self.is_magnetic else NONMAG_CONC[self.material]["name"]

    @property
    def sigma(self):
        """体电导率 (S/m)。磁性材料=0（叠片/绝缘颗粒压整体涡流，损耗走材料公式）；
        非磁导体=1/ρ（涡流由求解器直接解，物理上正是它的屏蔽机制）。"""
        if self.is_magnetic:
            return 0.0
        return 1.0 / NONMAG_CONC[self.material]["rho"]

    @property
    def rho_cp(self):
        return None if self.is_magnetic else NONMAG_CONC[self.material]["rho_cp"]

    def r_edges(self, g):
        return g.coil_r1 + self.back_t

    def mask(self, g, r, z):
        r = np.asarray(r); z = np.asarray(z)
        back = (r >= g.coil_r1) & (r <= g.coil_r1 + self.back_t) & \
               (z >= g.coil_z0 - self.leg_t) & (z <= g.coil_z1 + self.leg_t)
        if self.leg_t > 0:
            legs = (r >= g.coil_r0) & (r <= g.coil_r1) & \
                   (((z >= g.coil_z1) & (z <= g.coil_z1 + self.leg_t)) |
                    ((z >= g.coil_z0 - self.leg_t) & (z <= g.coil_z0)))
            return back | legs
        return back
