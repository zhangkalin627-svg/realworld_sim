"""导磁体(磁集中器)几何: 线圈背面 + 上下两个"腿"，形成开口朝向工件的 U 形截面 (轴对称)。"""
from dataclasses import dataclass
import numpy as np
from .bh import FLUX_MATERIALS


@dataclass
class Concentrator:
    material: str = "smc"      # FLUX_MATERIALS 的键
    back_t: float = 4.0        # 线圈外侧(背面)厚度 mm
    leg_t: float = 3.0         # 线圈上下端面"腿"的厚度 mm (0 = 只有背面)

    @property
    def mat(self):
        return FLUX_MATERIALS[self.material]

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
