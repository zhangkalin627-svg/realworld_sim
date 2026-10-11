"""气缸动力学（简化但物理的）：
  m·a = p·A_eff − F_load − F_fric − F_back(v)
  排气节流（调速阀 meter-out）限速：背压随速度上升，v_max ∝ √(p)。
  另加电磁阀响应 + 腔室充气延迟。
得到：给定气压/负载下，每个气缸伸出、缩回各要多久；气压太低推不动就"卡住"。
单位：mm, s, N, MPa, kg。"""
import math
from dataclasses import dataclass

G = 9.81


@dataclass
class Cylinder:
    name: str
    bore: float          # 缸径 mm
    rod: float           # 杆径 mm
    stroke: float        # 行程 mm
    mass: float          # 运动质量 kg
    angle: str = "h"     # "up"=伸出向上举, "down"=伸出向下(缩回时举起), "h"=水平
    n_rods: int = 1      # 导杆气缸按单活塞计
    v_set: float = 300.0  # 0.5 MPa 时调速阀调好的最高速度 mm/s
    end_force: float = 0.0  # 伸出端需要额外输出的力（顶升压紧）N
    valve_ms: float = 20.0
    friction: float = 0.06  # 摩擦按理论推力的比例

    def area(self, extend=True):
        A = math.pi / 4 * self.bore**2
        return A if extend else A - math.pi / 4 * self.rod**2

    def load(self, extend):
        W = self.mass * G
        if self.angle == "up":
            return W if extend else -W
        if self.angle == "down":
            return -W if extend else W
        return 0.0

    def travel_time(self, p_MPa, extend=True, dt=1e-4):
        """返回 (时间 s, 能否到位, 末端可输出力 N)。"""
        A = self.area(extend)
        F_th = p_MPa * A
        F_load = self.load(extend)
        F_fr = self.friction * F_th
        F_net0 = F_th - F_load - F_fr
        if F_net0 <= 0:
            return math.inf, False, F_net0
        vmax = self.v_set * math.sqrt(p_MPa / 0.5)
        # 节流背压模型：F_back = F_net0 * (v/vmax)^2  → 终速 vmax
        # 充气延迟：腔室从 0 升到能克服负载所需压力的时间，按 10 ms×(F_load+F_fr)/F_th 估
        t = self.valve_ms / 1e3 + 0.03 * max(F_load + F_fr, 0) / F_th
        x = v = 0.0
        m = max(self.mass, 0.2)
        while x < self.stroke:
            F = F_net0 * (1 - (v / vmax) ** 2)
            a = F / m * 1000.0   # mm/s²
            v = max(v + a * dt, 0.0)
            x += v * dt
            t += dt
            if t > 30:
                return math.inf, False, F_net0
        F_end = F_net0 - (self.end_force if extend else 0.0)
        return t, F_end > 0, F_end


# 示例气缸组（常见亚德客规格）
def example_cylinders(m_coil=31.8):
    return {
        "coil":   Cylinder("感应圈升降 TCL50×80", 50, 20, 80, m_coil, "down", v_set=250),
        "spin":   Cylinder("自转头顶升 TCL32×50(减压0.2)", 32, 16, 50, 3.5, "up", v_set=220, end_force=0.0),
        "A_lift": Cylinder("取放A升降 TCL20×150", 20, 10, 150, 1.6, "down", v_set=450),
        "A_trav": Cylinder("取放A平移 RMT25×250", 25, 0, 250, 3.0, "h", v_set=600),
        "A_grip": Cylinder("取放A夹爪 HFZ25", 25, 10, 6, 0.3, "h", v_set=30),
        "B_lift": Cylinder("取放B升降 TCL20×150", 20, 10, 150, 1.6, "down", v_set=450),
        "B_trav": Cylinder("取放B平移 RMT25×150", 25, 0, 150, 3.0, "h", v_set=380),
        "B_grip": Cylinder("取放B夹爪 HFZ25", 25, 10, 6, 0.3, "h", v_set=30),
        "C_lift": Cylinder("取放C升降 TCL20×150", 20, 10, 150, 1.6, "down", v_set=450),
        "C_trav": Cylinder("取放C平移 RMT25×250", 25, 0, 250, 3.0, "h", v_set=600),
        "C_grip": Cylinder("取放C夹爪 HFZ25", 25, 10, 6, 0.3, "h", v_set=30),
    }


SPIN_REG = 0.2   # 自转头支路减压阀 MPa


def travel_table(p_main, cyls=None):
    cyls = cyls or example_cylinders()
    out = {}
    for k, c in cyls.items():
        p = min(p_main, SPIN_REG) if k == "spin" else p_main
        te, oke, Fe = c.travel_time(p, True)
        tr, okr, Fr = c.travel_time(p, False)
        out[k] = dict(name=c.name, p=p, t_ext=te, t_ret=tr, ok_ext=oke, ok_ret=okr, F_ext=Fe, F_ret=Fr)
    return out
