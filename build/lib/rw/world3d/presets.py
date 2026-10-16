"""示例场景（全部通用，可公开）。"""
import numpy as np
from .scene import Scene


def beam_bridge(load_kg=None, plank_mat="Q235"):
    """钢板两端搁在混凝土墩上，跨中放一个铜圆柱。"""
    s = Scene("钢板桥：两墩 + 重物")
    s.add("钢板", "box", [1200, 200, 30], plank_mat, pos=[0, 0, 315])
    s.add("墩L", "box", [100, 200, 300], "concrete", pos=[-500, 0, 150], fixed=True)
    s.add("墩R", "box", [100, 200, 300], "concrete", pos=[500, 0, 150], fixed=True)
    s.add("重物", "box", [200, 150, 160], "Cu" if load_kg is None else "Q235", pos=[0, 0, 410])
    return s


def harmonic_stack(factor=0.95, n=4, L=200.0):
    """叠积木悬臂：第 k 块（从上数）相对下一块外伸 L/(2k)×factor。factor<1 稳，>1 倒（理论临界 1.0）。"""
    s = Scene(f"叠积木 外伸系数 {factor:g}")
    h, w = 50.0, 100.0
    s.add("底座", "box", [400, 200, 100], "concrete", pos=[0, 0, 50], fixed=True)
    # 从下往上放 n 块，最下面一块的"右端"外伸 L/(2n)（相对底座边缘），依次累积
    # 令最下块右端对齐底座右边缘 x=200 → 其中心 xc = 200 - L/2 + 外伸 ... 这里直接用相对位置计算
    off = [L / (2 * k) * factor for k in range(1, n + 1)]    # off[k-1]: 上数 k 块整体的重心要落在下面一块边缘 → 外伸 L/(2k)
    xc = 200.0 - L / 2 + off[n - 1]      # 最下块相对底座右边缘外伸 L/(2n)
    xs = [xc]
    for i in range(1, n):
        xc = xc + off[n - i - 1]         # 第 i 块(从下数)相对下一块外伸 L/(2(n-i))
        xs.append(xc)
    for i in range(n):
        s.add(f"积木{i+1}", "box", [L, w, h], "wood" if False else "Q235", pos=[xs[i], 0, 100 + h / 2 + i * h])
    return s


def ramp(angle=22.0, mat="rubber", w=100.0, h=300.0):
    s = Scene(f"斜面 {angle:g}° 上的 {w:g}×{h:g} 块 ({mat})")
    th = np.radians(angle)
    s.add("斜面", "box", [2000, 600, 50], "concrete", pos=[0, 0, 300], rot=[0, angle, 0], fixed=True)
    n = np.array([np.sin(th), 0, np.cos(th)])
    c = np.array([0, 0, 300.0]) + n * (25 + h / 2)
    s.add("物块", "box", [w, w, h], mat, pos=list(c), rot=[0, angle, 0])
    return s


def drop_stack():
    s = Scene("落下的东西：钢块砸木板")
    s.add("木板", "box", [600, 300, 40], "wood", pos=[0, 0, 20 + 100], rot=[0, 0, 0])
    s.add("枕木L", "box", [60, 300, 100], "concrete", pos=[-250, 0, 50], fixed=True)
    s.add("枕木R", "box", [60, 300, 100], "concrete", pos=[250, 0, 50], fixed=True)
    s.add("钢块", "box", [200, 200, 200], "Q235", pos=[0, 0, 140 + 100 + 400])
    return s


def float_test():
    s = Scene("水里的浮沉（阿基米德，静态判断）")
    s.fluid = ("water", 400.0)
    for i, (n, mat) in enumerate([("松木", "wood"), ("冰", "ice"), ("铝", "6061"), ("钢", "Q235"), ("铜", "Cu"), ("PP", "PP")]):
        s.add(n, "box", [100, 100, 100], mat, pos=[-600 + 240 * i, 0, 50])
    return s


def table_cart():
    s = Scene("焊接台架 + 偏载")
    # 同组 = 焊成一体
    for sx in (-1, 1):
        for sy in (-1, 1):
            s.add(f"腿{sx}{sy}", "box", [50, 50, 700], "Q235", pos=[sx * 275, sy * 275, 350], group="台架")
    s.add("台面", "box", [700, 700, 25], "Q235", pos=[0, 0, 712.5], group="台架")
    s.add("偏载", "box", [200, 200, 300], "Q235", pos=[250, 0, 875], )
    return s


def ibeam_bridge():
    """H200×200 工字钢搁两墩，跨中压一个钢块。"""
    s = Scene("工字钢梁桥")
    s.add("工字钢", "ibeam", [200, 200, 8, 12, 2400], "Q235", pos=[0, 0, 400])
    s.add("墩L", "box", [150, 300, 300], "concrete", pos=[-1050, 0, 150], fixed=True)
    s.add("墩R", "box", [150, 300, 300], "concrete", pos=[1050, 0, 150], fixed=True)
    s.add("钢块", "box", [300, 200, 200], "Q235", pos=[0, 0, 600])
    return s


def wall_bracket():
    """壁挂支架：背板+悬臂（固定，相当于螺栓把支架锁在墙上），臂端压一个钢块；
    4×M12 螺栓组校核拉力/剪力。"""
    s = Scene("壁挂支架 + 4×M12 螺栓组")
    s.add("背板", "box", [20, 100, 200], "Q235", pos=[10, 0, 400], fixed=True, group="支架")
    s.add("悬臂", "box", [300, 80, 20], "Q235", pos=[170, 0, 490], fixed=True, group="支架")
    s.add("钢块", "box", [100, 100, 100], "Q235", pos=[270, 0, 550])
    s.add_joint(base="墙", part="悬臂", side=["背板", "悬臂"], size=12, grade="8.8",
                center=[0, 0, 400], normal=[1, 0, 0], e1=[0, 1, 0],
                coords=[(-35, -70), (35, -70), (-35, 70), (35, 70)])
    return s


def weld_bracket():
    """悬臂焊在墙板上：两道角焊缝（hf=6），臂端压钢块；焊缝群应力校核。"""
    s = Scene("焊缝悬臂支架")
    s.add("悬臂", "box", [300, 80, 20], "Q235", pos=[150, 0, 400], fixed=True, group="支架")
    s.add("钢块", "box", [100, 100, 100], "Q235", pos=[250, 0, 460])
    s.add_joint(base="墙板", part="悬臂", side=["悬臂"], weld=dict(hf=6, legs=[((-40, -10), (40, -10)), ((-40, 10), (40, 10))]),
                center=[0, 0, 400], normal=[1, 0, 0], e1=[0, 1, 0])
    return s


def bearing_rotor(rpm=3000):
    """轴 + 偏心块，6206 轴承铰接，按转速旋转：测轴承反力、不平衡力、L10 寿命。"""
    s = Scene(f"轴承转子 {rpm:g} rpm")
    s.add("轴", "cyl", [30, 500], "45", pos=[250, 0, 600], rot=[0, 90, 0], group="转子")
    s.add("偏心块", "box", [40, 40, 20], "45", pos=[250, 150, 600], group="转子")
    s.add_hinge(part="轴", anchor=[250, 0, 600], axis=[1, 0, 0], bearing="6206", damp=(0.05 if not rpm else 1e-4), spin_rpm=rpm)
    return s


def glass_drop(load_kg=20.0):
    """玻璃板搁两支座，跨中放钢块：应力超过玻璃抗拉强度 → 判断裂。"""
    m = load_kg
    a = (m / 7850e-9) ** (1 / 3)          # 钢块边长 mm
    s = Scene(f"玻璃板 + {m:g} kg 钢块（断裂判据）")
    s.add("玻璃板", "box", [400, 100, 6], "glass", pos=[0, 0, 203])
    s.add("座L", "box", [40, 100, 200], "concrete", pos=[-180, 0, 100], fixed=True)
    s.add("座R", "box", [40, 100, 200], "concrete", pos=[180, 0, 100], fixed=True)
    s.add("钢块", "box", [a, a, a], "Q235", pos=[0, 0, 206 + a / 2])
    return s


PRESETS = {
    "钢板桥：两墩 + 铜块": beam_bridge,
    "叠积木 外伸系数 0.95（应稳）": lambda: harmonic_stack(0.95),
    "叠积木 外伸系数 1.10（应倒）": lambda: harmonic_stack(1.10),
    "橡胶块在 22° 斜面（会倒）": lambda: ramp(22.0, "rubber", 100, 300),
    "钢块在 30° 斜面（会滑）": lambda: ramp(30.0, "Q235", 100, 100),
    "钢块砸木板": drop_stack,
    "水里的浮沉": float_test,
    "焊接台架 + 偏载": table_cart,
    "工字钢梁桥": ibeam_bridge,
    "壁挂支架 + 螺栓组": wall_bracket,
    "轴承转子 3000rpm": bearing_rotor,
    "玻璃板断裂演示": glass_drop,
    "焊缝悬臂支架": weld_bracket,
}
