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


PRESETS = {
    "钢板桥：两墩 + 铜块": beam_bridge,
    "叠积木 外伸系数 0.95（应稳）": lambda: harmonic_stack(0.95),
    "叠积木 外伸系数 1.10（应倒）": lambda: harmonic_stack(1.10),
    "橡胶块在 22° 斜面（会倒）": lambda: ramp(22.0, "rubber", 100, 300),
    "钢块在 30° 斜面（会滑）": lambda: ramp(30.0, "Q235", 100, 100),
    "钢块砸木板": drop_stack,
    "水里的浮沉": float_test,
    "焊接台架 + 偏载": table_cart,
}
