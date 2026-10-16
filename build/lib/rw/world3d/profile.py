"""标准型材（GB 热轧型钢/方管）：尺寸表、截面分解、截面特性。
约定：型材零件的局部坐标 = 长度沿局部 X 轴，截面在局部 Y-Z 平面内（高 H 沿局部 Z，宽 B 沿局部 Y）。
物理(MuJoCo)里每种型材分解成若干长方体 primitive（尖角近似，忽略圆角）→ 质量/惯量/接触都按分解体算。
截面特性（I/W）用同样分解解析计算，与国标理论重量对照（忽略圆角 → 偏差 ~1-3%，如实标注）。"""
import numpy as np

# ---------------------------------------------------------------- 国标标准尺寸表
# (label, dims...) -> 理论重量 kg/m（GB/T 11263 热轧H型钢、GB/T 706 槽钢、GB/T 9787 等边角钢、GB/T 6728 方管，常用规格）
H_BEAMS = {   # "H200x200x8x12": (H, B, tw, tf, kg/m)
    "H100x100x6x8":   (100, 100, 6, 8, 17.2),
    "H150x150x7x10":  (150, 150, 7, 10, 31.9),
    "H200x200x8x12":  (200, 200, 8, 12, 50.5),
    "H250x250x9x14":  (250, 250, 9, 14, 72.4),
    "H300x300x10x15": (300, 300, 10, 15, 94.5),
    "H150x75x5x7":    (150, 75, 5, 7, 14.3),
    "H200x100x5.5x8": (200, 100, 5.5, 8, 21.7),
    "H300x150x6.5x9": (300, 150, 6.5, 9, 37.3),
}
CHANNELS = {  # "C10": (H, B, tw, tf, kg/m)  热轧普通槽钢
    "C5":   (50, 37, 4.5, 7.0, 5.44),
    "C8":   (80, 43, 5.0, 8.0, 8.04),
    "C10":  (100, 48, 5.3, 8.5, 10.00),
    "C12.6": (126, 53, 5.5, 9.0, 12.37),
    "C14a": (140, 58, 6.0, 9.5, 14.53),
    "C16":  (160, 63, 6.5, 10.0, 17.23),
    "C20a": (200, 73, 7.0, 11.0, 22.63),
}
ANGLES = {    # "L50x5": (B, B, t, kg/m)  等边角钢
    "L30x3": (30, 30, 3, 1.373),
    "L40x4": (40, 40, 4, 2.422),
    "L50x5": (50, 50, 5, 3.770),
    "L63x6": (63, 63, 6, 5.721),
    "L75x8": (75, 75, 8, 9.030),
}
SQ_TUBES = {  # "SQ50x3": (B, B, t, kg/m)  方形冷弯空心型钢
    "SQ40x2.5": (40, 40, 2.5, 2.84),
    "SQ50x3":   (50, 50, 3, 4.49),
    "SQ60x4":   (60, 60, 4, 6.85),
    "SQ80x5":   (80, 80, 5, 11.5),
    "SQ100x6":  (100, 100, 6, 17.4),
}

# kind -> (表, dims 含义)
PROFILE_KINDS = {
    "ibeam":   dict(zh="工字钢/H型钢", dims=["高H", "宽B", "腹板厚tw", "翼缘厚tf", "长L"], table=H_BEAMS),
    "channel": dict(zh="槽钢",         dims=["高H", "宽B", "腹板厚tw", "翼缘厚tf", "长L"], table=CHANNELS),
    "angle":   dict(zh="等边角钢",     dims=["边宽B", "边宽B", "厚t", "长L"],              table=ANGLES),
    "sqtube":  dict(zh="方管",         dims=["边宽B", "边宽B", "壁厚t", "长L"],            table=SQ_TUBES),
}


def primitives(kind, dims):
    """分解成局部坐标下的长方体列表 [(dims3, offset3)]，offset 为长方体中心相对零件中心的位移。
    尖角分解、互不重叠 → 体积/惯量精确到尖角截面。"""
    if kind == "ibeam" or kind == "channel":
        H, B, tw, tf, L = dims
        out = [([L, B, tf], [0, 0, (H - tf) / 2]),      # 上翼缘
               ([L, B, tf], [0, 0, -(H - tf) / 2]),     # 下翼缘
               ([L, tw, max(H - 2 * tf, 1e-3)], [0, 0, 0])]  # 腹板
        if kind == "channel":
            out[2] = ([L, tw, max(H - 2 * tf, 1e-3)], [0, -(B - tw) / 2, 0])   # 腹板偏向一侧（槽钢开口朝 +Y）
        return out
    if kind == "angle":
        B, _, t, L = dims
        return [([L, t, B], [0, -(B - t) / 2, 0]),                # 立边（沿 -Y 侧）
                ([L, B - t, t], [0, t / 2, -(B - t) / 2])]        # 横边（沿 -Z 侧，扣除重叠角）
    if kind == "sqtube":
        B, _, t, L = dims
        return [([L, B, t], [0, 0, (B - t) / 2]),
                ([L, B, t], [0, 0, -(B - t) / 2]),
                ([L, t, B - 2 * t], [0, (B - t) / 2, 0]),
                ([L, t, B - 2 * t], [0, -(B - t) / 2, 0])]
    raise ValueError(kind)


def area_mm2(kind, dims):
    return float(sum(d[1] * d[2] for d, _ in primitives(kind, dims)))


def kg_per_m(kind, dims):
    """按 Q235 密度 7850 的理论重量（尖角近似）。"""
    return area_mm2(kind, dims) * 1e-6 * 7850


def section_moments(kind, dims):
    """截面特性。返回 dict: Iy, Iz (mm4) —— 绕"局部 Y 轴"与"局部 Z 轴"的形心惯性矩；
    Wy, Wz (mm3) 对应抗弯截面系数；yc, zc 形心（局部坐标）。
    局部截面: H 沿局部 Z，B 沿局部 Y。竖放工字钢受竖向力 → 绕局部 Y 轴弯（强轴 Iy）。
    用 primitive 分解 + 平行移轴公式解析计算（尖角，忽略圆角与剪切中心偏移）。"""
    prims = primitives(kind, dims)
    A = 0.0; cy = 0.0; cz = 0.0
    for d, o in prims:
        a = d[1] * d[2]; A += a; cy += a * o[1]; cz += a * o[2]
    cy /= A; cz /= A
    Iy = 0.0; Iz = 0.0   # Iy: 绕局部Y轴(对z分布积分)；Iz: 绕局部Z轴(对y分布积分)
    for d, o in prims:
        a = d[1] * d[2]
        Iy += d[1] * d[2] ** 3 / 12 + a * (o[2] - cz) ** 2
        Iz += d[2] * d[1] ** 3 / 12 + a * (o[1] - cy) ** 2
    zmax = max(abs(o[2] + s * d[2] / 2 - cz) for d, o in prims for s in (-1, 1))
    ymax = max(abs(o[1] + s * d[1] / 2 - cy) for d, o in prims for s in (-1, 1))
    return dict(Iy=Iy, Iz=Iz, Wy=Iy / zmax, Wz=Iz / ymax, yc=cy, zc=cz, A=A)
