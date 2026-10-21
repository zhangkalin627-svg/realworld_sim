#!/usr/bin/env python3
"""coil_inductance 自检（仿真/解析结果，无实测）。FEM 对照数字见 docs/线圈电感_coil_inductance_2026-10-21.md，
复跑脚本在 tools/coil_fem/（需 CloudFEM）。"""
import math
import rw.coil_inductance as ci

def main():
    # 1 Nagaoka 系数对公认表值
    for x, K in [(0.5, 0.8181), (1, 0.6884), (2, 0.5255), (5, 0.3198), (10, 0.2033)]:
        assert abs(ci.nagaoka_K(2.0 / x, 1.0) - K) < 5e-4, (x, K)
    # 2 Wheeler 与 Nagaoka 在 l>=r 时 0.5% 内；更短时偏低（仅作对照）
    for lr in [1, 2, 4, 8, 20]:
        r = 0.04
        assert abs(ci.solenoid_wheeler(10, r, lr * r) / ci.solenoid_nagaoka(10, r, lr * r) - 1) < 0.005
    # 3 薄带≈Nagaoka，厚带(4 mm)低 6%（FEM 磁能法 7.79 uH 同向）
    thin = ci.rect_section_ring(10, 0.04, 0.0004, 0.04, 2, 80) * 1e6
    thick = ci.rect_section_ring(10, 0.04, 0.004, 0.04, 8, 40) * 1e6
    nag = ci.solenoid_nagaoka(10, 0.04, 0.04) * 1e6
    assert abs(thin / nag - 1) < 0.015 and abs(thick - 7.81) < 0.05
    # 4 单匝 5x10 mm 截面、平均直径 100 mm：空心 175.1 nH（FEM 174.0 nH）
    v = ci.rect_section_ring(1, 0.05, 0.005, 0.010) * 1e9
    assert abs(v - 175.1) < 0.5
    # 5 单匝方截面线 s=6 mm, r=30 mm：GMD 闭式 94.0 nH（FEM 外推 93.6-93.7 nH）；片状公式在单匝上偏高
    g = ci.MU0 * 0.03 * (math.log(8 * 0.03 / (0.44705 * 0.006)) - 2) * 1e9
    assert abs(g - 94.0) < 0.2
    assert ci.solenoid_nagaoka(1, 0.03, 0.006) * 1e9 > 1.15 * g
    # 6 谐振换算
    L = ci.L_from_resonance(18.4e3, 1e-6)
    assert abs(ci.f_from_LC(L, 1e-6) - 18.4e3) < 1e-6
    print(f"空心单匝(5x10, 平均直径100 mm) = {v:.1f} nH；薄带/厚带/Nagaoka = {thin:.3f}/{thick:.3f}/{nag:.3f} uH")
    print("validate_coil_inductance: PASS（均为仿真/解析结果，非实测）")

if __name__ == "__main__":
    main()
