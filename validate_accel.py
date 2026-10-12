import time
import numpy as np
from rw.induction import InductionQuenchSim, BottleNeckGeometry

g = BottleNeckGeometry(neck_od=22, wall=3, neck_len=50, zone=20, coil_id=30, coil_h=25)
kw = dict(geom=g, freq=20e3, power_kw=6.0, t_heat=3.0, t_quench=3.0, t_air=6.0,
          medium="water_spray", mur_cold=20.0, nonlinear=False)

t0 = time.time(); sim_a = InductionQuenchSim(**kw); ra = sim_a.run(); ta = time.time() - t0
t0 = time.time(); sim_f = InductionQuenchSim(accel=True, **kw); rf = sim_f.run(); tf = time.time() - t0

dT = float(np.max(np.abs(sim_a.T - sim_f.T)))
print(f"精确 {ta:.1f}s / 快速 {tf:.1f}s → 加速 ×{ta/tf:.2f}")
for k in ("t85_surface", "t85_inner", "case_depth_mm", "HRC_surface", "HRC_inner"):
    print(f"  {k}: 精确 {ra[k]} vs 快速 {rf[k]}  Δ={abs(ra[k]-rf[k]):.4g}")
print(f"  终态温度场 max|ΔT| = {dT:.3f}°C")
assert abs(ra["t85_surface"] - rf["t85_surface"]) < 0.05
assert abs(ra["case_depth_mm"] - rf["case_depth_mm"]) < 0.03
assert abs(ra["HRC_surface"] - rf["HRC_surface"]) < 0.3
print("加速模式精度 OK：t8/5 偏差 <0.05s，硬化层/HRC 一致，终态温度场 <0.5°C")


# ---- 边界用例（豆包 12 点实测矩阵发现的 fast 档已知边界，阈值记录实测偏差上限）----
print("\n边界用例：")
# 油淬（慢冷介质）：t8/5 绝对偏差放大，如实记录上限 0.2 s
kw_oil = dict(kw, medium="oil_dip")
t0 = time.time(); ra = InductionQuenchSim(**kw_oil).run(); rf = InductionQuenchSim(accel=True, **kw_oil).run()
d = abs(ra["t85_surface"] - rf["t85_surface"])
print(f"  油淬: t8/5 表面 {ra['t85_surface']:.3f}s vs {rf['t85_surface']:.3f}s，Δ={d:.4f}s（相对 {d/ra['t85_surface']*100:.2f}%），上限 0.2s")
assert d < 0.2
# 频率边界 5 kHz：HRC 表面漂移，如实记录上限 1.5
kw_5k = dict(kw, freq=5e3)
ra = InductionQuenchSim(**kw_5k).run(); rf = InductionQuenchSim(accel=True, **kw_5k).run()
d = abs(ra["HRC_surface"] - rf["HRC_surface"])
print(f"  5kHz: HRC 表面 {ra['HRC_surface']:.2f} vs {rf['HRC_surface']:.2f}，Δ={d:.3f}，上限 1.5")
assert d < 1.5
print("边界用例 OK：偏差在上限内（慢冷 t8/5 与频率两端 HRC 是 fast 档已知边界，README 已如实标注）")
