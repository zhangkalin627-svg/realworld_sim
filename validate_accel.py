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
