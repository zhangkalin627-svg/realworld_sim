import json, time
import numpy as np
from rw.induction import InductionQuenchSim, BottleNeckGeometry

g = BottleNeckGeometry(neck_od=22, wall=3, neck_len=50, zone=20, coil_id=30, coil_h=25)
kw = dict(geom=g, freq=20e3, power_kw=6.0, t_heat=1.5, t_quench=1.0, t_air=0.5,
          medium="water_spray", mur_cold=20.0, nonlinear=False)

t0 = time.time()
snaps = []
sim1 = InductionQuenchSim(**kw)
r1 = sim1.run(checkpoint=lambda st: snaps.append(st), checkpoint_every=100)
t1 = time.time() - t0
print(f"全程运行 {t1:.1f}s，捕获 {len(snaps)} 个检查点")

# 取加热中期 + 淬火期各一个检查点，JSON 往返后恢复
for label, idx in (("加热中期", len(snaps)//3), ("淬火期", -2)):
    st = json.loads(json.dumps(snaps[idx]))   # JSON round-trip（NaN 由 python json 保留）
    sim2 = InductionQuenchSim(**kw)
    t0 = time.time()
    r2 = sim2.run(resume=st)
    dt = time.time() - t0
    dT = float(np.max(np.abs(sim1.T - sim2.T)))
    d_depth = abs(r1["case_depth_mm"] - r2["case_depth_mm"])
    d_hrc = abs(r1["HRC_surface"] - r2["HRC_surface"])
    print(f"[{label}] 从 t={st['t']:.2f}s 恢复，续跑 {dt:.1f}s：终态 max|ΔT|={dT:.4f}°C，硬化层差 {d_depth:.4f} mm，HRC 差 {d_hrc:.3f}")
    assert dT < 0.5, f"温度场偏差过大 {dT}"
    assert d_depth < 0.02, f"硬化层深度偏差过大 {d_depth}"
    assert d_hrc < 0.2, f"HRC 偏差过大 {d_hrc}"
print("断点续算等价性 OK：恢复运行与连续运行终态一致（<0.5°C / <0.02mm / <0.2HRC）")
