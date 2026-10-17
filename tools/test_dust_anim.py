"""粉尘动画模型回归（简化动力学，量级断言；2026-10-17 现场视频校准版）：
  现场工况=棒料水平连续过环：氧化皮从棒料底面剥落，重力⊥棒轴沿 +r，
  积灰面=线圈内壁+导磁体底侧（不再假设只粘腿尖）。
  - 硅钢案：粘附数随时间增长、层厚>0；
  - 木头基线（无导磁体）：线圈内壁同样积灰（现场：裸线圈也脏），stuck>0；
  - 两案粘附都应落在线圈窗口内的内壁带（水平过环的重力方向）；
  - 开气枪：空气中粒子数受控、已粘附的可被吹走；
  - 风险/统计字段齐全。
运行：python3 tools/test_dust_anim.py"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rw.em.dust import build_dust_sim

fails = []
ds_si, rec_si = build_dust_sim("sife")
ds_no, _ = build_dust_sim(None)

# 硅钢：跑 30 个 sim-秒（步长 0.5 s，共 60 步）
for _ in range(60):
    ds_si.step(0.5, blow=False)
st_si = ds_si.stats()
print("硅钢 30s:", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in st_si.items()})

# 木头基线：同样跑
for _ in range(60):
    ds_no.step(0.5, blow=False)
st_no = ds_no.stats()
print("木头 30s:", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in st_no.items()})

if not st_si["stuck"] > 0:
    fails.append("硅钢案 30s 无粘附")
if not st_no["stuck"] > 0:
    fails.append("木头基线（裸线圈内壁）应有粘附")
if not st_si["layer_um"] > 0:
    fails.append("硅钢案层厚应>0")
if st_si["risk"] not in ("低", "中", "高"):
    fails.append("风险分级缺失")

# 水平过环校准：粘附应落在线圈窗口内的内壁带（重力⊥棒轴，粉尘垂直下落积在内壁）
for name, ds in (("硅钢", ds_si), ("木头", ds_no)):
    if len(ds.stuck):
        g = ds.g
        z = ds.stuck[:, 1]
        frac_bore = np.mean((z >= g.coil_z0) & (z <= g.coil_z1))
        print(f"{name}案 粘附落在线圈窗口内壁带比例 {frac_bore:.2f}")
        if frac_bore < 0.8:
            fails.append(f"{name}案粘附不在线圈窗口内壁带 ({frac_bore:.2f})")

# 气枪：再跑 10 s，空气中和已粘附的都应减少
before_air, before_stuck = st_si["airborne"], st_si["stuck"]
for _ in range(20):
    ds_si.step(0.5, blow=True)
st_blow = ds_si.stats()
print("吹扫 10s 后:", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in st_blow.items()})
if not st_blow["airborne"] <= before_air + 30:    # 新生成的加上界宽限
    fails.append("吹扫后空气粒子未受控")
if not st_blow["stuck"] < before_stuck:
    fails.append(f"吹扫未释放任何粘附粒子 ({before_stuck}→{st_blow['stuck']})")

print("PASS" if not fails else f"FAIL {fails}")
sys.exit(0 if not fails else 1)
