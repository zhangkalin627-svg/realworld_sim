"""冶金扫描回归（物理趋势断言，参数取能奥氏体化的档）：
  - 同功率同时间：频率越高，奥氏体化深度越浅（趋肤效应）；
  - 同频率：加热越久，奥氏体化深度越深；
  - 喷淋淬硬层 > 空冷；
  - 扫描编排/判定/推荐字段齐全。
运行：python3 tools/test_scan.py（约 3-4 分钟）"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rw.scan import run_one, scan_freq_time

fails = []
wall = 3.0
r10 = run_one(wall, 10.0, 2.6)
r100 = run_one(wall, 100.0, 2.6)
keys = ("T_surf", "aus_depth_mm", "aus1_depth_mm", "case_depth_mm", "HRC_surf", "verdict", "micro")
print("10kHz:", {k: round(v, 2) if isinstance(v, float) else v for k, v in r10.items() if k in keys})
print("100kHz:", {k: round(v, 2) if isinstance(v, float) else v for k, v in r100.items() if k in keys})
# 物理：同功率同时间低频热得深（用 ≥Ac1 深度度量；10kHz 表面可能未达 Ac3，那是"欠热"判定的事）
if not r10["aus1_depth_mm"] > r100["aus1_depth_mm"]:
    fails.append(f"10kHz ≥Ac1 深度应比 100kHz 大: {r10['aus1_depth_mm']:.2f} vs {r100['aus1_depth_mm']:.2f}")

r_short = run_one(wall, 30.0, 0.7)
r_spr = run_one(wall, 30.0, 2.6)
if not r_spr["aus_depth_mm"] > r_short["aus_depth_mm"]:
    fails.append(f"加热久应更深: {r_short['aus_depth_mm']:.2f} vs {r_spr['aus_depth_mm']:.2f}")

r_air = run_one(wall, 30.0, 2.6, medium="air")
if not r_spr["case_depth_mm"] > r_air["case_depth_mm"]:
    fails.append(f"喷淋淬硬层应>空冷: {r_air['case_depth_mm']:.2f} vs {r_spr['case_depth_mm']:.2f}")

res = scan_freq_time(wall, freqs=(30.0,), times=(0.7,))
assert len(res["rows"]) == 1
for k in ("verdict", "micro", "HRC_surf", "case_depth_mm", "aus_depth_mm"):
    assert k in res["rows"][0], k
print("推荐语句:", res["reason"][:100])
print("PASS" if not fails else f"FAIL {fails}")
sys.exit(0 if not fails else 1)
