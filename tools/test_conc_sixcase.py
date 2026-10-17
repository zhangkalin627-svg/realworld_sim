"""六案导磁位对比回归（用户 2026-10-17 工况：木头/硅钢/SMC/铁氧体/铝/石墨）。
可反驳断言：
  - 六案齐全，关键指标都在；
  - 铝(整块导体)物理作用是屏蔽：同功率所需线圈电流必须高于无导磁体基线；
  - 铝/石墨自身损耗必须 > 0（求解器焦耳积分，非材料公式）；
  - 硅钢(高μ)所需电流必须低于基线（聚磁增效）；
  - 无导磁体基线：导磁位损耗=0、积灰指标=0。
运行：python3 tools/test_conc_sixcase.py（约 1-3 分钟）"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rw.em.check import compare_concentrators

t0 = time.time()
res = compare_concentrators(freq=30e3, power_kw=6.0)
dt = time.time() - t0
need = ("none", "sife", "smc", "ferrite", "alu", "graphite")
fails = []
for k in need:
    if k not in res:
        fails.append(f"缺案 {k}")
        continue
    v = res[k]
    for f in ("I_rms", "eff", "conc_loss_kW", "B_dust_T", "F_dust_uN", "V_turn"):
        if f not in v:
            fails.append(f"{k} 缺指标 {f}")
if fails:
    print("FAIL", fails); sys.exit(1)
b, al, gr, fe = res["none"], res["alu"], res["graphite"], res["sife"]
print(f"{'案例':<22}{'电流A':>8}{'效率%':>7}{'损耗W':>8}{'积灰B_T':>8}{'受力µN':>8}{'V/匝':>7}")
for k in need:
    v = res[k]
    print(f"{v['name'][:22]:<24}{v['I_rms']:>8.0f}{v['eff']*100:>7.1f}{v['conc_loss_kW']*1e3:>8.0f}{v['B_dust_T']:>8.2f}{v['F_dust_uN']:>8.1f}{v['V_turn']:>7.0f}")
ok = True
def chk(cond, msg):
    global ok
    print(("PASS " if cond else "FAIL ") + msg)
    ok = ok and cond
chk(al["I_rms"] > b["I_rms"], f"铝屏蔽→电流升高 ({al['I_rms']:.0f} > {b['I_rms']:.0f})")
chk(al["conc_loss_kW"] > 0, f"铝自身损耗>0 ({al['conc_loss_kW']*1e3:.0f} W)")
chk(gr["conc_loss_kW"] > 0, f"石墨自身损耗>0 ({gr['conc_loss_kW']*1e3:.0f} W)")
chk(fe["I_rms"] < b["I_rms"], f"硅钢聚磁→电流下降 ({fe['I_rms']:.0f} < {b['I_rms']:.0f})")
chk(b["conc_loss_kW"] == 0 and b["B_dust_T"] == 0, "基线无导磁体损耗/积灰指标为 0")
print(f"用时 {dt:.0f}s")
sys.exit(0 if ok else 1)
