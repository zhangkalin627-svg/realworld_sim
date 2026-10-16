# -*- coding: utf-8 -*-
"""PLL 频率跟踪验证（v0.31）：启动捕获 / 居里跟踪 / 负载阶跃 / 失锁跳闸。
直驱 Stage+Controller（1 kHz 控制周期，与整机 tick 一致），T_wp/k 由剧本给定。"""
import math, json
from rw.machine.power import Spec, Stage, Controller

def drive(track, T_prog, k_prog, t_end=4.0, sp_kw=None, fault_at=None, fault_C=None):
    """T_prog/k_prog: t→值 函数；fault_at: 该时刻 C_res×fault_C（失锁案）"""
    sp = Spec(track_mode=track, **(sp_kw or {})); st = Stage(sp); ctl = Controller(sp)
    Vbus = sp.V_src()
    dt = 1e-3; t = 0.0; rows = []
    ctl.tick(dt, False, Vbus, None, None)  # boot 起步
    while t < t_end:
        T = T_prog(t); k = k_prog(t)
        if fault_at and t >= fault_at: sp.C_res = fault_C
        stage_fn = lambda f: st.solve(Vbus, f, T, k, 40.0, ctl.m)
        f0_fn = lambda: st.f0(T, k)
        s = ctl.tick(dt, True, Vbus, stage_fn, f0_fn)
        if s: rows.append((t, ctl.f_q, getattr(ctl, "phi", None), s["P_out"], s["f0"], ctl.pll, ctl.fault))
        t += dt
    return sp, ctl, rows

def summ(rows):
    import statistics
    last = rows[-1]
    lock_t = next((r[0] for r in rows if r[5] == "locked"), None)
    phis = [r[2] for r in rows[len(rows)//2:] if r[2] is not None]
    phis_all = [r[2] for r in rows if r[2] is not None]
    return dict(lock_t=lock_t, f_end=last[1], f0_end=last[4], ratio=last[1]/last[4],
                phi_end=last[2], phi_min=(min(phis_all) if phis_all else None),
                P_end=last[3], fault=last[6])

out = {}
# A 启动捕获：冷态工件，pll vs ideal
for mode in ("pll", "ideal"):
    sp, ctl, rows = drive(mode, lambda t: 20.0, lambda t: 0.949173, t_end=2.0)
    out["A_" + mode] = summ(rows)
# B 居里跟踪：T 20→900°C 三秒（f0 上移约 1.8%）
sp, ctl, rows = drive("pll", lambda t: 20 + 880 * t / 3.0, lambda t: 0.949173, t_end=3.0)
out["B_curie"] = summ(rows)
# C 负载耦合阶跃 k 0.949→0.70 @1.5s
sp, ctl, rows = drive("pll", lambda t: 850.0, lambda t: 0.949173 if t < 1.5 else 0.70, t_end=3.0)
relock = None
for i, r in enumerate(rows):
    if r[0] > 1.5 and r[2] is not None and abs(r[2] - 0.22) < 0.03:
        relock = r[0] - 1.5; break
out["C_step"] = {**summ(rows), "relock_s": relock}
# D 失锁：C_res ×0.5 @1.5s（电容失效，f0 跃升越过当前开关频率 → 负载变容性）
sp, ctl, rows = drive("pll", lambda t: 850.0, lambda t: 0.949173, t_end=3.0, fault_at=1.5, fault_C=0.667e-6 * 0.5)  # 电容失效减半 → f0 跃过当前 f → 容性失锁
trip = next((r[0] for r in rows if r[6]), None)
out["D_loss"] = {"trip_t": trip, "delay_after_fault": (trip - 1.5) if trip else None, "fault": rows[-1][6]}

for k, v in out.items(): print(k, json.dumps(v, ensure_ascii=False))
# 断言
a = out["A_pll"]
assert a["lock_t"] and a["lock_t"] < 1.0, "锁定太慢"
assert abs(a["phi_end"] - 0.22) < 0.03, "锁定相位偏"
assert a["phi_min"] > -0.05, "捕获过程深入容性"
assert a["P_end"] >= 0.9 * out["A_ideal"]["P_end"], "PLL 模式功率明显低于 ideal 对照"
assert out["B_curie"]["phi_min"] > 0.0, "居里跟踪过程跌破容性"
assert abs(out["B_curie"]["phi_end"] - 0.22) < 0.05, "居里后相位偏出带"
assert out["C_step"]["relock_s"] and out["C_step"]["relock_s"] < 0.5, "阶跃重锁太慢"
assert out["D_loss"]["fault"] == "LOSS" and out["D_loss"]["delay_after_fault"] < 0.2, "失锁未跳闸或太慢"
print("ALL PASS")
