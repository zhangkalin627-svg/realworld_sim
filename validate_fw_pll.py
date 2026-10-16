# -*- coding: utf-8 -*-
"""真实固件在环验证（v0.32）：指令级 PLL 固件 vs validate_pll 行为级基线的四个场景。
A 启动捕获 / B 居里跟踪 / C 负载耦合阶跃 / D 电容失效失锁跳闸。
频率/相位真值 = Stage 在实测门极平均频率下的基波解；锁定/跳闸看固件引脚（PD7/PB0）。"""
import math, json, time, statistics
from rw.machine.power import Spec, Stage
from rw.machine.fwloop import FWLoop

PHI_REF = 9 / 256 * 2 * math.pi     # 固件目标相位 0.2209 rad（≈行为级 0.22）

def drive_fw(T_prog, k_prog, t_end, fault_at=None, fault_C=None):
    lp = FWLoop()
    def mk(tt):
        def f():
            lp.T_wp = T_prog(tt); lp.k = k_prog(tt)
            if fault_at and tt >= fault_at:
                lp.sp.C_res = fault_C
        return f
    t = 0.0
    while t <= t_end:
        lp.at(t, mk(t)); t += 1e-3
    t0 = time.time()
    lp.run(t_end)
    el = time.time() - t0
    lock_t = lp.first_edge("PD7", 1)
    fault_t = lp.first_edge("PB0", 1)
    hist = lp.hist
    phis = [h[3] for h in hist]
    tail = [h[3] for h in hist if h[0] > t_end - 0.1]
    gate_last = lp.gate_rise
    ptail = [h[5] for h in hist if h[0] > t_end - 0.1]
    return dict(lock_t=lock_t, fault_t=fault_t,
                f_end=hist[-1][2] if hist else None,
                phi_end=statistics.mean(tail) if tail else None,
                phi_min=min(phis) if phis else None,
                P_end=statistics.mean(ptail) if ptail else None,
                n_cycles=len(hist), sim_s=round(el, 1), gate_last=gate_last, lp=lp)

def ideal_power(T, k, C=None):
    sp = Spec(); st = Stage(sp)
    if C: sp.C_res = C
    f0 = st.f0(T, k)
    return st.solve(sp.V_src(), f0 * sp.f_zvs, T, k, 40.0, 1.0)["P_out"]

out = {}
# A 启动捕获：冷态工件
r = drive_fw(lambda t: 20.0, lambda t: 0.949173, 2.0)
r["P_ideal"] = ideal_power(20.0, 0.949173)
out["A_start"] = {k: v for k, v in r.items() if k != "lp"}
print("A_start", json.dumps(out["A_start"], ensure_ascii=False), flush=True)
# B 居里跟踪：T 20→900°C 三秒
r = drive_fw(lambda t: 20 + 880 * t / 3.0, lambda t: 0.949173, 3.0)
out["B_curie"] = {k: v for k, v in r.items() if k != "lp"}
print("B_curie", json.dumps(out["B_curie"], ensure_ascii=False), flush=True)
# C 耦合阶跃 k 0.949→0.85 @1.5s（m=1 定压下 0.70 阶跃必然过流——那是 C2 的保护验证）
r = drive_fw(lambda t: 850.0, lambda t: 0.949173 if t < 1.5 else 0.85, 3.0)
relock = None
for h in r["lp"].hist:
    if h[0] > 1.5 and abs(h[3] - PHI_REF) < 0.03:
        relock = h[0] - 1.5; break
out["C_step"] = {**{k: v for k, v in r.items() if k != "lp"}, "relock_s": relock}
print("C_step", json.dumps(out["C_step"], ensure_ascii=False), flush=True)
# C2 大阶跃 k→0.70 @1.5s：定压下电流越过 339 Arms → 固件 OC 跳闸（频率环不含功率环，保护应接管）
r = drive_fw(lambda t: 850.0, lambda t: 0.949173 if t < 1.5 else 0.70, 2.5)
out["C2_oc"] = {**{k: v for k, v in r.items() if k != "lp"},
                "delay_after_fault": (r["fault_t"] - 1.5) if r["fault_t"] else None,
                "gate_stopped": (r["gate_last"] < r["fault_t"] + 5e-3) if r["fault_t"] else None}
print("C2_oc", json.dumps(out["C2_oc"], ensure_ascii=False), flush=True)
# D 电容失效减半 @1.5s → 容性失锁 → LOSS 跳闸
r = drive_fw(lambda t: 850.0, lambda t: 0.949173, 3.0, fault_at=1.5, fault_C=0.667e-6 * 0.5)
out["D_loss"] = {**{k: v for k, v in r.items() if k != "lp"},
                 "delay_after_fault": (r["fault_t"] - 1.5) if r["fault_t"] else None,
                 "gate_stopped": (r["gate_last"] < r["fault_t"] + 5e-3) if r["fault_t"] else None}
print("D_loss", json.dumps(out["D_loss"], ensure_ascii=False), flush=True)

# 断言
a = out["A_start"]
assert a["lock_t"] and a["lock_t"] < 1.0, f"锁定太慢 {a['lock_t']}"
assert abs(a["phi_end"] - PHI_REF) < 0.05, f"锁定相位偏 {a['phi_end']}"
assert a["phi_min"] > -0.05, f"捕获深入容性 {a['phi_min']}"
assert a["P_end"] >= 0.9 * a["P_ideal"], f"功率偏低 {a['P_end']} vs {a['P_ideal']}"
b = out["B_curie"]
assert b["phi_min"] > 0.0, f"居里跟踪跌破容性 {b['phi_min']}"
assert abs(b["phi_end"] - PHI_REF) < 0.06, f"居里后相位偏 {b['phi_end']}"
c = out["C_step"]
assert c["relock_s"] and c["relock_s"] < 0.5, f"阶跃重锁太慢 {c['relock_s']}"
c2 = out["C2_oc"]
assert c2["fault_t"] and c2["delay_after_fault"] < 0.05, f"大阶跃过流未跳闸 {c2['delay_after_fault']}"
assert c2["gate_stopped"], "OC 跳闸后门极未停"
d = out["D_loss"]
assert d["fault_t"] and d["delay_after_fault"] < 0.2, f"失锁未跳闸或太慢 {d['delay_after_fault']}"
assert d["gate_stopped"], "跳闸后门极未停"
print("ALL PASS")
