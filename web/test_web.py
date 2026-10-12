"""Web 版冒烟测试 (v0.16)：python3 web/test_web.py
首页可达、模块清单、材料库即查、感应淬火小算例端到端跑通（缩短加热/淬火时间，约 10–30 s）。"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fastapi.testclient import TestClient
from web.server import app

c = TestClient(app)

r = c.get("/")
assert r.status_code == 200 and "真实的世界" in r.text, "首页不可达"
print("[web-a] 首页 200 OK")

mods = c.get("/api/modules").json()
keys = {m["key"] for m in mods}
assert {"induction", "materials", "world3d", "plc"} <= keys, f"模块清单缺项: {keys}"
assert any(f.get("name") == "freq" for f in mods[0]["fields"]), "感应淬火表单缺字段"
print(f"[web-b] 模块清单 {sorted(keys)}，表单 schema 完整")

def run_wait(key, params, timeout=240):
    jid = c.post(f"/api/sim/{key}", json={"params": params}).json()["job"]
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = c.get(f"/api/job/{jid}").json()
        if j["status"] != "running":
            return j
        time.sleep(1.0)
    raise TimeoutError(key)

j = run_wait("materials", {})
assert j["status"] == "done" and any("45钢" in l for l in j["result"]["lines"]), str(j)[:300]
print("[web-c] 材料库即查 OK")

t0 = time.time()
j = run_wait("induction", {"mode": "power", "power_kw": 6.0, "t_heat": 1.0, "t_quench": 0.5, "t_air": 0.5})
assert j["status"] == "done", j.get("error", "?")
r = j["result"]
assert len(r["plots"]) == 4 and all("<svg" in p["svg"] for p in r["plots"]), "图缺失"
T = r["raw"]["T_surf_end_heat"]
assert 200 < T < 1600, f"温度不合理 {T}"
print(f"[web-d] 感应淬火端到端 OK ({time.time()-t0:.0f}s): 加热 1 s 后外表面 {T:.0f}°C，4 张 SVG 图，警告 {len(r['warnings'])} 条")
t0 = time.time()
j = run_wait("world3d", {"preset": "工字钢梁桥", "tmax": 8})
assert j["status"] == "done", j.get("error", "?")
v = j["result"]["viewer"]
names = [p["name"] for p in v["parts"]]
assert len(names) == 4 and set(v["meshes"]) == set(names) == set(v["poses"]), "viewer 零件/网格/位姿不一致"
assert all(len(v["poses"][n]) == len(v["times"]) for n in names), "位姿帧数与时间轴不齐"
assert all(len(fr) == 7 for n in names for fr in v["poses"][n]), "位姿必须是 [x,y,z,qx,qy,qz,qw]"
assert len(v["times"]) >= 3, "帧数过少"
import json as _j; _j.dumps(v)  # 必须可 JSON 序列化（无 numpy 残留）
z0, zN = v["poses"]["工字钢"][0][2], v["poses"]["工字钢"][-1][2]
assert abs(zN - z0) < 5, f"工字钢应静置于桥墩上（z0={z0:.0f} zN={zN:.0f}）"   # v0.18 接触对修复回归
print(f"[web-e] 三维装配世界端到端 OK ({time.time()-t0:.0f}s): {len(names)} 零件, {len(v['times'])} 帧, 状态 {v['status']}")
t0 = time.time()
j = run_wait("plc", {"t_sim": 30})
assert j["status"] == "done", j.get("error", "?")
r = j["result"]
assert r["raw"]["done"] >= 2, f"30 s 至少应完成 2 件: {r['raw']}"
assert abs(r["raw"]["heat_s"] - 3.0) < 0.05 and abs(r["raw"]["quench_s"] - 3.0) < 0.05, "每件加热/淬火必须等于程序设定 3.0 s"
assert r["raw"]["step_seq"][:5] == ["S0", "S20", "S21", "S22", "S23"], f"步进顺序错: {r['raw']['step_seq']}"
assert len(r["plots"]) == 2 and all("<svg" in p["svg"] for p in r["plots"]), "时序图缺失"
j2 = run_wait("plc", {"t_sim": 20, "scenario": "estop", "estop_at": 6.0})
assert j2["status"] == "done" and any("急停" in l for l in j2["result"]["lines"]), "急停剧本未生效"
print(f"[web-f] PLC 虚拟调试端到端 OK ({time.time()-t0:.0f}s): 30 s 完成 {r['raw']['done']} 件, 加热 {r['raw']['heat_s']:.2f}s/淬火 {r['raw']['quench_s']:.2f}s, 急停剧本 OK")
t0 = time.time()
import math as _m
j = run_wait("flow_net", {})
assert j["status"] == "done", j.get("error", "?")
r = j["result"]["raw"]
assert abs(r["Q_A_Lmin"] - r["Q_B_Lmin"]) < 0.2, "阀全开时 A/B 两路应对称"
assert 30 < r["Q_pump_Lmin"] < 60 and r["v_jet"] > 10, f"流量/射速不合理: {r}"
j = run_wait("flow_hammer", {})
r = j["result"]["raw"]
assert j["status"] == "done", j.get("error", "?")
assert 15 < r["p_max_bar"] < 25, f"水锤峰值应在阀前 3 bar 基础上显著升高: {r}"
assert r["tL_ms"] < 25, "默认关阀 25 ms 应慢于 2L/a"
j = run_wait("flow_jet", {})
r = j["result"]["raw"]
assert j["status"] == "done", j.get("error", "?")
assert abs(r["v0"] - _m.sqrt(2 * 9.81 * 1.0)) / _m.sqrt(2 * 9.81 * 1.0) < 0.02, "射流初速应对照 Torricelli √(2gH)"
assert r["t_drain_min"] > 10, "放空时间不合理"
print(f"[web-g] 流体三算例端到端 OK ({time.time()-t0:.0f}s): 管网/水锤/放空射流全部对照通过")
t0 = time.time()
j = run_wait("mcu", {})
assert j["status"] == "done", j.get("error", "?")
r = j["result"]; s = r["raw"]
assert s["t_EN_off"] is not None, "过流保护必须动作（关 EN）"
assert s["i_peak"] > s["i_th"] > s["i_before"], f"电流关系错: {s}"
prot = (s["t_EN_off"] - s["t_cross"]) * 1e6
assert 0 < prot < 500, f"保护响应应 <500 µs: {prot:.0f}"
assert s["t_Y000_off"] > s["t_PD7_on"], "PLC 应在 MCU 报警后撤销 Y000"
assert len(r["plots"]) == 3 and all("<svg" in p["svg"] for p in r["plots"]), "时序图缺失"
print(f"[web-h] 单片机+PLC 联合仿真端到端 OK ({time.time()-t0:.0f}s): {s['instr']} 条指令, 保护响应 {prot:.1f} µs, 峰值 {s['i_peak']:.2f} A")
print("web 冒烟测试全部通过")
