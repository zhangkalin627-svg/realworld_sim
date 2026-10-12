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
assert {"induction", "materials"} <= keys, f"模块清单缺项: {keys}"
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
print("web 冒烟测试全部通过")
