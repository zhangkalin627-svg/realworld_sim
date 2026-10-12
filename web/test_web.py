"""Web 版冒烟测试 (v0.16)：python3 web/test_web.py
首页可达、模块清单、材料库即查、感应淬火小算例端到端跑通（缩短加热/淬火时间，约 10–30 s）。"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["RW_JOBS_DB"] = os.path.join("/tmp", f"rw_test_jobs_{os.getpid()}.db")   # v0.22 测试库隔离
if os.path.exists(os.environ["RW_JOBS_DB"]):
    os.remove(os.environ["RW_JOBS_DB"])
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

def run_wait_id(jid, timeout=240):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = c.get(f"/api/job/{jid}").json()
        if j["status"] != "running":
            return j
        time.sleep(1.0)
    raise TimeoutError(jid)

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
t0 = time.time()
from web.jobstore import JobStore
dbp = os.environ["RW_JOBS_DB"]
j = run_wait("materials", {})
assert j["status"] == "done"
lst = c.get("/api/jobs").json()
assert any(x["module"] == "materials" and x["status"] == "done" for x in lst["jobs"]), "历史面板缺已完成任务"
mt = c.get("/api/meta").json()
assert mt["version"] and mt["modules"] >= 8, f"meta 异常: {mt}"
print("[web-i2] /api/meta 版本与模块数 OK")
st2 = JobStore(dbp)                      # 模拟服务器重启：新实例读同一库
assert st2.get(list(lst["jobs"])[0]["id"]) is not None, "重启后任务历史丢失"
st2.create("fake1", "plc", {"t_sim": 5}) # 伪造一个"正在跑"的任务
st3 = JobStore(dbp)
n = st3.mark_interrupted()               # 重启时的清理动作
assert n == 1 and st3.get("fake1")["status"] == "interrupted", "running 任务应被标记 interrupted"
assert st3.get(list(lst["jobs"])[0]["id"])["status"] == "done", "已完成任务不应被误标"
assert "参数已保留" in st3.get("fake1")["error"]
print(f"[web-i] 任务持久化 OK ({time.time()-t0:.0f}s): SQLite 落盘, 重启历史保留, running→interrupted 清理正确")
t0 = time.time()
# 反馈 B-1：power_kw=0 时效率 0/0=NaN，摘要文本不得崩（runner 级，绕过 API 收口）
from web.modules import run_induction, MODULES as _M
r = run_induction({"mode": "power", "power_kw": 0.0, "t_heat": 0.5, "t_quench": 0.5, "t_air": 0.2}, lambda f: None)
assert any("—" in l for l in r["lines"]), "NaN 字段应显示为 —"
# 反馈 B-3a：非数字 → 400 友好提示
bad = c.post("/api/sim/induction", json={"params": {"power_kw": "abc"}})
assert bad.status_code == 400 and "需要数字" in bad.json()["detail"], bad.text[:200]
# 反馈 B-3b：非法枚举 → 400
bad = c.post("/api/sim/induction", json={"params": {"medium": "liquid_nitrogen"}})
assert bad.status_code == 400 and "不在可选范围" in bad.json()["detail"], bad.text[:200]
# 反馈 B-2：越界数字收口而不是除零崩溃
j = run_wait("flow_net", {"holes": 0, "dh": 0})
assert j["status"] == "done", f"holes/dh=0 应收口到最小值跑通: {j.get('error', '?')[:150]}"
j = run_wait("world3d", {"preset": "工字钢梁桥", "tmax": 0})
assert j["status"] == "done", f"tmax=0 应收口: {j.get('error', '?')[:150]}"
# 数字选项的 select 传字符串（前端就是字符串）必须过校验
from web.server import _validate_params
_mod = next(m for m in _M if m["key"] == "induction")
_validate_params(_mod, {"freq": "30"})
# 反馈 B-4：物理不可达目标温度 → 快速明确报错（不能挂住线程池）
t1 = time.time()
j = run_wait("induction", {"mode": "target", "target": 5000, "t_heat": 1.0}, timeout=60)
assert time.time() - t1 < 30, "不可达温度必须快速失败"
assert j["status"] == "error" and "熔点" in j["error"] or "固相线" in j["error"], j.get("error", "?")[:200]
# 反馈 B-5：错误不回 traceback/服务器路径
assert "Traceback" not in j["error"] and "/home" not in j["error"] and "web/" not in j["error"], j["error"][:200]
j = run_wait("plc", {"program": "LD X000\nOUT Y000"})
assert j["status"] == "error" and "END" in j["error"] and "Traceback" not in j["error"], j.get("error", "?")[:200]
# 豆包反馈：未知指令要带行号
j = run_wait("plc", {"program": "LD X000\nFOO X001\nEND"})
assert j["status"] == "error" and "第2行" in j["error"] and "FOO" in j["error"], j.get("error", "?")[:200]
print(f"[web-j] 两封反馈修复回归 OK ({time.time()-t0:.0f}s): NaN文本/400校验/越界收口/不可达温度快速失败/错误不泄堆栈")

# [web-k] 电路模块：无 ngspice 的服务器必须给友好报错（沙箱即此情形）
t1 = time.time()
j = run_wait("circuit", {"example": "rc"}, timeout=30)
from rw.circuit import spice as _sp
if _sp.find_ngspice():
    assert j["status"] == "done", j.get("error", "?")[:200]
    assert any("Vpp" in l for l in j["result"]["lines"]), j["result"]["lines"]
    assert any(p["title"] == "波形图" for p in j["result"]["plots"])
    # RLC 回归（卡布 v0.26.2 报告：频率响应节点成员判断 AttributeError）
    j = run_wait("circuit", {"example": "rlc"}, timeout=60)
    assert j["status"] == "done", j.get("error", "?")[:200]
    assert any(p["title"] == "频率响应" for p in j["result"]["plots"]), [p["title"] for p in j["result"]["plots"]]
    print(f"[web-k] 电路模块 OK ({time.time()-t1:.0f}s): 有 ngspice，RC 测量+图齐全，RLC 频率响应回归通过")
else:
    assert j["status"] == "error" and "ngspice" in j["error"] and "Traceback" not in j["error"], j.get("error", "?")[:200]
    print(f"[web-k] 电路模块 OK ({time.time()-t1:.0f}s): 无 ngspice → 友好报错（完整算例需有 ngspice 的服务器验证）")


# [web-l] 断点续算：跑完的任务带检查点 → 模拟中断 → 续算终态与连续运行一致
t1 = time.time()
j = run_wait("induction", {"mode": "power", "power_kw": 6, "t_heat": 1.5, "t_quench": 1.0, "t_air": 0.5}, timeout=120)
assert j["status"] == "done", j.get("error", "?")[:150]
full = j["result"]["raw"]
ja = c.get(f"/api/job/{j['id']}").json()
assert ja.get("has_checkpoint"), "感应淬火任务应周期落检查点"
# 模拟服务器重启中断（直接改库状态），再续算
import sqlite3 as _sq, os as _os
_db = _os.environ.get("RW_JOBS_DB", "realworld_jobs.db")
_sq.connect(_db).execute("UPDATE jobs SET status='interrupted' WHERE id=?", (j["id"],)).connection.commit()
r2 = c.post("/api/sim/induction", json={"params": {"mode": "power", "power_kw": 6, "t_heat": 1.5, "t_quench": 1.0, "t_air": 0.5, "resume_from": j["id"]}})
assert r2.status_code == 200, r2.text[:200]
j2 = run_wait_id(r2.json()["job"], timeout=120)
assert j2["status"] == "done", j2.get("error", "?")[:150]
res2 = j2["result"]["raw"]
assert abs(res2["case_depth_mm"] - full["case_depth_mm"]) < 0.02, (res2["case_depth_mm"], full["case_depth_mm"])
assert abs(res2["HRC_surface"] - full["HRC_surface"]) < 0.2
# 参数变了（功率不同）指纹不符 → 忽略旧检查点从头跑，也必须正常完成
r3 = c.post("/api/sim/induction", json={"params": {"mode": "power", "power_kw": 12, "t_heat": 1.5, "t_quench": 1.0, "t_air": 0.5, "resume_from": j["id"]}})
j3 = run_wait_id(r3.json()["job"], timeout=120)
assert j3["status"] == "done", j3.get("error", "?")[:150]
print(f"[web-l] 断点续算 OK ({time.time()-t1:.0f}s): 检查点落盘/中断后续算终态一致(硬化层差<0.02mm)/参数变更加头重跑")


# [web-m] 加速档：fast 模式跑通且关键指标与精确档一致
t1 = time.time()
cfg = {"mode": "power", "power_kw": 6, "t_heat": 1.5, "t_quench": 1.0, "t_air": 0.5}
j = run_wait("induction", {**cfg, "speed": "fast"}, timeout=120)
assert j["status"] == "done", j.get("error", "?")[:150]
fast = j["result"]["raw"]
j = run_wait("induction", cfg, timeout=120)
acc = j["result"]["raw"]
assert abs(fast["case_depth_mm"] - acc["case_depth_mm"]) < 0.03, (fast["case_depth_mm"], acc["case_depth_mm"])
assert abs(fast["HRC_surface"] - acc["HRC_surface"]) < 0.3
assert abs(fast["t85_surface"] - acc["t85_surface"]) < 0.05
print(f"[web-m] 加速档 OK ({time.time()-t1:.0f}s): fast vs 精确 硬化层差<0.03mm/HRC差<0.3/t85差<0.05s")

print("web 冒烟测试全部通过")
