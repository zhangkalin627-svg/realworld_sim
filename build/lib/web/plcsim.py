"""PLC 虚拟调试 Web 模块 (v0.19)：三菱 FX 指令表(FX3U 子集, 含 STL 步进) + 单工位感应淬火设备模型，
整段运行（非实时）→ 逻辑分析仪式时序图 + 完成/不合格统计 + 事件日志。
与桌面版 ⑥ PLC 页同一内核（rw.plc），区别：桌面版实时交互按钮，Web 版用"剧本"注入按钮事件后整段跑完。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .modules import _clean, _svg

INPUTS = ["X000", "X001", "X002", "X010", "X011", "X020", "X040"]
OUTPUTS = ["Y000", "Y001", "Y002", "Y003", "Y004"]
STEPS = ["S0", "S20", "S21", "S22", "S23"]


def run_plc(params, progress):
    from rw.plc.demo import DemoSim, DEMO_IL
    from rw import viz                                  # 注册中文字体
    il = (params.get("program") or "").strip() or DEMO_IL
    p_air = min(max(float(params.get("p_air", 0.5)), 0.1), 0.8)
    t_sim = min(max(float(params.get("t_sim", 60.0)), 5.0), 180.0)
    scen = params.get("scenario", "normal")
    estop_at = min(max(float(params.get("estop_at", 8.0)), 1.0), t_sim - 4)
    scan_ms = 10
    sim = DemoSim(il, p_air=p_air, thermal=None, scan_ms=scan_ms)
    n = int(t_sim * 1000 / scan_ms)

    chans = {d: [] for d in INPUTS + OUTPUTS + STEPS}
    Tt, Tpart = [], []
    events = []
    prev_step = None
    k_start = int(0.3 * 1000 / scan_ms)
    k_estop = int(estop_at * 1000 / scan_ms)
    for k in range(n):
        t = k * scan_ms / 1000
        if k == k_start:
            sim.press("X000"); events.append((t, "按 启动 X000"))
        if scen == "estop":
            if k == k_estop:
                sim.plant.ext["X002"] = False; events.append((t, "急停 X002 按下（保持 2 s）"))
            if k == k_estop + int(2.0 * 1000 / scan_ms):
                sim.plant.ext["X002"] = True; events.append((t, "急停 X002 复位"))
            if k == k_estop + int(3.0 * 1000 / scan_ms):
                sim.press("X000"); events.append((t, "重新按 启动 X000"))
        sim.scan()
        inp = sim.plant.inputs(sim.Y)
        for d in INPUTS:
            chans[d].append(bool(inp.get(d, False)))
        for d in OUTPUTS:
            chans[d].append(bool(sim.Y.get(d, False)))
        act = [d for d in sorted(sim.plc.bits) if d.startswith("S") and sim.plc.bits[d]]
        for d in STEPS:
            chans[d].append(d in act)
        sname = " ".join(act) or "（无）"
        if sname != prev_step:
            events.append((round(t + scan_ms / 1000, 2), f"步 → {sname}"))
            prev_step = sname
        Tt.append(t)
        Tpart.append(sim.plant.part.T if sim.plant.part else float("nan"))
        if k % 1000 == 0:
            progress(0.05 + 0.85 * k / n)

    w = sim.plc.words
    d0, d1 = w.get("D0", 0), w.get("D1", 0)
    done = sim.plant.done
    heat0, quench0 = (done[0].heat, done[0].quench) if done else (None, None)
    te, tr = sim.plant.te, sim.plant.tr

    lines = [f"程序 {len(sim.plc.prog)} 条指令（{', '.join(sim.plc.used_ops)}），扫描周期 {scan_ms} ms，整段运行 {t_sim:.0f} s",
             f"气压 {p_air:.2f} MPa：感应圈下降 {te:.2f} s / 上升 {tr:.2f} s（气缸动力学）",
             f"完成 {d0} 件，测温不合格 {d1} 件"]
    if done:
        lines.append(f"每件实测节拍：加热 {done[0].heat:.2f} s / 淬火 {done[0].quench:.2f} s（程序设定见 D200/D201，0.1 s 单位）")
    lines.append("")
    lines.append("事件日志（前 14 条）：")
    for t, msg in events[:14]:
        lines.append(f"· t={t:6.2f} s  {msg}")

    warnings = []
    if te == float("inf") or tr == float("inf"):
        warnings.append(f"气压 {p_air:.2f} MPa 推不动负载，气缸卡住到不了位——这就是虚拟调试要抓的问题")
    if d0 == 0:
        warnings.append(f"{t_sim:.0f} s 内一件也没完成：检查步进转移条件（X020 有料、X010/X011 到位信号）或延长仿真时长")
    if d1 > 0:
        warnings.append(f"{d1} 件测温不合格（加热结束未到 850 °C）：D200 加热时间偏短或功率不足")

    # 图1：数字通道时序（逻辑分析仪）
    fig = plt.figure(figsize=(7.2, 5.6))
    ax = fig.add_subplot(111)
    order = INPUTS + OUTPUTS + STEPS
    colors = (["#2b6cb0"] * len(INPUTS)) + (["#2f9e44"] * len(OUTPUTS)) + (["#9c36b5"] * len(STEPS))
    for i, d in enumerate(order):
        y = [len(order) - 1 - i + (0.8 if v else 0.0) for v in chans[d]]
        ax.step(Tt, y, where="post", color=colors[i], lw=1.2)
        ax.text(-0.01, len(order) - 1 - i + 0.35, d, ha="right", va="center", fontsize=8,
                color=colors[i], transform=ax.get_yaxis_transform())
    ax.set_xlim(0, t_sim); ax.set_ylim(-0.5, len(order) - 0.2)
    ax.set_yticks([]); ax.set_xlabel("t (s)")
    for t, msg in events:
        if "急停" in msg or "启动" in msg:
            ax.axvline(t, color="#c62828" if "急停" in msg else "#888", ls=":", lw=0.8)
    ax.set_title("数字通道时序：蓝=输入 X，绿=输出 Y，紫=步进 S")
    p1 = {"title": "I/O 与步进时序", "svg": _svg(fig)}

    # 图2：工件温度
    fig = plt.figure(figsize=(7.2, 3.2))
    ax = fig.add_subplot(111)
    ax.plot(Tt, Tpart, color="#d23c3c", lw=1.4)
    ax.axhline(850, color="#888", ls="--", lw=0.8)
    ax.text(t_sim * 0.99, 865, "测温合格线 850°C", ha="right", fontsize=8, color="#555")
    ax.set_xlabel("t (s)"); ax.set_ylabel("工件表面温度 °C"); ax.set_xlim(0, t_sim)
    ax.set_title("工件温度（加热 Y001 / 喷淋 Y002 作用结果）")
    p2 = {"title": "工件温度曲线", "svg": _svg(fig)}

    raw = {"done": d0, "reject": d1, "te_s": te if te != float("inf") else None,
           "tr_s": tr if tr != float("inf") else None,
           "heat_s": heat0, "quench_s": quench0,
           "step_seq": [m.split("→ ")[1] for _, m in events if "步 →" in m][:8]}
    progress(1.0)
    return _clean({"lines": lines, "warnings": warnings, "plots": [p1, p2], "raw": raw})
