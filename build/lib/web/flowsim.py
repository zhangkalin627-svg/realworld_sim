"""流体 Web 模块 (v0.20)：⑩ 流体页三个算例移植——管网(泵/阀/并联喷淋)、水锤、水箱放空+射流。
计算内核与桌面版完全一致（rw.flow），图同样用 matplotlib（服务器端 SVG）。校验见 validate_flow.py。"""
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .modules import _clean, _svg


def _chk_lines(chk):
    lines, warns = [], []
    for st, msg in chk:
        lines.append(f"[{st}] {msg}")
        if st in ("WARN", "FAIL"):
            warns.append(msg)
    return lines, warns


def run_flow_net(params, progress):
    """管网：两工位并联喷淋，泵/阀/过滤器/软管/喷淋孔 → 流量分配 + 体检 + 压力沿程 + 泵曲线工作点。"""
    from rw.world.fluids import FLUIDS
    from rw.flow.network import example_two_station
    from rw import viz  # 字体
    fk = params.get("fluid", "pag10")
    if fk not in ("pag10", "water", "oil_quench"):
        fk = "pag10"
    T = float(params.get("T", 30.0)); dirty = float(params.get("dirty", 1.0))
    spd = float(params.get("speed", 100.0)) / 100
    openB = float(params.get("openB", 100.0)) / 100
    holes = int(params.get("holes", 24)); dh = float(params.get("dh", 1.2)) * 1e-3
    hose = float(params.get("hose", 10.0)) * 1e-3
    progress(0.1)
    net = example_two_station(fk, T, dirty)
    for l in net.links:
        if l.kind == "pump": l.speed = spd
        if l.name == "电磁阀B": l.opening = openB
        if l.kind == "nozzle": l.n = holes; l.d = dh
        if l.name.startswith("软管"): l.D = hose
    res = net.solve()
    progress(0.55)
    chk = net.checks(res)
    L = res["links"]
    qa, qb, qp = L["软管A 内径10"]["Q_Lmin"], L["软管B 内径10"]["Q_Lmin"], L["离心泵"]["Q_Lmin"]
    pm = L["离心泵"]
    lines = [f"介质 {FLUIDS[fk]['name']} {T:.0f}°C，泵转速 {spd*100:.0f}% 额定，过滤器脏污 ×{dirty:g}",
             f"泵送 {qp:.1f} L/min（A 路 {qa:.1f}，B 路 {qb:.1f}），扬程 {pm['head']:.1f} m，轴功率 {pm['P_shaft']:.0f} W",
             f"射流 {L['喷淋孔A']['v_jet']:.1f} m/s，A 环内压 {res['nodes']['A环']['p_static']/1e5:.2f} bar", ""]
    cl, warns = _chk_lines(chk)
    lines += cl

    fig = plt.figure(figsize=(9.6, 4.0))
    ax = fig.add_subplot(121)
    path = ["水箱", "泵吸入", "泵出口", "分叉", "A阀后", "A环"]
    p = [res["nodes"][n]["p_static"] / 1e5 for n in path]
    ax.plot(range(len(path)), p, "o-", color="#1f77b4", lw=2)
    for i, v in enumerate(p):
        ax.annotate(f"{v:.2f}", (i, v), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(range(len(path))); ax.set_xticklabels(path, rotation=30, fontsize=8)
    ax.set_ylabel("静压 bar（表压）"); ax.set_title("沿 A 路的压力变化"); ax.grid(alpha=0.3)
    ax2 = fig.add_subplot(122)
    pump = [l for l in net.links if l.kind == "pump"][0]
    Q = np.linspace(0, math.sqrt(pump.H0 / pump.k) * 0.98, 100)
    ax2.plot(Q * 6e4, [pump.H0 - pump.k * q * q for q in Q], color="#999", ls="--", label="泵曲线(额定转速)")
    ax2.plot(Q * 6e4, [pump.gain(q) for q in Q], color="#1f77b4", label=f"泵曲线({pump.speed*100:.0f}% 转速)")
    pr = res["links"]["离心泵"]
    ax2.plot([pr["Q_Lmin"]], [pr["head"]], "o", ms=11, color="#d62728", label="当前工作点")
    sp, qs, hs = np.linspace(0.5, 1.1, 7), [], []
    old = pump.speed
    for s in sp:
        pump.speed = s; r2 = net.solve(); qs.append(r2["links"]["离心泵"]["Q_Lmin"]); hs.append(r2["links"]["离心泵"]["head"])
    pump.speed = old
    progress(0.9)
    ax2.plot(qs, hs, ":", color="#d62728", label="转速 50–110% 的工作点")
    ax2.set_xlabel("流量 L/min"); ax2.set_ylabel("扬程 m"); ax2.set_title("泵曲线与工作点")
    ax2.grid(alpha=0.3); ax2.legend(fontsize=8)
    fig.tight_layout()
    progress(1.0)
    return _clean({"lines": lines, "warnings": warns,
                   "plots": [{"title": "管网压力与泵工作点", "svg": _svg(fig)}],
                   "raw": {"Q_pump_Lmin": qp, "Q_A_Lmin": qa, "Q_B_Lmin": qb,
                           "head_m": pm["head"], "P_shaft_W": pm["P_shaft"], "v_jet": L["喷淋孔A"]["v_jet"]}})


HAMMER_MATS = ["不锈钢", "钢", "铜", "PVC", "PE", "增强软管", "橡胶软管"]


def run_flow_hammer(params, progress):
    """水锤：关阀冲击（特征线法），最高/最低压力 vs 管材额定，汽穴判据。"""
    from rw.flow import transient as TR
    from rw.world.fluids import G
    from rw import viz
    m = params.get("mat", "钢")
    if m not in HAMMER_MATS:
        m = "钢"
    D = float(params.get("D", 26.0)) * 1e-3; e = float(params.get("e", 2.0)) * 1e-3
    Lp = float(params.get("L", 3.0)); H = float(params.get("H", 3.0))
    K = float(params.get("K", 20.0)); tc = float(params.get("tc", 25.0))
    rate = float(params.get("rate", 10.0))
    tL = 2 * Lp / TR.wave_speed(1000, 2.2e9, D, e, TR.PIPE_E[m])
    r = TR.water_hammer(L=Lp, D=D, e=e, E=TR.PIPE_E[m], Hres=H * 1e5 / (1000 * G), K_valve=K,
                        t_close=tc * 1e-3, N=24, T=25.0, tmax=max(0.05, 12 * tL, 6 * tc * 1e-3))
    progress(0.7)
    rho = r["rho"]
    pk, pn, p0 = r["Hmax"] * rho * G / 1e5, r["Hmin"] * rho * G / 1e5, r["H0_valve"] * rho * G / 1e5
    lines = [f"管材 {m}，管长 {Lp:g} m，内径 {D*1e3:g} mm，壁厚 {e*1e3:g} mm；关阀时间 {tc:g} ms",
             f"关阀前流速 {r['v0']:.2f} m/s，阀前压力 {p0:.2f} bar；波速 a={r['a']:.0f} m/s（Korteweg）",
             f"水锤结果：最高 {pk:.1f} bar，最低（线性理论值）{pn:.1f} bar"]
    slow = tc * 1e-3 > r["tL"]
    frac = (r["Hmax"] - r["H0_valve"]) / r["dH_jou"]
    lines.append(f"2L/a = {r['tL']*1e3:.1f} ms；关阀 {tc:g} ms，冲击只有瞬关值 ρ·a·v 的 {100*min(frac,1.5):.0f}%"
                 + ("" if slow else "（关得比 2L/a 还快，等于瞬关）"))
    lines.append(f"关阀瞬间最高约 {pk:.1f} bar（理论上限 阀前+ρav = {p0 + r['dH_jou']*rho*G/1e5:.1f} bar），额定 {rate:g} bar")
    warns = []
    if pk > rate:
        warns.append(f"超压！峰值 {pk:.1f} bar > 额定 {rate:g} bar：用更长的关阀时间、软管/蓄能器或换更高额定的管")
    if r["cavitation"]:
        warns.append(f"负压波降到 {pn:.2f} bar，低于水的蒸气压 → 水柱拉断、再合拢时会有更大的撞击（本模型不算汽穴，真实峰值更高）")
    elif pn < 0:
        warns.append(f"回波最低 {pn:.2f} bar（负压），接头可能吸气")

    fig = plt.figure(figsize=(7.2, 4.2))
    ax = fig.add_subplot(111)
    t = r["t"] * 1e3
    ax.plot(t, r["H_valve"] * rho * G / 1e5, color="#d62728", label="阀前压力")
    ax.plot(t, r["H_up"] * rho * G / 1e5, color="#1f77b4", alpha=0.6, label="上游端")
    ax.axhline(rate, color="#ff7f0e", ls="--", label=f"额定 {rate:g} bar")
    ax.axhline(p0, color="k", lw=0.6, ls=":")
    ax.axvline(tc, color="g", ls=":", label="关阀完成")
    ax.set_ylim(bottom=max(pn, -3.0) - 0.5); ax.set_xlabel("时间 ms"); ax.set_ylabel("压力 bar（表压）")
    ax.set_title("关阀后阀前压力（特征线法）"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    progress(1.0)
    return _clean({"lines": lines, "warnings": warns,
                   "plots": [{"title": "水锤压力历程", "svg": _svg(fig)}],
                   "raw": {"p_max_bar": pk, "p_min_bar": pn, "v0": r["v0"], "a": r["a"], "tL_ms": r["tL"] * 1e3}})


JET_FLUIDS = ["water", "pag10", "oil_quench", "gasoline", "glycerin", "mercury"]


def run_flow_jet(params, progress):
    """水箱放空 + 射流：Torricelli 对照、放空时间、射程、冲击力。"""
    from rw.world.fluids import FLUIDS, G
    from rw.flow.network import Network, Nozzle
    from rw.flow import jet as JT
    from rw import viz
    fk = params.get("fluid", "water")
    if fk not in JET_FLUIDS:
        fk = "water"
    A = float(params.get("A", 0.5)); H0 = float(params.get("H0", 1.0))
    d = float(params.get("d", 12.0)) * 1e-3; z = float(params.get("zh", 1.0))
    ang = float(params.get("ang", 0.0))
    progress(0.1)
    net = Network(fk, 25.0)
    net.tank("T", z=z, level=H0, area=A); net.reservoir("O", z=z, level=0.0)
    net.add(Nozzle("hole", "T", "O", n=1, d=d, Cd=0.62))
    a_hole = math.pi * d ** 2 / 4
    t_end = JT.drain_time(A, a_hole, H0, 0.0)
    hist = net.run_quasi_steady(t_end, dt=max(t_end / 300, 0.05), stop_empty=True)
    progress(0.7)
    rho = net.rho
    q0 = hist["Q"]["hole"][0]; v0 = q0 / (0.62 * a_hole)
    F = JT.jet_force(rho, q0, v0)
    tj = JT.trajectory(v0, ang, z)
    lines = [f"{FLUIDS[fk]['name']} ρ={rho:.0f} kg/m³；水箱 {A:g} m²，初始液位 {H0:g} m，孔径 {d*1e3:g} mm（离地 {z:g} m，仰角 {ang:g}°）",
             f"初始射流 {v0:.2f} m/s，流量 {q0*6e4:.1f} L/min",
             f"放空时间 {t_end/60:.1f} min；落地射程 {tj['range']:.2f} m；垂直打在板上冲击力 {F:.1f} N",
             f"射流速度 v≈√(2gh)，与液体密度无关；同样 {H0:g} m 液位的底部压力 {rho*G*H0/1e3:.1f} kPa"]
    if abs(ang) > 0:
        lines.append(f"斜向 {ang:g}° 射出，最高点 {tj['apex']:.2f} m")
    warns = ["射流按无空气阻力的抛物线算；细水柱/雾化射流会被空气明显减速，实际射程更短"]

    fig = plt.figure(figsize=(9.6, 4.0))
    ax = fig.add_subplot(121)
    ax.plot(hist["t"] / 60, hist["level"]["T"], color="#1f77b4", lw=2, label="网络准稳态")
    tt = np.linspace(0, t_end, 100)
    ax.plot(tt / 60, (np.sqrt(H0) - tt / (A / (0.62 * a_hole) * math.sqrt(2 / G))).clip(0) ** 2, "k--", lw=1, label="Torricelli 解析")
    ax.set_xlabel("分钟"); ax.set_ylabel("液位 m"); ax.set_title("水箱放空"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax2 = fig.add_subplot(122)
    for frac, c in ((1.0, "#1f77b4"), (0.5, "#2ca02c"), (0.15, "#ff7f0e")):
        h = H0 * frac; v = 0.97 * math.sqrt(2 * G * h)
        j = JT.trajectory(v, ang, z)
        ax2.plot(j["x"], j["z"], color=c, label=f"液位 {h:.2f} m：v={v:.1f} m/s，射程 {j['range']:.2f} m")
    ax2.axhline(0, color="k"); ax2.set_xlabel("m"); ax2.set_ylabel("高度 m")
    ax2.set_title("射流轨迹（液位下降时变近）"); ax2.set_ylim(bottom=-0.1); ax2.grid(alpha=0.3); ax2.legend(fontsize=7)
    fig.tight_layout()
    progress(1.0)
    return _clean({"lines": lines, "warnings": warns,
                   "plots": [{"title": "放空与射流", "svg": _svg(fig)}],
                   "raw": {"v0": v0, "q0_Lmin": q0 * 6e4, "t_drain_min": t_end / 60,
                           "range_m": tj["range"], "F_N": F}})
