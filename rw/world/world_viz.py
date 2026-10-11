"""物理沙盒的图。"""
import math
import numpy as np
import matplotlib
from .. import viz  # noqa 字体
from .immersion import pressure_at, Env
from .fluids import get_fluid, G, P_ATM


def plot_immersion(fig, g, env, verdicts, vals):
    fig.clear()
    ax = fig.add_subplot(121)
    hmax = max(env.depth_m * 1.6, g.rated_depth_m * 1000 / vals["rho"] * 1.1, 10)
    h = np.linspace(0, hmax, 200)
    f = get_fluid(env.fluid)
    rho = vals["rho"]
    p = (env.p_surface + rho * G * h - P_ATM) / 1e6
    ax.plot(p, h, color="#1f77b4", lw=2, label=f"{f['name']} ρ={rho:.0f} 表压")
    ax.axvline(g.range_MPa, color="#e0322a", ls="--", label=f"量程 {g.range_MPa:g} MPa")
    ax.axvline(1000 * G * g.rated_depth_m / 1e6, color="#ff7f0e", ls=":", label=f"外壳额定 {g.rated_depth_m:g} m 水深")
    ax.axhline(env.depth_m, color="k", lw=0.8)
    ax.plot([vals["p_gauge"] / 1e6], [env.depth_m], "o", ms=10, color="#2ca02c" if all(v[0] != "FAIL" for v in verdicts) else "#e0322a")
    ax.invert_yaxis(); ax.set_xlabel("表压 (MPa)"); ax.set_ylabel("深度 (m)"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_title("压力随深度")
    ax2 = fig.add_subplot(122)
    Tb = [f["tsat"](env.p_surface + rho * G * hh) for hh in h]
    ax2.plot(Tb, h, color="#9467bd", lw=2, label="该深度的沸点")
    ax2.axvline(env.T, color="#d62728", label=f"液体温度 {env.T:g}°C")
    ax2.axvline(g.T_medium_max, color="#ff7f0e", ls=":", label=f"表耐温 {g.T_medium_max:g}°C")
    ax2.fill_betweenx(h, 0, Tb, color="#9467bd", alpha=0.08)
    ax2.axhline(env.depth_m, color="k", lw=0.8)
    ax2.invert_yaxis(); ax2.set_xlabel("温度 (°C)"); ax2.grid(alpha=0.3); ax2.legend(fontsize=8)
    ax2.set_xlim(0, max(max(Tb) * 1.05, env.T * 1.1, g.T_medium_max * 1.1))
    ax2.set_title("沸点线左边才是液体")
    fig.tight_layout()


def plot_circuit(fig, c, op, switch_on=None):
    fig.clear()
    ax = fig.add_subplot(121)
    Qs = np.linspace(1e-7, math.sqrt(c.pump.H0 / c.pump.k) * 0.98, 80)
    ax.plot(Qs * 60000, [c.pump.H(q) for q in Qs], label=f"泵 {c.pump.name}")
    ax.plot(Qs * 60000, [c.system_dp(q) / (c.rho * G) for q in Qs], label="管路阻力曲线")
    ax.plot(op["Q_Lmin"], op["H"], "ko"); ax.text(op["Q_Lmin"] * 1.03, op["H"], f"工作点 {op['Q_Lmin']:.1f} L/min", fontsize=9)
    if switch_on:
        ax.axvline(switch_on, color="#e0322a", ls="--"); ax.text(switch_on * 1.01, 2, f"流量开关最小动作 {switch_on:g} L/min", color="#b00", fontsize=8, rotation=90)
    ax.set_xlabel("流量 (L/min)"); ax.set_ylabel("扬程 / 阻力 (m)"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_title("泵和管路的交点 = 实际流量")
    ax2 = fig.add_subplot(122)
    names = [d[0] for d in op["details"]]
    dps = [d[1] / 1e3 for d in op["details"]]
    ax2.barh(names, dps, color="#4c78a8")
    for i, (n, dp, det) in enumerate(op["details"]):
        v = det.get("v", det.get("v_jet", 0))
        ax2.text(dp / 1e3 + 2, i, f"{dp/1e3:.0f} kPa, 流速 {v:.1f} m/s", va="center", fontsize=8)
    ax2.invert_yaxis(); ax2.set_xlabel("压降 (kPa)"); ax2.set_xlim(0, max(dps) * 1.8)
    ax2.set_title("压力都花在哪了")
    fig.tight_layout()


def plot_motor(fig, m, st=None, T_load=None):
    fig.clear()
    ax = fig.add_subplot(121)
    w = np.linspace(0, m.w_sync * 0.999, 300)
    n = w * 60 / (2 * math.pi)
    ax.plot(n, [m.torque(x) for x in w], label="电机转矩")
    if T_load is not None:
        ax.plot(n, [T_load] * len(n), "--", label="负载转矩(折到电机轴)")
    ax.axvline(m.n_rated, color="gray", ls=":"); ax.axhline(m.Tn, color="gray", ls=":")
    ax.set_xlabel("转速 (rpm)"); ax.set_ylabel("转矩 (N·m)"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_title(f"{m.name}  额定 {m.Tn:.2f} N·m @ {m.n_rated:g} rpm")
    if st:
        ax2 = fig.add_subplot(122)
        h = st["hist"]
        ax2.plot(h["t"] * 1000, h["n"], label="转速 rpm")
        ax3 = ax2.twinx(); ax3.plot(h["t"] * 1000, h["I"], color="#d62728", label="电流 A"); ax3.set_ylabel("电流 (A)")
        ax2.set_xlabel("时间 (ms)"); ax2.set_ylabel("转速 (rpm)"); ax2.grid(alpha=0.3)
        ta = st["t_accel"]
        ax2.set_title("直接起动" + (f"：{ta*1000:.0f} ms 到额定转速" if ta else "：起不来（堵转）"))
    fig.tight_layout()
