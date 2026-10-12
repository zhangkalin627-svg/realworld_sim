"""v0.9 电磁升级的图"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.patches import Rectangle
from .. import viz as _v           # 注册中文字体
from .. import materials as M
from .bh import STEEL45_BH, FLUX_MATERIALS, MU0


def plot_bh(fig):
    fig.clear()
    a1 = fig.add_subplot(121)
    H = np.linspace(0, 1.2e5, 400)
    for T, c in ((20, "#1f77b4"), (400, "#2ca02c"), (600, "#ff7f0e"), (700, "#d62728"), (750, "#9467bd"), (790, "#7f7f7f")):
        a1.plot(H / 1e3, STEEL45_BH.B(H, T), color=c, label=f"{T}°C")
    a1.set_xlabel("磁场强度 H (kA/m)"); a1.set_ylabel("磁感应强度 B (T)")
    a1.set_title("45钢 B-H 曲线随温度(模型)")
    a1.legend(); a1.grid(alpha=.3)
    a2 = fig.add_subplot(122)
    Ts = np.linspace(20, 820, 400)
    for H0, c in ((1e3, "#1f77b4"), (1e4, "#2ca02c"), (3e4, "#ff7f0e"), (1e5, "#d62728")):
        a2.semilogy(Ts, STEEL45_BH.mu_eff(H0, Ts), color=c, label=f"H={H0/1e3:g} kA/m")
    a2.semilogy(Ts, M.steel_mur(Ts, 20.0), "k--", label="旧版线性 μr(T), 冷态20")
    a2.axvline(STEEL45_BH.Tc, color="gray", ls=":"); a2.text(STEEL45_BH.Tc + 5, 30, "居里点", rotation=90, color="gray")
    a2.set_xlabel("温度 (°C)"); a2.set_ylabel("有效相对磁导率 μ_eff")
    a2.set_title("有效磁导率: 场越强越饱和 → 越低")
    a2.legend(fontsize=8); a2.grid(alpha=.3, which="both")
    fig.tight_layout()


def plot_conc(fig, res):
    fig.clear()
    keys = list(res.keys())
    names = [res[k]["name"].split(" ")[0] for k in keys]
    a1 = fig.add_subplot(131)
    cols = ["#888", "#1f77b4", "#2ca02c", "#ff7f0e"][:len(keys)]
    a1.bar(names, [res[k]["I_rms"] for k in keys], color=cols)
    a1.set_ylabel("线圈电流 (A 有效值)"); a1.set_title("同样 6 kW 吸收功率所需电流")
    a1.tick_params(axis="x", labelrotation=20, labelsize=8)
    a2 = fig.add_subplot(132)
    w = 0.38
    x = np.arange(len(keys))
    a2.bar(x - w / 2, [res[k]["eff"] * 100 for k in keys], w, color=cols, label="电效率 %")
    a2.set_ylim(70, 100); a2.set_xticks(x); a2.set_xticklabels(names, fontsize=8, rotation=20)
    a2.set_title("电效率(线圈铜损为估算)")
    a2b = a2.twinx()
    a2b.bar(x + w / 2, [res[k]["conc_loss_kW"] * 1e3 for k in keys], w, color="k", alpha=.5)
    a2b.set_ylabel("导磁体自身损耗 (W, 黑)")
    a3 = fig.add_subplot(133)
    for k, c in zip(keys, cols):
        a3.plot(res[k]["z"], np.asarray(res[k]["q"]) / 1e9, color=c, label=res[k]["name"].split(" ")[0])
    a3.set_xlabel("轴向位置 z (mm)"); a3.set_ylabel("外表面热源 (GW/m³)")
    a3.set_title("轴向加热图形(冷态,同功率)"); a3.legend(fontsize=8); a3.grid(alpha=.3)
    fig.tight_layout()


def plot_mur_map(ax, sim, title="有效磁导率分布 (log10 μ)"):
    if not hasattr(sim, "_mur_el"):
        ax.text(.5, .5, "线性模式无磁导率分布", transform=ax.transAxes, ha="center"); return
    mm = sim.mesh_mm
    sel = sim.steel_el | sim.conc_el
    tri = Triangulation(mm.p[0], mm.p[1], mm.t.T, mask=~sel)
    v = np.log10(np.maximum(sim._mur_el, 1.0))
    h = ax.tripcolor(tri, facecolors=v, cmap="viridis", vmin=0, vmax=max(2.4, v[sel].max()))
    g = sim.g
    ax.add_patch(Rectangle((g.coil_r0, g.coil_z0), g.coil_r1 - g.coil_r0, g.coil_z1 - g.coil_z0, fc="#b87333", ec="k", alpha=.85))
    ax.set_xlim(0, 30); ax.set_ylim(g.coil_z0 - 20, g.coil_z1 + 10); ax.set_aspect("equal")
    ax.set_xlabel("r (mm)"); ax.set_ylabel("z (mm)"); ax.set_title(title)
    ax.figure.colorbar(h, ax=ax, shrink=.8, label="log10 μ_eff")


def plot_3d(fig, res):
    fig.clear()
    a1 = fig.add_subplot(121)
    e0, e1 = res["ecc0"], res["ecc"]
    n = len(e0["sectors"]); ang = 360 * (np.arange(n) + .5) / n; wd = 360 / n * .4
    a1.bar(ang - wd / 2, np.array(e0["sectors"]) / np.mean(e0["sectors"]), wd, label="同心 (0 mm)", color="#1f77b4")
    a1.bar(ang + wd / 2, np.array(e1["sectors"]) / np.mean(e1["sectors"]), wd, label=f"偏心 {e1['ecc_mm']:g} mm (工件向 0° 方向偏)", color="#d62728")
    a1.set_ylim(0.85, 1.15); a1.axhline(1.0, color="gray", lw=.8)
    a1.set_xticks(ang); a1.set_xticklabels([f"{a:.0f}°" for a in ang], fontsize=8)
    a1.set_xlabel("管壁方位角"); a1.set_ylabel("该方位吸收功率 / 平均")
    a1.set_title("三维: 管壁各方位吸收功率"); a1.legend(fontsize=8, loc="upper right"); a1.grid(alpha=.3)
    a2 = fig.add_subplot(122)
    lab = ["三维", "二维轴对称"]
    a2.bar(lab, [e0["P_3d"] / 1e3, e0["P_2d"] / 1e3], color=["#2ca02c", "#888"])
    a2.set_ylabel("工件吸收功率 (kW, 1 kA 安匝)")
    a2.set_title(f"同心时 3D 对 2D 差 {100*(e0['P_3d']/e0['P_2d']-1):+.1f}%  (网格 {e0['tets']//1000}k 四面体)")
    fig.tight_layout()
