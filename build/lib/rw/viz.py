"""结果可视化：发光颜色外观、温度场、硬度分布、温度曲线。"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.tri import Triangulation
from matplotlib.patches import Rectangle

import os as _os
_fp = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "fonts", "NotoSansSC-Regular.otf")
if _os.path.exists(_fp):
    matplotlib.font_manager.fontManager.addfont(_fp)
for _f in ("Noto Sans SC", "Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Zen Hei", "Microsoft YaHei", "SimHei"):
    if any(_f in f.name for f in matplotlib.font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [_f, "DejaVu Sans"]
        break
plt.rcParams["axes.unicode_minus"] = False

# 钢的火色（热处理火色表，暗处观察的大致颜色）
_GLOW_T = [20, 450, 550, 630, 700, 780, 850, 900, 980, 1050, 1150, 1250, 1350, 1500]
_GLOW_C = ["#6e7378", "#5a5c60", "#3a1410", "#6a1208", "#8f1a0a", "#b8240c", "#d93a10", "#ec5a16",
           "#f87c1e", "#ff9c2a", "#ffbe3c", "#ffd968", "#fff0a8", "#fffbe8"]


def glow_rgb(T):
    T = np.asarray(T, float)
    rgb = np.array([matplotlib.colors.to_rgb(c) for c in _GLOW_C])
    return np.stack([np.interp(T, _GLOW_T, rgb[:, i]) for i in range(3)], axis=-1)


HRC_CMAP = LinearSegmentedColormap.from_list("hrc", ["#3b4cc0", "#8db0fe", "#f2cbb7", "#e7745b", "#b40426"])


def _tri(sim, mirror=True):
    p = sim.th_mesh_mm.p
    t = sim.th_mesh_mm.t
    if not mirror:
        return Triangulation(p[0], p[1], t.T), None
    n = p.shape[1]
    P = np.hstack([p, p * np.array([[-1], [1]])])
    Tt = np.hstack([t, t + n])
    return Triangulation(P[0], P[1], Tt.T), n


def _mir(v):
    return np.concatenate([v, v])


def _coil(ax, g):
    for s in (1, -1):
        x0 = g.coil_r0 if s > 0 else -g.coil_r1
        ax.add_patch(Rectangle((x0, g.coil_z0), g.coil_r1 - g.coil_r0, g.coil_z1 - g.coil_z0,
                               fc="#b87333", ec="#5a3510", lw=0.8, alpha=0.9))


def plot_glow(ax, sim, T=None, title=None):
    T = sim.T_end_heat if T is None else T
    tri, n = _tri(sim)
    col = glow_rgb(_mir(T)[tri.triangles].mean(axis=1))
    ax.set_facecolor("#0b0b0e")
    ax.tripcolor(tri, facecolors=np.zeros(len(tri.triangles)), alpha=0)  # 设定范围
    from matplotlib.collections import PolyCollection
    verts = np.stack([np.c_[tri.x[tr], tri.y[tr]] for tr in tri.triangles])
    ax.add_collection(PolyCollection(verts, facecolors=col, edgecolors=col, linewidths=0.2))
    _coil(ax, sim.g)
    ax.set_aspect("equal")
    ax.set_xlim(-26, 26)
    ax.set_ylim(-32, 58)
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=10)


def plot_field(ax, sim, values, cmap, label, vmin=None, vmax=None, fig=None):
    tri, n = _tri(sim)
    tc = ax.tripcolor(tri, _mir(values), shading="gouraud", cmap=cmap, vmin=vmin, vmax=vmax)
    _coil(ax, sim.g)
    ax.set_aspect("equal")
    ax.set_xlim(-26, 26)
    ax.set_ylim(-32, 58)
    ax.set_xlabel("r (mm)")
    ax.set_ylabel("z (mm)")
    if fig is not None:
        fig.colorbar(tc, ax=ax, label=label, shrink=0.8)
    return tc


def plot_curves(ax, sim):
    h = sim.hist
    ax.plot(h["t"], h["Tsurf"], label="外表面", color="#d62728")
    ax.plot(h["t"], h["Tcore"], label="壁厚中心", color="#ff7f0e")
    ax.plot(h["t"], h["Tin"], label="内孔表面", color="#1f77b4")
    for y, s in ((830, "Ac3(快速加热)"), (340, "Ms 马氏体开始")):
        ax.axhline(y, ls="--", lw=0.8, color="gray")
        ax.text(h["t"][-1], y, s, ha="right", va="bottom", fontsize=8, color="gray")
    ax.axvspan(0, sim.t_heat, color="#ffe0c0", alpha=0.5, lw=0)
    q0 = sim.t_heat + sim.t_delay
    ax.axvspan(q0, q0 + sim.t_quench, color="#cfe8ff", alpha=0.6, lw=0)
    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("温度 (°C)")
    ax.set_title(f"淬火区中部温度  （橙=加热，蓝=淬火）", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)


def plot_hardness_line(ax, sim, label=None):
    p = sim.th_mesh_mm.p
    ln = sim.line_nodes
    d = sim.g.ro - p[0, ln]
    ax.plot(d, sim.HRC[ln], marker=".", label=label)
    ax.set_xlabel("离外表面深度 (mm)")
    ax.set_ylabel("硬度 HRC (估算)")
    ax.axhline(50, ls="--", lw=0.8, color="gray")
    ax.set_ylim(15, 62)
    ax.grid(alpha=0.3)


def single_report(sim, res, path):
    fig, axs = plt.subplots(1, 4, figsize=(18, 5.2), gridspec_kw=dict(width_ratios=[1, 1.15, 1.15, 1.5]))
    plot_glow(axs[0], sim, title="加热结束时的样子（火色）")
    plot_field(axs[1], sim, sim.T_end_heat, "inferno", "°C", 25, max(1000, sim.T_end_heat.max()), fig)
    axs[1].set_title("加热结束温度场", fontsize=10)
    plot_field(axs[2], sim, sim.HRC, HRC_CMAP, "HRC", 18, 60, fig)
    axs[2].set_title("淬火后硬度分布 (估算)", fontsize=10)
    plot_curves(axs[3], sim)
    fig.suptitle(f"{res['freq_kHz']:.0f} kHz · {res['power_kW']:.1f} kW · 加热 {sim.t_heat:.1f}s · {res['medium']} {sim.t_quench:.1f}s   "
                 f"→ 硬化层 {res['case_depth_mm']:.2f} mm, 表面 {res['HRC_surface']:.0f} HRC", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
