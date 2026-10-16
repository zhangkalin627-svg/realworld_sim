# -*- coding: utf-8 -*-
"""整机联动图：5 行共用时间轴。"""
import numpy as np
from .. import viz  # noqa 字体

STEP_NAME = {0: "待机", 10: "开水", 11: "预充", 12: "合旁路", 13: "待料", 20: "下降", 21: "加热", 22: "淬火", 23: "上升"}


def plot_machine(fig, m, t0=None, t1=None):
    fig.clear()
    L = {k: np.array(v, float) for k, v in m.log.items()}
    t = L["t"]
    if t0 is None: t0 = 0.0
    if t1 is None: t1 = t[-1]
    axs = fig.subplots(5, 1, sharex=True, gridspec_kw=dict(height_ratios=[1.0, 1.0, 1.0, 0.9, 1.35], hspace=0.5))
    ax = axs[0]
    ax.plot(t, L["Vbus"], color="#1f77b4", lw=1.4, label="母线电压 V")
    ax.set_ylabel("母线 V", color="#1f77b4"); ax.set_ylim(0, 620)
    ax.axhline(m.sp.V_ready, color="#1f77b4", ls=":", lw=0.8)
    ax.axhline(m.sp.V_uv, color="#c62828", ls=":", lw=0.8)
    ax2 = ax.twinx(); ax2.plot(t, np.where(L["run"] > 0, L["f"], np.nan), color="#ff7f0e", lw=1.2); ax2.set_ylabel("开关频率 kHz", color="#ff7f0e"); ax2.set_ylim(0, 65)
    ax.set_title("母线电压（蓝，点线=就绪/欠压阈值）  开关频率（橙）", fontsize=9, loc="left")
    ax = axs[1]
    ax.plot(t, L["P_ref"], color="#999", lw=1, ls="--", label="设定(软启动后)")
    ax.plot(t, L["P_out"], color="#d62728", lw=1.3, label="输出功率")
    ax.plot(t, L["P_wp"], color="#2ca02c", lw=1.3, label="进工件")
    ax.set_ylabel("功率 kW"); ax.legend(fontsize=7, loc="upper right", ncol=3)
    ax.set_title("变频器输出功率", fontsize=9, loc="left")
    ax = axs[2]
    ax.plot(t, L["T_wp"], color="#e6550d", lw=1.4, label="工件表面 °C")
    ax.axhline(850, color="#e6550d", ls=":", lw=0.8)
    ax.set_ylabel("工件 °C", color="#e6550d"); ax.set_ylim(0, max(1100, np.nanmax(L["T_wp"]) * 1.05 if np.isfinite(L["T_wp"]).any() else 1100))
    ax2 = ax.twinx(); ax2.plot(t, L["T_coil"], color="#8c564b", lw=1.3); ax2.set_ylabel("线圈 °C", color="#8c564b")
    ax2.axhline(m.sp.coil_fail, color="#8c564b", ls=":", lw=0.8)
    ax2.set_ylim(0, max(m.sp.coil_fail * 1.1, np.nanmax(L["T_coil"]) * 1.1))
    ax.set_title("工件表面温度（橙，点线=红外测温 850°C）  线圈温度（棕，点线=损坏阈值）", fontsize=9, loc="left")
    ax = axs[3]
    ax.plot(t, L["cw"], color="#17becf", lw=1.2, label="冷却水 L/min")
    ax.plot(t, L["spray"], color="#9467bd", lw=1.2, label="喷淋 L/min")
    ax.set_ylabel("流量 L/min"); ax.legend(fontsize=7, loc="upper right", ncol=2)
    ax2 = ax.twinx(); ax2.plot(t, L["T_tank"], color="#7f7f7f", lw=1.2); ax2.set_ylabel("水箱 °C", color="#555")
    ax.set_title("冷却水/喷淋流量   水箱温度（灰）", fontsize=9, loc="left")
    ax = axs[4]
    ys = sorted(STEP_NAME); idx = {y: i for i, y in enumerate(ys)}
    ax.step(t, [idx.get(int(v), 0) for v in L["step"]], where="post", color="#333", lw=1.2)
    ax.set_yticks(range(len(ys))); ax.set_yticklabels([STEP_NAME[y] for y in ys], fontsize=6.5)
    ax2 = ax.twinx(); ax2.plot(t, L["pos"], color="#bcbd22", lw=1.0); ax2.set_ylim(-0.1, 1.15); ax2.set_ylabel("线圈位置", fontsize=8)
    ax2.set_yticks([0, 1]); ax2.set_yticklabels(["上", "下"], fontsize=7)
    ax.set_title("PLC 步进（黑）与线圈位置（黄绿）", fontsize=9, loc="left")
    ax.set_xlabel("仿真时间 s")
    # 事件标记
    for (te, txt) in m.events:
        if txt.startswith("★") or "报警总 ON" in txt or "故障 ON" in txt or "快进" in txt or "损坏" in txt:
            for a in axs:
                a.axvline(te, color="#c62828", lw=0.8, alpha=0.7)
    for a in axs:
        a.set_xlim(t0, t1); a.grid(alpha=0.25); a.tick_params(labelsize=8)
    # 变频器故障时间段着色
    flt = L["fault"] > 0
    if flt.any():
        for a in axs:
            a.fill_between(t, 0, 1, where=flt, transform=a.get_xaxis_transform(), color="#c62828", alpha=0.10)
    fig.subplots_adjust(left=0.07, right=0.92, top=0.965, bottom=0.07)
