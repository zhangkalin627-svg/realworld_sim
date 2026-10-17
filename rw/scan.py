"""频率/时间/冷却扫描：给定工件壁厚，直接仿真冶金结果并给出频率推荐（用户 2026-10-17 设计目标）。

每个网格点 = 一次完整 加热→延时→淬火→空冷 仿真（InductionQuenchSim），取：
  奥氏体化深度（Tpeak ≥ Ac3_fast）、有效淬硬层深（HRC≥50）、表面硬度/组织、
  表面温度、电效率、热态趋肤深度。
判定与推荐是工程经验规则（见 verdict / recommend 的注释），全部输出均为【仿真结果】。

模型边界（诚实声明）：45 钢 CCT 规则化硬度模型，未含具体钢号成分差异；
奥氏体化用温度门槛近似（未解碳化物溶解动力学）；换热系数为文献典型量级，需按现场流量校准。
"""
import numpy as np

from .induction import InductionQuenchSim, BottleNeckGeometry
from .materials import STEEL_45, QUENCHANTS

FREQ_GRID_KHZ = (3.0, 10.0, 30.0, 100.0, 250.0)     # 超音频→高频常用档
MEDIA_ORDER = ("water_spray", "polymer_spray", "water_dip", "oil_dip", "air")


def _microstructure_label(Tpeak_surf, t85_surf):
    """表面组织定性（与 materials.hardness_estimate 同一套 C45 CCT 规则）。"""
    if Tpeak_surf < STEEL_45["Ac1"]:
        return "未变（原始组织）"
    if Tpeak_surf < STEEL_45["Ac3_fast"]:
        part = "部分奥氏体→"
    else:
        part = ""
    if t85_surf <= 1.0:
        return part + "马氏体"
    if t85_surf <= 4.0:
        return part + "马氏体+贝氏体"
    if t85_surf <= 15.0:
        return part + "贝氏体/屈氏体"
    return part + "铁素体+珠光体"


def run_one(wall, freq_khz, t_heat, medium="water_spray", power_kw=6.0, od=22.0):
    """单点仿真，返回一行指标 dict。"""
    s = InductionQuenchSim(geom=BottleNeckGeometry(neck_od=od, wall=wall),
                           freq=freq_khz * 1e3, power_kw=power_kw, nonlinear=True,
                           medium=medium, t_heat=t_heat, t_delay=0.2, t_quench=1.5, t_air=1.0,
                           accel=True)
    res = s.run()
    g = s.g
    r = s.th_mesh_mm.p[0, s.line_nodes]
    depth_out = g.ro - r
    Tp = s.Tpeak[s.line_nodes]
    aus = float(depth_out[Tp >= STEEL_45["Ac3_fast"]].max()) if np.any(Tp >= STEEL_45["Ac3_fast"]) else 0.0
    aus1 = float(depth_out[Tp >= STEEL_45["Ac1"]].max()) if np.any(Tp >= STEEL_45["Ac1"]) else 0.0
    row = dict(
        wall_mm=wall, freq_kHz=freq_khz, t_heat_s=t_heat, medium=medium,
        medium_name=QUENCHANTS[medium]["name"],
        T_surf=float(res["T_surf_end_heat"]), T_inner=float(res["T_inner_end_heat"]), T_max=float(res["T_max"]),
        aus_depth_mm=aus, aus1_depth_mm=aus1, case_depth_mm=float(res["case_depth_mm"]),
        HRC_surf=float(res["HRC_surface"]), through=bool(res["through_hardened"]),
        eff=float(res["efficiency"]), skin_hot_mm=float(res["skin_hot_mm"]),
        I_rms_A=float(res["coil_current_A"]),
        micro=_microstructure_label(res["T_surf_end_heat"], res["t85_surface"]),
        melted=bool(res["melted"]),
    )
    row["verdict"] = verdict(row, wall)
    return row


def verdict(row, wall, target_depth=None):
    """工程判定（经验规则）：
      过烧/过热：表面>980°C 或熔化；欠热：表面<830°C（未完全奥氏体化）；
      层深不当：目标层深给定时偏离 ±50% 以上；薄壁透烧提示；否则 合适。"""
    if row["melted"] or row["T_surf"] > 980:
        return "过热"
    if row["T_surf"] < STEEL_45["Ac3_fast"]:
        return "欠热"
    if target_depth is not None and target_depth > 0:
        if row["case_depth_mm"] < 0.5 * target_depth:
            return "层深不足"
        if row["case_depth_mm"] > 1.5 * target_depth:
            return "层深过深"
    if row["aus_depth_mm"] >= 0.95 * wall:
        return "透烧(薄壁注意)"
    return "合适"


def recommend(rows, wall, target_depth=None):
    """从扫描行中给出推荐（经验规则，均为仿真结果）：
      1) 只要 合适/层深合格 的行；2) 目标层深给定时 |层深-目标| 最小优先；
      3) 效率优先；4) 同条件取频率更高者（加热快、变形小）。
      返回 (推荐行 或 None, 理由 str)。"""
    ok = [r for r in rows if r["verdict"] in ("合适",)]
    if not ok:
        cand = [r for r in rows if r["verdict"] not in ("过热",) and r["T_surf"] >= STEEL_45["Ac3_fast"]]
        if not cand:
            return None, "所有网格点都未达完全奥氏体化或过热：加大功率/延长时间或降低频率后重扫。"
        return None, "没有完全合适的组合；可试表中接近的行并微调时间。"
    if target_depth:
        ok.sort(key=lambda r: (abs(r["case_depth_mm"] - target_depth), -r["eff"], -r["freq_kHz"]))
    else:
        ok.sort(key=lambda r: (-r["eff"], -r["freq_kHz"]))
    best = ok[0]
    reason = (f"{best['freq_kHz']:.0f} kHz / {best['t_heat_s']:.1f} s / {best['medium_name']}："
              f"热态趋肤 {best['skin_hot_mm']:.1f} mm，奥氏体化深 {best['aus_depth_mm']:.1f} mm，"
              f"淬硬层 {best['case_depth_mm']:.1f} mm，表面 {best['HRC_surf']:.0f} HRC（{best['micro']}），"
              f"电效率 {best['eff']*100:.0f}%。规律：层深≈(1~3)×热态趋肤深度；"
              f"壁 {wall:.0f} mm " + ("宜透热，选较低频。" if wall <= 3 else "宜表面淬火，趋肤深度应≤目标层深。"))
    return best, reason


def scan_freq_time(wall, medium="water_spray", power_kw=6.0, freqs=None, times=None,
                   target_depth=None, od=22.0, progress=None):
    """主扫描：频率×加热时间网格（介质固定）。times 缺省按壁厚²比例给 3 档。"""
    freqs = tuple(freqs) if freqs else FREQ_GRID_KHZ
    if times is None:
        base = (0.7, 1.5, 3.0)
        scale = (wall / 3.0) ** 2
        times = tuple(round(b * scale, 2) for b in base)
    rows = []
    n = len(freqs) * len(times)
    i = 0
    for f in freqs:
        for t in times:
            rows.append(run_one(wall, f, t, medium=medium, power_kw=power_kw, od=od))
            i += 1
            if progress:
                progress(i / n)
    best, reason = recommend(rows, wall, target_depth)
    return dict(wall=wall, medium=medium, rows=rows, best=best, reason=reason, times=times, freqs=freqs)


def scan_media(wall, freq_khz, t_heat, media=None, power_kw=6.0, od=22.0, progress=None):
    """介质对比：固定频率/时间，扫各冷却介质。"""
    media = tuple(media) if media else MEDIA_ORDER
    rows = []
    for i, m in enumerate(media):
        rows.append(run_one(wall, freq_khz, t_heat, medium=m, power_kw=power_kw, od=od))
        if progress:
            progress((i + 1) / len(media))
    return dict(wall=wall, freq_kHz=freq_khz, t_heat=t_heat, rows=rows)
