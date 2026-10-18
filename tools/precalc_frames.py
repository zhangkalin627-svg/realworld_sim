"""L0 预计算帧包生成器：输出 rw-frames/1 JSON（docs/动画帧流契约_rw-frames-1_2026-10-18.md）。
用法: python3 tools/precalc_frames.py [freq_kHz ...]  # 默认 20 -> default.json，其余 fXXX.json
一切输出为仿真结果，页面须带"仿真结果"徽章（展示规范 v1）。"""
import json, sys, time
from pathlib import Path
import numpy as np
from matplotlib.tri import Triangulation, LinearTriInterpolator
from rw.induction import InductionQuenchSim, BottleNeckGeometry, find_power

DEFAULT_GEOM = dict(neck_od=22, wall=3, neck_len=50, zone=20, coil_id=30, coil_h=25)
OUT = Path(__file__).resolve().parent.parent / "web" / "static" / "precalc"

def build(freq_khz, target=900.0, nr=30, nz=100, record_every=0.1):
    g = BottleNeckGeometry(**DEFAULT_GEOM)
    kw = dict(geom=g, freq=freq_khz * 1e3, t_heat=3.0, t_quench=3.0, t_air=6.0,
              medium="water_spray", mur_cold=20.0, accel=True)
    t0 = time.time()
    power = find_power(target, **kw)
    sim = InductionQuenchSim(power_kw=power, **kw)
    r = sim.run(record_every=record_every)
    p, tt = sim.th_mesh_mm.p, sim.th_mesh_mm.t
    tri = Triangulation(p[0], p[1], tt.T)
    rr = np.linspace(0, float(p[0].max()), nr)
    zz = np.linspace(0, float(p[1].max()), nz)
    RR, ZZ = np.meshgrid(rr, zz)
    # 帧抽取：加热段全留，淬火段 1/2，空冷段 1/5
    t_h, t_q = kw["t_heat"], kw["t_heat"] + kw["t_quench"]
    frames = []
    for t, T in sim.frames:
        keep = (t <= t_h) or (t <= t_q and len(frames) % 2 == 0) or (t > t_q and len(frames) % 5 == 0)
        if not keep:
            continue
        Ti = np.ma.filled(LinearTriInterpolator(tri, T)(RR, ZZ), np.nan)
        Ti = np.round(Ti, 1)
        frames.append({"t": round(float(t), 3),
                       "T_C": [[None if np.isnan(v) else float(v) for v in row] for row in Ti]})
    h = sim.hist
    idx = np.linspace(0, len(h["t"]) - 1, min(120, len(h["t"]))).astype(int)
    monitor = [{"t": round(float(h["t"][i]), 3), "T_surf": round(float(h["Tsurf"][i]), 1),
                "T_inner": round(float(h["Tin"][i]), 1)} for i in idx]
    pack = {
        "schema": "rw-frames/1", "mode": "precalc",
        "params": {"freq_kHz": freq_khz, "power_kW": round(float(r["power_kW"]), 2),
                   "t_heat": kw["t_heat"], "t_quench": kw["t_quench"], "t_air": kw["t_air"],
                   "medium": kw["medium"], "mur_cold": kw["mur_cold"], "nonlinear": False,
                   "target_C": target, "geom": DEFAULT_GEOM},
        "grid": {"nr": nr, "nz": nz, "r_mm": [round(float(x), 3) for x in rr],
                 "z_mm": [round(float(x), 3) for x in zz]},
        "frames": frames, "monitor": monitor,
        "coil": {"z0_mm": float(g.coil_z0), "z1_mm": float(g.coil_z1), "r_in_mm": float(g.coil_r0)},
        "result": {"T_max": round(float(r["T_max"]), 1), "t85_surface_s": round(float(r["t85_surface"]), 2),
                   "HRC_surface": round(float(r["HRC_surface"]), 0), "case_depth_mm": round(float(r["case_depth_mm"]), 2),
                   "I_coil_A_rms": round(float(r["coil_current_A"]), 0)},
        "meta": {"badge": "仿真结果", "solver": "fem-fast", "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                 "gen_seconds": round(time.time() - t0, 1)},
    }
    return pack

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    freqs = [float(x) for x in sys.argv[1:]] or [20.0]
    for f in freqs:
        pack = build(f)
        name = "default.json" if f == 20.0 else f"f{int(f):03d}.json"
        path = OUT / name
        path.write_text(json.dumps(pack, separators=(",", ":")), encoding="utf-8")
        print(f"{name}: {path.stat().st_size/1e6:.2f} MB, {len(pack['frames'])} 帧, {pack['meta']['gen_seconds']}s")
