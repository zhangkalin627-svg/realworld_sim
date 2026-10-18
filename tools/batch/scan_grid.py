"""冶金扫描大网格（全部为仿真结果，非实测）。每点 = 一次 rw.scan.run_one；结果追加到 tools/batch/out/scan_grid.csv，
重启时跳过已有的点（断点续跑）。网格可改：WALLS/FREQS/THEATS/MEDIA。"""
import csv, os, sys, itertools, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from rw.scan import run_one
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "scan_grid.csv")
WALLS = [2.0, 3.0, 4.0, 6.0, 8.0]; FREQS = [5, 10, 20, 30, 50, 100, 200]
THEATS = [0.7, 1.5, 2.6, 4.0]; MEDIA = ["water_spray", "polymer_spray", "oil_dip"]; POWERS = [6.0, 15.0]
os.makedirs(os.path.dirname(OUT), exist_ok=True)
done = set()
if os.path.exists(OUT):
    with open(OUT) as f:
        for r in csv.DictReader(f): done.add((r["wall"], r["freq"], r["t_heat"], r["medium"], r["power"]))
cols = ["wall", "freq", "t_heat", "medium", "power", "T_surf", "T_inner", "aus_depth_mm", "case_depth_mm", "HRC_surf", "verdict", "micro", "secs", "err"]
new = not os.path.exists(OUT)
with open(OUT, "a", newline="") as f:
    w = csv.DictWriter(f, cols)
    if new: w.writeheader()
    for wall, fr, th, med, pw in itertools.product(WALLS, FREQS, THEATS, MEDIA, POWERS):
        key = (str(wall), str(fr), str(th), med, str(pw))
        if key in done: continue
        t0 = time.time(); row = dict(wall=wall, freq=fr, t_heat=th, medium=med, power=pw, err="")
        try:
            r = run_one(wall, fr, th, medium=med, power_kw=pw)
            for k in cols[5:12]:
                v = r.get(k); row[k] = round(v, 3) if isinstance(v, float) else v
        except Exception as e:
            row["err"] = repr(e)[:120]
        row["secs"] = round(time.time() - t0, 1)
        w.writerow(row); f.flush()
print("scan_grid 完成或超时中断（可重启续跑）")
