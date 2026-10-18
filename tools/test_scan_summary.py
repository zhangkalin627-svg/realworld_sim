"""tools/scan_summary.py 的回归测试（外部 AI 生成的工具经我方评审后采用；本测试由我方编写，含对照参考实现）。程序运行结果，非实测。"""
import sys, os, csv, io, time, random, tempfile, subprocess, importlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collections import defaultdict
H = "wall,freq,t_heat,medium,power,T_surf,T_inner,aus_depth_mm,case_depth_mm,HRC_surf,verdict,micro,secs,err".split(",")
def ref(path):
    rows = errs = 0; vd = defaultdict(int); pn = defaultdict(int); po = defaultdict(int); cd = defaultdict(list)
    for r in csv.DictReader(open(path, encoding="utf-8")):
        rows += 1
        if (r["err"] or "").strip(): errs += 1; continue
        vd[r["verdict"]] += 1
        w, f = float(r["wall"]), float(r["freq"]); pn[(w, f)] += 1; po[(w, f)] += (r["verdict"] == "合适")
        if r["case_depth_mm"].strip(): cd[w].append(float(r["case_depth_mm"]))
    bw = {}
    for w in sorted({k[0] for k in pn}):
        fs = sorted([(-po[k] / pn[k], k[1]) for k in pn if k[0] == w])
        d = cd.get(w)
        bw[w] = {"n": sum(pn[k] for k in pn if k[0] == w), "ok": sum(po[k] for k in pn if k[0] == w), "best_freqs": [f for _, f in fs[:3]], "case_depth_range": (min(d), max(d)) if d else None}
    return {"rows": rows, "errors": errs, "verdicts": dict(vd), "by_wall": bw}
def mk(rows, name=None):
    p = tempfile.mktemp(suffix=".csv"); 
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(H)
        for r in rows: w.writerow(r)
    return p
def gen(n, seed=0):
    R = random.Random(seed); out = []
    for _ in range(n):
        wall = R.choice([2, 3, 4, 6, 10]); fr = R.choice([2, 5, 10, 20, 30, 100]); v = R.choice(["合适", "合适", "偏浅", "过烧", "偏深"])
        cdp = "" if R.random() < .2 else round(R.uniform(.2, 4), 3); err = "boom" if R.random() < .05 else ""
        out.append([wall, fr, R.choice([.7, 2.6]), "water_spray", 6.0, 800, 600, 1.2, cdp, 55, v, "M", 1.0, err])
    return out
def run(modname):
    m = importlib.import_module(modname); res = []
    def chk(n, c, i=""): res.append((n, bool(c), i))
    def safe(n, fn):
        try: fn()
        except Exception as e: chk(n, False, repr(e)[:90])
    def eq(n, rows):
        p = mk(rows)
        try: s = m.summarize(p); r = ref(p); 
        finally: pass
        ok = s["rows"] == r["rows"] and s["errors"] == r["errors"] and s["verdicts"] == r["verdicts"] and set(s["by_wall"]) == set(r["by_wall"])
        bad = []
        if ok:
            for w in r["by_wall"]:
                a, b = s["by_wall"][w], r["by_wall"][w]
                for k in ("n", "ok", "best_freqs"):
                    if a[k] != b[k]: bad.append((w, k, a[k], b[k]))
                ra, rb = a["case_depth_range"], b["case_depth_range"]
                if (ra is None) != (rb is None) or (ra and (abs(ra[0]-rb[0]) > 1e-9 or abs(ra[1]-rb[1]) > 1e-9)): bad.append((w, "range", ra, rb))
        chk(n, ok and not bad, str(bad[:2]) if bad else "")
        return p
    safe("1 随机5000行 对照参考实现", lambda: eq("1 随机5000行 对照参考实现", gen(5000)))
    safe("2 随机5000行 另一种子", lambda: eq("2 随机5000行 另一种子", gen(5000, 7)))
    safe("3 排序:10 在 2 之后(数值排序)", lambda: chk("3 排序:10 在 2 之后(数值排序)", list(m.summarize(mk(gen(300)))["by_wall"].keys()) == sorted(m.summarize(mk(gen(300)))["by_wall"].keys())))
    def t4():
        p = mk([[3, 10, 1, "w", 6, 1, 1, 1, "", 1, "合适", "", 1, ""], [3, 10, 1, "w", 6, 1, 1, 1, "", 1, "偏浅", "", 1, ""], [3, 20, 1, "w", 6, 1, 1, 1, "", 1, "合适", "", 1, ""], [3, 20, 1, "w", 6, 1, 1, 1, "", 1, "合适", "", 1, ""], [3, 5, 1, "w", 6, 1, 1, 1, "", 1, "偏浅", "", 1, ""]])
        s = m.summarize(p); chk("4 同占比按freq升序+深度全空为None", s["by_wall"][3.0]["best_freqs"] == [20.0, 10.0, 5.0] and s["by_wall"][3.0]["case_depth_range"] is None, str(s["by_wall"]))
    safe("4", t4)
    def t5():
        p = mk([[3, 10, 1, "w", 6, 1, 1, 1, 1, 1, "新判据XYZ", "", 1, ""], [3, 10, 1, "w", 6, 1, 1, 1, 1, 1, "合适", "", 1, "fail"]]); s = m.summarize(p)
        chk("5 新verdict被统计、err行排除并计数", s["verdicts"] == {"新判据XYZ": 1} and s["errors"] == 1 and s["rows"] == 2, str(s["verdicts"]))
    safe("5", t5)
    def t6():
        p = tempfile.mktemp(suffix=".csv"); open(p, "w", encoding="utf-8").write(",".join(H) + "\n3,10,1,w,6,1,1,1,1,1,合适\n3,10\n\n3,20,1,w,6,1,1,1,2,1,偏浅,,,,\n")
        s = m.summarize(p); chk("6 短行/空行不崩", s["rows"] >= 2, "rows=%s verdicts=%s" % (s["rows"], s["verdicts"]))
    safe("6 短行/空行不崩", t6)
    def t7():
        p = tempfile.mktemp(suffix=".csv"); open(p, "w", encoding="utf-8").write(",".join(H) + "\n"); s = m.summarize(p); chk("7 只有表头", s["rows"] == 0 and s["by_wall"] == {} and s["verdicts"] == {})
        md = m.to_markdown(s); chk("7b 空汇总能出Markdown", "仿真结果" in md)
    safe("7", t7)
    def t8():
        p = mk(gen(200)); md = m.to_markdown(m.summarize(p))
        chk("8 Markdown含标题/表头/声明", all(x in md for x in ["扫描结果汇总（仿真结果）", "壁厚(mm)", "有效点数", "推荐频率(kHz)", "淬硬层深范围(mm)", "以上均为仿真结果，非实测"]), "")
    safe("8", t8)
    def t9():
        p = mk(gen(200)); o = tempfile.mktemp(suffix=".md"); f = m.__file__
        r = subprocess.run([sys.executable, f, p, "--out", o], capture_output=True, text=True); chk("9 CLI --out", r.returncode == 0 and os.path.exists(o) and "仿真结果" in open(o, encoding="utf-8").read(), r.stderr[:80])
        r = subprocess.run([sys.executable, f, "/nonexistent/x.csv"], capture_output=True, text=True); chk("9b CLI 文件不存在退出码2且中文报错", r.returncode == 2 and any('\u4e00' <= c <= '\u9fff' for c in r.stdout + r.stderr), "rc=%s %s" % (r.returncode, (r.stdout + r.stderr)[:60]))
        r = subprocess.run([sys.executable, f, p], capture_output=True, text=True); chk("9c CLI 无 --out 打印到屏幕", r.returncode == 0 and "仿真结果" in r.stdout)
    safe("9", t9)
    def t10():
        p = mk(gen(200000, 3)); t = time.time(); s = m.summarize(p); d = time.time() - t; chk("10 20万行速度", d < 10, "%.2fs" % d)
    safe("10", t10)
    def t11():
        p = tempfile.mktemp(suffix=".csv"); open(p, "w", encoding="utf-8-sig").write(",".join(H) + "\n3,10,1,w,6,1,1,1,1,1,合适,M,1,\n"); s = m.summarize(p); chk("11 带BOM的CSV", s["rows"] == 1 and 3.0 in s["by_wall"])
    safe("11", t11)
    return res
bad = 0
for n, ok, i in run("scan_summary"):
    print(("PASS " if ok else "FAIL ") + n, i); bad += (not ok)
print("ALL PASS" if not bad else "SOME FAIL"); sys.exit(1 if bad else 0)
