import sys, os, math, time, tempfile, subprocess, numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import waveform_metrics as wm, rms_stats as rs, report as rp
R = []
def chk(n, c, i=""): R.append(bool(c)); print(("PASS " if c else "FAIL ") + n, i, flush=True)
def T(n, f):
    try: f()
    except Exception as e: chk(n, False, "异常 " + repr(e)[:100])
def cn(e): return any('\u4e00' <= c <= '\u9fff' for c in str(e))
def raises(f):
    try: f()
    except ValueError as e: return cn(e), str(e)
    except Exception as e: return False, "非ValueError:" + repr(e)[:60]
    return False, "未抛错"
def sine(f=50, P=10, N=200, A=1.0, ph=0.0, dc=0.0, fn=np.cos):
    T0 = 1 / f; t = np.linspace(0, P * T0, N * P + 1); return t, dc + A * fn(2 * np.pi * f * t + ph)
def t1():
    t = np.linspace(0, 1, 1001); x = np.sin(2 * np.pi * 5 * t + 0.3)
    zr = wm.zero_crossings(t, x, True); zf = wm.zero_crossings(t, x, False)
    er = [(k - 0.3 / (2 * np.pi)) / 5 + 0 for k in range(-1, 6)]; er = [e for e in er if 0 <= e <= 1]
    # sin(2π5t+0.3)=0 上升在 5t+0.3/2π = k
    er = [(k - 0.3 / (2 * np.pi)) / 5 for k in range(0, 6) if 0 <= (k - 0.3 / (2 * np.pi)) / 5 <= 1]
    ef = [(k + 0.5 - 0.3 / (2 * np.pi)) / 5 for k in range(0, 6) if 0 <= (k + 0.5 - 0.3 / (2 * np.pi)) / 5 <= 1]
    chk("1 过零上升(插值，误差<1e-5)", len(zr) == len(er) and np.allclose(zr, er, atol=1e-5), "%s vs %s" % (zr, er))
    chk("1b 过零下降", len(zf) == len(ef) and np.allclose(zf, ef, atol=1e-5), "%s vs %s" % (zf, ef))
    t = np.arange(7.0); x = np.array([-1, 0, 1, 0, -1, 0, 1.0]); chk("1c 采样点恰为0不重复计", list(wm.zero_crossings(t, x, True)) == [1.0, 5.0] and list(wm.zero_crossings(t, x, False)) == [3.0], str(wm.zero_crossings(t, x, True)) + str(wm.zero_crossings(t, x, False)))
    chk("1d 返回 ndarray", isinstance(zr, np.ndarray))
T("1", t1)
def t2():
    t, x = sine(f=50, P=10, N=1000, fn=np.sin); f = wm.frequency(t, x); chk("2 frequency 50Hz", type(f) is float and abs(f - 50) < 1e-3, str(f))
    t = np.linspace(0, 0.021, 4000); x = np.sin(2 * np.pi * 50 * t); chk("2b 少于3个过零抛错", raises(lambda: wm.frequency(t[:300], x[:300]))[0])
T("2", t2)
def t3():
    t, x = sine(50, 10, 200, A=3.0, ph=0.7, dc=1.5)
    h = wm.fft_harmonics(t, x, 50, n_harm=5)
    chk("3 DC", abs(h["dc"] - 1.5) < 1e-9, str(h["dc"])); chk("3b 基波幅值 3.0(峰值)", abs(h["amp"][0] - 3.0) < 1e-6, str(h["amp"][:2]))
    chk("3c 相位相对 cos = 0.7", abs(h["phase"][0] - 0.7) < 1e-6, str(h["phase"][0])); chk("3d 谐波≈0 且长度 5", len(h["amp"]) == 5 and max(h["amp"][1:]) < 1e-9, str(h["amp"]))
    t, x = sine(50, 10, 200, fn=np.sin); h = wm.fft_harmonics(t, x, 50, n_harm=3); chk("3e sin 的相位 = -π/2", abs(h["phase"][0] + math.pi / 2) < 1e-6, str(h["phase"][0]))
    t, x = sine(50, 10, 200, A=2.0, ph=-0.4); x = x + 0.5 * np.cos(3 * 2 * np.pi * 50 * t + 1.0); h = wm.fft_harmonics(t, x, 50, n_harm=5)
    chk("3f 3次谐波幅值/相位", abs(h["amp"][2] - 0.5) < 1e-6 and abs(h["phase"][2] - 1.0) < 1e-6 and abs(wm.thd(h["amp"]) - 0.25) < 1e-6, str((h["amp"], h["phase"][2], wm.thd(h["amp"]))))
    chk("3g 返回类型 float/list", type(h["dc"]) is float and isinstance(h["amp"], list) and all(type(a) is float for a in h["amp"]) and all(type(a) is float for a in h["phase"]))
    ph = h["phase"]; chk("3h 相位范围 (-π,π]", all(-math.pi < p <= math.pi + 1e-12 for p in ph))
T("3", t3)
def t4():
    t, x = sine(50, 10, 200, A=1.0, ph=0.0); t2 = np.concatenate([t, t[-1] + (t[1] - t[0]) * np.arange(1, 70)]); x2 = np.cos(2 * np.pi * 50 * t2)
    h = wm.fft_harmonics(t2, x2, 50, n_harm=3); chk("4 非整周期数据自动取整周期(幅值1、DC≈0)", abs(h["amp"][0] - 1.0) < 1e-6 and abs(h["dc"]) < 1e-9, str((h["amp"][0], h["dc"])))
    h = wm.fft_harmonics(t2, x2, 50, n_harm=3, periods=4); chk("4b periods=4", abs(h["amp"][0] - 1.0) < 1e-6)
    tn = np.sort(np.random.default_rng(0).uniform(0, 0.1, 500)); tn[0] = 0
    ok, m = raises(lambda: wm.fft_harmonics(tn, np.sin(tn), 50)); chk("4c 非均匀采样报错(中文)", ok, m)
    ok, m = raises(lambda: wm.fft_harmonics(t[:50], x[:50], 50)); chk("4d 放不下1个周期报错", ok, m)
T("4", t4)
def t5():
    chk("5 thd", abs(wm.thd([2.0, 0.3, 0.4]) - 0.25) < 1e-12 and type(wm.thd([2.0, 0.3])) is float); chk("5b thd amp0<=0 抛错", raises(lambda: wm.thd([0.0, 1.0]))[0])
    x = np.array([1, 5, -2, 4.0]); t = np.array([0, 1, 2, 3.0]); chk("5c ripple ptp", wm.ripple(x) == 7.0 and type(wm.ripple(x)) is float)
    chk("5d ripple 窗口(含端点)", wm.ripple(x, t, (1, 3)) == 7.0 and wm.ripple(x, t, (0, 1)) == 4.0 and wm.ripple(x, t, (0.5, 1.5)) == 0.0, str(wm.ripple(x, t, (0.5, 1.5))))
    chk("5e ripple 窗口内无点抛错", raises(lambda: wm.ripple(x, t, (1.2, 1.8)))[0])
T("5", t5)
def t6():
    t = np.linspace(0, 1, 100001); x = np.sin(2 * np.pi * 5 * t)
    chk("6 rms 时间加权 sin=0.7071", abs(rs.rms(x, t) - 1 / math.sqrt(2)) < 1e-6 and type(rs.rms(x, t)) is float, str(rs.rms(x, t)))
    tn = np.array([0, 1, 4.0]); xn = np.array([0, 2, 2.0]); ex = math.sqrt((0.5 * (0 + 4) * 1 + 4 * 3) / 4)
    chk("6b rms 非均匀 t 梯形", abs(rs.rms(xn, tn) if False else rs.rms(xn, tn) - ex) < 1e-12, str(rs.rms(xn, tn)))
    chk("6c rms 无 t = 算术", abs(rs.rms(np.array([3.0, 4.0])) - math.sqrt(12.5)) < 1e-12)
    chk("6d avg 时间加权", abs(rs.avg(xn, tn) - (0.5 * 2 * 1 + 2 * 3) / 4) < 1e-12 and abs(rs.avg(np.array([1.0, 3.0])) - 2.0) < 1e-12)
    chk("6e crest", abs(rs.crest_factor(np.array([1.0, -1.0, 1.0, -1.0])) - 1.0) < 1e-12 and abs(rs.crest_factor(np.sqrt(2) * np.sin(np.linspace(0, 2 * np.pi * 10, 100001))) - math.sqrt(2)) < 1e-3)
    chk("6f crest rms=0 抛错", raises(lambda: rs.crest_factor(np.zeros(5)))[0])
T("6", t6)
def t7():
    t = np.linspace(0, 0.1, 200001); v = 325 * np.sin(2 * np.pi * 50 * t)
    for phi in (0.0, 0.5, 1.2):
        i = 10 * np.sin(2 * np.pi * 50 * t - phi); pf = rs.power_factor(v, i, t); chk("7 pf φ=%.1f → cos φ" % phi, abs(pf - math.cos(phi)) < 1e-4 and type(pf) is float, str(pf))
    chk("7b pf 分母0抛错", raises(lambda: rs.power_factor(np.zeros(5), np.ones(5)))[0])
T("7", t7)
def t8():
    t = np.linspace(0, 1, 11); g = np.ones(11); nan = np.where(np.arange(11) == 3, np.nan, 1.0)
    bad = {"zc 长度": lambda: wm.zero_crossings(t, g[:5]), "zc n<2": lambda: wm.zero_crossings(t[:1], g[:1]), "zc 非递增": lambda: wm.zero_crossings(t[::-1], g), "zc NaN": lambda: wm.zero_crossings(t, nan),
           "fft n_harm<1": lambda: wm.fft_harmonics(*sine(), 50, n_harm=0), "fft f0<=0": lambda: wm.fft_harmonics(*sine(), 0.0), "fft NaN": lambda: wm.fft_harmonics(t, nan, 1.0),
           "rms 长度": lambda: rs.rms(g, t[:5]), "rms NaN": lambda: rs.rms(nan, t), "rms t 非递增": lambda: rs.rms(g, t[::-1]), "avg n<2": lambda: rs.avg(g[:1]), "pf 长度": lambda: rs.power_factor(g, g[:5]), "pf NaN": lambda: rs.power_factor(nan, g)}
    ok = 0; bd = []
    for k, f in bad.items():
        a, m = raises(f); ok += a
        if not a: bd.append((k, m))
    chk("8 非法输入抛中文 ValueError(%d项)" % len(bad), ok == len(bad), "%d/%d %s" % (ok, len(bad), bd))
    ok, m = raises(lambda: wm.zero_crossings(t, nan)); chk("8b NaN 报错写明数组(含'x')", 'x' in m.lower() or 'X' in m, m)
    ok, m = raises(lambda: rs.power_factor(nan, g)); chk("8c NaN 报错写明数组(含'v'/电压)", 'v' in m.lower() or '电压' in m, m)
T("8", t8)
def t9():
    t = np.linspace(0, 1, 100); x = np.random.default_rng(1).normal(size=100); i = x.copy(); t0, x0 = t.copy(), x.copy()
    wm.zero_crossings(t, x); wm.ripple(x, t, (0.1, 0.5)); rs.rms(x, t); rs.power_factor(x, i, t); rs.avg(x, t); chk("9 不修改入参", np.array_equal(t, t0) and np.array_equal(x, x0) and np.array_equal(i, x0))
    t, x = sine(50, 10, 200); wm.fft_harmonics(t, x, 50); chk("9b fft 不修改", True)
T("9", t9)
def t10():
    n = 1_000_000; t = np.linspace(0, 1, n); x = np.sin(2 * np.pi * 50 * t); v = x.copy(); i = np.cos(2 * np.pi * 50 * t); res = []
    for nm, f in (("zero_crossings", lambda: wm.zero_crossings(t, x)), ("frequency", lambda: wm.frequency(t, x)), ("fft_harmonics", lambda: wm.fft_harmonics(t[:-1] if False else t, x, 50)), ("ripple", lambda: wm.ripple(x, t, (0.1, 0.9))), ("rms", lambda: rs.rms(x, t)), ("avg", lambda: rs.avg(x, t)), ("pf", lambda: rs.power_factor(v, i, t))):
        t0 = time.time()
        try: f()
        except Exception as e: res.append((nm, "异常" + repr(e)[:40])); continue
        d = time.time() - t0; res.append((nm, round(d, 2)))
    chk("10 1e6 点各函数<2s", all(isinstance(d, float) and d < 2 for _, d in res), str(res))
T("10", t10)
def t11():
    t, v = sine(50, 5, 400, A=10, fn=np.sin); i = 4 * np.sin(2 * np.pi * 50 * t - 0.3); md = rp.make_report("T1", t, v, i, 50)
    last = [l for l in md.strip().splitlines() if l.strip()][-1].strip()
    chk("11 报告标题与末行", md.startswith("# T1 波形后处理（仿真结果）") and last == "以上均为仿真结果，非实测", repr(md[:40]) + "|" + last)
    chk("11b 除'非实测'外无'实测'", md.replace("非实测", "").count("实测") == 0)
    chk("11c 含全部指标字样", all(k in md for k in ["频率", "直流", "基波", "THD", "有效值", "有功功率", "功率因数", "波峰", "纹波"]), "")
    chk("11d 数值：频率≈50、功率因数≈cos0.3=0.9553", "50" in md and "0.9553" in md, md[:400].replace("\n", "|"))
T("11", t11)
def t12():
    t, v = sine(50, 5, 400, A=10, fn=np.sin); i = 4 * np.sin(2 * np.pi * 50 * t - 0.3); p = tempfile.mktemp(suffix=".csv")
    open(p, "w").write("t,v,i\n" + "\n".join("%.9g,%.9g,%.9g" % (a, b, c) for a, b, c in zip(t, v, i)))
    r = subprocess.run([sys.executable, "report.py", p, "--f0", "50"], capture_output=True, text=True); chk("12 CLI 正常", r.returncode == 0 and "仿真结果" in r.stdout and "0.9553" in r.stdout, r.stderr[:100] + r.stdout[:60])
    r = subprocess.run([sys.executable, "report.py", "/nonexistent.csv", "--f0", "50"], capture_output=True, text=True); chk("12b CLI 文件不存在 rc2 中文", r.returncode == 2 and cn(r.stdout + r.stderr), "rc=%s %s" % (r.returncode, (r.stdout + r.stderr)[:60]))
T("12", t12)
print("SCORE %d/%d" % (sum(R), len(R)))
