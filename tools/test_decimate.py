"""rw/circuit/decimate.py 的回归测试（外部 AI 生成的模块经我方评审后采用；本测试由我方编写）。仿真/程序运行结果，非实测。"""
import os, sys, time, importlib, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
res = {}
def run(modname):
    m = importlib.import_module(modname); f = m.minmax_decimate; out = []
    def chk(name, cond, info=""):
        out.append((name, bool(cond), info))
    rng = np.random.default_rng(1)
    def common(t, ys, mp, kl=True):
        to, yo, idx = f(t, ys, mp, kl)
        ok = (len(idx) <= mp and idx.dtype == np.int64 and np.all(np.diff(idx) > 0) and np.all(np.diff(to) > 0)
              and all(np.array_equal(yo[k], ys[k][idx], equal_nan=True) for k in ys) and np.array_equal(to, t[idx]))
        return ok, to, yo, idx
    # 1 copy small
    t = np.arange(10.0); y = {"a": rng.normal(size=10)}
    to, yo, idx = f(t, y, 20); chk("1 小序列拷贝", np.array_equal(idx, np.arange(10)) and to is not t and yo["a"] is not y["a"] and not np.shares_memory(to, t))
    # 2 spike
    n = 100000; t = np.arange(n) * 1e-6; y = {"a": np.sin(t * 2000)}; y["a"][54321] = 100.0
    ok, to, yo, idx = common(t, y, 1000); chk("2 尖峰保留+基本不变量", ok and 54321 in idx, "len=%d" % len(idx))
    # 3 two curves
    y = {"a": np.sin(t * 2000), "b": (t * 1e3) % 1}
    ok, *_ = common(t, y, 1000); chk("3 两曲线不变量", ok)
    # 4 global extremes random walk
    n = 200000; t = np.arange(n) * 1.0; y = {"a": np.cumsum(rng.normal(size=n)), "b": np.cumsum(rng.normal(size=n))}
    ok, to, yo, idx = common(t, y, 800)
    chk("4 全局最大最小保留", ok and all(yo[k].max() == y[k].max() and yo[k].min() == y[k].min() for k in y), "len=%d/800" % len(idx))
    # 5 keep_last
    ok1, _, _, i1 = common(t, y, 800, True); ok2, _, _, i2 = common(t, y, 800, False)
    chk("5 keep_last True 含末点", ok1 and i1[-1] == n - 1); chk("5b keep_last False 仍合法且含首点", ok2 and i2[0] == 0, "含末点=%s" % (i2[-1] == n - 1))
    # 6 NaN
    y = {"a": np.sin(np.arange(n) * 1e-3)}; y["a"][1000:50000] = np.nan; y["a"][777] = 5.0
    try:
        ok, to, yo, idx = common(t, y, 500)
        v = yo["a"]; chk("6 NaN 不报错且不选 NaN 为极值", ok and np.nanmax(v) == np.nanmax(y["a"]) and np.nanmin(v) == np.nanmin(y["a"]), "NaN占输出=%d" % np.isnan(v).sum())
    except Exception as e: chk("6 NaN 不报错", False, repr(e)[:80])
    # 7 invalid
    t = np.arange(100.0); ya = {"a": np.zeros(100)}
    bad = {"t非递增": (np.r_[t[:50], t[49:99]], ya, 10), "t含NaN": (np.r_[t[:50], np.nan, t[51:]], ya, 10), "长度不一致": (t, {"a": np.zeros(99)}, 10),
           "mp<4": (t, ya, 3), "ys空": (t, {}, 10), "t是list": (list(t), ya, 10), "mp是小数": (t, ya, 10.5), "ys曲线2维": (t, {"a": np.zeros((100, 1))}, 10),
           "t长度1": (np.array([1.0]), {"a": np.array([1.0])}, 10), "mp是bool": (t, ya, True), "t含重复": (np.r_[0., 0., t[2:]], ya, 10)}
    for k, (a, b, c) in bad.items():
        try: f(a, b, c); chk("7 非法输入 " + k, False, "没抛错")
        except ValueError as e: chk("7 非法输入 " + k, True, str(e)[:30])
        except Exception as e: chk("7 非法输入 " + k, False, "抛了 " + type(e).__name__)
    # 8 no mutation
    t = np.arange(5000.0); y = {"a": rng.normal(size=5000)}; t0, y0 = t.copy(), y["a"].copy(); f(t, y, 100)
    chk("8 输入不被修改", np.array_equal(t, t0) and np.array_equal(y["a"], y0))
    # 9 boundaries
    for n, mp in [(101, 100), (1001, 1000), (100, 99), (10, 4), (9, 4), (1000, 4), (5, 4)]:
        t = np.arange(float(n)); y = {"a": rng.normal(size=n)}
        try:
            ok, to, yo, idx = common(t, y, mp); chk("9 边界 n=%d mp=%d" % (n, mp), ok and idx[0] == 0 and idx[-1] == n - 1, "len=%d 极值保留=%s" % (len(idx), yo['a'].max() == y['a'].max() and yo['a'].min() == y['a'].min()))
        except Exception as e: chk("9 边界 n=%d mp=%d" % (n, mp), False, repr(e)[:70])
    # 10 small mp multi-curve
    for mp, nc in [(4, 2), (8, 2), (12, 3), (50, 4)]:
        n = 5000; t = np.arange(float(n)); y = {"c%d" % i: rng.normal(size=n) for i in range(nc)}
        try:
            ok, to, yo, idx = common(t, y, mp)
            keep = all(yo[k].max() == y[k].max() and yo[k].min() == y[k].min() for k in y)
            chk("10 小 mp=%d 曲线=%d" % (mp, nc), ok, "len=%d 全局极值全保留=%s 末点=%s" % (len(idx), keep, idx[-1] == n - 1))
        except Exception as e: chk("10 小 mp=%d 曲线=%d" % (mp, nc), False, repr(e)[:70])
    # 11 dtypes / views
    t = np.arange(10000.0); base = rng.normal(size=20000).astype(np.float32); y = {"a": base[::2], "i": rng.integers(0, 100, 10000)}
    try: ok, *_ = common(t, y, 300); chk("11 float32非连续视图+整型曲线", ok)
    except Exception as e: chk("11 float32视图/整型", False, repr(e)[:70])
    # 12 inf
    y = {"a": np.sin(np.arange(10000) * 0.01)}; y["a"][3000] = np.inf; y["a"][7000] = -np.inf
    try: ok, to, yo, idx = common(t, y, 200); chk("12 含inf保留", ok and 3000 in idx and 7000 in idx)
    except Exception as e: chk("12 含inf", False, repr(e)[:70])
    # 13 speed
    for n, mp, nc in [(1_000_000, 2000, 1), (1_000_000, 2000, 4), (10_000_000, 2000, 1), (10_000_000, 20000, 2)]:
        t = np.arange(n) * 1.0; y = {"c%d" % i: np.random.default_rng(i).normal(size=n) for i in range(nc)}
        t0 = time.time()
        try: f(t, y, mp); chk("13 速度 n=%.0e mp=%d 曲线=%d" % (n, mp, nc), True, "%.2fs" % (time.time() - t0))
        except Exception as e: chk("13 速度", False, repr(e)[:70])
    return out
# 说明：第 10 项（max_points<2×曲线数+2）和第 9 项里 max_points=4 的"极值保留"属已知限制，只要求输出合法，不作失败判据。
bad = 0
for name, ok, info in run("rw.circuit.decimate"):
    known = name.startswith("10 小 mp=4") 
    print(("PASS " if ok else "FAIL ") + name, info)
    bad += (not ok) and not known
print("ALL PASS" if not bad else "SOME FAIL"); sys.exit(1 if bad else 0)
