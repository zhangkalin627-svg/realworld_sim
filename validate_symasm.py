"""v0.28 符号组装验证：核级 asm 对比 + 单步 A/B + 短程全 run A/B（RW_NO_SYM 对照）。"""
import os, time
import numpy as np

def one(run_sym):
    env = dict(os.environ)
    if not run_sym:
        os.environ["RW_NO_SYM"] = "1"
    else:
        os.environ.pop("RW_NO_SYM", None)
    import importlib, rw.induction as ind
    importlib.reload(ind)
    s = ind.InductionQuenchSim(freq=20e3, power_kw=6, t_heat=1.0, t_quench=1.0, t_air=1.0, nonlinear=True)
    t0 = time.time(); r = s.run(); dt = time.time() - t0
    os.environ.pop("RW_NO_SYM", None)
    return s, r, dt

# 1) 核级验证（_build_sym 内已做 rel<1e-12 校验，构建失败会回退；这里确认 sym 真的启用了）
from rw.induction import InductionQuenchSim
s0 = InductionQuenchSim(freq=20e3, power_kw=6, nonlinear=True)
sym = s0._build_sym()
assert sym, "符号组装构建或校验失败（应自动回退，但这里要求成功）"
print("符号模板构建 + 核级校验 OK（K/C/Q/H/hlin 对 asm rel<1e-12）")

# 2) 短程全 run A/B
s_a, r_a, t_a = one(True)
s_b, r_b, t_b = one(False)
dT = float(np.abs(s_a.T - s_b.T).max())
dHRC = float(np.abs(s_a.HRC - s_b.HRC).max())
m = np.isfinite(s_a.t85) & np.isfinite(s_b.t85)
assert np.array_equal(np.isfinite(s_a.t85), np.isfinite(s_b.t85)), "t85 有限性不一致"
dt85 = float(np.abs(s_a.t85[m] - s_b.t85[m]).max()) if m.any() else 0.0
print(f"全 run A/B：sym {t_a:.1f}s vs 通用 {t_b:.1f}s（省 {t_b-t_a:.1f}s, {(t_b-t_a)/t_b*100:.0f}%）")
print(f"  终态 max|ΔT| = {dT:.2e}°C，max|ΔHRC| = {dHRC:.2e}，max|Δt85| = {dt85:.2e}s")
assert dT < 1e-8 and dHRC < 1e-8 and dt85 < 1e-9
print("符号组装 OK：结果偏差在浮点噪声量级")
