"""分解复用门限 v2（thermal-factor-reuse-opt）验收自检。

A) 终态偏差：opt 默认门 vs 通用组装+逐点分解（RW_NO_SYM=1），要求 ~1e-12 量级
B) 计数器快照（重分解/复用/细化/回退/重锚/降级）
C) tol_A=0 ⇒ 每步重分解，与 v0.28 基线（RW_NO_FACTOR_REUSE=1）终态逐位一致
D) 自校验冒烟：强制 refine 失败 ⇒ 回退计数+warn+回退解逐位一致+下一步重锚；
   NaN 注入 ⇒ degraded 置位
子进程隔离 env；全部数字为仿真环境结果对照，非实测。
"""
import json
import os
import subprocess
import sys

import numpy as np

PY = sys.executable
ROOT = os.path.dirname(os.path.abspath(__file__))
RUN = ("import rw.induction as ind;"
       "s=ind.InductionQuenchSim(freq=20e3,power_kw=6,t_heat=1.0,t_quench=1.0,t_air=1.0,nonlinear=True);"
       "s.run();import numpy as np;np.save('%s',s.T);"
       "import json;print('INFO', json.dumps(s._fr.info() if getattr(s,'_fr',None) else {}))")

fails = []


def run_case(tag, env_extra):
    env = dict(os.environ)
    env.update(env_extra)
    env["PYTHONPATH"] = ROOT
    out = subprocess.run([PY, "-c", RUN % os.path.join(ROOT, "_fo_%s.npy" % tag)],
                         capture_output=True, text=True, env=env, cwd=ROOT)
    if out.returncode != 0:
        print(tag, "RUN FAILED rc=", out.returncode)
        print(out.stdout[-1500:])
        print(out.stderr[-1500:])
        sys.exit(1)
    info = [l[5:] for l in out.stdout.splitlines() if l.startswith("INFO")]
    T = np.load(os.path.join(ROOT, "_fo_%s.npy" % tag))
    os.unlink(os.path.join(ROOT, "_fo_%s.npy" % tag))
    return T, json.loads(info[0]) if info else {}


T_opt, info_opt = run_case("opt", {})
T_ref, _ = run_case("ref", {"RW_NO_SYM": "1"})
T_base, _ = run_case("base", {"RW_NO_FACTOR_REUSE": "1"})

d_ref = float(np.abs(T_opt - T_ref).max())
d_base = float(np.abs(T_opt - T_base).max())
print("A) opt vs 通用组装+逐点分解: max|dT| = %.3e C" % d_ref)
print("   opt vs v0.28 基线:      max|dT| = %.3e C" % d_base)
if not d_ref < 2e-11:
    fails.append("A: 偏差 %.3e 超出 ~1e-12 量级带（2e-11）" % d_ref)
print("B) opt manager info:", json.dumps(info_opt))
if info_opt.get("degraded") is not None:
    fails.append("B: 不应降级 degraded=%r" % info_opt.get("degraded"))
if info_opt.get("n_refactor", 999) > 40:
    fails.append("B: 重分解次数 %d 未显著低于初版 93" % info_opt.get("n_refactor", -1))

# C) tol_A=0 逐位对照
import rw.induction as ind
from rw import factor_reuse as frm

orig_init = frm.FactorReuseCache.__init__


def zero_init(self, *a, **k):
    orig_init(self, *a, **k)
    self.tol_A = 0.0
    self.tol_block_loose = 0.0


frm.FactorReuseCache.__init__ = zero_init
s0 = ind.InductionQuenchSim(freq=20e3, power_kw=6, t_heat=1.0, t_quench=1.0,
                            t_air=1.0, nonlinear=True)
s0.run()
d0 = float(np.abs(s0.T - T_base).max())
print("C) tol_A=0: n_refactor=%d n_reuse=%d ; vs 基线 max|dT| = %.3e"
      % (s0._fr.n_refactor, s0._fr.n_reuse, d0))
if d0 != 0.0 or s0._fr.n_reuse != 0:
    fails.append("C: tol_A=0 未逐位一致 (dT=%.3e, reuse=%d)" % (d0, s0._fr.n_reuse))
frm.FactorReuseCache.__init__ = orig_init

# D) 回退/重锚/降级冒烟（合成小矩阵）
import warnings

from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

rng = np.random.default_rng(1)
n = 60
Mm = rng.standard_normal((n, n))
M = csr_matrix(Mm @ Mm.T + n * np.eye(n))
comps0 = {"K": M}
c = frm.FactorReuseCache(tol_A=1e18, tol_block_loose=1e18, max_refine=0)
b = rng.standard_normal(n)
c.solve(M, b, comps0, 0.01)
M2 = M.copy()
M2.data = M.data * 1.05
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    x2 = c.solve(M2, b, comps0, 0.01)
    warned = len(w) > 0
x2_ref = spsolve(M2.tocsc(), b)
c.solve(M2, b, comps0, 0.01)
d_fb = float(np.abs(x2 - x2_ref).max())
print("D) fallback: n_fallback=%d warned=%s |x2-spsolve|=%.2e ; reanchor=%d"
      % (c.n_fallback, warned, d_fb, c.n_reanchor))
if not (c.n_fallback == 1 and warned and d_fb == 0.0 and c.n_reanchor == 1):
    fails.append("D: 回退/重锚冒烟不符 (nfb=%d warned=%s d=%.2e reanchor=%d)"
                 % (c.n_fallback, warned, d_fb, c.n_reanchor))
Mb = M.copy()
Mb.data = Mb.data.copy()
Mb.data[0] = np.nan
with warnings.catch_warnings(record=True):
    warnings.simplefilter("always")
    c.solve(Mb, b, comps0, 0.01)
print("D) degrade on NaN: degraded=%r" % (c.degraded,))
if c.degraded != "non-finite":
    fails.append("D: NaN 未触发降级 (degraded=%r)" % (c.degraded,))

if fails:
    print("\nFAIL:")
    for f in fails:
        print(" -", f)
    sys.exit(1)
print("\nPASS: validate_factor_opt 全部通过")
