"""热步矩阵分解复用（v0.28 符号组装之后的下一段）。

派单（2026-10-22 管家，卡林已批准）：realworld_sim 热求解器"矩阵分解复用"。
基线 v0.28 = 符号组装热步（sparsity pattern 一次构建 + P1 几何张量预计算）；
本单做分解复用：符号分解一次构建，数值分解仅在矩阵值变化时重做。

派单措辞 → 实现的映射（逐条可审计）：
1. "符号分解一次构建"：网格/基函数在整个 run 中不变 ⇒ CSR pattern 恒定
   （v0.28 的 SymAssembler 构建时已断言 pattern 稳定）。SuperLU 的符号分析
   （列置换）只与 pattern 有关；本实现让 factorization 对象常驻、每步断言
   pattern 一致（自校验之一），即"符号一次"的语义。注：scipy 未暴露符号/
   数值分离接口，数值重做时 SuperLU 内部会附带重做一次符号分析——这是接口
   限制，非设计冗余，如实标注。
2. "数值分解仅在矩阵值变化时重做"：每步按"块"（K 传导 / C 热容 / Ho 外对流 /
   Hi 内对流）比较材料性质驱动的矩阵值相对变化；任一块变化超过 tol_reuse
   或 dt 变化 ⇒ 数值重分解；否则复用旧分解。
3. "非线性 Picard 迭代内矩阵结构不变时复用分解"：管理器以 solve(A, b) 为通用
   接口，天然支持迭代作用域内的复用。当前 _thermal_step 无内层 Picard 迭代
   （物性按旧温显式线性化，一步一解），该作用域在结构上已支持、尚未启用；
   EM 的 mur 不动点迭代是后续候选，未在本单范围（见 docs/FACTOR_REUSE.md）。
4. "相邻热步材料性质未变的块复用"：块级变化门——K/C/Ho/Hi 逐块比对相对变化，
   全块未变（≤ tol_reuse）才复用；并记录是哪一块触发了重分解（可审计）。
5. "任何自校验失败自动回退逐点分解，不得静默继续"：自校验 =
   {pattern 一致， 全有限， splu 成功， 复用解残差达标}。
   任一失败 ⇒ 本步改用久经验证的 spsolve（逐点分解）求解，并大声记录
   （warnings.warn + 计数器 + info() 可查），绝不静默。pattern/有限性/splu
   异常属系统性故障 ⇒ 置 degraded 标志，此后各步直接走逐点分解（不再尝试
   复用，避免刷屏；状态与原因可查）。
   精度自校验：复用分解的解必须经混合精度迭代细化——残差用 float128
   （eps~1e-19）计算、修正步用 float64 的 SuperLU 分解，把解的精度下限从
   cond*eps64 压到 cond*eps128；细化至残差停滞（tol_res 默认 1e-17）；
   细化后残差仍高于 tol_accept（默认 1e-15）⇒ 视为自校验失败
   （本步回退逐点分解并大声记录；下一步的块级门限比较会自然触发重分解恢复）。

环境开关：
- RW_NO_SYM=1：沿用旧行为（通用组装 + 逐点分解），复用管理器不启用。
  这也是验收口径①"通用组装 + 逐点分解"对照的参考路径。
- RW_NO_FACTOR_REUSE=1：符号组装启用、分解复用关闭（= v0.28 基线行为），
  用于隔离"复用"带来的提速（验收口径②的对照）。

所有计时/偏差数字均为"仿真环境计时/结果对照，非实测"。
"""

import os
import time
import warnings

import numpy as np
from scipy.sparse.linalg import splu, spsolve


class FactorReuseCache:
    """热步线性求解的分解复用管理器（每 sim 实例一个，随实例消亡，不进 checkpoint）。"""

    def __init__(self, tol_reuse=0.02, tol_res=1e-17, tol_accept=1e-15, max_refine=15):
        self.tol_reuse = float(tol_reuse)
        self.tol_res = float(tol_res)       # 细化停止目标（高精度残差下）
        self.tol_accept = float(tol_accept)  # 细化后验收线：仍不达标 = 自校验失败
        self.max_refine = int(max_refine)
        # 混合精度细化：残差用 float128（eps~1e-19）计算，把解的精度下限
        # 从 cond*eps64 压到 cond*eps128；修正步仍用 float64 的 SuperLU 分解。
        try:
            _ = np.float128(1.0)
            self._have_f128 = True
        except AttributeError:
            self._have_f128 = False
            # 无 float128：细化目标退回 float64 可达水平（如实标注）
            self.tol_res = max(self.tol_res, 2e-15)
            self.tol_accept = max(self.tol_accept, 1e-14)
        self.lu = None
        self.pat = None            # (indptr.copy(), indices.copy()) at factorization
        self.comp_data = None      # {name: data.copy()} at factorization
        self.dt_fac = None
        self.degraded = None       # reason str once degraded, else None
        # 计数器（审计用）
        self.n_solve = 0
        self.n_reuse = 0
        self.n_refactor = 0
        self.n_refine_iter = 0
        self.n_fallback = 0
        self.fallback_reasons = []
        self.last_changed_blocks = []
        # 轻量计时（审计用，单位 s）
        self.t_refactor = 0.0
        self.t_reuse_solve = 0.0

    # ---------------------------------------------------------- 对外接口
    def solve(self, A, b, comps, dt):
        """求解 A x = b。

        A: csr（= C/dt + K + Ho + Hi）；b: rhs；comps: {name: csr} 各块
        （K/C/Ho/Hi，与 A 同一次组装产物）；dt: 本步时间步长（标量）。
        返回 x。绝不静默失败：任何回退都计数 + warn + 可查。
        """
        self.n_solve += 1
        if self.degraded is not None or os.environ.get("RW_NO_FACTOR_REUSE"):
            # 已降级或被环境关闭：走久经验证的逐点分解；这是预期路径，不计 fallback、不刷屏
            return spsolve(A.tocsc(), b)
        # 自校验 1：pattern 一致（符号一次的语义落地）
        if self.pat is not None:
            if not (A.indptr.shape == self.pat[0].shape
                    and np.array_equal(A.indptr, self.pat[0])
                    and np.array_equal(A.indices, self.pat[1])):
                return self._degrade(A, b, "pattern-mismatch")
        # 自校验 2：有限性
        if not (np.all(np.isfinite(A.data)) and np.all(np.isfinite(b))):
            return self._degrade(A, b, "non-finite")
        if self.lu is None:
            return self._refactor(A, b, comps, dt, why="first")
        # 变化门：块级比对 + dt
        if dt != self.dt_fac:
            return self._refactor(A, b, comps, dt, why="dt-changed")
        changed = [name for name, c in comps.items()
                   if self._rel_change(c.data, self.comp_data[name]) > self.tol_reuse]
        self.last_changed_blocks = changed
        if changed:
            return self._refactor(A, b, comps, dt, why="blocks-changed:" + ",".join(changed))
        # 复用分解求解 + 精度自校验（迭代细化至浮点噪声级）
        t0 = time.perf_counter()
        x = self.lu.solve(b)
        x, rn = self._refine(A, b, x)
        self.t_reuse_solve += time.perf_counter() - t0
        if rn > self.tol_accept:
            # 细化后残差仍不达标 = 自校验失败：本步回退逐点分解并大声记录；
            # 下一步块级门限会自然触发重分解恢复（不清缓存、不降级）。
            return self._fallback_once(A, b, "refine-residual-%.2e" % rn)
        self.n_reuse += 1
        return x

    def info(self):
        """审计快照：计数器、回退记录、降级状态、计时。"""
        return dict(n_solve=self.n_solve, n_reuse=self.n_reuse,
                    n_refactor=self.n_refactor, n_refine_iter=self.n_refine_iter,
                    n_fallback=self.n_fallback,
                    fallback_reasons=list(self.fallback_reasons),
                    degraded=self.degraded, tol_reuse=self.tol_reuse,
                    tol_res=self.tol_res, t_refactor=self.t_refactor,
                    t_reuse_solve=self.t_reuse_solve)

    # ---------------------------------------------------------- 内部
    @staticmethod
    def _rel_change(new, old):
        denom = max(float(np.abs(old).max(initial=0.0)), 1e-300)
        return float(np.abs(new - old).max(initial=0.0) / denom)

    @staticmethod
    def _nres(A, b, x):
        """归一化无穷范数残差 ||b-Ax||_inf / (||A||_inf||x||_inf + ||b||_inf)。
        纯 float64 版（用于新鲜分解的健康检查等非关键路径）。"""
        r = b - A.dot(x)
        nA = float(np.abs(A).sum(axis=1).max())
        denom = nA * float(np.abs(x).max(initial=0.0)) + float(np.abs(b).max(initial=0.0))
        return float(np.linalg.norm(r, np.inf) / max(denom, 1e-300))

    def _refine(self, A, b, x):
        """混合精度迭代细化：残差用 float128（eps~1e-19）计算，修正步仍用
        float64 的 SuperLU 分解。返回 (x_float64, rn_final)。
        细化至残差停滞（触及舍入下限）或达 tol_res；若停滞在 tol_accept 之上，
        调用方视为自校验失败。"""
        if not self._have_f128:
            return self._refine_f64(A, b, x)
        f128 = np.float128
        A8 = A.astype(f128)
        b8 = b.astype(f128)
        x8 = x.astype(f128)
        nA8 = float(np.abs(A8).sum(axis=1).max())
        nb8 = float(np.abs(b8).max(initial=0.0))

        def rn8(x8):
            r8 = b8 - A8.dot(x8)
            denom = nA8 * float(np.abs(x8).max(initial=0.0)) + nb8
            return float(np.linalg.norm(r8, np.inf) / max(denom, 1e-300)), r8

        rn, r8 = rn8(x8)
        best = rn
        it = 0
        while it < self.max_refine:
            if rn <= self.tol_res:
                break
            dx = self.lu.solve(np.asarray(r8, dtype=np.float64))
            x8 = x8 + dx.astype(f128)
            self.n_refine_iter += 1
            it += 1
            rn, r8 = rn8(x8)
            if rn >= best * 0.5:  # 停滞：残差不再有效下降（已触及舍入下限）
                break
            if rn < best:
                best = rn
        return np.asarray(x8, dtype=np.float64), rn

    def _refine_f64(self, A, b, x):
        """纯 float64 细化（无 float128 环境的回退路径；精度下限 ~cond*eps64）。"""
        rn = self._nres(A, b, x)
        best = rn
        it = 0
        while it < self.max_refine:
            if rn <= self.tol_res:
                break
            r = b - A.dot(x)
            x = x + self.lu.solve(r)
            self.n_refine_iter += 1
            it += 1
            rn = self._nres(A, b, x)
            if rn >= best * 0.5:
                break
            if rn < best:
                best = rn
        return x, rn

    def _refactor(self, A, b, comps, dt, why):
        """数值（重）分解：矩阵值变化时的正常路径（非失败）。"""
        t0 = time.perf_counter()
        try:
            lu = splu(A.tocsc())
        except Exception as e:  # 自校验失败：splu 异常
            return self._degrade(A, b, "splu-failed:%r" % (e,))
        self.t_refactor += time.perf_counter() - t0
        self.lu = lu
        self.pat = (A.indptr.copy(), A.indices.copy())
        self.comp_data = {k: v.data.copy() for k, v in comps.items()}
        self.dt_fac = dt
        self.n_refactor += 1
        x = lu.solve(b)
        # 自校验：新鲜分解的残差也应健康（float64 下 ~1e-16；>1e-13 说明矩阵本身病态）
        if self._nres(A, b, x) > 1e-13:
            return self._degrade(A, b, "fresh-residual-too-large")
        return x

    def _fallback_once(self, A, b, reason):
        """本步回退逐点分解：大声记录，不降级（下步门限自然恢复）。"""
        self.n_fallback += 1
        self.fallback_reasons.append(reason)
        warnings.warn("[factor_reuse] 自校验失败，本步回退逐点分解: %s（已记录，未静默）" % reason)
        return spsolve(A.tocsc(), b)

    def _degrade(self, A, b, reason):
        """系统性自校验失败：永久回退逐点分解（大声记录一次，不刷屏）。"""
        if self.degraded is None:
            self.degraded = reason
            warnings.warn("[factor_reuse] 自校验失败，永久回退逐点分解: %s（已记录，未静默）" % reason)
        self.n_fallback += 1
        self.fallback_reasons.append(reason)
        return spsolve(A.tocsc(), b)
