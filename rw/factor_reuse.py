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
2. "数值分解仅在矩阵值变化时重做"：门限判据（v2，thermal-factor-reuse-opt
   优化迭代；初版为逐块自身尺度 0.02 门，见 docs/FACTOR_REUSE.md §7 的根因
   实测）——每步计算总矩阵相对锚点（上次分解的 A）的 Frobenius 尺度变化
   δ = ||A.data − A_fac.data||₂ / ||A_fac.data||₂（pattern 恒定、data 逐元
   对齐，此差为精确 ΔA）；δ ≤ tol_A（默认 0.02）且 dt 不变且无块发生
   自身尺度 >tol_block_loose（默认 1.0，工况切换级跳变）的变化 ⇒ 复用旧
   分解；另设锚点年龄上限 max_age（默认 16 步）：慢变相位（如空冷段）δ
   长期不达门限时仍强制周期重锚——实测无年龄上限时空冷段锚点老化使终态
   轨迹偏差漂移至 ~3e-11 °C，加 max_age=16 后回落到 5.6e-12 °C（与初版
   5.7e-12 同级，见 §7）。理由：复用的正确性由细化残差验收（见 5）守住，
   门限只是性能启发式；初版逐块自身尺度门把量级远小于 A 主导项（C/dt）的
   边界块 Ho/Hi 的相对变化误判为"矩阵已变"——实测触发重分解的步上
   |ΔHo|max/|A|max 中位数仅 5.8e-5、总 A 范数相对变化中位数 0.0014，
   旧分解作预条件子的细化收缩比 ρ≈3.4δ（δ=0.02 时 ρ≈0.06），复用完全
   可行。逐块 0.02 信息仍记录于 last_changed_blocks 供审计。
3. "非线性 Picard 迭代内矩阵结构不变时复用分解"：管理器以 solve(A, b) 为通用
   接口，天然支持迭代作用域内的复用。当前 _thermal_step 无内层 Picard 迭代
   （物性按旧温显式线性化，一步一解），该作用域在结构上已支持、尚未启用；
   EM 的 mur 不动点迭代是后续候选，未在本单范围（见 docs/FACTOR_REUSE.md）。
4. "相邻热步材料性质未变的块复用"：v2 门以总矩阵 δ 为准（理由见 2）；逐块
   比对保留两项职责——自身尺度 >tol_block_loose 的块作为工况切换级跳变的
   后备触发（如淬火介质切换时 Ho 跳变），以及 last_changed_blocks 审计记录
   （>tol_reuse 的块名）。触发重分解的原因（delta-A / 块名 / dt）可审计。
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
   （本步回退逐点分解并大声记录）。v2 恢复策略（加强而非削弱）：发生
   refine-residual 回退后置 force_refactor 标志，下一步无条件重分解
   重新锚定（回退步本身的处理与初版完全一致：spsolve + warn + 计数；
   重新锚定避免对已证伪的锚点反复空烧细化迭代——v2 门放宽后初版赖以恢复
   的"块门自然触发"不再必然成立，故显式化）。
6. opt2（分支 thermal-factor-reuse-opt2，基于 4d56326）：云端验收确定性
   全过但提速仅 3.1%——1764 次细化迭代的 float128 setup/求值单价吃掉了
   少分解 70 次的收益。本轮只动细化内部（门限、验收线 tol_res/tol_accept、
   spsolve 回退机制一律不动）：
   方向① 混合精度两段式：残差远高于验收线时先用 float64 残差/修正快速
   推进（每轮成本约 float128 轮的 1/3），进入警戒带（10×tol_accept）或
   float64 停滞后交 float128 段按初版口径裁决（验收 rn 仍为 float128
   度量）；float128 段的归一化常数 nA8 改用 CSR 行段 reduceat 求和
   （与稀疏 sum(axis=1) 按 data 顺序逐行累加、逐位一致，交付验证已比对）。
   方向② 停滞判据：float128 段内无改善立即停；已达 tol_accept 且连续
   两轮改善 <2× 时提前停，不再为追 tol_res 空烧慢尾（tol_res 仍是第一
   停机线）；两段共享 max_refine 总预算（与初版语义一致）。新增计数器
   n_f64_prescreen / n_f128_verify / n_refine_f64_iter / n_refine_f128_iter
   供审计两段的实际分工。

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

    def __init__(self, tol_reuse=0.02, tol_A=0.02, tol_block_loose=1.0, max_age=16,
                 tol_res=1e-17, tol_accept=1e-15, max_refine=15):
        self.tol_reuse = float(tol_reuse)  # v2 起仅作审计标注：块自身尺度变化 >此值记入 last_changed_blocks
        self.tol_A = float(tol_A)          # v2 主门限：总矩阵 Frobenius 尺度相对变化 δ ≤ 此值才复用
        self.tol_block_loose = float(tol_block_loose)  # v2 后备门限：任一块自身尺度变化 >此值（工况切换级）⇒ 重分解
        self.max_age = int(max_age)       # v2 锚点年龄上限（步）：δ 达标也强制周期重锚，防慢变相位锚点无限老化
        self.steps_since_fac = 0
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
        self.A_data = None         # A.data.copy() at factorization（v2 主门限的锚点）
        self.A_norm = None         # ||A_fac.data||_2 at factorization
        self.dt_fac = None
        self.force_refactor = False  # v2：refine-residual 回退后置位，下一步无条件重分解重新锚定
        self.degraded = None       # reason str once degraded, else None
        # 计数器（审计用）
        self.n_solve = 0
        self.n_reuse = 0
        self.n_refactor = 0
        self.n_refine_iter = 0
        self.n_f64_prescreen = 0   # opt2：进入 float64 预筛段的复用求解次数
        self.n_f128_verify = 0     # opt2：进入 float128 裁决段的复用求解次数
        self.n_refine_f64_iter = 0   # opt2：float64 段细化轮数
        self.n_refine_f128_iter = 0  # opt2：float128 段细化轮数
        self.n_fallback = 0
        self.n_reanchor = 0
        self.fallback_reasons = []
        self.last_changed_blocks = []
        self.last_delta_A = None
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
        # 变化门（v2）：dt + 总矩阵 δ（主） + 块级跳变（后备）
        if dt != self.dt_fac:
            return self._refactor(A, b, comps, dt, why="dt-changed")
        if self.force_refactor:
            # 上一步 refine-residual 回退 ⇒ 本步无条件重分解重新锚定（见 docstring 5）
            return self._refactor(A, b, comps, dt, why="post-fallback-reanchor")
        self.steps_since_fac += 1
        if self.steps_since_fac >= self.max_age:
            return self._refactor(A, b, comps, dt, why="max-age")
        blk_rel = {name: self._rel_change(c.data, self.comp_data[name])
                   for name, c in comps.items()}
        self.last_changed_blocks = [n for n, v in blk_rel.items() if v > self.tol_reuse]
        loose = [n for n, v in blk_rel.items() if v > self.tol_block_loose]
        if loose:
            return self._refactor(A, b, comps, dt, why="blocks-jump:" + ",".join(loose))
        # δ：总矩阵相对锚点的 Frobenius 尺度变化（pattern 已断言一致，data 逐元对齐）
        delta = float(np.linalg.norm(A.data - self.A_data) / max(self.A_norm, 1e-300))
        self.last_delta_A = delta
        if delta > self.tol_A:
            return self._refactor(A, b, comps, dt, why="delta-A:%.3g" % delta)
        # 复用分解求解 + 精度自校验（迭代细化至浮点噪声级）
        t0 = time.perf_counter()
        x = self.lu.solve(b)
        x, rn = self._refine(A, b, x)
        self.t_reuse_solve += time.perf_counter() - t0
        if rn > self.tol_accept:
            # 细化后残差仍不达标 = 自校验失败：本步回退逐点分解并大声记录（与初版一致）；
            # 并置 force_refactor，下一步重分解重新锚定（不清其余缓存、不降级）。
            self.force_refactor = True
            return self._fallback_once(A, b, "refine-residual-%.2e" % rn)
        self.n_reuse += 1
        return x

    def info(self):
        """审计快照：计数器、回退记录、降级状态、计时。"""
        return dict(n_solve=self.n_solve, n_reuse=self.n_reuse,
                    n_refactor=self.n_refactor, n_refine_iter=self.n_refine_iter,
                    n_f64_prescreen=self.n_f64_prescreen,
                    n_f128_verify=self.n_f128_verify,
                    n_refine_f64_iter=self.n_refine_f64_iter,
                    n_refine_f128_iter=self.n_refine_f128_iter,
                    n_fallback=self.n_fallback, n_reanchor=self.n_reanchor,
                    fallback_reasons=list(self.fallback_reasons),
                    degraded=self.degraded, tol_reuse=self.tol_reuse,
                    tol_A=self.tol_A, last_delta_A=self.last_delta_A,
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

    # opt2 方向①警戒带倍数（写死，不许事后调）：float64 预筛只在残差高于
    # _GUARD_ACCEPT × tol_accept 时推进。依据：float64 残差的计算噪声下限
    # 为 eps64 量级（归一化后 ~1e-16），警戒带取 10×tol_accept = 1e-14：
    # 带外时 float64 残差相对误差 ≤ ~1e-2，修正方向与 float128 残差一致；
    # 带内一律交 float128 段精确裁决，验收线与度量口径不动。
    _GUARD_ACCEPT = 10.0
    # opt2 方向②停滞判据（写死，不许事后调）：连续 _STALL_ROUNDS 轮改善
    # 因子 < _STALL_FACTOR（新残差 ≥ 历史最优 / _STALL_FACTOR）视为停滞。
    _STALL_FACTOR = 2.0
    _STALL_ROUNDS = 2

    @staticmethod
    def _nA_inf(A):
        """||A||_inf = 各行 |data| 段和的最大值（A 须为 CSR）。

        按 CSR 的 indptr 行段对 |data| 做 reduceat 求和：与
        np.abs(A).sum(axis=1).max() 同为按 data 顺序逐行累加，
        float128 下逐位一致（opt2 交付验证在标准用例全程矩阵上比对相等），
        但免去稀疏 sum(axis=1) 的矩阵构造开销。空行（段长 0）行和为 0。
        """
        ad = np.abs(A.data)
        if ad.size == 0:
            return 0.0
        idx = np.minimum(A.indptr[:-1], ad.size - 1)
        rs = np.add.reduceat(ad, idx)
        empty = np.diff(A.indptr) == 0
        if np.any(empty):
            rs = np.where(empty, ad.dtype.type(0), rs)
        return float(rs.max())

    def _refine(self, A, b, x):
        """混合精度迭代细化（opt2 两段式；初版为全程 float128 残差）。

        第一段 float64 预筛（方向①）：残差远高于验收线时用 float64
        残差/修正快速推进（每轮约 float128 轮成本的 1/3）；进入警戒带、
        float64 停滞或预算耗尽后交第二段。
        第二段 float128 裁决（方向②）：口径与初版一致——残差用 float128
        计算、修正步用 float64 SuperLU 分解、验收 rn 为 float128 度量、
        tol_res/tol_accept 不动；仅停滞判据收紧慢尾、nA8 换逐位一致的
        reduceat 求和（见 _nA_inf）。
        两段共享 max_refine 总预算（与初版"最多 max_refine 次修正"语义
        一致；max_refine=0 时两段均不修正、只做 float128 度量）。
        返回 (x_float64, rn_final)。"""
        if not self._have_f128:
            return self._refine_f64(A, b, x)
        self.n_f64_prescreen += 1
        x, used = self._refine_f64_phase(A, b, x, self.max_refine)
        self.n_f128_verify += 1
        return self._refine_f128(A, b, x, self.max_refine - used)

    def _refine_f64_phase(self, A, b, x, budget):
        """方向①预筛段：全程 float64。返回 (x, used_iters)。

        推进条件：归一化残差（float64 度量）高于警戒带
        _GUARD_ACCEPT × tol_accept。停机：进入警戒带、连续
        _STALL_ROUNDS 轮改善 <_STALL_FACTOR（float64 舍入下限停滞）、
        预算耗尽、残差非有限（交 float128 段按原口径处置）。"""
        Ac = A.tocsr()
        nA = self._nA_inf(Ac)
        nb = float(np.abs(b).max(initial=0.0))
        guard = self._GUARD_ACCEPT * self.tol_accept

        def rn64(x):
            r = b - Ac.dot(x)
            denom = nA * float(np.abs(x).max(initial=0.0)) + nb
            return float(np.linalg.norm(r, np.inf) / max(denom, 1e-300)), r

        rn, r = rn64(x)
        best = rn
        slow = 0
        used = 0
        while used < budget and np.isfinite(rn) and rn > guard:
            x = x + self.lu.solve(r)
            used += 1
            self.n_refine_iter += 1
            self.n_refine_f64_iter += 1
            rn_new, r = rn64(x)
            if not np.isfinite(rn_new):
                break
            if rn_new >= best / self._STALL_FACTOR:
                slow += 1
            else:
                slow = 0
            if rn_new < best:
                best = rn_new
            rn = rn_new
            if slow >= self._STALL_ROUNDS:
                break
        return x, used

    def _refine_f128(self, A, b, x, budget):
        """方向②裁决段：float128 残差 + float64 SuperLU 修正（初版口径）。

        与初版 _refine 的差异只有两处：nA8 改用 _nA_inf（逐位一致）；
        停滞判据——无改善（新残差 ≥ 历史最优）立即停，已达 tol_accept
        且连续 _STALL_ROUNDS 轮改善 <_STALL_FACTOR 时提前停（不为追
        tol_res 空烧慢尾）；tol_res 仍是第一停机线，验收语义不变。"""
        f128 = np.float128
        A8 = A.tocsr().astype(f128)
        b8 = b.astype(f128)
        x8 = x.astype(f128)
        # 归一化常数 ||A||_inf 按当前 A 在 float128 下计算（口径同初版；
        # §7 记录的 float64 行和/锚点缓存两种改值省算均已否决，不再试）。
        nA8 = self._nA_inf(A8)
        nb8 = float(np.abs(b8).max(initial=0.0))

        def rn8(x8):
            r8 = b8 - A8.dot(x8)
            denom = nA8 * float(np.abs(x8).max(initial=0.0)) + nb8
            return float(np.linalg.norm(r8, np.inf) / max(denom, 1e-300)), r8

        rn, r8 = rn8(x8)
        best = rn
        slow = 0
        it = 0
        while it < budget:
            if rn <= self.tol_res:
                break
            dx = self.lu.solve(np.asarray(r8, dtype=np.float64))
            x8 = x8 + dx.astype(f128)
            self.n_refine_iter += 1
            self.n_refine_f128_iter += 1
            it += 1
            rn_new, r8 = rn8(x8)
            if rn_new >= best:  # 无改善：已触及舍入下限
                rn = rn_new
                break
            if rn_new >= best / self._STALL_FACTOR:
                slow += 1
            else:
                slow = 0
            if rn_new < best:
                best = rn_new
            rn = rn_new
            if rn <= self.tol_accept and slow >= self._STALL_ROUNDS:
                break  # 方向②：已达验收线且慢尾，提前停
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
        self.A_data = A.data.copy()
        self.A_norm = float(np.linalg.norm(A.data))
        self.dt_fac = dt
        if why == "post-fallback-reanchor":
            self.n_reanchor += 1
        self.force_refactor = False
        self.steps_since_fac = 0
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
