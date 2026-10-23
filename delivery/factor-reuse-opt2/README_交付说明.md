# 交付说明：热求解器分解复用提速优化第二轮（thermal-factor-reuse-opt2）

- 分支：`thermal-factor-reuse-opt2`，基线 `thermal-factor-reuse-opt@4d56326`
- 改动文件：`rw/factor_reuse.py`（细化内部两段式改造）、`docs/FACTOR_REUSE.md`（新增 §8）
- 口径声明：本文全部计时/偏差数字均为**仿真环境计时/结果对照，非实测**；
  本地计时取自本轮实际执行的命令输出，本机 2 核共享 VM 负载方差 ±30%，
  只作定性参考，30% 提速线的最终判定归管家云端同机同例复跑。

## 背景

v2 候选（4d56326）云端验收：确定性全对上，但提速仅 3.1%（baseline 均值
0.9222 s vs reuse 0.8933 s），未达 30% 线、不合 main。机制读数：少 70 次
分解只省约 0.04 s，而 276 次复用带出 1764 次细化、t_reuse_solve=0.325 s。
本轮按 §7.4 预告的两方向执行，**只动细化内部**：门限判据、残差验收线
（tol_res=1e-17 / tol_accept=1e-15）、spsolve 回退机制一律不动。

## _refine 开销剖析（本轮实测归属，标准短用例 300 步，n=1560、nnz=10388）

- 原版分段计时（277 次调用）：setup 0.66 ms/次、迭代段 3.77 ms/次、
  平均 6.37 轮/次（约 0.59 ms/轮）。
- 单项微基准：A.astype(float128) 0.255 ms；|A8| 稀疏行和 nA8 独立测法
  1.09 ms；float128 SpMV 0.063 ms；float64 SpMV 0.017 ms；lu.solve
  0.167 ms（在位更高）。
- 残差轨迹：rn0 中位数 5.8e-05，每轮改善约 2–3 个量级，约 5 轮穿过
  tol_accept，约 2 轮耗在 accept 线以下的抛光段。
- 结论：成本主体是 1764 轮 float128 残差求值与向量运算（lu.solve 两
  精度同价、不可省）；setup 次之。

## 两方向落实

**方向① 混合精度两段式**（`FactorReuseCache._refine`）：

1. float64 预筛段（`_refine_f64_phase`）：全程 float64 残差/修正推进；
   停机 = 残差进入警戒带（`_GUARD_ACCEPT=10` × tol_accept = 1e-14，
   写死）、连续 2 轮改善 <2×（float64 舍入下限停滞）、预算耗尽、
   残差非有限。带外 float64 残差相对误差 ≤~1e-2，修正方向与
   float128 一致。
2. float128 裁决段（`_refine_f128`）：验收 rn 仍为 float128 度量、
   tol_res/tol_accept 不动、口径与初版一致。
3. 两段共享 max_refine 总预算（max_refine=0 时两段均不修正、只做
   float128 度量，初版语义不变）。
4. nA8 改用 CSR 行段 reduceat 求和（`_nA_inf`）：与稀疏 sum(axis=1)
   同为按 data 顺序逐行累加；标准用例全程 300 步、两精度下与原式
   逐位比对 **0 不一致**（一次性校验脚本，非常驻断言）。与 §7 已否决
   的两项改值省算（float64 行和、锚点缓存）不同类——不改变数值。

**方向② 停滞判据**（float128 段，参数写死）：无改善（新残差 ≥ 历史
最优）立即停；已达 tol_accept 且连续 `_STALL_ROUNDS=2` 轮改善
<`_STALL_FACTOR=2` 时提前停，不再为追 tol_res 空烧慢尾；tol_res
仍是第一停机线。行为差如实标注：初版在 accept 线之上遇单轮慢改善
即停，本版继续推进至预算/停滞——只会减少、不会增加回退。

**新增计数器**：n_f64_prescreen / n_f128_verify / n_refine_f64_iter /
n_refine_f128_iter（info() 可查）；n_refine_iter 保持总量口径。

## 确定性自检结果（本地冒烟，本轮实际执行）

- `validate_factor_opt.py`：**PASS**
  - A) max|dT|：vs 通用组装+逐点分解 **5.443e-12 °C**、vs v0.28 基线
    **4.803e-12 °C**（线 <2e-11；v2 为 5.571/4.462e-12，同级）
  - C) tol_A=0：300 重分解、0 复用、与基线逐位 0.000e+00
  - D) 强制回退：n_fallback=1、warn 发出、回退解与 spsolve 逐位一致、
    下一步重锚 n_reanchor=1；NaN 注入 → degraded='non-finite'
- 回归冒烟：validate_symasm.py 通过（终态 max|ΔT|=5.44e-12 °C，
  ΔHRC=0，Δt85=0）；validate_ckpt.py 通过（恢复 vs 连续 0.0006 °C，
  与 v2 一致）；validate_em.py 本地运行期报 OSError: libGLU.so.1
  （既有环境问题、与 v2 本地同错、非回归），已列入 run_all.sh 云端全量。

## 计数对比（标准短用例，v2 → opt2）

| 计数器 | v2 (4d56326) | opt2 |
|---|---|---|
| n_solve / n_reuse / n_refactor | 300 / 276 / 23 | 300 / 276 / 23（不变） |
| n_fallback / n_reanchor / degraded | 1 / 1 / null | 1 / 1 / null（不变） |
| 回退原因 | refine-residual-1.32e-05 | refine-residual-5.66e-08（同属约 1/300 步局部收缩失效，验收拦截+重锚自愈） |
| n_refine_iter 总轮数 | 1764 | 1785（+1.2%） |
| 其中 float128 轮 | 1764 | **417（−76%）** |
| 其中 float64 轮 | 0 | 1368 |
| n_f64_prescreen / n_f128_verify | — | 277 / 277 |

## 计时（定性，本地交替 A/B 各 2 次，t_reuse_solve）

- 新版：1.163 / 1.139 s；旧版：1.350 / 2.098 s（旧版第二样本遇负载
  尖峰）。两组交替均新版更低，定性约 −15%（取第一组比值）。
- 新版分段：float64 段 0.51–0.59 s（1368 轮 ≈ 0.40 ms/轮）、
  float128 段 0.47–0.53 s（277 次 setup + 417 轮）。
- 注：validate_factor_opt 本轮运行时段 t_reuse_solve=1.609 s 为高负载
  读数，与 A/B 时段不可直接比，印证本地计时不可作定量结论。

## 预期与未验项（如实标注）

- 按云端 v2 读数折算：t_reuse_solve 0.325 s 若同比例（−15%）下降，
  总时约 0.845 s、提速约 8%——**预计仍未过 30% 线**，终判归云端。
- 剩余瓶颈：每复用步的修正 lu.solve 本身（1785 轮两精度同价）与
  float128 裁决 setup；只动细化内部时复用步成本下限约为逐点分解的
  1/3 量级。后续候选（不在本轮范围）：门限侧按预测细化轮数分流
  （高轮数步直接重分解）、tol_A / max_age 再标定——均需另行立项
  并重新过确定性验收。
- 全套件回归：本地冒烟 3 项（见上），其余由 run_all.sh 云端全量执行。

## 复跑方法

解压源码包（或 checkout 分支）后：`bash run_all.sh`（解释器可用
`PYTHON` 环境变量指定，需 scikit-fem 等依赖）。
