#!/bin/bash
# realworld_sim 热求解器分解复用优化第二轮（thermal-factor-reuse-opt2）一键回归
# 用法：先解压源码包并 cd 到源码根目录，再 bash run_all.sh
# 解释器可用 $PYTHON 指定（默认 python3）；依赖见 requirements.txt（scikit-fem 等）。
set -u
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
export PYTHONPATH=.
fail=0
for t in validate_factor_opt.py validate_symasm.py validate_ckpt.py validate_em.py \
         validate_energy.py validate_flow.py validate_pll.py validate_fw_pll.py \
         validate_power_baseline.py validate_accel.py validate_circuit.py \
         validate_circuit_v012.py validate_power_dmatrix.py \
         validate_scenec_inductance.py validate_w3d.py validate.py; do
  echo "===== $t ====="
  "$PY" "$t" || fail=1
done
# 超长项（默认不跑，需要时单独执行）：validate_power_longrun.py
if [ "$fail" = 0 ]; then echo "ALL PASS"; else echo "SOME FAILED"; fi
exit $fail
