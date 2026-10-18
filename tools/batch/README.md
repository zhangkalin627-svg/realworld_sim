# 批量队列（公开）
把算量大的作业放进 `jobs/*.job`，用 `bash tools/batch/queue_worker.sh <名>` 起 worker（可多开）。作业自己做断点续跑；
结果在 `out/`，状态在 `done/`。全部输出为仿真结果，非实测。队列里现有：扫描大网格、回归稳定性、确定性复核、基准计时。
