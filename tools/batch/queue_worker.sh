#!/bin/bash
# 通用批量队列 worker（公开，只跑本仓库脚本）。用法：bash tools/batch/queue_worker.sh [worker名]
# 队列：tools/batch/jobs/*.job（每个是一个 bash 脚本）。认领=把文件 mv 到 run/<worker>/，
# 跑完把 exit 码写入 done/<job>.status。作业自己做断点续跑（见各 job 注释）。队列空就每 5 分钟再扫一次。
W=${1:-w$$}; R=$(cd "$(dirname "$0")/../.." && pwd); B=$R/tools/batch
mkdir -p $B/run/$W $B/done $B/out
cd $R
while true; do
  for j in $B/jobs/*.job; do
    [ -e "$j" ] || continue
    n=$(basename $j .job)
    mv "$j" "$B/run/$W/" 2>/dev/null || continue     # 抢占失败（别的 worker 已拿走）就跳过
    s=$(date +%s)
    timeout ${JOB_TIMEOUT:-14000} bash "$B/run/$W/$n.job" > "$B/out/$n.log" 2>&1
    rc=$?
    echo "$(date +%FT%T) $W exit=$rc secs=$(( $(date +%s)-s ))" > "$B/done/$n.status"
    # 超时(124)的作业放回队列，靠断点续跑接着干
    if [ $rc -eq 124 ]; then mv "$B/run/$W/$n.job" "$B/jobs/"; else mv "$B/run/$W/$n.job" "$B/done/"; fi
  done
  sleep 300
done
