#!/usr/bin/env python3
"""感应淬火参数扫描 CSV 汇总工具（纯标准库，逐行读取）。

来源标注：外部 AI（Qwen）按我方规格生成，经我方独立评审（tools/test_scan_summary.py，14 项验收，含对照参考实现、短行、BOM、20 万行速度）后采用。
读 tools/batch/scan_grid.py 产出的 CSV；全部为仿真结果，非实测。verdict 为空的行记为"(空)"。
"""

import argparse
import csv
import math
import os
import sys
from collections import defaultdict


# ---------- 内部辅助 ----------

def _to_float(value):
    """把字符串转成 float，空值或无法转换时返回 None。"""
    if value is None:
        return None
    value = value.strip()
    if value == "":
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _num(value):
    """数值列的规范化：空字符串视为 None。"""
    if value is None:
        return None
    value = value.strip()
    return None if value == "" else value


# ---------- 核心逻辑 ----------

def summarize(path):
    """读取感应淬火扫描 CSV，返回汇总字典。

    返回：
        {
            "rows": 总行数,
            "errors": err 非空行数,
            "verdicts": {verdict: 计数},
            "by_wall": {
                wall: {
                    "n": 有效点数,
                    "ok": verdict 为 "合适" 的点数,
                    "best_freqs": [按"合适"占比从高到低排序的最多 3 个 freq],
                    "case_depth_range": (最小, 最大) 或 None,
                }
            },
        }

    wall、freq 按数值排序；占比 = 该壁厚下该频率的"合适"点数 / 该壁厚下该频率的有效点数。
    """
    rows = 0
    errors = 0
    verdicts = defaultdict(int)

    # 按 (wall, freq) 统计有效点数与"合适"点数
    pair_ok = defaultdict(int)
    pair_n = defaultdict(int)

    # 按 wall 统计 case_depth 有效值
    wall_case_depths = defaultdict(list)

    # 收集 wall、freq 的数值，用于后续数值排序
    wall_values = set()
    freq_values = set()

    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows += 1

            err = (row.get("err") or "").strip()
            if err:
                errors += 1
                continue

            verdict = (row.get("verdict") or "").strip() or "(空)"
            verdicts[verdict] += 1

            wall_raw = row.get("wall", "")
            freq_raw = row.get("freq", "")
            wall_val = _to_float(wall_raw)
            freq_val = _to_float(freq_raw)

            if wall_val is None or freq_val is None:
                continue

            wall_values.add(wall_val)
            freq_values.add(freq_val)

            key = (wall_val, freq_val)
            pair_n[key] += 1
            if verdict == "合适":
                pair_ok[key] += 1

            case_depth_raw = row.get("case_depth_mm", "")
            case_depth = _to_float(case_depth_raw)
            if case_depth is not None:
                wall_case_depths[wall_val].append(case_depth)

    # 构建 by_wall
    by_wall = {}
    for wall in sorted(wall_values):
        ok_count = 0
        n_count = 0
        freq_stats = []

        for freq in sorted(freq_values):
            key = (wall, freq)
            n = pair_n.get(key, 0)
            if n == 0:
                continue
            ok = pair_ok.get(key, 0)
            n_count += n
            ok_count += ok
            ratio = ok / n
            freq_stats.append((ratio, freq))

        # 按"合适"占比降序，同占比按 freq 升序
        freq_stats.sort(key=lambda x: (-x[0], x[1]))
        best_freqs = [f for _, f in freq_stats[:3]]

        depths = wall_case_depths.get(wall, [])
        case_depth_range = (min(depths), max(depths)) if depths else None

        by_wall[wall] = {
            "n": n_count,
            "ok": ok_count,
            "best_freqs": best_freqs,
            "case_depth_range": case_depth_range,
        }

    return {
        "rows": rows,
        "errors": errors,
        "verdicts": dict(verdicts),
        "by_wall": by_wall,
    }


def to_markdown(summary):
    """把汇总字典转成中文 Markdown 字符串。"""
    lines = []
    lines.append("# 扫描结果汇总（仿真结果）")
    lines.append("")

    # 总览
    rows = summary["rows"]
    errors = summary["errors"]
    verdicts = summary["verdicts"]

    lines.append("## 总览")
    lines.append("")
    lines.append(f"- 总行数：{rows}")
    lines.append(f"- 失败点数（err 非空）：{errors}")

    if verdicts:
        verdict_parts = []
        for v, c in sorted(verdicts.items()):
            verdict_parts.append(f"{v}：{c}")
        lines.append(f"- 判定分布：{', '.join(verdict_parts)}")
    else:
        lines.append("- 判定分布：无")

    lines.append("")

    # 汇总表
    lines.append("## 按壁厚汇总")
    lines.append("")
    lines.append(
        "| 壁厚(mm) | 有效点数 | 合适占比 | 推荐频率(kHz) | 淬硬层深范围(mm) |"
    )
    lines.append("| --- | --- | --- | --- | --- |")

    by_wall = summary["by_wall"]
    for wall in sorted(by_wall.keys()):
        info = by_wall[wall]
        n = info["n"]
        ok = info["ok"]
        ratio = ok / n if n else 0.0
        freqs = info["best_freqs"]
        freq_str = "、".join(_format_num(f) for f in freqs) if freqs else "-"

        case_range = info["case_depth_range"]
        if case_range is None:
            range_str = "-"
        else:
            lo, hi = case_range
            range_str = f"{_format_num(lo)} ~ {_format_num(hi)}"

        lines.append(
            f"| {_format_num(wall)} | {n} | {ratio:.1%} | {freq_str} | {range_str} |"
        )

    lines.append("")
    lines.append("以上均为仿真结果，非实测；推荐频率仅为该网格内的统计倾向。")
    lines.append("")
    return "\n".join(lines)


def _format_num(value):
    """把数值转成展示字符串：整数不显示小数点。"""
    if isinstance(value, float) and value == math.floor(value):
        return str(int(value))
    return str(value)


# ---------- 命令行入口 ----------

def main():
    parser = argparse.ArgumentParser(description="感应淬火参数扫描 CSV 汇总工具")
    parser.add_argument("csv", help="输入 CSV 路径")
    parser.add_argument("--out", default=None, help="输出 Markdown 路径，不提供则打印到屏幕")
    args = parser.parse_args()

    if not os.path.isfile(args.csv):
        print(f"错误：文件不存在：{args.csv}", file=sys.stderr)
        sys.exit(2)

    summary = summarize(args.csv)
    md = to_markdown(summary)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(md)
    else:
        sys.stdout.write(md)


if __name__ == "__main__":
    main()