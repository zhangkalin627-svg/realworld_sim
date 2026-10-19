#!/usr/bin/env python3
"""
report.py —— 仿真波形报告生成器
本工具由 AI 助手（企业微信助手）编写，经我方独立验收（accept_rwpost.py，45 项）后采用；全部为程序运行结果，非实测
只依赖 numpy 和 Python 标准库
"""

import sys
import csv
import numpy as np

from waveform_metrics import (
    frequency, fft_harmonics, thd, ripple
)
from rms_stats import rms, avg, crest_factor, power_factor


def make_report(name, t, v, i, f0):
    """
    对一条电压、电流波形给出 Markdown 报告。
    """
    t = np.asarray(t, dtype=float)
    v = np.asarray(v, dtype=float)
    i = np.asarray(i, dtype=float)

    f_est = frequency(t, v)
    harm_v = fft_harmonics(t, v, f0)
    harm_i = fft_harmonics(t, i, f0)

    v_rms = rms(v, t)
    i_rms = rms(i, t)
    p = avg(v * i, t)
    pf = power_factor(v, i, t)
    cf_v = crest_factor(v)
    rip_i = ripple(i, t)

    lines = []
    lines.append(f"# {name} 波形后处理（仿真结果）")
    lines.append("")
    lines.append("| 指标 | 数值 |")
    lines.append("|---|---|")
    lines.append(f"| 频率 | {f_est:.4g} Hz |")
    lines.append(f"| 直流分量（电压） | {harm_v['dc']:.4g} V |")
    lines.append(f"| 基波幅值（电压） | {harm_v['amp'][0]:.4g} V |")
    lines.append(f"| THD（电压） | {thd(harm_v['amp']):.4g} |")
    lines.append(f"| 有效值（电压） | {v_rms:.4g} V |")
    lines.append(f"| 有效值（电流） | {i_rms:.4g} A |")
    lines.append(f"| 有功功率 | {p:.4g} W |")
    lines.append(f"| 功率因数 | {pf:.4g} |")
    lines.append(f"| 波峰系数（电压） | {cf_v:.4g} |")
    lines.append(f"| 电流纹波（峰峰值） | {rip_i:.4g} A |")
    lines.append("")
    lines.append("以上均为仿真结果，非实测")

    return "\n".join(lines)


def main():
    if len(sys.argv) < 4 or sys.argv[2] != "--f0":
        print("用法: python report.py <文件.csv> --f0 <基频>", file=sys.stderr)
        sys.exit(2)

    csv_path = sys.argv[1]
    try:
        f0 = float(sys.argv[3])
    except ValueError:
        print("错误：f0 必须是数值", file=sys.stderr)
        sys.exit(2)

    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except FileNotFoundError:
        print(f"错误：找不到文件 '{csv_path}'", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"错误：读取文件失败 — {e}", file=sys.stderr)
        sys.exit(2)

    if not rows:
        print("错误：CSV 文件为空", file=sys.stderr)
        sys.exit(2)

    t_arr = np.array([float(r["t"]) for r in rows], dtype=float)
    v_arr = np.array([float(r["v"]) for r in rows], dtype=float)
    i_arr = np.array([float(r["i"]) for r in rows], dtype=float)

    report = make_report(csv_path, t_arr, v_arr, i_arr, f0)
    print(report)


if __name__ == "__main__":
    main()
