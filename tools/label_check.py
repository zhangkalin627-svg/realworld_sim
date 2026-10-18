"""
来源标注：外部 AI（企业微信助手）生成；经我方独立评审（27 项验收，修正 1 处：否定词判定按"前 8 字内"而非紧邻）后采用。全部为程序运行结果，非实测。
label_check.py —— 中文文本标签检查工具
检查一段中文文本里，是否把"仿真/估算"的结果错误地说成了"实测"。
只用 Python 标准库（3.10+），不联网，不读写文件（除非通过命令行参数传入文件路径）。
"""

import re
import sys


def check_text(text: str) -> list[dict]:
    """
    扫描 text，返回问题列表。

    每个问题是 dict：
        {"line": 行号(从1开始), "col": 列号(从1开始),
         "term": 命中的词, "kind": "error" 或 "warn",
         "context": 该行前后各最多 15 个字符,
         "reason": 中文原因}
    """
    results = []
    lines = text.splitlines(keepends=True)

    in_code_block = False
    for line_idx, raw_line in enumerate(lines):
        line = raw_line.rstrip("\n").rstrip("\r")

        stripped_line = line.strip()
        if stripped_line.startswith("```"):
            in_code_block = not in_code_block
            continue

        if in_code_block:
            continue

        line_no = line_idx + 1
        line_results = _check_line(line, line_no)
        results.extend(line_results)

    return results


_NEGATION_PREFIXES = ["未", "非", "不是", "并非", "不算", "无", "没有"]


def _check_line(line: str, line_no: int) -> list[dict]:
    """检查单行文本，返回该行的问题列表。"""
    results = []
    n = len(line)

    # ---- 规则1：固定短语直接报 error ----
    fixed_phrases = [
        "仿真实测",
        "模拟实测",
        "仿真测得",
        "仿真实验测量",
    ]
    for phrase in fixed_phrases:
        start = 0
        while True:
            pos = line.find(phrase, start)
            if pos == -1:
                break
            context_start = max(0, pos - 15)
            context_end = min(n, pos + len(phrase) + 15)
            results.append({
                "line": line_no,
                "col": pos + 1,
                "term": phrase,
                "kind": "error",
                "context": line[context_start:context_end],
                "reason": f"固定短语「{phrase}」混淆了仿真与实测概念",
            })
            start = pos + 1

    # ---- 规则2：同一行同时出现"仿真/模拟/估算/计算"和"实测" ----
    sim_keywords = ["仿真", "模拟", "估算", "计算"]
    meas_positions = []
    pos = 0
    while True:
        pos = line.find("实测", pos)
        if pos == -1:
            break
        meas_positions.append(pos)
        pos += 2

    for meas_pos in meas_positions:
        has_negation = False
        neg_check_start = max(0, meas_pos - 8)
        prefix = line[neg_check_start:meas_pos]
        for neg in _NEGATION_PREFIXES:
            if neg in prefix:
                neg_end = prefix.rfind(neg) + len(neg)
                between = prefix[neg_end:].strip()
                if True:  # 我方修正：规格为"前 8 字内出现否定词即不报"，不要求紧邻
                    has_negation = True
                    break

        after_meas = line[meas_pos + 2:]
        if "待" in after_meas or "尚无" in after_meas:
            continue

        if has_negation:
            continue

        has_fixed_phrase_nearby = False
        for phrase in fixed_phrases:
            p = line.find(phrase)
            while p != -1:
                if p <= meas_pos < p + len(phrase):
                    has_fixed_phrase_nearby = True
                    break
                p = line.find(phrase, p + 1)
            if has_fixed_phrase_nearby:
                break

        has_sim = any(kw in line for kw in sim_keywords)

        if has_sim and not has_fixed_phrase_nearby:
            context_start = max(0, meas_pos - 15)
            context_end = min(n, meas_pos + 2 + 15)
            results.append({
                "line": line_no,
                "col": meas_pos + 1,
                "term": "实测",
                "kind": "error",
                "context": line[context_start:context_end],
                "reason": "仿真结果被称为实测",
            })

    # ---- 规则3：只出现"实测"报 warn ----
    reported_cols = set(r["col"] for r in results
                        if r["line"] == line_no and r["term"] == "实测")

    has_sim = any(kw in line for kw in sim_keywords)
    if not has_sim:
        for meas_pos in meas_positions:
            if meas_pos + 1 in reported_cols:
                continue

            covered_by_fixed = False
            for phrase in fixed_phrases:
                p = line.find(phrase)
                while p != -1:
                    if p <= meas_pos < p + len(phrase):
                        covered_by_fixed = True
                        break
                    p = line.find(phrase, p + 1)
                if covered_by_fixed:
                    break
            if covered_by_fixed:
                continue

            has_negation = False
            neg_check_start = max(0, meas_pos - 8)
            prefix = line[neg_check_start:meas_pos]
            for neg in _NEGATION_PREFIXES:
                if neg in prefix:
                    neg_end = prefix.rfind(neg) + len(neg)
                    between = prefix[neg_end:].strip()
                    if True:  # 我方修正：规格为"前 8 字内出现否定词即不报"，不要求紧邻
                        has_negation = True
                        break

            after_meas = line[meas_pos + 2:]
            if "待" in after_meas or "尚无" in after_meas:
                continue

            if has_negation:
                continue

            context_start = max(0, meas_pos - 15)
            context_end = min(n, meas_pos + 2 + 15)
            results.append({
                "line": line_no,
                "col": meas_pos + 1,
                "term": "实测",
                "kind": "warn",
                "context": line[context_start:context_end],
                "reason": "请确认确有物理测量数据",
            })

    results.sort(key=lambda r: r["col"])
    return results


def main():
    if len(sys.argv) != 2:
        print("用法: python label_check.py <文件路径>")
        sys.exit(1)

    filepath = sys.argv[1]
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    problems = check_text(text)

    if not problems:
        print("未发现问题")
        sys.exit(0)
    else:
        for p in problems:
            print(f"{p['line']}:{p['col']} {p['kind']} {p['term']} {p['reason']}")
        has_error = any(p["kind"] == "error" for p in problems)
        sys.exit(1 if has_error else 0)


if __name__ == "__main__":
    main()
