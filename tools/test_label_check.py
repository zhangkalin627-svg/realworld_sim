"""test_label_check.py —— label_check 模块的单元测试"""

import unittest
import tempfile
import os
import sys
import subprocess
from label_check import check_text


class TestLabelCheck(unittest.TestCase):

    def test_rule1_fixed_phrases(self):
        text = "仿真实测结果与预期一致。\n模拟实测值偏高。\n仿真测得数据。\n仿真实验测量完成。"
        results = check_text(text)
        self.assertEqual(len(results), 4)
        for r in results:
            self.assertEqual(r["kind"], "error")

    def test_rule1_no_fixed_phrases(self):
        text = "仿真结果与实测值进行了对比。"
        results = check_text(text)
        self.assertFalse(any(
            r["term"] in ["仿真实测", "模拟实测", "仿真测得", "仿真实验测量"]
            for r in results))

    def test_rule2_sim_and_meas_same_line(self):
        text = "仿真模型显示温度升高，实测数据为50度。"
        results = check_text(text)
        errors = [r for r in results
                  if r["kind"] == "error" and r["reason"] == "仿真结果被称为实测"]
        self.assertGreaterEqual(len(errors), 1)

    def test_rule2_negation_exempt(self):
        text = "仿真结果并非实测数据。"
        results = check_text(text)
        self.assertFalse(any(r["reason"] == "仿真结果被称为实测" for r in results))

    def test_rule2_not_sim_but_meas(self):
        text = "我们进行了实测，结果正常。"
        results = check_text(text)
        self.assertFalse(any(r["reason"] == "仿真结果被称为实测" for r in results))

    def test_rule3_only_meas_warn(self):
        text = "根据实测结果，设备运行正常。"
        results = check_text(text)
        warns = [r for r in results
                 if r["kind"] == "warn" and r["reason"] == "请确认确有物理测量数据"]
        self.assertGreaterEqual(len(warns), 1)

    def test_rule3_meas_with_sim_no_warn(self):
        text = "仿真结果与实测结果一致。"
        results = check_text(text)
        warns = [r for r in results
                 if r["kind"] == "warn" and r["reason"] == "请确认确有物理测量数据"]
        self.assertEqual(len(warns), 0)

    def test_rule4_negation_not_reported(self):
        texts = [
            "非实测数据", "未实测", "不是实测值", "并非实测结果",
            "无实测数据", "没有实测数据", "实测值待补", "实测尚无结果",
        ]
        for text in texts:
            results = check_text(text)
            self.assertEqual(len(results), 0, f"不应报错: {text}")

    def test_rule5_code_block_skipped(self):
        text = "```\n仿真结果与实测一致\n```"
        results = check_text(text)
        self.assertEqual(len(results), 0)

    def test_rule5_code_block_mixed(self):
        text = "根据实测结果\n```\n仿真实测\n```\n仿真测得数据"
        results = check_text(text)
        self.assertGreaterEqual(len(results), 2)

    def test_rule6_multiple_hits_one_line(self):
        text = "仿真实测和模拟实测都是错误的。"
        results = check_text(text)
        cols = [r["col"] for r in results]
        self.assertEqual(cols, sorted(cols))

    def test_fullwidth_punctuation(self):
        text = "仿真模型显示，温度升高；实测数据为５０度。"
        results = check_text(text)
        errors = [r for r in results if r["kind"] == "error"]
        self.assertGreaterEqual(len(errors), 1)

    def test_empty_string(self):
        text = ""
        results = check_text(text)
        self.assertEqual(len(results), 0)

    def test_line_number_correct(self):
        text = "正常行\n仿真实测\n模拟实测\n正常行"
        results = check_text(text)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["line"], 2)
        self.assertEqual(results[1]["line"], 3)

    def test_col_correct(self):
        text = "数据仿真实测结果"
        results = check_text(text)
        self.assertGreaterEqual(len(results), 1)
        self.assertEqual(results[0]["col"], 3)

    def test_command_exit_code_no_problem(self):
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8",
                                         delete=False, suffix=".txt") as f:
            f.write("这是一个正常测试文本，没有敏感词。\n")
            tmp_path = f.name
        try:
            result = subprocess.run(
                [sys.executable, "label_check.py", tmp_path],
                capture_output=True, text=True,
                cwd=os.path.dirname(__file__) or ".")
            self.assertEqual(result.returncode, 0)
            self.assertIn("未发现问题", result.stdout)
        finally:
            os.unlink(tmp_path)

    def test_command_exit_code_with_error(self):
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8",
                                         delete=False, suffix=".txt") as f:
            f.write("仿真实测结果。\n")
            tmp_path = f.name
        try:
            result = subprocess.run(
                [sys.executable, "label_check.py", tmp_path],
                capture_output=True, text=True,
                cwd=os.path.dirname(__file__) or ".")
            self.assertEqual(result.returncode, 1)
        finally:
            os.unlink(tmp_path)

    def test_command_exit_code_with_warn(self):
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8",
                                         delete=False, suffix=".txt") as f:
            f.write("根据实测结果。\n")
            tmp_path = f.name
        try:
            result = subprocess.run(
                [sys.executable, "label_check.py", tmp_path],
                capture_output=True, text=True,
                cwd=os.path.dirname(__file__) or ".")
            self.assertEqual(result.returncode, 0)
        finally:
            os.unlink(tmp_path)


if __name__ == "__main__":
    unittest.main()
