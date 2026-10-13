"""PLC load validation regression: python tools/test_plc_load.py."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rw.plc.fx3u import FX3U


class EmptyProgramTests(unittest.TestCase):
    def test_empty_or_comment_only_reports_business_error(self):
        for text in ("", " ", "; comment", "\n; comment\n\t"):
            with self.subTest(text=text):
                with self.assertRaisesRegex(ValueError, "第1行.*程序为空"):
                    FX3U(text.splitlines())

    def test_end_only_remains_valid(self):
        plc = FX3U(["END"])
        plc.scan({})
        self.assertEqual(plc.t_ms, 10)

    def test_valid_program_still_scans(self):
        plc = FX3U(["LD X000", "OUT Y000", "END"])
        plc.scan({"X000": True})
        self.assertTrue(plc.get("Y000"))
        plc.scan({"X000": False})
        self.assertFalse(plc.get("Y000"))

    def test_missing_end_still_rejected(self):
        with self.assertRaisesRegex(ValueError, "END"):
            FX3U(["LD X000"])


if __name__ == "__main__":
    unittest.main()
