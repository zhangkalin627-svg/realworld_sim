"""STL 步进门控回归：非活动步内的边沿触点(LDP/LDF)不得触发。python tools/test_stl_gate.py"""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rw.plc.fx3u import FX3U


def mk(text):
    return FX3U(text.splitlines(), scan_ms=10)


class StlGateTests(unittest.TestCase):
    def test_ldp_in_inactive_step_does_not_fire(self):
        # 卡布11号发现：旧实现 LDP/LDF 未与 STL gate 相与，
        # 非活动步内输入上升沿会把 Y 打通一个扫描（真实 FX 不允许）。
        plc = mk("LD M8002\nSET S0\nSTL S1\nLDP X0\nOUT Y0\nRET\nEND")
        plc.scan({"X0": True})
        self.assertFalse(plc.get("Y0"))
        plc.scan({"X0": False})
        plc.scan({"X0": True})
        self.assertFalse(plc.get("Y0"))

    def test_ldf_in_inactive_step_does_not_fire(self):
        plc = mk("STL S2\nLDF X0\nOUT Y2\nRET\nEND")
        plc.scan({"X0": True})
        plc.scan({"X0": False})
        self.assertFalse(plc.get("Y2"))

    def test_ldp_in_active_step_still_fires_once(self):
        plc = mk("LD M8002\nSET S1\nSTL S1\nLDP X0\nOUT Y0\nRET\nEND")
        plc.scan({"X0": False})
        plc.scan({"X0": True})
        self.assertTrue(plc.get("Y0"))

    def test_level_contacts_unaffected(self):
        plc = mk("STL S1\nLD X0\nOUT Y1\nRET\nEND")
        plc.scan({"X0": True})
        self.assertFalse(plc.get("Y1"))


if __name__ == "__main__":
    unittest.main()
