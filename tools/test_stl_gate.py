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


    def test_or_in_inactive_step_does_not_energize(self):
        # kabu499 发现：OR/ORI/ORP/ORF 未与 STL gate 相与，
        # 失活步内 OR 真触点会把 Y 打通（真实 FX：失活步内条件全按 OFF）。
        plc = mk("LD M8002\nSET S0\nSTL S0\nLD X0\nOR X1\nOUT Y0\nRET\nEND")
        plc.scan({"X0": False, "X1": True})
        self.assertTrue(plc.get("Y0"))          # 活动步：OR 正常
        plc2 = mk("STL S0\nLD X0\nOR X1\nOUT Y0\nRET\nEND")   # S0 从未激活
        for _ in range(3):
            plc2.scan({"X0": False, "X1": True})
            self.assertFalse(plc2.get("Y0"))

    def test_ori_in_inactive_step_does_not_energize(self):
        plc = mk("STL S0\nLD X0\nORI X1\nOUT Y1\nRET\nEND")
        for _ in range(3):
            plc.scan({"X0": False, "X1": False})
            self.assertFalse(plc.get("Y1"))

    def test_orp_orf_in_inactive_step_do_not_fire(self):
        plc = mk("STL S0\nLD X0\nORP X1\nOUT Y2\nRET\nEND")
        plc.scan({"X1": False}); plc.scan({"X1": True})
        self.assertFalse(plc.get("Y2"))
        plc = mk("STL S0\nLD X0\nORF X1\nOUT Y3\nRET\nEND")
        plc.scan({"X1": True}); plc.scan({"X1": False})
        self.assertFalse(plc.get("Y3"))

    def test_or_chain_in_active_step_unaffected(self):
        plc = mk("LD M8002\nSET S0\nSTL S0\nLD X0\nORI X1\nOUT Y4\nRET\nEND")
        plc.scan({"X0": False, "X1": False})
        self.assertTrue(plc.get("Y4"))

    def test_or_leak_cannot_trigger_step_transition(self):
        # 失活步内 OR 泄漏曾经连 SET S 步转移都能误触发
        plc = mk("STL S0\nLD X0\nOR X1\nSET S2\nRET\nSTL S2\nLD M8000\nOUT Y5\nRET\nEND")
        for _ in range(3):
            plc.scan({"X0": False, "X1": True})
        self.assertFalse(plc.get("S2"))
        self.assertFalse(plc.get("Y5"))


if __name__ == "__main__":
    unittest.main()
