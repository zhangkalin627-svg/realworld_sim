"""Batch-control completion and failure guards. Requires ngspice."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rw.circuit.spice import Circuit, SpiceError, check_exit

class BatchTests(unittest.TestCase):
    def test_passive_transient(self):
        c = Circuit("passive")
        c.R("1", "a", "0", 1000)
        c.C("1", "a", "b", 1e-6)
        c.R("2", "b", "0", 1000)
        w = c.tran(1e-6, 1e-3)
        self.assertGreater(len(w.time), 10)
        self.assertTrue(any("无激励源" in x for x in w.warnings))

    def test_valid_voltage(self):
        c = Circuit("valid")
        c.V("1", "a", "0", dc=1)
        c.R("1", "a", "0", 1000)
        w = c.tran(1e-6, 1e-3)
        self.assertAlmostEqual(float(w["a"][-1]), 1)

    def test_invalid_model_rejected(self):
        c = Circuit("invalid")
        c.V("1", "a", "0", dc=1)
        c.D("1", "a", "0", "NO_SUCH_MODEL")
        with self.assertRaises(SpiceError):
            c.tran(1e-6, 1e-3)

    def test_invalid_analysis_rejected(self):
        c = Circuit("invalid analysis")
        c.V("1", "a", "0", dc=1)
        c.R("1", "a", "0", 1000)
        with self.assertRaises(SpiceError):
            c.run(".tran -1 1m")

    def test_nonzero_and_kill_remain_rejected(self):
        for rc in (1, 2, -9, 137):
            with self.subTest(rc=rc), self.assertRaises(SpiceError):
                check_exit(SimpleNamespace(returncode=rc, stdout="raw output exists", stderr=""))

if __name__ == "__main__":
    unittest.main()
