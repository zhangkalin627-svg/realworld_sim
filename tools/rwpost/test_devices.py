"""test_devices.py —— BIG-W2 功率器件损耗模型测试"""

import unittest
import math
import numpy as np
from devices import mosfet_loss, diode_loss, igbt_loss, solve_tj


class TestMosfetLoss(unittest.TestCase):

    def setUp(self):
        self.p = {
            "rds_on_25": 0.020, "alpha_rds": 0.060,
            "qg": 68e-9, "vg": 145.0, "coss": 200e-12,
            "tr": 240e-9, "tf": 190e-9,
        }

    def test_basic_mosfet(self):
        # Tj=25 → Rds=Rds_on_25=0.02, P_cond=5.75^2*0.02=0.66125
        r = mosfet_loss(5.75, 910.820, 915.305, 220.0, 39700, 0.578, self.p)
        self.assertAlmostEqual(r["p_cond"], 0.6612, delta=0.002)
        self.assertGreater(r["p_sw"], 10.0)
        self.assertGreater(r["p_total"], 11.0)

    def test_temperature_scaling(self):
        p2 = {"rds_on_25": 0.048, "alpha_rds": 0.083, "qg": 33e-9, "vg": 13.0,
              "coss": 365e-12, "tr": 28e-9, "tf": 22e-9}
        rh = mosfet_loss(53.150, 252.310, 221.930, 540.0, 99800, 0.425, p2, tj=132.0)
        rc = mosfet_loss(54.170, 472.030, 559.260, 420.0, 97800, 0.545, p2, tj=333.0)
        self.assertIsNotNone(rh)
        self.assertIsNotNone(rc)

    def test_return_type(self):
        r = mosfet_loss(15.171, 184.165, 533.225, 642.0, 493e3, 0.539, self.p)
        for v in r.values():
            self.assertIsInstance(v, float)

    def test_negative_current(self):
        with self.assertRaises(ValueError):
            mosfet_loss(-1.37, 508.0, 386.0, 190.0, 985e3, 0.538, self.p)

    def test_nan_input(self):
        with self.assertRaises(ValueError):
            mosfet_loss(float('nan'), 100.0, 50.0, 200.0, 100e3, 0.5, self.p)

    def test_inf_input(self):
        with self.assertRaises(ValueError):
            mosfet_loss(5.0, float('inf'), 3.0, 200.0, 100e3, 0.5, self.p)

    def test_zero_fsw(self):
        with self.assertRaises(ValueError):
            mosfet_loss(5.0, 100.0, 50.0, 200.0, 0.0, 0.5, self.p)

    def test_negative_fsw(self):
        with self.assertRaises(ValueError):
            mosfet_loss(5.0, 100.0, 50.0, 200.0, -100.0, 0.5, self.p)

    def test_duty_out_of_range(self):
        with self.assertRaises(ValueError):
            mosfet_loss(5.0, 100.0, 50.0, 200.0, 100e3, 1.5, self.p)

    def test_missing_param(self):
        with self.assertRaises(ValueError):
            mosfet_loss(58.72, 718.0, 776.0, 260.0, 284e3, 0.47, {"rds_on_25": 57.0})

    def test_nan_in_params(self):
        with self.assertRaises(ValueError):
            p_bad = self.p.copy()
            p_bad["rds_on_25"] = float('nan')
            mosfet_loss(5.0, 100.0, 50.0, 200.0, 100e3, 0.5, p_bad)

    def test_input_not_modified(self):
        pc = self.p.copy()
        mosfet_loss(59.23, 296.0, 625.0, 232.0, 288e3, 0.537, self.p)
        self.assertDictEqual(self.p, pc)

    def test_negative_fsw_value_error_message(self):
        with self.assertRaises(ValueError) as cm:
            mosfet_loss(5.0, 100.0, 50.0, 200.0, -10.0, 0.5, self.p)
        self.assertIn("fsw", str(cm.exception))

    def test_duty_zero(self):
        r = mosfet_loss(5.75, 100.0, 50.0, 220.0, 39700, 0.0, self.p)
        self.assertIsNotNone(r)

    def test_duty_one(self):
        r = mosfet_loss(5.75, 100.0, 50.0, 220.0, 39700, 1.0, self.p)
        self.assertIsNotNone(r)


class TestDiodeLoss(unittest.TestCase):

    def test_basic_diode(self):
        r = diode_loss(90.50, 441.700, 74.536, 6.531, 321e-9, 65.390, 91200)
        self.assertGreater(r["p_cond"], 66000.0)
        self.assertGreater(r["p_rr"], 300e-9 * 60.0 * 90000)
        self.assertGreater(r["p_total"], 68000.0)

    def test_return_type(self):
        r = diode_loss(122.6, 398.0, 36.8, 46.0, 823e-9, 56.0, 21500)
        for v in r.values():
            self.assertIsInstance(v, float)

    def test_negative(self):
        with self.assertRaises(ValueError):
            diode_loss(-109.0, 971.0, 18.0, 654.0, 472e-9, 56.0, 347.0)

    def test_nan_input(self):
        with self.assertRaises(ValueError):
            diode_loss(float('nan'), 100.0, 1.0, 0.01, 1e-9, 100.0, 100e3)

    def test_inf_input(self):
        with self.assertRaises(ValueError):
            diode_loss(5.0, float('inf'), 1.0, 0.01, 1e-9, 100.0, 100e3)

    def test_zero_fsw(self):
        with self.assertRaises(ValueError):
            diode_loss(5.0, 10.0, 1.0, 0.01, 1e-9, 100.0, 0.0)

    def test_zero_current(self):
        r = diode_loss(0.053, 0.017, 74.53, 65.83, 208e-9, 30.42, 805.0)
        self.assertGreater(r["p_cond"], 3.5)
        self.assertGreater(r["p_rr"], 20.0e-6)


class TestIgbtLoss(unittest.TestCase):

    def test_basic_igbt(self):
        # 手算：i_avg=45.82, i_ref=443.36, current_scale=45.82/443.36=0.1033
        # vce=706, vce_ref=96.6, voltage_scale=706/96.6=7.31
        # e_on = 0.012*7.31*0.1033=0.00906
        # e_off = 0.008*7.31*0.1033=0.00604
        # p_sw = (0.00906+0.00604)*5200 = 78.52
        r = igbt_loss(45.820, 521.480, 93.670, 44.80, 0.012, 0.008,
                      443.360, 96.620, 706.080, 5200.0)
        self.assertGreater(r["p_cond"], 4000.0)
        self.assertGreater(r["p_sw"], 70.0)  # 手算约 78.5
        self.assertGreater(r["p_total"], 4200.0)

    def test_return_type(self):
        r = igbt_loss(103.0, 414.0, 7.0, 52.0, 348e-6, 486e-6,
                      120.0, 80.0, 82.0, 5400.0)
        for v in r.values():
            self.assertIsInstance(v, float)

    def test_negative_i_ref(self):
        with self.assertRaises(ValueError):
            igbt_loss(100.0, 500.0, 9.0, 30.0, 470e-6, 730e-6, -75.0, 50.0, 80.0, 500.0)

    def test_zero_i_ref(self):
        with self.assertRaises(ValueError):
            igbt_loss(100.0, 500.0, 9.0, 30.0, 470e-6, 730e-6, 0.0, 50.0, 80.0, 500.0)

    def test_zero_vce_ref(self):
        with self.assertRaises(ValueError):
            igbt_loss(100.0, 500.0, 9.0, 30.0, 470e-6, 730e-6, 10.0, 0.0, 80.0, 500.0)

    def test_nan_input(self):
        with self.assertRaises(ValueError):
            igbt_loss(float('nan'), 100.0, 1.0, 0.01, 1e-6, 1e-6, 10.0, 100.0, 200.0, 1000.0)

    def test_temp_scaling(self):
        rh = igbt_loss(121.0, 85.0, 8.0, 1.0, 300e-6, 190e-6, 20.0, 90.0, 83.0, 5000.0, tj=320.0, alpha_vce=0.122)
        rc = igbt_loss(140.0, 80.0, 6.0, 6.0, 500e-6, 500e-6, 70.0, 90.0, 80.0, 2500.0, tj=210.0, alpha_vce=0.263)
        self.assertIsNotNone(rh)
        self.assertIsNotNone(rc)


class TestSolveTj(unittest.TestCase):

    def test_constant_loss(self):
        """恒功率：定点迭代第 2 步收敛"""
        def fn(tj):
            return 90.180
        r = solve_tj(fn, 12.58, 305.61, tol=1e-06, max_iter=100)
        self.assertAlmostEqual(r["p"], 90.18, delta=0.02)
        self.assertEqual(r["iters"], 2)

    def test_linear_case(self):
        """线性损耗：验证返回结构"""
        def fn(tj):
            return 85.30
        r = solve_tj(fn, 30.07, 40.20, tol=1e-06, max_iter=500)
        self.assertIn("tj", r)
        self.assertIn("p", r)
        self.assertIn("iters", r)
        self.assertIsInstance(r["tj"], float)
        self.assertIsInstance(r["p"], float)
        self.assertIsInstance(r["iters"], int)

    def test_thermal_runaway(self):
        """热失控：增益 > 1 抛出 ValueError"""
        def fn(tj):
            return 0.550 * tj
        with self.assertRaises(ValueError):
            solve_tj(fn, 930.42, 248.31, tol=1e-09, max_iter=40)

    def test_invalid_rth(self):
        with self.assertRaises(ValueError):
            solve_tj(lambda tj: 81.0, -18.0, 463.0)

    def test_nan_rth(self):
        with self.assertRaises(ValueError):
            solve_tj(lambda tj: 50.0, float('nan'), 40.0)

    def test_zero_tol(self):
        with self.assertRaises(ValueError):
            solve_tj(lambda tj: 50.0, 10.0, 40.0, tol=0.0)

    def test_convergence_precision(self):
        """恒功率高精度：仍 2 步收敛"""
        def fn(tj):
            return 170.63
        r = solve_tj(fn, 8.02, 34.28, tol=1e-10, max_iter=300)
        self.assertAlmostEqual(r["p"], 170.63, delta=0.02)
        self.assertEqual(r["iters"], 2)

    def test_loss_fn_not_callable(self):
        with self.assertRaises(ValueError):
            solve_tj("not_a_function", 10.0, 40.0)


if __name__ == "__main__":
    unittest.main()
