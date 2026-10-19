"""
test_rwpost.py —— BIG-W1 波形后处理工具包集成测试
本工具由 AI 助手（企业微信助手）编写，经我方独立验收（accept_rwpost.py，45 项）后采用；全部为程序运行结果，非实测
"""

import unittest
import numpy as np
import sys
import os
import subprocess
import tempfile
import csv
import time
import math

from waveform_metrics import (
    zero_crossings, frequency, fft_harmonics, thd, ripple
)
from rms_stats import rms, avg, crest_factor, power_factor
from report import make_report


class TestWaveformMetrics(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)

    def test_zero_crossings_sine(self):
        t = np.linspace(0.001, 0.041, 2000)
        x = np.sin(2 * np.pi * 50 * t)
        zc = zero_crossings(t, x, rising=True)
        self.assertAlmostEqual(len(zc), 2)

    def test_zero_crossings_falling(self):
        t = np.linspace(0.001, 0.041, 4000)
        x = np.sin(2 * np.pi * 50 * t)
        zc = zero_crossings(t, x, rising=False)
        self.assertAlmostEqual(len(zc), 2)

    def test_zero_crossings_no_repeat(self):
        t = np.array([0.0, 0.005, 0.01, 0.015, 0.02])
        x = np.array([0.0, 1.0, 0.0, -1.0, 0.0])
        zc = zero_crossings(t, x, rising=True)
        self.assertEqual(len(zc), 1)
        self.assertAlmostEqual(zc[0], 0.02, delta=1e-6)

    def test_frequency_sine(self):
        fs = 5000
        f_signal = 60.0
        t = np.arange(0.0005, 0.055, 1.0 / fs)
        x = np.sin(2 * np.pi * f_signal * t)
        f_est = frequency(t, x)
        self.assertAlmostEqual(f_est, f_signal, delta=0.5)

    def test_frequency_few_zero_crossings(self):
        t = np.array([0.0, 0.002, 0.004])
        x = np.array([1.0, 0.5, -1.0])
        with self.assertRaises(ValueError):
            frequency(t, x)

    def test_fft_harmonics_sine(self):
        f0 = 125.0
        T0 = 1.0 / f0
        n_periods = 10
        fs = 2500
        dt = 1.0 / fs
        n_total = int(n_periods * T0 / dt) + 1
        t = np.linspace(0, n_periods * T0, n_total)
        x = 10.0 * np.cos(2 * np.pi * f0 * t)
        harm = fft_harmonics(t, x, f0, n_harm=5)
        self.assertAlmostEqual(harm["dc"], 0.0, delta=1e-12)
        self.assertAlmostEqual(harm["amp"][0], 10.0, delta=1e-6)
        self.assertAlmostEqual(harm["phase"][0], 0.0, delta=1e-6)

    def test_fft_harmonics_with_dc(self):
        f0 = 100.0
        T0 = 1.0 / f0
        n_periods = 8
        fs = 2000
        dt = 1.0 / fs
        n_total = int(n_periods * T0 / dt) + 1
        t = np.linspace(0, n_periods * T0, n_total)
        x = 5.0 + 10.0 * np.cos(2 * np.pi * f0 * t)
        harm = fft_harmonics(t, x, f0, n_harm=5)
        self.assertAlmostEqual(harm["dc"], 5.0, delta=1e-8)

    def test_fft_harmonics_third_harmonic(self):
        f0 = 150.0
        T0 = 1.0 / f0
        n_periods = 8
        fs = 3000
        dt = 1.0 / fs
        n_total = int(n_periods * T0 / dt) + 1
        t = np.linspace(0, n_periods * T0, n_total)
        x = 10.0 * np.cos(2 * np.pi * f0 * t) + 3.0 * np.cos(2 * np.pi * 3 * f0 * t)
        harm = fft_harmonics(t, x, f0, n_harm=5)
        self.assertAlmostEqual(harm["amp"][0], 10.0, delta=0.1)
        self.assertAlmostEqual(harm["amp"][2], 3.0, delta=0.1)
        self.assertAlmostEqual(thd(harm["amp"]), 0.3, delta=1e-4)

    def test_fft_nonuniform_sampling(self):
        t = np.array([0.0, 0.001, 0.003, 0.004])
        x = np.sin(2 * np.pi * 50 * t)
        with self.assertRaises(ValueError):
            fft_harmonics(t, x, 50.0)

    def test_thd_sine(self):
        self.assertAlmostEqual(thd([10.0]), 0.0)

    def test_thd_zero_fundamental(self):
        with self.assertRaises(ValueError):
            thd([0.0, 1.0])

    def test_ripple_basic(self):
        x = np.array([1.0, 2.0, 3.0, 0.0, 5.0])
        self.assertAlmostEqual(ripple(x), 5.0)

    def test_ripple_window(self):
        t = np.array([0.15, 0.20, 0.25, 0.30, 0.35])
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertAlmostEqual(ripple(x, t, (0.17, 0.33)), 2.0)

    def test_ripple_window_empty(self):
        t = np.array([0.0, 0.1, 0.2])
        x = np.array([1.0, 2.0, 3.0])
        with self.assertRaises(ValueError):
            ripple(x, t, (10.0, 20.0))

    def test_invalid_length_mismatch(self):
        t = np.array([0.0, 0.1, 0.2])
        x = np.array([1.0, 2.0])
        with self.assertRaises(ValueError):
            zero_crossings(t, x)

    def test_invalid_not_strictly_increasing(self):
        t = np.array([0.0, 0.1, 0.1, 0.2])
        x = np.array([1.0, 2.0, 3.0, 4.0])
        with self.assertRaises(ValueError):
            zero_crossings(t, x)

    def test_invalid_nan(self):
        t = np.array([0.0, 0.1, 0.2])
        x = np.array([1.0, np.nan, 3.0])
        with self.assertRaises(ValueError):
            zero_crossings(t, x)

    def test_invalid_n_harm(self):
        t = np.linspace(0.001, 0.061, 1200)
        x = np.sin(2 * np.pi * 50 * t)
        with self.assertRaises(ValueError):
            fft_harmonics(t, x, 50.0, n_harm=0)

    def test_invalid_f0_nonpositive(self):
        t = np.linspace(0.001, 0.061, 1200)
        x = np.sin(2 * np.pi * 50 * t)
        with self.assertRaises(ValueError):
            fft_harmonics(t, x, 0.0)

    def test_input_not_modified(self):
        t = np.linspace(0.003, 0.083, 8000)
        x = np.sin(2 * np.pi * 50 * t)
        t_copy = t.copy()
        x_copy = x.copy()
        zero_crossings(t, x)
        np.testing.assert_array_equal(t, t_copy)
        np.testing.assert_array_equal(x, x_copy)
        frequency(t, x)
        np.testing.assert_array_equal(t, t_copy)
        np.testing.assert_array_equal(x, x_copy)

    def test_return_type_python_float(self):
        t = np.linspace(0.005, 0.085, 16000)
        x = np.sin(2 * np.pi * 50 * t)
        r = ripple(x)
        self.assertIsInstance(r, float)
        f = frequency(t, x)
        self.assertIsInstance(f, float)
        h = thd([10.0, 1.0])
        self.assertIsInstance(h, float)

    def test_speed_1e6(self):
        n = 1_000_000
        t = np.linspace(0.007, 0.167, n)
        x = np.sin(2 * np.pi * 50 * t)
        start = time.time()
        zc = zero_crossings(t, x)
        elapsed = time.time() - start
        self.assertLess(elapsed, 2.0, f"zero_crossings 耗时 {elapsed:.2f}s")

        start = time.time()
        f = frequency(t, x)
        elapsed = time.time() - start
        self.assertLess(elapsed, 2.0, f"frequency 耗时 {elapsed:.2f}s")


class TestRmsStats(unittest.TestCase):

    def test_rms_sine(self):
        t = np.linspace(0, 0.1, 10000)
        v = 10.0 * np.sin(2 * np.pi * 50 * t)
        self.assertAlmostEqual(rms(v, t), 10.0 / np.sqrt(2), delta=0.01)

    def test_rms_no_time(self):
        x = np.array([1.0, -1.0, 1.0, -1.0])
        self.assertAlmostEqual(rms(x), 1.0, delta=1e-10)

    def test_avg_sine(self):
        t = np.linspace(0, 0.1, 10000)
        v = 10.0 * np.sin(2 * np.pi * 50 * t)
        self.assertAlmostEqual(avg(v, t), 0.0, delta=0.01)

    def test_avg_no_time(self):
        x = np.array([1.0, 2.0, 3.0])
        self.assertAlmostEqual(avg(x), 2.0)

    def test_crest_factor_sine(self):
        t = np.linspace(0, 0.1, 10000)
        v = 10.0 * np.sin(2 * np.pi * 50 * t)
        self.assertAlmostEqual(crest_factor(v), np.sqrt(2), delta=0.01)

    def test_crest_factor_zero_rms(self):
        x = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        with self.assertRaises(ValueError):
            crest_factor(x)

    def test_power_factor_resistive(self):
        t = np.linspace(0, 0.1, 10000)
        v = 10.0 * np.sin(2 * np.pi * 50 * t)
        i = v / 10.0
        self.assertAlmostEqual(power_factor(v, i, t), 1.0, delta=0.01)

    def test_power_factor_lagging(self):
        t = np.linspace(0, 0.1, 10000)
        v = 10.0 * np.sin(2 * np.pi * 50 * t)
        phi = np.pi / 6
        i = 5.0 * np.sin(2 * np.pi * 50 * t - phi)
        pf = power_factor(v, i, t)
        self.assertAlmostEqual(pf, np.cos(phi), delta=0.02)

    def test_power_factor_zero_voltage(self):
        t = np.linspace(0, 0.1, 100)
        v = np.zeros(100)
        i = np.ones(100)
        with self.assertRaises(ValueError):
            power_factor(v, i, t)

    def test_input_not_modified(self):
        t = np.linspace(0, 0.1, 1000)
        v = np.sin(2 * np.pi * 50 * t)
        t_copy = t.copy()
        v_copy = v.copy()
        rms(v, t)
        np.testing.assert_array_equal(t, t_copy)
        np.testing.assert_array_equal(v, v_copy)

    def test_return_type_python_float(self):
        t = np.linspace(0, 0.1, 1000)
        v = np.sin(2 * np.pi * 50 * t)
        self.assertIsInstance(rms(v, t), float)
        self.assertIsInstance(avg(v, t), float)
        self.assertIsInstance(crest_factor(v), float)
        i = np.cos(2 * np.pi * 50 * t)
        self.assertIsInstance(power_factor(v, i, t), float)


class TestReport(unittest.TestCase):

    def test_report_contains_warning(self):
        fs = 10000
        t = np.linspace(0, 0.1, int(fs * 0.1))
        v = 10.0 * np.sin(2 * np.pi * 50 * t)
        i = 5.0 * np.sin(2 * np.pi * 50 * t)
        report = make_report("test", t, v, i, 50.0)
        self.assertIn("以上均为仿真结果，非实测", report)
        self.assertIn("test", report)
        self.assertIn("功率因数", report)
        self.assertIn("THD", report)

    def test_cli_csv(self):
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8",
                                         delete=False, suffix=".csv") as f:
            f.write("t,v,i\n")
            fs = 10000
            for n in range(int(fs * 0.1)):
                t_val = n / fs
                v_val = 10.0 * np.sin(2 * np.pi * 50 * t_val)
                i_val = 5.0 * np.sin(2 * np.pi * 50 * t_val)
                f.write(f"{t_val},{v_val},{i_val}\n")
            tmp_path = f.name
        try:
            result = subprocess.run(
                [sys.executable, "report.py", tmp_path, "--f0", "50"],
                capture_output=True, text=True,
                cwd=os.path.dirname(__file__) or "."
            )
            self.assertIn("以上均为仿真结果，非实测", result.stdout)
            self.assertEqual(result.returncode, 0)
        finally:
            os.unlink(tmp_path)

    def test_cli_file_not_found(self):
        result = subprocess.run(
            [sys.executable, "report.py", "nonexistent.csv", "--f0", "50"],
            capture_output=True, text=True,
            cwd=os.path.dirname(__file__) or "."
        )
        self.assertEqual(result.returncode, 2)


if __name__ == "__main__":
    unittest.main()
