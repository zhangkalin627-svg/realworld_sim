"""Real API -> job -> ngspice custom-netlist regression (no solver doubles)."""
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
_tmp = tempfile.TemporaryDirectory(prefix='rw-custom-netlist-')
os.environ['RW_JOBS_DB'] = str(Path(_tmp.name) / 'jobs.db')
from fastapi.testclient import TestClient
from web.server import app, _pool
from rw.circuit.spice import find_ngspice


@unittest.skipUnless(find_ngspice(), 'requires real ngspice')
class CustomNetlistTests(unittest.TestCase):
    client = TestClient(app)
    valid = 'Custom DC oracle\nV1 out 0 DC 2\nR1 out 0 1000\n.end'

    def run_job(self, params):
        response = self.client.post('/api/sim/circuit', json={'params': params})
        self.assertEqual(response.status_code, 200, response.text)
        jid = response.json()['job']
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            result = self.client.get('/api/job/' + jid)
            self.assertEqual(result.status_code, 200)
            job = result.json()
            if job['status'] in ('done', 'error'):
                self.assertEqual(job['module'], 'circuit')
                return job
            time.sleep(0.02)
        self.fail('custom-netlist job timed out')

    def assert_dc(self, params):
        job = self.run_job(params)
        self.assertEqual(job['status'], 'done', job.get('error'))
        result = job['result']
        # An ideal DC voltage source is an independent 2 V RMS, 0 Vpp oracle.
        measure = result['raw']['measurements']['out']
        self.assertAlmostEqual(measure['Vrms'], 2.0, places=8)
        self.assertAlmostEqual(measure['Vpp'], 0.0, places=8)
        self.assertGreater(result['raw']['points'], 10)
        self.assertIn('<svg', result['plots'][0]['svg'])

    def test_valid_default_time_parameters(self):
        self.assert_dc({'custom_netlist': self.valid})

    def test_valid_explicit_string_time_parameters(self):
        self.assert_dc({'custom_netlist': self.valid, 'tstep': '0.000002', 'tstop': '0.0005'})

    def test_invalid_netlist_reaches_ngspice_then_valid_recovers(self):
        job = self.run_job({'custom_netlist': 'Bad model\nV1 out 0 DC 1\nD1 out 0 NO_SUCH_MODEL\n.end'})
        self.assertEqual(job['status'], 'error')
        self.assertIn('ngspice', job['error'])
        self.assertNotIn('could not convert string to float', job['error'])
        self.assert_dc({'custom_netlist': self.valid})


if __name__ == '__main__':
    try:
        unittest.main()
    finally:
        _pool.shutdown(wait=True)
        _tmp.cleanup()
