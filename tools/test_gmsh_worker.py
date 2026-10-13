"""Optional gmsh worker regression; mesh build only (no expensive solve)."""
import sys
import unittest
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    import gmsh
except ImportError:
    gmsh = None
from rw.em.eddy3d_gmsh import BilletInCoil3D

@unittest.skipIf(gmsh is None, "optional gmsh is not installed")
class WorkerTests(unittest.TestCase):
    def test_two_worker_meshes_are_serialized(self):
        with ThreadPoolExecutor(2) as pool:
            futures = [pool.submit(BilletInCoil3D().build) for _ in range(2)]
            meshes = [f.result(timeout=60) for f in futures]
        for m in meshes:
            self.assertGreater(m.mesh.nelements, 0)
            self.assertTrue(m.tube.any())
        self.assertFalse(gmsh.isInitialized())

if __name__ == "__main__":
    unittest.main()
