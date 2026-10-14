"""Real OCC pre-mesh material-disjointness guard, no EM solve."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
try:import gmsh
except ImportError:gmsh=None
from rw.em.eddy3d_gmsh import BilletInCoil3D

@unittest.skipIf(gmsh is None,'requires optional gmsh')
class GeometryTests(unittest.TestCase):
    def test_overlapping_square_rejected(self):
        with self.assertRaisesRegex(ValueError,'重叠'):
            BilletInCoil3D(a=20,ecc_mm=3,h_skin=6,h_coil=6,h_air=35).build()
        self.assertFalse(gmsh.isInitialized())
    def test_overlapping_round_rejected(self):
        with self.assertRaisesRegex(ValueError,'重叠'):
            BilletInCoil3D(a=0,ro=12,ecc_mm=5,h_skin=6,h_coil=6,h_air=35).build()
        self.assertFalse(gmsh.isInitialized())
    def test_disjoint_default_remains_meshable(self):
        s=BilletInCoil3D(h_skin=5,h_coil=5,h_air=30).build()
        import numpy as np
        t=np.sort(s.mesh.t.T,axis=1)
        self.assertEqual(len(np.unique(t,axis=0)),len(t))
        self.assertTrue(s.tube.any())
        self.assertFalse(gmsh.isInitialized())

if __name__=='__main__':unittest.main()
