import sys,os
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),".."))
import numpy as np
from rw.em.eddy3d import *
from rw.em.eddy3d import _cc
p=TubeInCoil3D(nth=12,fine=3.0,ecc_mm=3.0).build()
b=p.basis; K=asm(_cc,b,nu=p._el_to_qp(p.nu)); G=p._incidence()
ph=np.random.rand(G.shape[1]); v=G@ph
print('|K G phi|/|K|*..', np.linalg.norm(K@v)/ (np.linalg.norm(v)*abs(K).max()), 'edges',b.N)
G2=G.copy(); 
