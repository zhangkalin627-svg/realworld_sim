import sys,os,time
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),".."))
import numpy as np
from rw.induction import *
s=InductionQuenchSim(freq=30e3,power_kw=6,nonlinear=True)
for T in (25,25,60,100,150,200,300,400,500,600,700,740,760,780,800):
    t=time.time(); s.solve_em(np.full(s.tb.N,float(T))); print(T,s.n_iter_em,round(time.time()-t,1),flush=True)
