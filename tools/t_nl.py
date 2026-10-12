import sys,os,time
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),".."))
from rw.induction import *
t=time.time()
s=InductionQuenchSim(freq=30e3,power_kw=6,t_heat=3.0,t_quench=0.5,t_air=0.2,nonlinear=True)
r=s.run(); print(round(time.time()-t),{k:round(r[k],2) for k in ['T_surf_end_heat','T_inner_end_heat','coil_current_A','efficiency']},[round(i) for i in s.I_hist[:3]],flush=True)
