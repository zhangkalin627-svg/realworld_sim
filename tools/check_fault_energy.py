"""Independent Ri^2 ledger for a pinned 100 kW short-circuit operating point.
Run: python tools/check_fault_energy.py. Not hardware or device certification.
"""
import math,numpy as np
from types import SimpleNamespace
from scipy.linalg import expm
from numpy.polynomial.legendre import leggauss
# Pinned source-derived benchmark, see models/fault_transient_energy.md.
r=SimpleNamespace(L=48.6e-6,C=.932e-6,L_SHORT=1e-9,R_SWITCH=1e-4,
                  R_WP=1.206428095660906,R_TOTAL=.315854+1.206428095660906)
f,V=25123.31431594058,499.80034406002545

def run(dt,delay):
 # Independent loss quadrature, not loss defined by residual.
 nodes,weights=leggauss(48)
 def matrices(after):
  L1=r.L-r.L_SHORT;R1=r.R_TOTAL-r.R_WP
  if after:
   A=np.array([[-(R1+r.R_SWITCH)/L1,-1/L1,r.R_SWITCH/L1],[1/r.C,0,0],[r.R_SWITCH/r.L_SHORT,0,-(r.R_WP+r.R_SWITCH)/r.L_SHORT]])
   B=np.array([1/L1,0,0]);Q=np.array([[R1+r.R_SWITCH,0,-r.R_SWITCH],[0,0,0],[-r.R_SWITCH,0,r.R_WP+r.R_SWITCH]])
  else:
   A=np.array([[-r.R_TOTAL/r.L,-1/r.L,0],[1/r.C,0,0],[0,0,0]])
   B=np.array([1/r.L,0,0]);Q=np.diag([r.R_TOTAL,0,0])
  F=np.zeros((4,4));F[:3,:3]=A;F[:3,3]=B
  EQ=np.zeros((4,4));EQ[:3,:3]=Q
  S=np.zeros((4,4))
  for node,w in zip(nodes,weights):
   E=expm(F*(node+1)*dt/2);S+=E.T@EQ@E*(w*dt/2)
  return expm(F*dt),S
 pre,post=matrices(False),matrices(True)
 def energy(x,after):
  return .5*r.C*x[1]**2+.5*(r.L-r.L_SHORT if after else r.L)*x[0]**2+(.5*r.L_SHORT*x[2]**2 if after else 0)
 x=np.zeros(3);trip=None;off=None;stop=False;src=loss=clipped=0.;parts=np.zeros((3,2));vf=0.;offenergy=None
 for k in range(round(1.8e-3/dt)):
  t=k*dt;after=t>=.4e-3
  if after and k and (k-1)*dt<.4e-3:x[2]=x[0]
  ix=0 if not after else 1 if off is None else 2
  old=x.copy();E,S=post if after else pre
  if stop:
   x[0]=0;x[2]*=math.exp(-dt*(r.R_WP+r.R_SWITCH)/r.L_SHORT) if after else 1
   dl=.5*r.L_SHORT*(old[2]**2-x[2]**2) if after else 0;dw=0.
  else:
   u=(V if t%(1/f)<.5/f else -V) if off is None else -math.copysign(V,x[0]) if abs(x[0])>1e-6 else 0.
   z=np.r_[x,u];xn=(E@z)[:3];dl=float(z@S@z);dw=u*r.C*(xn[1]-x[1]);x=xn
   if off is not None and old[0] and np.sign(old[0])!=np.sign(x[0]):
    clipped+=.5*(r.L-r.L_SHORT if after else r.L)*x[0]**2;x[0]=0;stop=True
  loss+=dl;src+=dw;parts[ix]+=[dw,dl];vf=max(vf,abs(x[1]))
  if off is None:
   if trip is None and (abs(x[0])>480 or abs(x[1])>.95*2000*2**.5):trip=t
   if trip is not None and t>=trip+delay:off=t;offenergy=energy(x,after)
 final=energy(x,True);res=src-loss-clipped-final
 print('dt',dt,'delay',delay,'peak',vf,'off',off,'offenergy',offenergy,'source',src,'integrated_Ri2',loss,'clipped_zero_cross',clipped,'final',final,'residual',res,'segments source/loss',parts.tolist())
 assert abs(res)<2e-8,(res,src,loss)
if __name__ == '__main__':
 for dt in [25e-9,12.5e-9]:run(dt,2e-6)
 run(25e-9,20e-6)
