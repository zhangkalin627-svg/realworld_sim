"""L = 2W/I^2 magnetostatic axisym check with CloudFEM (sigma=0, omega=0), half-domain z>=0 (mirror symmetry)."""
import sys,time,math,numpy as np
sys.path.insert(0,'.')
from cloudfem.mesh.tri_mesh import rect_mesh
from cloudfem.physics.axisym import solve_eddy_current_axi, MU0
import rw.coil_inductance as ci
def fem_L(rc,length,t,N,h,Rb,Zb):
    I=1.0; area=t*length; J=N*I/area
    mesh=rect_mesh(int(round(Rb/h)),int(round(Zb/h)),xlim=(0,Rb),ylim=(0,Zb))
    def Jf(r,z): return J if (rc-t/2<=r<=rc+t/2 and z<=length/2) else 0.0
    u=solve_eddy_current_axi(mesh,J_phi=Jf,bc={'left':0.0,'right':0.0,'top':0.0})
    W=0.0
    for el in mesh.elements:
        p=mesh.nodes[el]; r=p[:,0].mean(); z=p[:,1].mean(); j=Jf(r,z)
        if j==0: continue
        a=np.real(u[el]).mean()/r
        ar=0.5*abs((p[1][0]-p[0][0])*(p[2][1]-p[0][1])-(p[1][1]-p[0][1])*(p[2][0]-p[0][0]))
        W+=0.5*j*a*2*math.pi*r*ar
    return 2*(2*W)/I**2      # x2 mirror half, L=2W/I^2
if __name__=='__main__':
    rc,t,N=0.04,0.004,10
    for length in [0.04,0.08,0.16]:
        for h in [0.004,0.002]:
            t0=time.time(); L=fem_L(rc,length,t,N,h,0.24,0.24)
            print(f"l={length} h={h} FEM L={L*1e6:.4f} uH  Nagaoka={ci.solenoid_nagaoka(N,rc,length)*1e6:.4f} Wheeler={ci.solenoid_wheeler(N,rc,length)*1e6:.4f}  t={time.time()-t0:.0f}s",flush=True)
