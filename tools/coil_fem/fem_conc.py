import sys,time,math,numpy as np
sys.path.insert(0,'.')
import warnings; warnings.filterwarnings('ignore')
from cloudfem.mesh.tri_mesh import rect_mesh
from cloudfem.physics.axisym import solve_eddy_current_axi, MU0
def conc_L(rc,t,hz,mur,variant,h,B,g=1e-3,T=5e-3):
    """single turn, radial thickness t, axial height hz, mean radius rc; half domain z>=0. variant: none|back|U"""
    I=1.0; J=I/(t*hz)
    r0,r1=rc-t/2,rc+t/2
    def inC(r,z): return r0<=r<=r1 and z<=hz/2
    def inM(r,z):
        if variant=='none': return False
        back = (r1+g<=r<=r1+g+T) and z<=hz/2
        if variant=='back': return back
        flank = (r0<=r<=r1+g+T) and (hz/2+g<=z<=hz/2+g+T)
        return back or (variant=='U' and (back or flank)) or (back and True)
    def nu(r,z): return 1/(MU0*mur) if inM(r,z) else 1/MU0
    def Jf(r,z): return J if inC(r,z) else 0.0
    mesh=rect_mesh(int(round(B/h)),int(round(B/h)),xlim=(0,B),ylim=(0,B))
    u=solve_eddy_current_axi(mesh,nu=nu,J_phi=Jf,bc={'left':0.0,'right':0.0,'top':0.0})
    W=0.0
    for el in mesh.elements:
        p=mesh.nodes[el]; r=p[:,0].mean(); z=p[:,1].mean()
        if not inC(r,z): continue
        a=np.real(u[el]).mean()/r
        ar=0.5*abs((p[1][0]-p[0][0])*(p[2][1]-p[0][1])-(p[1][1]-p[0][1])*(p[2][0]-p[0][0]))
        W+=0.5*J*a*2*math.pi*r*ar
    return 2*(2*W)/I**2
if __name__=='__main__':
    rc=float(sys.argv[1]); t=float(sys.argv[2]); hz=float(sys.argv[3]); h=float(sys.argv[4])
    Bs=[0.16,0.2,0.24]
    cases=[('none',1)]+[(v,m) for v in ('back','U') for m in (10,50,200,1000)]
    for v,m in cases:
        Ls=[conc_L(rc,t,hz,m,v,h,B)*1e9 for B in Bs]
        x=np.array(Bs)**-3.0; A=np.vstack([np.ones(3),-x]).T
        inf=np.linalg.lstsq(A,np.array(Ls),rcond=None)[0][0]
        print(f"rc={rc} t={t} hz={hz} h={h} {v} mur={m}: boxes {[round(l,3) for l in Ls]} -> L_inf={inf:.3f} nH",flush=True)
