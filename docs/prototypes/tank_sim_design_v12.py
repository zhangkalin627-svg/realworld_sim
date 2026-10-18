"""设计稿 v1.2 数据生成:串联谐振槽真实瞬态仿真(矩阵指数精确解,分段常数输入)。
模型: v_ab(方波 ±Vdc) -- L -- C -- R 串联。状态 x=[i, vC]。
这是真实数值仿真输出(ODE 精确积分解),不是手画示意。
"""
import numpy as np, json

VDC = 510.0; TRAFO = 2.0; VT = VDC/TRAFO; L0 = 60e-6; R = 0.9

def tank_run(C, f_sw, L=L0, T=6e-3, steps_per_half=400):
    w0 = 1/np.sqrt(L*C)
    A = np.array([[-R/L, -1/L],[1/C, 0]])
    Th = 0.5/f_sw
    dt = Th/steps_per_half
    # 精确离散化(每 dt 内输入恒定)
    M = np.zeros((3,3)); M[:2,:2]=A; M[:2,2]=np.array([1/L,0]); M*=dt
    eM = np.zeros((3,3))
    # 泰勒求 expm(3x3 很小)
    k=np.eye(3); eM=np.eye(3).copy(); fact=1.0
    for n in range(1,40):
        k = k@M; fact*=n; eM += k/fact
    Ad=eM[:2,:2]; Bd=eM[:2,2]
    n_half = int(T/Th)
    N = n_half*steps_per_half
    ts = np.empty(N); iL=np.empty(N); vC=np.empty(N); vab=np.empty(N)
    x=np.zeros(2); idx=0; t=0.0
    for h in range(n_half):
        u = VT if h%2==0 else -VT
        for s in range(steps_per_half):
            x = Ad@x + Bd*u
            ts[idx]=t; iL[idx]=x[0]; vC[idx]=x[1]; vab[idx]=u
            t+=dt; idx+=1
    return ts, iL, vC, vab, w0/(2*np.pi)

def metrics(ts, iL, vC, vab, R, f_sw, tail=1e-3):
    m = ts > ts[-1]-tail
    i=iL[m]; v=vC[m]; vv=vab[m]; tt=ts[m]
    ipk = np.max(np.abs(i)); irms=np.sqrt(np.mean(i**2))
    P = np.mean(i**2)*R
    vcpk = np.max(np.abs(v))
    # 相位: v_ab 上升沿 → i_L 正向过零 的时延
    edges = np.where((vv[:-1]<0)&(vv[1:]>0))[0]
    zeros = np.where((i[:-1]<0)&(i[1:]>=0))[0]
    phase_deg=None
    if len(edges) and len(zeros):
        e=edges[-3]; z=zeros[zeros>e][0]
        phase_deg = (tt[z]-tt[e])*f_sw*360
    pf = P/(np.sqrt(np.mean(vv**2))*irms)
    return dict(i_peak=float(ipk), i_rms=float(irms), P_load=float(P),
                vC_peak=float(vcpk), phase_deg=float(phase_deg), pf=float(pf))

def f0(C,L=L0): return 1/(2*np.pi*np.sqrt(L*C))

runs={}
for tag,C in [("base",2.2e-6),("cur",1.8e-6)]:
    f=f0(C); fsw=1.03*f
    ts,iL,vC,vab,freal = tank_run(C,fsw)
    m = metrics(ts,iL,vC,vab,R,fsw)
    m.update(C=C, f0=freal, f_sw=fsw)
    runs[tag]=dict(m=m, ts=ts,iL=iL,vC=vC,vab=vab)
    print(tag, {k:(round(v,4) if isinstance(v,float) else v) for k,v in m.items()})

b,c = runs["base"]["m"], runs["cur"]["m"]
print("df0_pct %.2f"%(100*(c["f0"]-b["f0"])/b["f0"]))
print("dIpeak_pct %.2f"%(100*(c["i_peak"]-b["i_peak"])/b["i_peak"]))
print("dP_pct %.2f"%(100*(c["P_load"]-b["P_load"])/b["P_load"]))

# 居里塌缩包络: 15ms 处 L 60→54µH(-10%), 驱动固定 15.8kHz(PLL 不跟踪 → 失谐演示)
fsw=15.8e3; T=30e-3
ts2=[]; env_t=[]; env_a=[]
t=0; x=np.zeros(2)
Th=0.5/fsw; dt=Th/200
tseg=[]
seg_bound=15e-3
allt=[]; alli=[]
for L in [60e-6, 54e-6]:
    C=1.8e-6
    A=np.array([[-R/L,-1/L],[1/C,0]])
    M=np.zeros((3,3)); M[:2,:2]=A; M[:2,2]=[1/L,0]; M*=dt
    k=np.eye(3); eM=np.eye(3).copy(); fact=1.0
    for n in range(1,40):
        k=k@M; fact*=n; eM+=k/fact
    Ad=eM[:2,:2]; Bd=eM[:2,2]
    Tend = seg_bound if L==60e-6 else T
    while t < Tend:
        h=int(t/Th)
        u=VT if h%2==0 else -VT
        x=Ad@x+Bd*u; t+=dt
        allt.append(t); alli.append(x[0])
allt=np.array(allt); alli=np.array(alli)
# 包络: 每 150µs 取 |i| 最大
win=150e-6
env_t=[]; env_a=[]
for t0 in np.arange(0,T,win):
    m=(allt>=t0)&(allt<t0+win)
    if m.any(): env_t.append(t0+win/2); env_a.append(np.max(np.abs(alli[m])))
env_t=np.array(env_t); env_a=np.array(env_a)
pre=env_a[env_t<12e-3].mean(); post=env_a[env_t>25e-3].mean()
print("curie_env pre=%.1fA post=%.1fA d=%.1f pct"%(pre,post,100*(post-pre)/pre))

np.savez("sim_traces.npz",
    p1_t=runs["cur"]["ts"], p1_i=runs["cur"]["iL"], p1_vab=runs["cur"]["vab"], p1_vC=runs["cur"]["vC"],
    b_t=runs["base"]["ts"], b_i=runs["base"]["iL"],
    c_t=runs["cur"]["ts"], c_i=runs["cur"]["iL"],
    env_t=env_t, env_a=env_a)
json.dump({"base":b,"cur":c,
  "df":100*(c["f0"]-b["f0"])/b["f0"],"di":100*(c["i_peak"]-b["i_peak"])/b["i_peak"],"dp":100*(c["P_load"]-b["P_load"])/b["P_load"],
  "curie":{"pre":float(pre),"post":float(post),"d":100*(post-pre)/pre}}, open("sim_metrics.json","w"), indent=1)
print("saved")
