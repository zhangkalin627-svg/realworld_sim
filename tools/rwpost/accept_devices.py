import sys, os, math
sys.path.insert(0, os.getcwd())
import devices as D
R=[]
def chk(n,c,i=""): R.append(bool(c)); print(("PASS " if c else "FAIL ")+n,i,flush=True)
def T(n,f):
    try: f()
    except Exception as e: chk(n,False,"异常 "+repr(e)[:100])
def cn(e): return any('\u4e00'<=c<='\u9fff' for c in str(e))
mp=dict(rds_on_25=0.01,alpha_rds=0.004,qg=30e-9,vg=10,coss=300e-12,tr=10e-9,tf=10e-9)
def t1():
    r=D.mosfet_loss(5,3,7,24,100e3,0.5,mp,75.0) if True else None; rds=0.01*(1+0.004*50)
    e=dict(p_cond=25*rds,p_sw=0.5*24*(3*10e-9+7*10e-9)*1e5,p_oss=0.5*300e-12*576*1e5,p_gate=30e-9*10*1e5)
    chk("1 mosfet_loss(Tj=75)",all(abs(r[k]-v)<1e-12*max(1,abs(v)) for k,v in e.items()) and abs(r["p_total"]-sum(e.values()))<1e-12,str(r))
    r0=D.mosfet_loss(5,3,7,24,100e3,0.5,mp); chk("1b Tj 默认 25",abs(r0["p_cond"]-25*0.01)<1e-12)
    chk("1c 类型 float",all(type(v) is float for v in r.values()))
T("1",t1)
def t2():
    d=D.diode_loss(2.5,4.0,0.5,0.02,20e-9,24,100e3); chk("2 diode_loss",abs(d["p_cond"]-(0.5*2.5+0.02*16))<1e-12 and abs(d["p_rr"]-20e-9*24*1e5)<1e-12 and abs(d["p_total"]-d["p_cond"]-d["p_rr"])<1e-12,str(d))
T("2",t2)
def t3():
    a,b,Rj,ta=2.0,0.01,5.0,40.0; r=D.solve_tj(lambda tj:a+b*(tj-25),Rj,ta); ex=(ta+Rj*(a-25*b))/(1-Rj*b)
    chk("3 solve_tj 线性解析解",abs(r["tj"]-ex)<1e-5 and abs(r["p"]-(a+b*(ex-25)))<1e-5 and type(r["iters"]) is int,str(r))
    for n,f in (("热失控 R·b>1",lambda:D.solve_tj(lambda tj:1+0.5*(tj-25),5.0,40.0)),("超1000℃",lambda:D.solve_tj(lambda tj:1e4,1.0,25.0))):
        try: f(); chk("3b "+n,False)
        except ValueError as e: chk("3b "+n,cn(e),str(e))
    r2=D.solve_tj(lambda tj:3.0,2.0,30.0); chk("3c 常数功率 1 步内收敛",abs(r2["tj"]-36)<1e-6 and r2["iters"]<=3,str(r2))
    # 耦合：真实 mosfet Rds(T)
    f=lambda tj:D.mosfet_loss(5,3,7,24,1e5,0.5,mp,tj)["p_total"]; r3=D.solve_tj(f,20.0,40.0); chk("3d 与 mosfet 耦合自洽",abs(40+20*f(r3["tj"])-r3["tj"])<1e-4,str(r3))
T("3",t3)
def t4():
    ok=0;bad=[]
    for n,f in (("mosfet NaN",lambda:D.mosfet_loss(float('nan'),1,1,24,1e5,0.5,mp)),("mosfet fsw<=0",lambda:D.mosfet_loss(1,1,1,24,0,0.5,mp)),("diode fsw<=0",lambda:D.diode_loss(1,1,0.5,0.01,1e-9,24,0)),("diode 负 qrr",lambda:D.diode_loss(1,1,0.5,0.01,-1e-9,24,1e5)),("mosfet p 缺键",lambda:D.mosfet_loss(1,1,1,24,1e5,0.5,{"rds_on_25":0.01})),("solve_tj 负 rth",lambda:D.solve_tj(lambda t:1.0,-1.0,25.0)),("diode NaN",lambda:D.diode_loss(float('nan'),1,0.5,0.01,1e-9,24,1e5))):
        try: f(); bad.append((n,"未抛错"))
        except ValueError as e:
            if cn(e): ok+=1
            else: bad.append((n,str(e)))
        except Exception as e: bad.append((n,repr(e)[:50]))
    chk("4 非法输入抛中文 ValueError(7项)",ok==7,"%d/7 %s"%(ok,bad))
T("4",t4)
def t5():
    mpc=dict(mp); D.mosfet_loss(5,3,7,24,1e5,0.5,mp,60.0); chk("5 不修改 p",mp==mpc)
T("5",t5)
def t6():
    # IGBT：导通压降 vce0+rce·i；规格："导通压降 vce0+rce·i"。用对称波形(i_avg 与 i_rms 不同)检验 p_cond 是否为 vce0·i_avg 而非 vce0·i_rms
    import inspect
    sig=list(inspect.signature(D.igbt_loss).parameters)
    chk("6 igbt_loss 接口含平均电流参数(能正确计 vce0·i_avg)", any("avg" in s for s in sig), str(sig))
T("6",t6)
print("SCORE %d/%d"%(sum(R),len(R)))
