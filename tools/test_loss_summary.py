import sys, os, math, time, importlib, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
def ref(t, v, i, t0=None, t1=None):
    t=np.asarray(t,float); v=np.asarray(v,float); i=np.asarray(i,float)
    a=t[0] if t0 is None else max(t0,t[0]); b=t[-1] if t1 is None else min(t1,t[-1])
    m=(t>a)&(t<b); tt=np.concatenate([[a],t[m],[b]]); vv=np.interp(tt,t,v); ii=np.interp(tt,t,i); p=vv*ii
    E=float(np.sum(0.5*(p[1:]+p[:-1])*np.diff(tt))); W=b-a
    return dict(energy=E,avg_power=E/W,peak_power=float(np.abs(p).max()),rms_current=math.sqrt(float(np.sum(0.5*(ii[1:]**2+ii[:-1]**2)*np.diff(tt)))/W))
def run(mn):
    m=importlib.import_module(mn); R=[]
    def chk(n,c,i=""): R.append(bool(c)); print(("PASS " if c else "FAIL ")+n,i,flush=True)
    def T(n,f):
        try: f()
        except Exception as e: chk(n,False,"异常 "+repr(e)[:90])
    def close(a,b,tol=1e-9): return all(abs(a[k]-b[k])<=tol*max(1,abs(b[k])) for k in b)
    def t1():
        t=np.linspace(0,1,1001); r=m.device_power(t,10*np.ones_like(t),5*np.ones_like(t))
        chk("1 恒定 R=2Ω",abs(r["avg_power"]-50)<1e-9 and abs(r["energy"]-50)<1e-9 and abs(r["rms_current"]-5)<1e-9,str(r))
    T("1",t1)
    def t2():
        t=np.linspace(0,0.1,200001); w=2*np.pi*50; r=m.device_power(t,325*np.sin(w*t),10*np.sin(w*t))
        chk("2 正弦同相 1625W",abs(r["avg_power"]-1625)<1.625 and abs(r["rms_current"]-10/math.sqrt(2))<0.01,str(r))
    T("2",t2)
    def t3():
        rng=np.random.default_rng(1); t=np.sort(rng.uniform(0,1,60)); t[0]=0; t[-1]=1; t=np.unique(t); v=rng.normal(size=len(t)); i=rng.normal(size=len(t))
        ok=True; bad=""
        for a,b in [(0.13,0.77),(0.0,0.5),(0.31,1.0),(0.2,0.2000001+0.3),(t[3],t[9]),(t[3]+1e-9,t[4]-1e-9)]:
            r=m.device_power(t,v,i,a,b); e=ref(t,v,i,a,b)
            if not close(r,e,1e-9): ok=False; bad=str((a,b,r,e))
        chk("3 窗口端点插值(随机数据,6窗口)对照参考",ok,bad)
    T("3",t3)
    def t3b():
        t=np.array([0.,1.,2.]); v=np.array([0.,2.,4.]); i=np.array([0.,1.,2.]); r=m.device_power(t,v,i,0.5,1.5); e=ref(t,v,i,0.5,1.5)
        chk("3b 线性函数窗口",close(r,e),str(r)+" vs "+str(e))
    T("3b",t3b)
    def t4():
        t=np.linspace(0,1,101); r=m.device_power(t,np.ones_like(t),np.ones_like(t),-5,0.5); chk("4a 部分超出截断",abs(r["energy"]-0.5)<1e-9 and abs(r["avg_power"]-1.0)<1e-9,str(r))
        for a,b in [(2,3),(-3,-1)]:
            try: m.device_power(t,t,t,a,b); chk("4b 完全在外应 ValueError",False)
            except ValueError: chk("4b 完全在外 ValueError",True)
    T("4",t4)
    def t5():
        t=np.linspace(0,1,1001); d={"Q1":(t,10+0*t,2+0*t),"Q2":(t,5+0*t,1+0*t),"D1":(t,2+0*t,5+0*t),"X":(t,3+0*t,-2+0*t)}
        lt=m.loss_table(d,p_in=40.0,p_out=30.0); rows=lt["rows"]; names=[r["name"] for r in rows]
        chk("5a 排序",names==["Q1","D1","Q2","X"],str(names)); chk("5b total=20+10+5-6=29",abs(lt["total_loss"]-29)<1e-9,str(lt["total_loss"]))
        chk("5c share 之和 1",abs(sum(r["share"] for r in rows)-1)<1e-9); chk("5d efficiency/balance",abs(lt["efficiency"]-0.75)<1e-12 and abs(lt["balance"]-(40-30-29))<1e-9,str((lt["efficiency"],lt["balance"])))
        lt2=m.loss_table(d); chk("5e 缺 p_in 时 None",lt2["efficiency"] is None and lt2["balance"] is None)
        z={"A":(t,0*t,0*t)}; lz=m.loss_table(z); chk("5f total=0 时 share=0.0",lz["rows"][0]["share"]==0.0 and type(lz["rows"][0]["share"]) is float)
        e={"B":(t,1+0*t,1+0*t),"A":(t,1+0*t,1+0*t)}; chk("5g 同值按名字升序",[r["name"] for r in m.loss_table(e)["rows"]]==["A","B"])
        tw=m.loss_table(d,t0=0.25,t1=0.75); chk("5h loss_table 窗口生效",abs(tw["rows"][0]["energy"]-10)<1e-9,str(tw["rows"][0]["energy"]))
    T("5",t5)
    def t6():
        t=np.linspace(0,1,11); g=np.ones(11); bad=[lambda:m.device_power(t[::-1],g,g),lambda:m.device_power(np.array([0,1,1,2.]),np.ones(4),np.ones(4)),lambda:m.device_power(t,g[:5],g),lambda:m.device_power(t[:1],g[:1],g[:1]),
          lambda:m.device_power(t,g,g,0.6,0.4),lambda:m.device_power(t,g,g,0.5,0.5),lambda:m.loss_table({}),lambda:m.loss_table({1:(t,g,g)}),
          lambda:m.device_power(t,np.where(np.arange(11)==3,np.nan,1.),g),lambda:m.loss_table({"A":(t,g,np.where(np.arange(11)==3,np.nan,1.))})]
        n=0; hz=0
        for f in bad:
            try: f()
            except ValueError as e: n+=1; hz+=any('\u4e00'<=c<='\u9fff' for c in str(e))
            except Exception: pass
        chk("6 非法输入 ValueError(10项)",n==10,"%d/10 中文信息%d"%(n,hz))
        chk("6b 中文信息",hz==10)
        try: m.loss_table({"A":(t,g,np.where(np.arange(11)==3,np.nan,1.))})
        except ValueError as e: chk("6c NaN 信息含器件名",'A' in str(e),str(e))
    T("6",t6)
    def t7():
        t=np.linspace(0,1,101); v=np.random.default_rng(0).normal(size=101); i=v.copy(); t0,v0,i0=t.copy(),v.copy(),i.copy(); r=m.device_power(t,v,i,0.1,0.9)
        chk("7 不修改输入",np.array_equal(t,t0) and np.array_equal(v,v0) and np.array_equal(i,i0))
        chk("7b 返回 Python float",all(type(r[k]) is float for k in ("energy","avg_power","peak_power","rms_current")),str([type(r[k]) for k in r]))
        lt=m.loss_table({"A":(t,v,i)}); rw=lt["rows"][0]; chk("7c 表内 float",all(type(rw[k]) is float for k in ("energy","avg_power","peak_power","rms_current","share")) and type(lt["total_loss"]) is float)
        r2=m.device_power(list(t),list(v),list(i)); chk("7d 接受 list",abs(r2["energy"]-m.device_power(t,v,i)["energy"])<1e-12)
    T("7",t7)
    def t8():
        n=10_000_000; t=np.linspace(0,1,n); v=np.sin(t); i=np.cos(t); t0=time.time(); m.device_power(t,v,i,0.1,0.9); d=time.time()-t0; chk("8 1e7 点 <5s",d<5,"%.2fs"%d)
        t=np.linspace(0,1,1_000_000); t0=time.time(); m.device_power(t,t,t); d=time.time()-t0; chk("8b 1e6 点 <5s",d<5,"%.2fs"%d)
    T("8",t8)
    def t9():
        t=np.array([0.,1.]); r=m.device_power(t,np.array([-3.,-3.]),np.array([2.,2.])); chk("9 峰值取绝对值，吸收为负",abs(r["peak_power"]-6)<1e-12 and r["energy"]<0,str(r))
        r=m.device_power(np.array([0.,1.,2.]),np.array([0,0,0.]),np.array([5,5,5.]),0.3,1.7); chk("9b 窗口两端插值入峰值",r["peak_power"]==0.0)
        t=np.array([0.,1.,2.]); r=m.device_power(t,np.array([0.,10.,0.]),np.array([1.,1.,1.]),0.5,1.0); chk("9c 峰值含插值端点",abs(r["peak_power"]-10)<1e-9,str(r))
    T("9",t9)
    print("SCORE",sum(R),"/",len(R)); return R
R = run(sys.argv[1] if len(sys.argv) > 1 else "loss_summary"); sys.exit(0 if all(R) else 1)
