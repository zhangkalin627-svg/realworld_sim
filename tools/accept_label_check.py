import subprocess,sys,tempfile
import os,sys
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from label_check import check_text as c
def k(t): return [(r["line"],r["col"],r["kind"],r["term"]) for r in c(t)]
T=[("仿真实测结果",[(1,1,"error","仿真实测")]),("本文用仿真算出的值，称为实测。",[(1,13,"error","实测")]),("现场实测电流3A",[(1,3,"warn","实测")]),
("这是非实测数据",[]),("尚未实测",[]),("未实测",[]),("不是实测",[]),("并非实测",[]),("无实测",[]),("没有实测",[]),("实测值待补",[]),("实测值尚无",[]),
("```\n仿真实测\n```\n正常",[]),("",[]),("第一行\n仿真值\n仿真与实测对比",[(3,4,"error","实测")]),
("仿真实测，仿真测得，模拟实测",[(1,1,"error","仿真实测"),(1,6,"error","仿真测得"),(1,11,"error","模拟实测")]),
("仿真结果，并非仿真的实测",[]),("估算值：非常接近实测",[])  # 已知限制：规格按字符匹配，"非常"的"非"会触发否定豁免,("仿真结果不能当作实测。",[(1,9,"error","实测")]),
("仿真实验测量得到", [(1,1,"error","仿真实验测量")]),("计算结果（实测）",[(1,6,"error","实测")]),("实测A，实测B",[(1,1,"warn","实测"),(1,5,"warn","实测")]),
("仿真\r\n实测\r\n",[(2,1,"warn","实测")]),("无仿真实测",[(1,2,"error","仿真实测")])]
ok=0
for t,e in T:
    try: g=k(t)
    except Exception as x: g=repr(x)
    p=g==e; ok+=p; print("PASS" if p else "FAIL",repr(t),"" if p else "got=%s exp=%s"%(g,e))
for t,rc in (("仿真实测",1),("正常文本",0),("实测A",0)):
    f=tempfile.NamedTemporaryFile("w",suffix=".md",delete=False,encoding="utf-8");f.write(t);f.close()
    r=subprocess.run([sys.executable,os.path.join(os.path.dirname(os.path.abspath(__file__)),"label_check.py"),f.name],capture_output=True,text=True);p=r.returncode==rc;ok+=p;print("PASS" if p else "FAIL","CLI",t,r.returncode,r.stdout.strip()[:40])
print("SCORE",ok,"/",len(T)+3)
