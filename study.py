"""对比研究：5 / 20 / 100 kHz × 清水喷淋 / PAG喷淋 / 淬火油"""
import json, warnings, sys; warnings.filterwarnings('ignore')
import numpy as np, matplotlib.pyplot as plt
from rw.induction import InductionQuenchSim, find_power
from rw import viz
FREQS=[5e3,20e3,100e3]; MEDIA=["water_spray","polymer_spray","oil_dip"]
T_TARGET=900; T_HEAT=3.0
rows=[]; sims={}
for f in FREQS:
    P=find_power(T_TARGET, freq=f, t_heat=T_HEAT)
    for m in MEDIA:
        s=InductionQuenchSim(freq=f,power_kw=P,t_heat=T_HEAT,medium=m,t_quench=3.0,t_air=4.0)
        r=s.run(); r['medium_key']=m; rows.append(r); sims[(f,m)]=s
        viz.single_report(s,r,f"out/run_{int(f/1e3)}k_{m}.png")
        print(f/1e3,m,round(P,2),round(r['case_depth_mm'],2),round(r['T_inner_end_heat']),round(r['efficiency'],2),round(r['coil_current_A']),flush=True)
json.dump(rows,open('out/study.json','w'),ensure_ascii=False,indent=1,default=float)
# 图1：频率对比（清水喷淋）
fig,axs=plt.subplots(2,3,figsize=(14,11))
for j,f in enumerate(FREQS):
    s=sims[(f,'water_spray')]; r=s.summary()
    viz.plot_glow(axs[0,j],s,title=f"{f/1e3:.0f} kHz  {r['power_kW']:.1f} kW×{T_HEAT:.0f}s\n外表面 {r['T_surf_end_heat']:.0f}°C / 内孔 {r['T_inner_end_heat']:.0f}°C")
    viz.plot_field(axs[1,j],s,s.HRC,viz.HRC_CMAP,"HRC",18,60,fig)
    axs[1,j].set_title(f"清水喷淋后硬度：硬化层 {r['case_depth_mm']:.2f} mm\n线圈电流 {r['coil_current_A']:.0f} A，电效率约 {r['efficiency']*100:.0f}%",fontsize=10)
fig.suptitle("同一工件、同样加热 3 秒到表面 900°C：频率不同，烧出来不一样",fontsize=13)
fig.tight_layout(); fig.savefig('out/compare_frequency.png',dpi=105); plt.close(fig)
# 图2：介质对比（20 kHz）
fig,axs=plt.subplots(1,4,figsize=(18,5.5),gridspec_kw=dict(width_ratios=[1,1,1,1.4]))
cols={'water_spray':'#1f77b4','polymer_spray':'#2ca02c','oil_dip':'#d62728'}
for j,m in enumerate(MEDIA):
    s=sims[(20e3,m)]; r=s.summary()
    viz.plot_field(axs[j],s,s.HRC,viz.HRC_CMAP,"HRC",18,60,fig)
    axs[j].set_title(f"{r['medium']}\n表面 {r['HRC_surface']:.0f} HRC，硬化层 {r['case_depth_mm']:.2f} mm",fontsize=10)
    h=s.hist; axs[3].plot(h['t'],h['Tsurf'],color=cols[m],label=r['medium'])
axs[3].axhline(800,ls=':',c='gray');axs[3].axhline(500,ls=':',c='gray');axs[3].axhline(340,ls='--',c='gray')
axs[3].set_xlabel('时间 (s)');axs[3].set_ylabel('外表面温度 (°C)');axs[3].legend(fontsize=8);axs[3].grid(alpha=.3)
axs[3].set_title('冷却曲线（虚线 340°C = 马氏体开始）',fontsize=10)
fig.suptitle("20 kHz 加热后，换不同冷却液",fontsize=13)
fig.tight_layout(); fig.savefig('out/compare_quench.png',dpi=105); plt.close(fig)
print("done")
