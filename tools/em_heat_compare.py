"""完整加热-淬火: 线性有效 μr / 非线性 B-H / 非线性+SMC导磁体 对比 (同样 6 kW, 30 kHz, 3 s)"""
import sys, time, os, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from rw.induction import InductionQuenchSim
from rw.em.conc import Concentrator
out = {}
for name, kw in [("lin", {}), ("nl", dict(nonlinear=True)), ("nl+smc", dict(nonlinear=True, conc=Concentrator("smc")))]:
    t = time.time()
    s = InductionQuenchSim(freq=30e3, power_kw=6, t_heat=3.0, t_quench=0.5, t_air=0.2, **kw)
    r = s.run()
    keys = ['T_surf_end_heat', 'T_inner_end_heat', 'T_max', 'coil_current_A', 'efficiency', 'conc_loss_kW', 'HRC_surface', 'case_depth_mm']
    out[name] = {k: r[k] for k in keys}
    print(name, round(time.time() - t), {k: round(r[k], 2) for k in keys}, flush=True)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out", "em_heat_compare.json"), "w"), indent=1)
