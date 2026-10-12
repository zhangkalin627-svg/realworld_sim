"""生成三维涡流缓存结果 rw/em/eddy3d_results.json (约 3–5 分钟, 内存 ~1 GB)"""
import sys, json, time, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from rw.em.eddy3d import study_3d
t = time.time()
r = study_3d(ecc_mm=3.0, sweep_sigma=True)
json.dump(r, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rw", "em", "eddy3d_results.json"), "w"), indent=1)
print(json.dumps(r)[:1200], time.time() - t)
