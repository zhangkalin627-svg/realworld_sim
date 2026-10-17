"""离屏生成粉尘动画 GIF（报告附件用）：硅钢案 90 sim-秒 + 末尾 10 s 吹扫。
运行：python3 tools/make_dust_gif.py out.gif"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = ["Droid Sans Fallback", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
from matplotlib.figure import Figure
from PIL import Image
from rw.em.dust import build_dust_sim, draw_static, update_scatter

out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/dust_anim.gif"
ds, rec = build_dust_sim("sife")
fig = Figure(figsize=(6.4, 4.8), dpi=90)
ax = fig.add_subplot(111)
sc_air, sc_stk = draw_static(ax, ds)
frames = []
def snap():
    import io
    b = io.BytesIO()
    fig.savefig(b, format="png")
    b.seek(0)
    frames.append(Image.open(b).convert("RGB").copy())

for i in range(180):                    # 90 s：dt=0.5
    ds.step(0.5, blow=False)
    if i % 3 == 0:
        update_scatter(ax, sc_air, sc_stk, ds, "硅钢导磁位｜")
        snap()
for i in range(20):                     # 末尾 10 s 开气枪
    ds.step(0.5, blow=True)
    if i % 2 == 0:
        st = update_scatter(ax, sc_air, sc_stk, ds, "硅钢导磁位｜气枪吹扫｜")
        snap()
frames[0].save(out, save_all=True, append_images=frames[1:], duration=150, loop=0)
st = ds.stats()
print(f"frames={len(frames)} final: stuck={st['stuck']} layer={st['layer_um']:.0f}µm risk={st['risk']} -> {out}")
