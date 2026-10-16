"""管件局部阻力系数 K 与标准管径。优先调用 `fluids`（MIT，pip 装）；没装则用常用手册值。"""
import math
try:
    import fluids as _fl
    from fluids import fittings as _ft
    HAVE_FLUIDS = True
except Exception:       # pragma: no cover
    HAVE_FLUIDS = False


def K_elbow90(D=0.026, r_over_d=1.5, fd=0.02):
    """90° 圆弧弯头"""
    if HAVE_FLUIDS:
        return float(_ft.bend_rounded(Di=D, angle=90, fd=fd, bend_diameters=r_over_d))
    return 0.3 if r_over_d >= 1.5 else 0.5


def K_entrance(sharp=True):
    return float(_ft.entrance_sharp()) if HAVE_FLUIDS and sharp else (0.5 if sharp else 0.04)


def K_exit():
    return 1.0


def K_contraction(D1, D2):
    if HAVE_FLUIDS:
        return float(_ft.contraction_sharp(D1, D2))
    return 0.5 * (1 - (D2 / D1) ** 2)


def K_expansion(D1, D2):
    if HAVE_FLUIDS:
        return float(_ft.diffuser_sharp(D1, D2))
    return (1 - (D1 / D2) ** 2) ** 2


def pipe_id(nps, schedule="40"):
    """公称管径(英寸 NPS)+壁厚号 → 内径 m（ASME B36.10，来自 fluids.piping）"""
    if HAVE_FLUIDS:
        from fluids.piping import nearest_pipe
        r = nearest_pipe(NPS=nps, schedule=schedule)
        return r[1]
    raise RuntimeError("需要 pip install fluids 才能查标准管径")
