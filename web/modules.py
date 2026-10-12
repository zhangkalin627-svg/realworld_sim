"""Web 版模块注册表：每个模块声明参数表单 schema + 运行函数。加新模块 = 在 MODULES 里加一项。
架构约定（面向中期目标"自由组合拼设备 + 长时连续运行"）：参数是纯 JSON dict、运行函数无状态(JSON in → JSON out)，
未来"拼设备"= 部件图/连接拓扑的 JSON 由图执行器组合多个模块runner；长任务靠任务表持久化 + 加速仿真，runner 契约不变。
运行函数签名: run(params: dict, progress: callable(fraction)) -> {"lines": [str], "warnings": [str], "plots": [{"title", "svg"}], "raw": dict}
图用服务器端 matplotlib 渲染成 SVG 字符串，前端内联显示，不需要 JS 图表库。"""
import io
import matplotlib
matplotlib.use("Agg")                    # 服务器无显示环境
import matplotlib.pyplot as plt


def _clean(o):
    """递归清理: 非有限浮点(NaN/Inf, 如未达温的 t8/5) 转 None，保证 JSON 可序列化。"""
    import math
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    return o


def _f2(x, nd=2):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) and x == x else "—"   # NaN 显示为 —


def _svg(fig):
    b = io.StringIO()
    fig.savefig(b, format="svg", bbox_inches="tight")
    plt.close(fig)
    return b.getvalue()


def _plot(sim, kind, title):
    from rw import viz
    fig = plt.figure(figsize=(5.2, 4.0))
    ax = fig.add_subplot(111)
    if kind == "temp":
        viz.plot_field(ax, sim, sim.T_end_heat, "inferno", "°C", 25, max(1000, float(sim.T_end_heat.max())), fig)
    elif kind == "hrc":
        viz.plot_field(ax, sim, sim.HRC, viz.HRC_CMAP, "HRC", 18, 60, fig)
    elif kind == "curve":
        viz.plot_curves(ax, sim)
    elif kind == "line":
        viz.plot_hardness_line(ax, sim)
    ax.set_title(title)
    fig.tight_layout()
    return {"title": title, "svg": _svg(fig)}


def run_induction(params, progress):
    """感应淬火：参数对齐桌面版①页。慢任务（非线性约 ×3），由任务线程跑。"""
    from rw.induction import InductionQuenchSim, BottleNeckGeometry, find_power
    f = lambda k, d: float(params.get(k, d))
    g = BottleNeckGeometry(neck_od=f("neck_od", 22), wall=f("wall", 3), neck_len=f("neck_len", 50),
                           zone=f("zone", 20), coil_id=f("coil_id", 30), coil_h=f("coil_h", 25))
    kw = dict(geom=g, freq=f("freq", 20) * 1e3, t_heat=f("t_heat", 3.0),
              t_quench=f("t_quench", 3.0), t_air=f("t_air", 6.0),
              medium=params.get("medium", "water_spray"),
              mur_cold=f("mur_cold", 20.0), nonlinear=bool(params.get("nonlinear", False)))
    ck = params.get("conc")
    if ck:
        from rw.em.conc import Concentrator
        kw["conc"] = Concentrator(ck)
    if params.get("mode", "target") == "target":
        target = f("target", 900.0)
        progress(0.02)
        kw["power_kw"] = find_power(target, **kw)
    else:
        kw["power_kw"] = f("power_kw", 6.0)
    sim = InductionQuenchSim(**kw)
    r = _clean(sim.run(progress=lambda f: progress(0.05 + 0.95 * f)))
    lines = [
        f"频率 {r['freq_kHz']:.0f} kHz，工件吸收 {r['power_kW']:.2f} kW",
        f"透入深度：冷态 {r['skin_cold_mm']:.2f} mm，热态(>居里点) {r['skin_hot_mm']:.2f} mm",
        f"加热结束：外表面 {r['T_surf_end_heat']:.0f}°C，内孔 {r['T_inner_end_heat']:.0f}°C，最高 {r['T_max']:.0f}°C",
        f"线圈电流约 {r['coil_current_A']:.0f} A(有效值)，线圈铜损约 {r['coil_loss_kW']:.2f} kW，电效率约 {r['efficiency']*100:.0f}%",
        f"淬火：{r['medium']}，t8/5 外表面 {_f2(r['t85_surface'])} s / 内孔 {_f2(r['t85_inner'])} s",
        f"硬度(估算)：外表面 {r['HRC_surface']:.0f} HRC，内孔 {r['HRC_inner']:.0f} HRC",
        f"有效硬化层 (≥50 HRC)：{r['case_depth_mm']:.2f} mm" + ("，已淬透" if r['through_hardened'] else ""),
    ]
    if r.get("nonlinear"):
        lines.insert(1, "磁导率：非线性 B-H(含居里点)" + (f"；导磁体 {r['conc']}，其损耗 {r['conc_loss_kW']:.2f} kW" if r.get("conc") else ""))
    plots = [_plot(sim, "temp", "加热结束温度场"), _plot(sim, "hrc", "淬火后硬度（估算）"),
             _plot(sim, "curve", "监测点温度 / 线圈电流"), _plot(sim, "line", "淬火区中部：硬度随深度")]
    return {"lines": lines, "warnings": r["warnings"], "plots": plots, "raw": r}


def run_materials(params, progress):
    """材料库：45 钢关键参数 + 淬火介质表（只读）。"""
    from rw import materials as M
    progress(1.0)
    s = M.STEEL_45
    lines = [f"{s['name']}：密度 {s['density']:.0f} kg/m³，Ac1 {s['Ac1']:.0f}°C，Ac3 {s['Ac3']:.0f}°C，"
             f"居里点 {s['T_curie']:.0f}°C，Ms {s['Ms']:.0f}°C，熔点 {s['T_melt']:.0f}°C", "", "淬火介质："]
    for k, q in M.QUENCHANTS.items():
        lines.append(f"· {q['name']}（{k}）：槽温 {q['T_bath']:.0f}°C，峰值换热系数约 {max(q['h']):.0f} W/m²K")
    return {"lines": lines, "warnings": [], "plots": [], "raw": {}}


def _w3d(params, progress):
    from .w3d import run_world3d
    return run_world3d(params, progress)


def _plc(params, progress):
    from .plcsim import run_plc
    return run_plc(params, progress)


def _flow_net(params, progress):
    from .flowsim import run_flow_net
    return run_flow_net(params, progress)


def _flow_hammer(params, progress):
    from .flowsim import run_flow_hammer
    return run_flow_hammer(params, progress)


def _flow_jet(params, progress):
    from .flowsim import run_flow_jet
    return run_flow_jet(params, progress)


def _mcu(params, progress):
    from .mcusim import run_mcu
    return run_mcu(params, progress)


def _mcu_default_asm():
    from rw.mcu import firmware as FW
    return FW.HEATER_ASM


def _mcu_default_plc():
    from rw.mcu import firmware as FW
    return FW.PLC_IL


def _fluid_options(keys):
    from rw.world.fluids import FLUIDS
    return [{"v": k, "t": FLUIDS[k]["name"]} for k in keys]


def _plc_default_program():
    from rw.plc.demo import DEMO_IL
    return DEMO_IL


MODULES = [
    {"key": "induction", "name": "感应加热淬火", "desc": "瓶坯感应加热 + 喷淋淬火全过程：温度场、硬度、硬化层深度",
     "fields": [
         {"g": "工件 / 线圈 (mm)"},
         {"name": "neck_od", "label": "瓶口外径", "type": "number", "default": 22, "min": 5, "max": 200, "step": 0.5},
         {"name": "wall", "label": "壁厚", "type": "number", "default": 3, "min": 0.5, "max": 30, "step": 0.5},
         {"name": "neck_len", "label": "瓶颈长", "type": "number", "default": 50, "min": 10, "max": 300, "step": 1},
         {"name": "zone", "label": "淬火区长", "type": "number", "default": 20, "min": 2, "max": 100, "step": 1},
         {"name": "coil_id", "label": "线圈内径", "type": "number", "default": 30, "min": 6, "max": 300, "step": 0.5},
         {"name": "coil_h", "label": "线圈高度", "type": "number", "default": 25, "min": 2, "max": 200, "step": 1},
         {"g": "加热"},
         {"name": "freq", "label": "频率 (kHz)", "type": "select", "options": [5, 10, 20, 30, 50, 100, 200], "default": 20},
         {"name": "mode", "label": "方式", "type": "select", "options": [{"v": "target", "t": "给定目标温度（自动调功率）"}, {"v": "power", "t": "给定功率"}], "default": "target"},
         {"name": "target", "label": "外表面目标 °C", "type": "number", "default": 900, "min": 200, "max": 1600, "step": 10},
         {"name": "power_kw", "label": "工件吸收功率 kW", "type": "number", "default": 6, "min": 0.1, "max": 500, "step": 0.5},
         {"name": "t_heat", "label": "加热时间 s", "type": "number", "default": 3, "min": 0.1, "max": 60, "step": 0.1},
         {"name": "mur_cold", "label": "冷态有效磁导率 μr", "type": "number", "default": 20, "min": 1, "max": 200, "step": 1},
         {"name": "nonlinear", "label": "非线性 B-H + 居里点 (更准, 慢约 3 倍)", "type": "check", "default": False},
         {"g": "冷却"},
         {"name": "medium", "label": "淬火介质", "type": "select", "default": "water_spray",
          "options": [{"v": "water_spray", "t": "清水喷淋"}, {"v": "polymer_spray", "t": "聚合物喷淋"}, {"v": "water_dip", "t": "清水浸入"}, {"v": "oil_dip", "t": "油浸"}, {"v": "air", "t": "空冷"}]},
         {"name": "t_quench", "label": "淬火时间 s", "type": "number", "default": 3, "min": 0.1, "max": 60, "step": 0.1},
     ],
     "run": run_induction},
    {"key": "materials", "name": "材料库", "desc": "45 钢关键相变点 + 淬火介质换热参数（只读查询）",
     "fields": [], "run": run_materials},
    {"key": "world3d", "name": "三维装配世界", "desc": "零件摆放 → 刚体物理仿真 → 受力体检（稳定性/梁应力/螺栓/轴承/赫兹），浏览器 3D 动画回放",
     "fields": [
         {"name": "preset", "label": "示例场景", "type": "select", "default": "工字钢梁桥",
          "options": ["钢板桥：两墩 + 铜块", "叠积木 外伸系数 0.95（应稳）", "叠积木 外伸系数 1.10（应倒）", "橡胶块在 22° 斜面（会倒）",
                      "钢块在 30° 斜面（会滑）", "钢块砸木板", "水里的浮沉", "焊接台架 + 偏载", "工字钢梁桥",
                      "壁挂支架 + 螺栓组", "轴承转子 3000rpm", "玻璃板断裂演示", "焊缝悬臂支架"]},
         {"name": "tmax", "label": "仿真时长上限 s", "type": "number", "default": 8, "min": 1, "max": 15, "step": 1},
     ],
     "run": _w3d},
    {"key": "plc", "name": "PLC 虚拟调试", "desc": "三菱 FX 指令表（含 STL 步进）+ 单工位感应淬火设备模型：整段运行 → 逻辑分析仪时序图 + 节拍/不合格统计",
     "fields": [
         {"name": "program", "label": "FX 指令表程序（可直接改，; 后为注释）", "type": "textarea", "default": _plc_default_program()},
         {"name": "p_air", "label": "气源气压 MPa", "type": "number", "default": 0.5, "min": 0.1, "max": 0.8, "step": 0.05},
         {"name": "scenario", "label": "演示剧本", "type": "select", "default": "normal",
          "options": [{"v": "normal", "t": "正常循环（0.3 s 按启动）"}, {"v": "estop", "t": "急停演示（中途急停 2 s 后复位重启）"}]},
         {"name": "estop_at", "label": "急停时刻 s（仅急停剧本）", "type": "number", "default": 8, "min": 1, "max": 170, "step": 1},
         {"name": "t_sim", "label": "运行时长 s", "type": "number", "default": 60, "min": 5, "max": 180, "step": 5},
     ],
     "run": _plc},
    {"key": "flow_net", "name": "流体·管网", "desc": "两工位并联喷淋管网：泵/阀/过滤器/软管/喷淋孔 → 流量分配、压力沿程、泵工作点、汽蚀/超速体检",
     "fields": [
         {"name": "fluid", "label": "介质", "type": "select", "default": "pag10", "options": _fluid_options(("pag10", "water", "oil_quench"))},
         {"name": "T", "label": "液温 °C", "type": "number", "default": 30, "min": 5, "max": 90, "step": 5},
         {"name": "speed", "label": "泵转速 % 额定", "type": "number", "default": 100, "min": 30, "max": 120, "step": 5},
         {"name": "dirty", "label": "吸入过滤器脏污 ×（1=干净）", "type": "number", "default": 1, "min": 0.5, "max": 40, "step": 1},
         {"name": "openB", "label": "B 路电磁阀开度 %", "type": "number", "default": 100, "min": 0, "max": 100, "step": 10},
         {"name": "holes", "label": "每路喷淋孔数", "type": "number", "default": 24, "min": 4, "max": 100, "step": 2},
         {"name": "dh", "label": "孔径 mm", "type": "number", "default": 1.2, "min": 0.5, "max": 4, "step": 0.1},
         {"name": "hose", "label": "软管内径 mm", "type": "number", "default": 10, "min": 5, "max": 20, "step": 1},
     ],
     "run": _flow_net},
    {"key": "flow_hammer", "name": "流体·水锤", "desc": "关阀冲击（特征线法）：最高/最低压力 vs 管材额定，汽穴判据，关阀时间影响",
     "fields": [
         {"name": "mat", "label": "管材", "type": "select", "default": "钢",
          "options": ["不锈钢", "钢", "铜", "PVC", "PE", "增强软管", "橡胶软管"]},
         {"name": "L", "label": "管长 m（阀到上游稳压点）", "type": "number", "default": 3, "min": 0.2, "max": 2000, "step": 1},
         {"name": "D", "label": "内径 mm", "type": "number", "default": 26, "min": 3, "max": 500, "step": 1},
         {"name": "e", "label": "壁厚 mm", "type": "number", "default": 2.0, "min": 0.2, "max": 30, "step": 0.5},
         {"name": "H", "label": "阀前稳压 bar（表压）", "type": "number", "default": 3.0, "min": 0.1, "max": 40, "step": 0.5},
         {"name": "K", "label": "阀全开阻力系数 K", "type": "number", "default": 20, "min": 0.5, "max": 5000, "step": 5},
         {"name": "tc", "label": "关阀时间 ms", "type": "number", "default": 25, "min": 1, "max": 20000, "step": 5},
         {"name": "rate", "label": "管/软管额定压力 bar", "type": "number", "default": 10, "min": 1, "max": 400, "step": 1},
     ],
     "run": _flow_hammer},
    {"key": "flow_jet", "name": "流体·放空与射流", "desc": "水箱底部开孔：放空时间（对照 Torricelli 解析）、射流轨迹/射程/冲击力",
     "fields": [
         {"name": "fluid", "label": "液体", "type": "select", "default": "water",
          "options": _fluid_options(("water", "pag10", "oil_quench", "gasoline", "glycerin", "mercury"))},
         {"name": "A", "label": "水箱截面积 m²", "type": "number", "default": 0.5, "min": 0.01, "max": 100, "step": 0.1},
         {"name": "H0", "label": "初始液位 m", "type": "number", "default": 1.0, "min": 0.05, "max": 30, "step": 0.1},
         {"name": "d", "label": "孔径 mm", "type": "number", "default": 12, "min": 1, "max": 100, "step": 1},
         {"name": "zh", "label": "孔离地高度 m", "type": "number", "default": 1.0, "min": 0, "max": 20, "step": 0.1},
         {"name": "ang", "label": "孔朝向 仰角 °", "type": "number", "default": 0, "min": -60, "max": 80, "step": 5},
     ],
     "run": _flow_jet},
    {"key": "mcu", "name": "单片机 + PLC 联合仿真", "desc": "ATmega328P 风格 AVR + FX PLC + 半桥电路：启动握手 → 软启动 → 短路过流保护 → 复位重启，关键时刻全部从波形量出",
     "fields": [
         {"name": "fw", "label": "固件", "type": "select", "default": "asm",
          "options": [{"v": "asm", "t": "汇编固件（默认加热器 PWM 控制）"}, {"v": "behav", "t": "行为级 Python 固件（快，不含指令计时）"}, {"v": "custom", "t": "自定义汇编（填下面）"}]},
         {"name": "custom_asm", "label": "自定义汇编固件（选了自定义才用）", "type": "textarea", "default": ""},
         {"name": "custom_plc", "label": "自定义 PLC 指令表（留空用默认加热工位程序）", "type": "textarea", "default": ""},
         {"g": "演示剧本时刻 (s)"},
         {"name": "fault_at", "label": "短路时刻 s", "type": "number", "default": 0.45, "min": 0.1, "max": 2.5, "step": 0.05},
         {"name": "fault_R", "label": "短路电阻 Ω", "type": "number", "default": 0.4, "min": 0.05, "max": 5, "step": 0.05},
         {"name": "reset_at", "label": "报警复位时刻 s", "type": "number", "default": 0.75, "min": 0.2, "max": 2.8, "step": 0.05},
         {"name": "restart_at", "label": "再启动时刻 s", "type": "number", "default": 0.85, "min": 0.3, "max": 2.9, "step": 0.05},
         {"name": "t_end", "label": "仿真时长 s", "type": "number", "default": 1.6, "min": 0.5, "max": 3.0, "step": 0.1},
     ],
     "run": _mcu},
]
