"""通用 PLC 演示：单工位感应淬火（公开示例，不涉及任何具体设备图纸）。
程序用三菱 FX 指令表写（含 STL 步进），设备模型：感应圈升降气缸 + 高频加热 + 喷淋 + 测温仪 + 上下料。
你可以在 GUI 里直接改这段程序再运行，看设备怎么动。"""
from .fx3u import FX3U
from .pneumatics import Cylinder

DEMO_IL = """; ===== 单工位感应淬火 演示程序 (FX3U 指令表) =====
; X000 启动  X001 停止(常闭)  X002 急停(常闭)  X010 感应圈上位  X011 感应圈下位
; X020 工位有料  X040 测温合格(红外测温仪)
; Y000 感应圈下降阀  Y001 高频加热  Y002 淬火喷淋  Y003 绿灯  Y004 红灯
; D0 完成件数  D1 测温不合格件数  D200 加热时间(0.1s)  D201 淬火时间(0.1s)
LD M8002
SET S0
MOV K30 D200
MOV K30 D201
; ---- 启停
LD X000
AND X001
AND X002
SET M0
LDI X001
ORI X002
RST M0
; ---- 急停：回到初始步
LDI X002
ZRST S20 S23
LDI X002
SET S0
; ---- 步0：等待
STL S0
LD M0
AND X020
AND X010
SET S20
; ---- 步20：感应圈下降
STL S20
OUT Y000
LD X011
SET S21
; ---- 步21：加热
STL S21
OUT Y000
OUT Y001
OUT T0 D200
LD T0
ANI X040
INC D1
LD T0
SET S22
; ---- 步22：喷淋淬火
STL S22
OUT Y000
OUT Y002
OUT T1 D201
LD T1
SET S23
; ---- 步23：感应圈上升，计数
STL S23
LD X010
INC D0
LD X010
SET S0
RET
; ---- 指示灯
LD M0
OUT Y003
LDI X002
OUT Y004
END
"""


class DemoPart:
    def __init__(self, pid):
        self.id, self.heat, self.quench, self.T = pid, 0.0, 0.0, 25.0


class DemoPlant:
    def __init__(self, p_air=0.5, thermal=None, load_time=1.5):
        self.cyl = Cylinder("感应圈升降 TCL50×80", 50, 20, 80, 31.8, "down", v_set=250)
        self.p_air = p_air
        self.thermal = thermal
        self.pos = 0.0          # 0=上 1=下
        self.part = DemoPart(1)
        self.n = 1
        self.t = 0.0
        self.load_time = load_time
        self.t_empty = None
        self.done = []
        self.ext = {"X000": False, "X001": True, "X002": True}
        self._tt()

    def _tt(self):
        te, *_ = self.cyl.travel_time(self.p_air, True)
        tr, *_ = self.cyl.travel_time(self.p_air, False)
        self.te, self.tr = te, tr

    def set_air(self, p):
        self.p_air = p; self._tt()

    def step(self, dt, Y):
        self.t += dt
        if Y.get("Y000"):
            if self.te < 1e6:
                self.pos = min(1.0, self.pos + dt / self.te)
        elif self.tr < 1e6:
            self.pos = max(0.0, self.pos - dt / self.tr)
        p = self.part
        if p is not None:
            if Y.get("Y001") and self.pos >= 1.0:
                p.heat += dt
                p.T = self.thermal.T_surface(p.heat) if self.thermal else min(25 + 290 * p.heat, 1200)
            if Y.get("Y002") and self.pos >= 1.0:
                p.quench += dt
                p.T = max(25.0, p.T - 600 * dt)
            # 圈回到上位且已淬火 → 下料
            if self.pos <= 0 and p.quench > 0.5:
                self.done.append(p); self.part = None; self.t_empty = self.t
        elif self.t - self.t_empty >= self.load_time:
            self.n += 1; self.part = DemoPart(self.n)

    def inputs(self, Y):
        x = dict(self.ext)
        x["X010"] = self.pos <= 0.0
        x["X011"] = self.pos >= 1.0
        x["X020"] = self.part is not None
        x["X040"] = bool(self.part and Y.get("Y001") and self.part.T >= 850)
        return x


class DemoSim:
    def __init__(self, il_text=DEMO_IL, p_air=0.5, thermal=None, scan_ms=10):
        self.plc = FX3U(il_text.splitlines(), scan_ms=scan_ms)
        self.plant = DemoPlant(p_air, thermal)
        self.dt = scan_ms / 1000
        self.t = 0.0
        self.Y = {}
        self.pulses = {}

    def press(self, x, dur=0.2):
        nc = x in ("X001", "X002")
        self.plant.ext[x] = not nc
        self.pulses[x] = (self.t + dur, nc)

    def scan(self):
        for x, (tr, v) in list(self.pulses.items()):
            if self.t >= tr:
                self.plant.ext[x] = v; del self.pulses[x]
        inp = self.plant.inputs(self.Y)
        self.plc.scan(inp)
        self.Y = {f"Y{i:03o}": self.plc.bits.get(f"Y{i:03o}", False) for i in range(8)}
        self.plant.step(self.dt, self.Y)
        self.t = round(self.t + self.dt, 4)
