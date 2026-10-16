# -*- coding: utf-8 -*-
"""通用整机 PLC 程序（FX3U 指令表，公开示例）：上电顺序 + 单工位淬火循环 + 报警联锁。
可以在 GUI 里直接改，再跑一遍看整机怎么反应。"""

IO_MAP = """输入  X000 启动  X001 停止(常闭)  X002 急停(常闭)  X003 复位
      X004 冷却水流量开关  X005 气压开关  X010 线圈上位  X011 线圈下位
      X012 变频器就绪  X013 变频器故障  X014 功率到达  X020 工位有料  X040 红外测温≥850°C
输出  Y000 K1 主接触器(经预充电阻)  Y001 K2 旁路接触器  Y002 冷却水泵  Y003 线圈下降阀
      Y004 高频使能(到变频器)  Y005 喷淋泵+阀  Y006 红灯  Y007 绿灯"""

MACHINE_IL = """; ===== 整机联动演示程序 (FX3U 指令表) =====
; D200 加热时间(0.1s)  D201 淬火时间(0.1s)  D202 预充电等待(0.1s)  D0 完成件数  D1 测温不合格件数
LD M8002
SET S0
MOV K30 D200
MOV K30 D201
MOV K20 D202
; ---- 启停
LD X000
AND X001
AND X002
ANI M9
SET M0
LDI X001
ORI X002
OR M9
RST M0
LDI M0
ZRST S10 S23
LDI M0
SET S0
LDI M0
ZRST M20 M22
; ---- 上电顺序：冷却水 -> 预充电 -> 旁路 -> 等变频器就绪
STL S0
LD M0
AND X005
SET S10
STL S10
SET M20
OUT T10 K10
LD T10
AND X004
SET S11
STL S11
SET M21
OUT T11 D202
LD T11
SET S12
STL S12
SET M22
OUT T13 K80
LD T13
SET M304
LD X012
SET S13
; ---- 工作循环
STL S13
LD X020
AND X010
AND X012
SET S20
STL S20
OUT Y003
OUT T14 K60
LD T14
SET M303
LD X011
SET S21
STL S21
OUT Y003
OUT Y004
OUT T0 D200
LD X040
SET M30
LD T0
ANI M30
INC D1
LD T0
SET S22
STL S22
OUT Y003
OUT Y005
OUT T1 D201
LD T1
SET S23
STL S23
RST M30
LD X010
INC D0
LD X010
SET S13
RET
; ---- 输出
LD M20
OR M9
OUT Y002
LD M21
OUT Y000
LD M22
OUT Y001
LD M9
OUT Y006
LD M0
ANI M9
OUT Y007
; ---- 报警锁存
LD M0
ANI X004
OUT T20 K5
LD T20
SET M300
LD M0
AND X013
SET M301
LD M0
ANI X005
OUT T21 K3
LD T21
SET M302
LD M300
OR M301
OR M302
OR M303
OR M304
OUT M9
LD X003
ZRST M300 M304
END
"""

# 常见的“现场改动”，用于故障场景
def without_water_interlock(il=MACHINE_IL):
    """冷却水流量联锁被短接（现场常见）：缺水不再报警。"""
    return il.replace("LD T20\nSET M300\n", "LD T20\n; (水流联锁已短接)\n")


def no_wait_for_coil(il=MACHINE_IL):
    """PLC 编程错误：线圈开始下降的同时就开高频（没等下限位）。"""
    return il.replace("STL S20\nOUT Y003\n", "STL S20\nOUT Y003\nOUT Y004\n")
