# 公开示例网表

- `npn_switch_kicad.net`：按 KiCad 8 网表格式（Export Netlist → KiCad，S 表达式）**手写**的示例：
  1 kHz 方波 → 10 kΩ → NPN 开关管，集电极 1 kΩ 上拉到 12 V，100 nF 到地。
  不是从真实 KiCad 工程导出的（沙盒里没有 KiCad），字段名按 KiCad 8 的 Sim.Device / Sim.Type / Sim.Params / pinfunction 写。
- `npn_switch.cir`：同一电路的 SPICE 网表（KiCad「导出 SPICE 网表」的样子），带 .tran。

两份网表 MIT 许可，可随意使用。
