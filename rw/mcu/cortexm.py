# -*- coding: utf-8 -*-
"""可选：ARM Cortex-M 核心（Thumb 指令），用 unicorn 引擎执行（实验性）。

unicorn 是外部依赖：`pip install unicorn`。引擎本体 GPLv2（Python 绑定标注 BSD）；本仓库不拷贝它的任何代码，
只在运行时 import。如果你要把 unicorn 和本软件打包在一起分发，打包产物需遵守 GPLv2。

这里给的是最小可用接口：装载裸机二进制（向量表在 0x08000000），一个通用的 GPIO 外设映射在 0x40000000
（ODR 输出寄存器 +0x00，IDR 输入寄存器 +0x04；这是本软件自定义的简化外设，不是任何厂家的寄存器表）。
计时：unicorn 不是周期精确的，这里按“每条指令 1 个周期”（cpi 可改）估算时间 —— 只能当估算。
"""
try:
    import unicorn as _uc
    from unicorn import arm_const as _arm
    HAVE_UNICORN = True
except Exception:          # noqa: BLE001
    HAVE_UNICORN = False

FLASH, SRAM, GPIO = 0x08000000, 0x20000000, 0x40000000

# 演示程序（手工编码的 Thumb 机器码）：r0=0x40000000，r1 在 0/1 之间翻转并写到 ODR
#   08: movs r0,#0x40   0A: lsls r0,r0,#24   0C: movs r1,#1   0E: movs r2,#1
#   10: str r1,[r0]     12: eors r1,r2       14: b 0x10
BLINK_THUMB = bytes.fromhex("00100020" "09000008") + bytes(
    b for hw in (0x2040, 0x0600, 0x2101, 0x2201, 0x6001, 0x4051, 0xE7FC) for b in hw.to_bytes(2, "little"))


class CortexM:
    def __init__(self, image=BLINK_THUMB, f_cpu=16e6, cpi=1.0):
        if not HAVE_UNICORN:
            raise RuntimeError("没有安装 unicorn（可选依赖）：pip install unicorn")
        self.f, self.cpi = f_cpu, cpi
        self.uc = _uc.Uc(_uc.UC_ARCH_ARM, _uc.UC_MODE_THUMB | _uc.UC_MODE_MCLASS)
        self.uc.mem_map(FLASH, 0x10000); self.uc.mem_map(SRAM, 0x10000); self.uc.mem_map(GPIO, 0x1000)
        self.uc.mem_write(FLASH, image)
        self.sp = int.from_bytes(image[0:4], "little"); self.entry = int.from_bytes(image[4:8], "little")
        self.uc.reg_write(_arm.UC_ARM_REG_SP, self.sp)
        self.n = 0; self.odr = 0; self.idr = 0
        self.pin_log = []            # (估算时间 s, ODR 值)
        self.uc.hook_add(_uc.UC_HOOK_CODE, self._code)
        self.uc.hook_add(_uc.UC_HOOK_MEM_WRITE, self._mw, begin=GPIO, end=GPIO + 0xFFF)
        self.uc.hook_add(_uc.UC_HOOK_MEM_READ, self._mr, begin=GPIO, end=GPIO + 0xFFF)

    def _code(self, uc, addr, size, ud):
        self.n += 1

    def _mw(self, uc, access, addr, size, value, ud):
        if addr == GPIO:
            self.odr = value & 0xFFFFFFFF
            self.pin_log.append((self.n * self.cpi / self.f, self.odr))

    def _mr(self, uc, access, addr, size, value, ud):
        if addr == GPIO + 4:
            uc.mem_write(GPIO + 4, int(self.idr).to_bytes(4, "little"))

    def run(self, n_instr):
        pc = self.uc.reg_read(_arm.UC_ARM_REG_PC) if self.n else self.entry
        self.uc.emu_start(pc | 1, 0xFFFFFFFF, count=n_instr)
        return self.n
