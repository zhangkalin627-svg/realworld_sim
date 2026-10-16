# -*- coding: utf-8 -*-
"""示例固件（AVR 汇编，通用演示，不对应任何具体产品）。"""

# 引脚分配（联合仿真演示）
PINMAP = """PD2 (INT0) ← PLC Y000 启动加热      PD3 (INT1) ← PLC Y003 故障复位
PD4 → PLC X002 MCU 就绪               PD7 → PLC X003 过流报警          PB0 → PLC X004 软启动完成
PD5 (OC0B) → 半桥驱动 IN（20 kHz PWM）  PD6 → 半桥驱动 EN               ADC0 ← 电流采样放大器 0.2 V/A
PD1 (TXD) → 串口 9600 波特（S 启动 / R 到达 / O C 过流 / r 复位）"""

HEATER_ASM = r"""; ===== 加热器 PWM 控制：软启动 + 过流保护 + PLC 握手（ATmega328P 风格，16 MHz）=====
; 定时器0 Fast PWM（WGM=7，TOP=OCR0A=99，预分频 8）→ 20 kHz；占空比 = (OCR0B+1)/100
; 每个 PWM 周期（溢出中断）启动一次 ADC0 转换（预分频 32 → 500 kHz，13 个 ADC 时钟 = 26 µs）
; ADC 中断：电流 ≥ OC_TH → 立刻关 EN、断开 PWM、报警给 PLC、串口发 "OC"
; 软启动：每 2 ms（40 个 PWM 周期）占空比 +1%，直到 50%；电流 ≥ SOFT_TH 时暂停加大（限流）
.equ TOP      = 99
.equ DUTY_MAX = 49          ; OCR0B=49 → 50 %
.equ OC_TH    = 328         ; 328/1024·5 V = 1.602 V → 8.01 A（0.2 V/A）
.equ SOFT_TH  = 287         ; 7.0 A
.equ RAMP_DIV = 40
.equ UBRR     = 103         ; 16 MHz/(16·104) = 9615 波特
.def sreg_s = r15
.def state  = r20           ; 0 停止 1 软启动 2 运行 3 过流故障
.def duty   = r21
.def tick   = r22
.def adl    = r24
.def adh    = r25

.org 0
    rjmp reset
.org INT0addr
    rjmp isr_int0
.org INT1addr
    rjmp isr_int1
.org OVF0addr
    rjmp isr_t0ovf
.org ADCCaddr
    rjmp isr_adc

.org INT_VECTORS_SIZE
reset:
    ldi r16, low(RAMEND)
    out SPL, r16
    ldi r16, high(RAMEND)
    out SPH, r16
    clr state
    clr duty
    clr tick
    ; GPIO：PD4 PD5 PD6 PD7 PB0 输出
    ldi r16, (1<<PD4)|(1<<PD5)|(1<<PD6)|(1<<PD7)
    out DDRD, r16
    sbi DDRB, 0
    ; 串口 9600 8N1，只发送
    ldi r16, high(UBRR)
    sts UBRR0H, r16
    ldi r16, low(UBRR)
    sts UBRR0L, r16
    ldi r16, (1<<UCSZ01)|(1<<UCSZ00)
    sts UCSR0C, r16
    ldi r16, 1<<TXEN0
    sts UCSR0B, r16
    ; 定时器0：Fast PWM，TOP=OCR0A，先不接 OC0B
    ldi r16, TOP
    out OCR0A, r16
    out OCR0B, duty
    ldi r16, (1<<WGM01)|(1<<WGM00)
    out TCCR0A, r16
    ldi r16, (1<<WGM02)|(1<<CS01)
    out TCCR0B, r16
    ldi r16, 1<<TOIE0
    sts TIMSK0, r16
    ; ADC：AVCC 参考，通道 0，中断，预分频 32
    ldi r16, 1<<REFS0
    sts ADMUX, r16
    ldi r16, (1<<ADEN)|(1<<ADIE)|(1<<ADPS2)|(1<<ADPS0)
    sts ADCSRA, r16
    ; 外部中断：INT0 任意电平变化，INT1 上升沿
    ldi r16, (1<<ISC00)|(1<<ISC11)|(1<<ISC10)
    sts EICRA, r16
    ldi r16, (1<<INT0)|(1<<INT1)
    out EIMSK, r16
    ; 睡眠模式 idle
    ldi r16, 1<<SE
    out SMCR, r16
    sbi PORTD, 4            ; MCU 就绪 → PLC X002
    sei
main:
    sleep
    rjmp main

; ---------- 定时器0 溢出：每个 PWM 周期启动 ADC，软启动斜坡
isr_t0ovf:
    in sreg_s, SREG
    ldi r16, (1<<ADEN)|(1<<ADSC)|(1<<ADIE)|(1<<ADPS2)|(1<<ADPS0)
    sts ADCSRA, r16
    cpi state, 1
    brne t0_done
    inc tick
    cpi tick, RAMP_DIV
    brlo t0_done
    clr tick
    ldi r16, low(SOFT_TH)   ; 限流：电流 ≥ 7 A 时不再加大
    ldi r17, high(SOFT_TH)
    cp adl, r16
    cpc adh, r17
    brsh t0_done
    cpi duty, DUTY_MAX
    brsh t0_reached
    inc duty
    out OCR0B, duty
    rjmp t0_done
t0_reached:
    ldi state, 2
    sbi PORTB, 0            ; 软启动完成 → PLC X004
    ldi r16, 'R'
    sts UDR0, r16
t0_done:
    out SREG, sreg_s
    reti

; ---------- ADC 完成：过流保护
isr_adc:
    in sreg_s, SREG
    lds adl, ADCL
    lds adh, ADCH
    cpi state, 0
    breq adc_done
    cpi state, 3
    breq adc_done
    ldi r16, low(OC_TH)
    ldi r17, high(OC_TH)
    cp adl, r16
    cpc adh, r17
    brlo adc_done
    cbi PORTD, 6            ; ① 先关驱动 EN（最快的动作）
    ldi r16, (1<<WGM01)|(1<<WGM00)
    out TCCR0A, r16         ; ② 断开 OC0B → PD5 回到 PORT=0
    cbi PORTD, 5
    sbi PORTD, 7            ; ③ 报警 → PLC X003
    cbi PORTB, 0
    ldi state, 3
    ldi r16, 'O'
    sts UDR0, r16
    ldi r16, 'C'
    sts UDR0, r16
adc_done:
    out SREG, sreg_s
    reti

; ---------- INT0：PLC 启动/停止命令（PD2 电平）
isr_int0:
    in sreg_s, SREG
    sbis PIND, 2
    rjmp int0_stop
    cpi state, 0
    brne int0_done          ; 故障中/已运行：忽略
    clr duty
    out OCR0B, duty
    clr tick
    ldi r16, (1<<COM0B1)|(1<<WGM01)|(1<<WGM00)
    out TCCR0A, r16         ; 接上 OC0B
    sbi PORTD, 6            ; EN
    ldi state, 1
    ldi r16, 'S'
    sts UDR0, r16
    rjmp int0_done
int0_stop:
    cbi PORTD, 6
    ldi r16, (1<<WGM01)|(1<<WGM00)
    out TCCR0A, r16
    cbi PORTD, 5
    cbi PORTB, 0
    cpi state, 3
    breq int0_done
    clr state
int0_done:
    out SREG, sreg_s
    reti

; ---------- INT1：PLC 故障复位（PD3 上升沿；启动命令已撤销才允许复位）
isr_int1:
    in sreg_s, SREG
    cpi state, 3
    brne int1_done
    sbic PIND, 2
    rjmp int1_done
    cbi PORTD, 7
    clr state
    ldi r16, 'r'
    sts UDR0, r16
int1_done:
    out SREG, sreg_s
    reti
"""

# 通用 PLC 程序（FX 指令表）：启动/停止、加热计时、报警锁存与复位
PLC_IL = """; ===== 加热工位 PLC 程序（FX3U 指令表，通用演示）=====
; X000 启动按钮  X001 停止按钮(常闭)  X002 MCU 就绪  X003 MCU 过流报警  X004 MCU 软启动完成  X005 报警复位按钮
; Y000 启动加热 → MCU PD2   Y001 报警灯   Y002 加热中   Y003 故障复位 → MCU PD3
; T0 加热时间（100 ms 单位）  D0 报警次数
; ---- 报警锁存放在前面：同一个扫描里就能切断启动
LD X003
SET M10
LDP X003
INC D0
LD X005
ANI X003
RST M10
; ---- 启停自保持
LD X000
OR M0
AND X001
AND X002
ANI M10
ANI T0
OUT M0
LD M0
OUT Y000
LD M0
AND X004
OUT Y002
LD M0
OUT T0 K5
LD M10
OUT Y001
LD X005
OUT Y003
END
"""

# ---------- validate.py 用的小程序
TIMING_ASM = r"""; 指令计时：PB5 翻转，延时循环 dec/brne（3 周期/次，最后一次 2 周期）
.org 0
    rjmp reset
.org INT_VECTORS_SIZE
reset:
    ldi r16, low(RAMEND)
    out SPL, r16
    ldi r16, high(RAMEND)
    out SPH, r16
    sbi DDRB, 5
loop:
    sbi PINB, 5             ; 2 周期（写 PINx 翻转）
    ldi r16, 200            ; 1
d1: dec r16                 ; 1
    brne d1                 ; 2/1
    ldi r24, low(1000)      ; 1
    ldi r25, high(1000)     ; 1
d2: sbiw r24, 1             ; 2
    brne d2                 ; 2/1
    rcall sub               ; 3 + RET 4
    rjmp loop               ; 2
sub:
    nop
    ret
"""
TIMING_CYCLES = 2 + 1 + (200 * 3 - 1) + 1 + 1 + (1000 * 4 - 1) + 3 + 1 + 4 + 2   # 一次循环（相邻两次翻转）


def timer_asm(mode, presc_cs, ocra, ocrb=0, com="", toggle_pin="PB0"):
    """生成定时器0 测试程序：溢出中断里翻转 PB0（测溢出频率），OC0A/OC0B 引脚出 PWM。"""
    return f"""
.org 0
    rjmp reset
.org OVF0addr
    rjmp ovf
.org OC0Aaddr
    rjmp cmpa
.org INT_VECTORS_SIZE
reset:
    ldi r16, low(RAMEND)
    out SPL, r16
    ldi r16, high(RAMEND)
    out SPH, r16
    ldi r16, (1<<PD5)|(1<<PD6)
    out DDRD, r16
    sbi DDRB, 0
    sbi DDRB, 1
    ldi r16, {ocra}
    out OCR0A, r16
    ldi r16, {ocrb}
    out OCR0B, r16
    ldi r16, {com or 0}|({mode & 3})
    out TCCR0A, r16
    ldi r16, ({(mode >> 2) & 1}<<WGM02)|{presc_cs}
    out TCCR0B, r16
    ldi r16, (1<<TOIE0)|(1<<OCIE0A)
    sts TIMSK0, r16
    ldi r16, 1<<SE
    out SMCR, r16
    sei
main:
    sleep
    rjmp main
ovf:
    sbi PINB, 0
    reti
cmpa:
    sbi PINB, 1
    reti
"""


ADC_ASM = r"""; ADC 单次转换：通道 0，AVCC 参考，预分频 128，轮询 ADIF，结果存 SRAM 0x0100
.org 0
    rjmp reset
.org INT_VECTORS_SIZE
reset:
    ldi r16, low(RAMEND)
    out SPL, r16
    ldi r16, high(RAMEND)
    out SPH, r16
    ldi r16, 1<<REFS0
    sts ADMUX, r16
    ldi r16, (1<<ADEN)|(1<<ADSC)|7
    sts ADCSRA, r16
wait:
    lds r16, ADCSRA
    sbrs r16, ADIF
    rjmp wait
    lds r24, ADCL
    lds r25, ADCH
    sts 0x0100, r24
    sts 0x0101, r25
    break
"""

UART_ASM = r"""; 串口 9600 8N1 发送 "U" 和 "OK\n"（轮询 UDRE）
.equ UBRR = 103
.org 0
    rjmp reset
.org INT_VECTORS_SIZE
reset:
    ldi r16, low(RAMEND)
    out SPL, r16
    ldi r16, high(RAMEND)
    out SPH, r16
    ldi r16, high(UBRR)
    sts UBRR0H, r16
    ldi r16, low(UBRR)
    sts UBRR0L, r16
    ldi r16, (1<<UCSZ01)|(1<<UCSZ00)
    sts UCSR0C, r16
    ldi r16, 1<<TXEN0
    sts UCSR0B, r16
    ldi r17, 'U'
    rcall putc
    ldi r17, 'O'
    rcall putc
    ldi r17, 'K'
    rcall putc
    ldi r17, 10
    rcall putc
idle:
    rjmp idle
putc:
    lds r16, UCSR0A
    sbrs r16, UDRE0
    rjmp putc
    sts UDR0, r17
    ret
"""

INT_ASM = r"""; 外部中断响应：PD2 上升沿 → INT0 → ISR 第一条指令置 PB1（主循环在睡眠）
.org 0
    rjmp reset
.org INT0addr
    rjmp isr
.org INT_VECTORS_SIZE
reset:
    ldi r16, low(RAMEND)
    out SPL, r16
    ldi r16, high(RAMEND)
    out SPH, r16
    sbi DDRB, 1
    ldi r16, (1<<ISC01)|(1<<ISC00)
    sts EICRA, r16
    sbi EIMSK, 0
    ldi r16, 1<<SE
    out SMCR, r16
    sei
main:
    sleep
    rjmp main
isr:
    sbi PORTB, 1
    reti
"""
