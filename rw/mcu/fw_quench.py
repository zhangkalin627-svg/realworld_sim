# -*- coding: utf-8 -*-
"""淬火电源逆变器控制固件（AVR 汇编，通用教学示例，不对应任何具体产品）。

相位锁定环频率跟踪：模拟鉴相器方案（v2，取代过零时间戳方案——ATmega328P 无输入捕获，
多中断源竞争使时间戳抖动远超相位目标，真实控制器此处也用模拟鉴相，CD4046 式）。
V/I 方波 XOR 占空 ∝ |φ|/π，RC 积分成直流 → ADC1 读相位幅值；超前/滞后由 D 触发器
（i 沿采样 v 电平）给 PD4 数字量。固件全程无定时中断级测量，只有 T0 溢出计数控制拍。
独立设计：教科书原理（XOR 鉴相 + I 环 + 一阶 DSM 频率细调），参数从 rw.machine.power 推导。
"""

PINMAP = """PD5 (OC0B) → 门极方波（T0 Fast PWM 模式7，预分频 8，f = 2MHz/(OCR0A+1)，占空 ≈50%）
PD6 → 门极使能 EN        PD7 → LOCK 锁定指示        PB0 → FAULT 故障锁存
PD4 ← 超前/滞后（1=容性超前，0=感性滞后，鉴相器 D 触发器输出）
ADC1 ← 相位幅值（XOR+RC：0..π → 0..5 V，φ_ref 0.221 rad ≈ 码 18）
ADC0 ← 电流包络 0.01 V/A（ADLAR=1，读 ADCH 得 8 位）"""

QUENCH_PLL_ASM = r"""; ===== 淬火电源逆变器控制固件：PLL 频率跟踪（ATmega328P，16 MHz，模拟鉴相 v2）=====
; 测量：ADC1 = 相位幅值直流（XOR 占空 φ/π × 5 V，8 位码）；PD4 = 超前(1)/滞后(0)。
;       无中断级时间戳——控制拍里阻塞读 ADC（26 µs，主循环里无所谓）。
; 目标：φ_ref = 0.221 rad → 占空 0.0704 → 0.352 V → REF = 18 码。
; 误差：滞后 e = REF − code；超前 e = REF + code（推频率上行脱离容性）。±32 限幅
;       （≈扫频 57 kHz/s，对齐行为级基线 150 kHz/s 量级并压过冲），IIR(1/8) 用 Q3
;       定点（e_filt 存 e·8）消整数死区，I 环：每控制拍（30 PWM 周期 ≈1 ms）
;       top_f −= e_filt>>3。
; 执行：8 位 TOP 在 30 kHz 步进约 1.5% —— 8.8 定点频率字 + 每 PWM 周期一阶 DSM，
;       平均频率分辨率 < 2 Hz（等效分数分频门极驱动）。
; 保护：超前（容性）持续 50 拍 → LOSS；包络 ADC0 ≥ OC_TH → OC。跳闸 = 封门极 + FAULT 锁存。

.equ TOP_START = 43          ; 46.5 kHz ≈ 0.775·f_max（对齐行为级 0.75·f_max 起步）
.equ TOP_MIN   = 33          ; 58.8 kHz（f_max = 60 kHz）
.equ TOP_MAX   = 249         ; 8.0 kHz
.equ E_LIM     = 32          ; e 限幅（ADC 码）
.equ REF       = 18          ; φ_ref = 18/256·5V/5·π = 0.2209 rad
.equ LOCK_BAND = 2           ; |e_filt| ≤ 2 码 → LOCK（≈0.025 rad 滞回）
.equ DWELL_TRIP = 50         ; 容性驻留拍数（≈50 ms）
.equ OC_TH     = 173         ; ADCH 8 位阈值：3.39 V ≈ 339 Arms（I_trip_pk 480 A 峰值）
.equ TICKS_PER_CTL = 30      ; 每 30 个 PWM 周期一个控制拍（30 kHz 时 = 1 ms）

; SRAM 变量（复位后 bytearray 全零，依赖零初始化）
.equ v_state  = 0x0100       ; 0 boot / 1 run / 3 trip
.equ v_topf   = 0x0101       ; 2B，8.8 定点 TOP
.equ v_dsm    = 0x0103       ; DSM 累加器
.equ v_tick   = 0x0104
.equ v_dwell  = 0x0105
.equ v_efilt  = 0x0106       ; 2B，有符号（Q3：码·8，小数累加消 IIR 死区）
.equ v_boot   = 0x0108
.equ v_flag   = 0x0109       ; 控制拍标志（OVF0 置位，主循环清）

.org 0
    rjmp reset
.org OVF0addr
    rjmp isr_t0ovf

.org INT_VECTORS_SIZE
reset:
    ldi r16, low(RAMEND)
    out SPL, r16
    ldi r16, high(RAMEND)
    out SPH, r16
    ; 端口：PD5 门极 / PD6 EN / PD7 LOCK 输出，PB0 FAULT 输出，初值全低；PD4 输入
    ldi r16, (1<<PD5)|(1<<PD6)|(1<<PD7)
    out DDRD, r16
    clr r16
    out PORTD, r16
    sbi DDRB, 0
    cbi PORTB, 0
    ; T0：Fast PWM 模式 7，OC0B 暂断开，预分频 8，TOP = TOP_START
    ldi r16, (1<<WGM01)|(1<<WGM00)
    out TCCR0A, r16
    ldi r16, TOP_START
    out OCR0A, r16
    ldi r16, TOP_START/2
    out OCR0B, r16
    ldi r16, (1<<WGM02)|(1<<CS01)
    out TCCR0B, r16
    ldi r16, 1<<TOIE0
    sts TIMSK0, r16
    ; 频率字 = TOP_START·256
    clr r16
    sts v_topf, r16
    ldi r16, TOP_START
    sts v_topf+1, r16
    ; ADC：AVcc 基准，左对齐，预分频 32（500 kHz，转换 26 µs）
    ldi r16, (1<<ADEN)|(1<<ADPS2)|(1<<ADPS0)
    sts ADCSRA, r16
    ; idle 睡眠，控制拍在主循环
    ldi r16, 1<<SE
    out SMCR, r16
    sei
main:
    sleep
    lds r16, v_flag
    tst r16
    breq main
    clr r16
    sts v_flag, r16
    rcall ctrl_tick
    rjmp main

; ---- T0 溢出（每个 PWM 周期）：DSM 更新 TOP/OCR0B；每 30 周期置控制拍标志
isr_t0ovf:
    in  r15, SREG
    push r16
    push r17
    lds r16, v_dsm
    lds r17, v_topf
    add r16, r17
    sts v_dsm, r16
    lds r16, v_topf+1
    brcc t0_top
    inc r16
t0_top:
    out OCR0A, r16
    lsr r16
    out OCR0B, r16
    lds r16, v_tick
    inc r16
    cpi r16, TICKS_PER_CTL
    brlo t0_save
    clr r16
    ldi r17, 1
    sts v_flag, r17
t0_save:
    sts v_tick, r16
    pop r17
    pop r16
    out SREG, r15
    reti

; ---- 阻塞读 ADC 通道 r17 → r16 = ADCH（8 位）
read_adc:
    ldi r16, (1<<REFS0)|(1<<ADLAR)
    or  r16, r17
    sts ADMUX, r16
    lds r16, ADCSRA
    ori r16, 1<<ADSC
    sts ADCSRA, r16
ra_wait:
    lds r16, ADCSRA
    sbrc r16, ADSC
    rjmp ra_wait
    lds r16, ADCH
    ret

; ---- 控制拍（≈1 ms，主循环调用）：boot → run（I 环 + 指示 + 保护）→ trip
ctrl_tick:
    push r17
    push r18
    push r19
    push r20
    push r21
    lds r16, v_state
    cpi r16, 3
    brne ct_alive
    rjmp ct_ret
ct_alive:
    cpi r16, 1
    brne ct_boot
    rjmp ct_run
ct_boot:
    lds r16, v_boot
    inc r16
    sts v_boot, r16
    cpi r16, 5
    brsh ct_go
    rjmp ct_ret
ct_go:
    ldi r16, 1
    sts v_state, r16
    ldi r16, (2<<COM0B0)|(1<<WGM01)|(1<<WGM00)
    out TCCR0A, r16
    sbi PORTD, 6
    rjmp ct_ret
ct_run:
    ; 相位：ADC1 → e（滞后 e = REF − code；超前 e = REF + code）
    ldi r17, 1
    rcall read_adc          ; r16 = 相位码
    sbis PIND, 4            ; PD4=1 → 超前（容性）
    rjmp ct_lag
    ; 超前：e = REF + code
    subi r16, -REF          ; r16 += REF
    clr r17
    rjmp ct_clamp
ct_lag:
    ; 滞后：e = REF − code
    ldi r17, REF
    sub r17, r16
    mov r16, r17
    clr r17
    sbrc r16, 7             ; 负结果 → 符号扩展
    ser r17
ct_clamp:
    ; r17:r16 = e（有符号），限幅 ±E_LIM
    cpi r16, E_LIM
    clr r20
    cpc r17, r20
    brge ct_ehi
    cpi r16, 0xE0           ; -32
    ldi r20, 0xFF
    cpc r17, r20
    brge ct_iir
    ldi r16, 0xE0
    ldi r17, 0xFF
    rjmp ct_iir
ct_ehi:
    ldi r16, E_LIM
    clr r17
ct_iir:
    ; e_filt(Q3) += e − (e_filt>>3)：小数位累加，|e|<8 也能收敛（原整数 IIR 有 ±7 码死区）
    lds r18, v_efilt
    lds r19, v_efilt+1
    mov r20, r18
    mov r21, r19
    asr r21
    ror r20
    asr r21
    ror r20
    asr r21
    ror r20
    sub r16, r20
    sbc r17, r21
    add r18, r16
    adc r19, r17
    sts v_efilt, r18
    sts v_efilt+1, r19
    ; I 环：top_f −= (e_filt>>3)
    lds r16, v_topf
    lds r17, v_topf+1
    asr r19
    ror r18
    asr r19
    ror r18
    asr r19
    ror r18
    sub r16, r18
    sbc r17, r19
    cpi r16, 0
    ldi r20, TOP_MIN
    cpc r17, r20
    brsh ct_hi
    ldi r16, 0
    ldi r17, TOP_MIN
    rjmp ct_save
ct_hi:
    cpi r16, 0
    ldi r20, TOP_MAX
    cpc r17, r20
    brlo ct_save
    ldi r16, 0
    ldi r17, TOP_MAX
ct_save:
    sts v_topf, r16
    sts v_topf+1, r17
    ; LOCK：|e_filt| ≤ LOCK_BAND（Q3：阈值 ·8）
    lds r18, v_efilt
    lds r19, v_efilt+1
    cpi r18, 24             ; (LOCK_BAND+1)·8
    clr r20
    cpc r19, r20
    brge ct_ul
    cpi r18, 0xF0           ; -16 = -LOCK_BAND·8
    ldi r20, 0xFF
    cpc r19, r20
    brlt ct_ul
    sbi PORTD, 7
    rjmp ct_dw
ct_ul:
    cbi PORTD, 7
ct_dw:
    ; 容性驻留 → LOSS
    sbis PIND, 4
    rjmp ct_cd
    lds r16, v_dwell
    inc r16
    sts v_dwell, r16
    cpi r16, DWELL_TRIP
    brlo ct_oc
    rjmp ct_trip
ct_cd:
    clr r16
    sts v_dwell, r16
ct_oc:
    ; 过流：包络 ADC0 ≥ OC_TH
    clr r17
    rcall read_adc
    cpi r16, OC_TH
    brsh ct_trip
    rjmp ct_ret
ct_trip:
    ldi r16, 3
    sts v_state, r16
    ldi r16, (1<<WGM01)|(1<<WGM00)   ; 断开 OC0B，门极拉低
    out TCCR0A, r16
    cbi PORTD, 5
    cbi PORTD, 6
    cbi PORTD, 7
    sbi PORTB, 0
ct_ret:
    pop r21
    pop r20
    pop r19
    pop r18
    pop r17
    ret
"""
