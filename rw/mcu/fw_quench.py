# -*- coding: utf-8 -*-
"""淬火电源逆变器控制固件（AVR 汇编，通用教学示例，不对应任何具体产品）。

相位锁定环频率跟踪：只用 V/I 过零比相驱动开关频率，不知谐振点（与 rw.machine.power
的 track_mode="pll" 行为级控制器同一思路，这里是指令级真实固件）。独立设计：教科书原理
（过零比相 + I 环 + 一阶 DSM 频率细调），参数从 rw.machine.power 的负载模型推导。
"""

PINMAP = """PD5 (OC0B) → 门极方波（T0 Fast PWM 模式7，预分频 8，f = 2MHz/(OCR0A+1)，占空 ≈50%）
PD6 → 门极使能 EN        PD7 → LOCK 锁定指示        PB0 → FAULT 故障锁存
PD2 (INT0) ← V 过零比较器（上升沿）   PD3 (INT1) ← I 过零比较器（上升沿）
ADC0 ← 电流包络 0.01 V/A（ADLAR=1，读 ADCH 得 8 位）"""

QUENCH_PLL_ASM = r"""; ===== 淬火电源逆变器控制固件：PLL 频率跟踪（ATmega328P，16 MHz）=====
; 测量：T2 自由运行（预分频 1，62.5 ns/字）；INT0(v 过零) 记周期 per，INT1(i 过零) 算 Δ = t_i − t_v。
; 相位：Δ < per/2 → 感性（滞后），e = Δ_ref − Δ；Δ ≥ per/2 → 容性，e = Δ_ref + (per − Δ)。
; 目标：Δ_ref = per·9>>8（φ_ref = 9/256·2π ≈ 0.221 rad，感性 ZVS 裕度）。
; 执行：8 位 TOP 在 30 kHz 步进约 1.5% —— 8.8 定点频率字 + 每 PWM 周期一阶 DSM，
;       平均频率分辨率 < 2 Hz（等效于分数分频门极驱动）。
; 环：e 限幅 ±96 tick（≈扫频 170 kHz/s，对齐行为级基线 150 kHz/s 量级），IIR(1/8) 滤波，
;     每控制拍（30 个 PWM 周期 ≈ 1 ms）top_f −= e_filt（I 环，无 P 项，同行为级）。
; 保护：容性持续 50 拍 → LOSS；I 过零丢失 100 拍 → LOSS；包络 ≥ OC_TH → OC。跳闸锁存。

.equ TOP_START = 43          ; 46.5 kHz ≈ 0.775·f_max（对齐行为级 0.75·f_max 起步）
.equ TOP_MIN   = 33          ; 58.8 kHz（f_max = 60 kHz）
.equ TOP_MAX   = 249         ; 8.0 kHz
.equ E_LIM     = 32          ; e 限幅（T2 tick，≈扫频 57 kHz/s；压过冲 + 抗时间戳抖动）
.equ GUARD     = 24          ; Δ 或 per−Δ 小于此值 = ISR 碰撞区，该周测量作废（保持频率）
.equ LOCK_BAND = 2           ; |e_filt| ≤ 2 → LOCK（≈0.024 rad 滞回）
.equ DWELL_TRIP = 50         ; 容性驻留拍数（≈50 ms）
.equ AGE_TRIP  = 100         ; I 过零丢失拍数（≈100 ms）
.equ OC_TH     = 173         ; ADCH 8 位阈值：3.39 V ≈ 339 Arms（I_trip_pk 480 A 峰值）
.equ TICKS_PER_CTL = 30      ; 每 30 个 PWM 周期一个控制拍（30 kHz 时 = 1 ms）

; SRAM 变量（复位后 bytearray 全零，依赖零初始化）
.equ v_state  = 0x0100       ; 0 boot / 1 run / 3 trip
.equ v_topf   = 0x0101       ; 2B，8.8 定点 TOP
.equ v_dsm    = 0x0103       ; DSM 累加器
.equ v_tick   = 0x0104
.equ v_dwell  = 0x0105
.equ v_age    = 0x0106
.equ v_tv     = 0x0107       ; 2B，上次 v 过零时间戳（16 位回绕 4.096 ms » 周期）
.equ v_per    = 0x0109       ; 2B，门极周期（T2 字）
.equ v_dref   = 0x010B       ; Δ_ref（< 64，1B 足够）
.equ v_perh   = 0x010C       ; 2B，per/2
.equ v_efilt  = 0x010E       ; 2B，有符号
.equ v_capf   = 0x0110       ; 最近一周容性标志
.equ v_boot   = 0x0111
.equ v_ovf2   = 0x0112       ; T2 溢出计数（低 8 位参与时间戳）
.equ v_delta  = 0x0113       ; 2B，最近一周 Δ（INT1 原始值，控制拍里处理）
.equ v_newd   = 0x0115       ; 新 Δ 标志
.equ v_perf   = 0x0116       ; 2B，per 滤波（4 周平均）
.equ v_tv2    = 0x0118       ; 2B，上上周 v 时间戳（INT0/INT1 排队抢跑修正用）

.org 0
    rjmp reset
.org INT0addr
    rjmp isr_int0
.org INT1addr
    rjmp isr_int1
.org OVF2addr
    rjmp isr_t2ovf
.org OVF0addr
    rjmp isr_t0ovf

.org INT_VECTORS_SIZE
reset:
    ldi r16, low(RAMEND)
    out SPL, r16
    ldi r16, high(RAMEND)
    out SPH, r16
    ; 端口：PD5 门极 / PD6 EN / PD7 LOCK 输出，PB0 FAULT 输出，初值全低
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
    ; T2：普通模式，预分频 1，溢出中断（时间基准）
    clr r16
    sts TCCR2A, r16
    ldi r16, 1<<CS20
    sts TCCR2B, r16
    ldi r16, 1<<TOIE2
    sts TIMSK2, r16
    ; INT0/INT1 上升沿
    ldi r16, (1<<ISC11)|(1<<ISC10)|(1<<ISC01)|(1<<ISC00)
    sts EICRA, r16
    ldi r16, (1<<INT1)|(1<<INT0)
    out EIMSK, r16
    ; ADC0：AVcc 基准，左对齐，预分频 32（500 kHz，转换 26 µs）
    ldi r16, (1<<REFS0)|(1<<ADLAR)
    sts ADMUX, r16
    ldi r16, (1<<ADEN)|(1<<ADPS2)|(1<<ADPS0)
    sts ADCSRA, r16
    ; idle 睡眠，逻辑全在中断里
    ldi r16, 1<<SE
    out SMCR, r16
    sei
main:
    sleep
    rjmp main

; ---- 时间戳：r18 = TCNT2，r19 = ovf2（16 位）；溢出挂起且低字节尚小 → 补 1
read_t:
    lds r18, TCNT2
    lds r19, v_ovf2
    in  r16, TIFR2
    sbrs r16, TOV2
    ret
    sbrc r18, 7
    ret
    inc r19
    ret

isr_t2ovf:
    in  r15, SREG
    push r16
    lds r16, v_ovf2
    inc r16
    sts v_ovf2, r16
    pop r16
    out SREG, r15
    reti

; ---- V 过零：只记时间戳与 per（短 ISR = 窄碰撞窗）；dref/perh 在控制拍里算
isr_int0:
    in  r15, SREG
    push r16
    push r17
    push r18
    push r19
    rcall read_t
    lds r16, v_tv
    lds r17, v_tv+1
    sts v_tv2, r16
    sts v_tv2+1, r17
    sts v_tv, r18
    sts v_tv+1, r19
    sub r18, r16
    sbc r19, r17
    cpi r18, 0xB9           ; 3001 = 0x0BB9
    ldi r16, 0x0B
    cpc r19, r16
    brsh i0_out
    sts v_per, r18
    sts v_per+1, r19
i0_out:
    pop r19
    pop r18
    pop r17
    pop r16
    out SREG, r15
    reti

; ---- I 过零：只算 Δ = t_i − t_v 存原始值（处理全在控制拍）；age 清零
isr_int1:
    in  r15, SREG
    push r16
    push r17
    push r18
    push r19
    push r20
    push r21
    rcall read_t
    lds r16, v_tv
    lds r17, v_tv+1
    sub r18, r16
    sbc r19, r17           ; Δ1（按当前 t_v）
    ; Δ1 ≥ per → i 沿其实在 v 沿之前（INT0 抢跑先更新了 t_v）→ 用上上周 t_v2
    lds r16, v_per
    lds r17, v_per+1
    cp  r18, r16
    cpc r19, r17
    brlo i1_ok
    lds r16, v_tv2
    lds r17, v_tv2+1
    mov r20, r18
    mov r21, r19
    sub r20, r16
    sbc r21, r17
    mov r18, r20
    mov r19, r21
i1_ok:
    sts v_delta, r18
    sts v_delta+1, r19
    ldi r16, 1
    sts v_newd, r16
    clr r16
    sts v_age, r16
    pop r21
    pop r20
    pop r19
    pop r18
    pop r17
    pop r16
    out SREG, r15
    reti

; ---- T0 溢出（每个 PWM 周期）：DSM 更新 TOP/OCR0B；每 30 周期一个控制拍
isr_t0ovf:
    in  r15, SREG
    push r16
    push r17
    ; 一阶 DSM：acc += frac(top_f)；TOP = base + 进位
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
    sts v_tick, r16
    rcall ctrl_tick
    rjmp t0_out
t0_save:
    sts v_tick, r16
t0_out:
    pop r17
    pop r16
    out SREG, r15
    reti

; ---- 控制拍（≈1 ms）：boot → run（I 环 + 指示 + 保护）→ trip
ctrl_tick:
    push r18
    push r19
    push r20
    lds r16, v_state
    cpi r16, 3
    brne ct_alive
    rjmp ct_ret
ct_alive:
    cpi r16, 1
    breq ct_run
    ; boot：5 拍后接管门极
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
    ; --- 测量处理：per 滤波（4 周）→ dref/perh → 碰撞守卫 → e → IIR
    lds r16, v_newd
    tst r16
    brne ct_meas
    rjmp ct_pi              ; 没有新 Δ：跳过更新
ct_meas:
    clr r16
    sts v_newd, r16
    ; per_filt += (per − per_filt)>>2（首周从 0 直接灌入）
    lds r16, v_per
    lds r17, v_per+1
    lds r18, v_perf
    lds r19, v_perf+1
    mov r20, r18
    or  r20, r19
    brne ct_f1
    sts v_perf, r16
    sts v_perf+1, r17
    rjmp ct_f2
ct_f1:
    sub r16, r18
    sbc r17, r19
    asr r17
    ror r16
    asr r17
    ror r16
    add r18, r16
    adc r19, r17
    sts v_perf, r18
    sts v_perf+1, r19
ct_f2:
    ; dref = (per_filt·9)>>8；perh = per_filt/2
    lds r18, v_perf
    lds r19, v_perf+1
    lsr r19
    ror r18
    sts v_perh, r18
    sts v_perh+1, r19
    lds r16, v_perf
    lds r17, v_perf+1
    mov r18, r16
    mov r19, r17
    lsl r18
    rol r19
    lsl r18
    rol r19
    lsl r18
    rol r19
    add r18, r16
    adc r19, r17
    sts v_dref, r19
    ; 守卫：Δ < GUARD 或 per−Δ < GUARD → 作废（保持频率）；否则按象限算 e
    lds r16, v_delta
    lds r17, v_delta+1
    cpi r16, GUARD
    clr r20
    cpc r17, r20
    brsh ct_g1
    rjmp ct_pi             ; Δ < GUARD：作废
ct_g1:
    ; 象限：Δ ≥ perh → 容性
    lds r18, v_perh
    lds r19, v_perh+1
    cp  r16, r18
    cpc r17, r19
    brsh ct_cape
    ; 感性：e = dref − Δ
    lds r18, v_dref
    clr r19
    sub r18, r16
    sbc r19, r17
    clr r16
    sts v_capf, r16
    rjmp ct_e
ct_cape:
    ; 容性：e = dref + (per_filt − Δ)
    lds r18, v_perf
    lds r19, v_perf+1
    sub r18, r16
    sbc r19, r17
    lds r16, v_dref
    clr r17
    add r18, r16
    adc r19, r17
    ldi r16, 1
    sts v_capf, r16
    rjmp ct_e
ct_e:
    ; e 限幅 ±E_LIM（e 在 r19:r18）
    cpi r18, E_LIM
    clr r20
    cpc r19, r20
    brge ct_ehi
    cpi r18, 0xE0          ; -32 = 0xFFE0
    ldi r20, 0xFF
    cpc r19, r20
    brge ct_iir
    ldi r18, 0xE0
    ldi r19, 0xFF
    rjmp ct_iir
ct_ehi:
    ldi r18, E_LIM
    clr r19
ct_iir:
    ; e_filt += (e − e_filt)>>3
    lds r16, v_efilt
    lds r17, v_efilt+1
    sub r18, r16
    sbc r19, r17
    asr r19
    ror r18
    asr r19
    ror r18
    asr r19
    ror r18
    add r16, r18
    adc r17, r19
    sts v_efilt, r16
    sts v_efilt+1, r17
ct_pi:
    ; I 环：top_f −= e_filt
    lds r16, v_topf
    lds r17, v_topf+1
    lds r18, v_efilt
    lds r19, v_efilt+1
    sub r16, r18
    sbc r17, r19
    ; 无符号夹紧 [TOP_MIN·256, TOP_MAX·256]
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
    ; LOCK：|e_filt| ≤ LOCK_BAND
    lds r18, v_efilt
    lds r19, v_efilt+1
    cpi r18, LOCK_BAND+1
    clr r20
    cpc r19, r20
    brge ct_ul
    cpi r18, 0xFE          ; -2
    ldi r20, 0xFF
    cpc r19, r20
    brlt ct_ul
    sbi PORTD, 7
    rjmp ct_dw
ct_ul:
    cbi PORTD, 7
ct_dw:
    ; 容性驻留 → LOSS
    lds r16, v_capf
    tst r16
    breq ct_cd
    lds r16, v_dwell
    inc r16
    sts v_dwell, r16
    cpi r16, DWELL_TRIP
    brsh ct_trip
    rjmp ct_ag
ct_cd:
    clr r16
    sts v_dwell, r16
ct_ag:
    ; I 过零丢失看门狗
    lds r16, v_age
    inc r16
    sts v_age, r16
    cpi r16, AGE_TRIP
    brsh ct_trip
    ; 过流：包络 ADC0 ≥ OC_TH
    lds r16, ADCSRA
    sbrc r16, ADSC
    rjmp ct_ret
    lds r16, ADCH
    cpi r16, OC_TH
    brsh ct_trip
    lds r16, ADCSRA
    ori r16, 1<<ADSC
    sts ADCSRA, r16
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
    pop r20
    pop r19
    pop r18
    ret
"""
