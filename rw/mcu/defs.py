# -*- coding: utf-8 -*-
"""ATmega328P 风格的寄存器/位/中断向量名（地址取自公开数据手册的寄存器总表；只列本模拟器实现的部分）。
I/O 空间寄存器给 I/O 地址（IN/OUT/SBI/CBI 用）；扩展 I/O 给数据空间地址（LDS/STS 用），和 m328Pdef.inc 习惯一致。"""
F_CPU_DEFAULT = 16_000_000
IO = dict(PINB=0x03, DDRB=0x04, PORTB=0x05, PINC=0x06, DDRC=0x07, PORTC=0x08, PIND=0x09, DDRD=0x0A, PORTD=0x0B,
          TIFR0=0x15, TIFR2=0x17, EIFR=0x1C, EIMSK=0x1D, GPIOR0=0x1E,
          TCCR0A=0x24, TCCR0B=0x25, TCNT0=0x26, OCR0A=0x27, OCR0B=0x28, GPIOR1=0x2A, GPIOR2=0x2B,
          SMCR=0x33, SPL=0x3D, SPH=0x3E, SREG=0x3F)
MEM = dict(EICRA=0x69, TIMSK0=0x6E, TIMSK2=0x70, ADCL=0x78, ADCH=0x79, ADCSRA=0x7A, ADCSRB=0x7B, ADMUX=0x7C, DIDR0=0x7E,
           TCCR2A=0xB0, TCCR2B=0xB1, TCNT2=0xB2, OCR2A=0xB3, OCR2B=0xB4,
           UCSR0A=0xC0, UCSR0B=0xC1, UCSR0C=0xC2, UBRR0L=0xC4, UBRR0H=0xC5, UDR0=0xC6)
BITS = dict(COM0A1=7, COM0A0=6, COM0B1=5, COM0B0=4, WGM01=1, WGM00=0, FOC0A=7, FOC0B=6, WGM02=3, CS02=2, CS01=1, CS00=0,
            OCIE0B=2, OCIE0A=1, TOIE0=0, OCF0B=2, OCF0A=1, TOV0=0,
            COM2A1=7, COM2A0=6, COM2B1=5, COM2B0=4, WGM21=1, WGM20=0, WGM22=3, CS22=2, CS21=1, CS20=0,
            OCIE2B=2, OCIE2A=1, TOIE2=0, OCF2B=2, OCF2A=1, TOV2=0,
            REFS1=7, REFS0=6, ADLAR=5, MUX3=3, MUX2=2, MUX1=1, MUX0=0,
            ADEN=7, ADSC=6, ADATE=5, ADIF=4, ADIE=3, ADPS2=2, ADPS1=1, ADPS0=0,
            RXC0=7, TXC0=6, UDRE0=5, FE0=4, DOR0=3, UPE0=2, U2X0=1, MPCM0=0,
            RXCIE0=7, TXCIE0=6, UDRIE0=5, RXEN0=4, TXEN0=3, UCSZ02=2,
            UMSEL01=7, UMSEL00=6, UPM01=5, UPM00=4, USBS0=3, UCSZ01=2, UCSZ00=1,
            ISC11=3, ISC10=2, ISC01=1, ISC00=0, INT1=1, INT0=0, INTF1=1, INTF0=0,
            SM2=3, SM1=2, SM0=1, SE=0,
            SREG_C=0, SREG_Z=1, SREG_N=2, SREG_V=3, SREG_S=4, SREG_H=5, SREG_T=6, SREG_I=7)
for p, n in (("B", 8), ("C", 7), ("D", 8)):
    for i in range(n):
        BITS[f"P{p}{i}"] = i; BITS[f"PORT{p}{i}"] = i; BITS[f"DD{p}{i}"] = i; BITS[f"PIN{p}{i}"] = i
VECTORS = dict(RESETaddr=0x00, INT0addr=0x02, INT1addr=0x04, OC2Aaddr=0x0E, OC2Baddr=0x10, OVF2addr=0x12,
               OC0Aaddr=0x1C, OC0Baddr=0x1E, OVF0addr=0x20, URXCaddr=0x24, UDREaddr=0x26, UTXCaddr=0x28, ADCCaddr=0x2A,
               INT_VECTORS_SIZE=0x34)
SYMBOLS = {**IO, **MEM, **BITS, **VECTORS, "RAMEND": 0x08FF, "SRAM_START": 0x0100}
# 引脚 → (端口, 位)；ATmega328P TQFP/DIP 的复用功能
PIN_FUNCS = {"PD0": "RXD", "PD1": "TXD", "PD2": "INT0", "PD3": "INT1/OC2B", "PD5": "OC0B", "PD6": "OC0A",
             "PB3": "OC2A", "PC0": "ADC0", "PC1": "ADC1", "PC2": "ADC2", "PC3": "ADC3", "PC4": "ADC4", "PC5": "ADC5"}
