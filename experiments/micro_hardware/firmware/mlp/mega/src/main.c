/* ATmega2560, UART0 115200, Timer1 2 MHz; no heap or external peripherals. */
#include <avr/io.h>
#include <avr/interrupt.h>
#include <avr/pgmspace.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "kernel.h"

static float batch[MEGA_BATCH][MEGA_INPUTS];
static uint32_t loaded;
static volatile uint32_t sink_bits;
static volatile uint16_t timer_wraps;
static char line[512];
extern uint8_t __heap_start;
static uint8_t *paint_end;
extern uint8_t mega_reset_flags;

ISR(TIMER1_OVF_vect) {++timer_wraps;}
static uint32_t clock_ticks(void) {
    const uint8_t saved=SREG;cli();
    uint16_t hi=timer_wraps,lo=TCNT1;
    if ((TIFR1&_BV(TOV1)) && lo<0x8000U) ++hi;
    SREG=saved;return ((uint32_t)hi<<16)|lo;
}
static int uart_put(char c,FILE *f) {
    (void)f;while (!(UCSR0A&_BV(UDRE0))) {}
    UDR0=(uint8_t)c;return 0;
}
static int uart_get(void) {
    while (!(UCSR0A&_BV(RXC0))) {}
    const uint8_t state=UCSR0A,data=UDR0;
    return state&(_BV(FE0)|_BV(DOR0)|_BV(UPE0)) ? -1 : data;
}
static FILE uart_stream=FDEV_SETUP_STREAM(uart_put,NULL,_FDEV_SETUP_WRITE);
static uint32_t bits(float x) {uint32_t u;memcpy(&u,&x,4);return u;}
static uint16_t untouched_ram(void) {
    uint8_t *p=&__heap_start;
    while(p<paint_end && *p==0xa5) ++p;
    return (uint16_t)(p-&__heap_start);
}
static void error(const char *why) {printf_P(PSTR("{\"kind\":\"error\",\"reason\":\"%S\"}\n"),why);}
#define ERROR(s) error(PSTR(s))
static uint8_t number(char *s,uint32_t *out,uint32_t max) {
    if (!s || !*s) return 0;
    uint32_t n=0;
    for (;*s;++s) {
        if (*s<'0'||*s>'9') return 0;
        const uint8_t d=(uint8_t)(*s-'0');
        if (n>max/10 || (n==max/10 && d>max%10)) return 0;
        n=n*10+d;
    }
    *out=n;return 1;
}
static int8_t hex_digit(char c) {
    if (c>='0'&&c<='9') return c-'0';
    if (c>='a'&&c<='f') return c-'a'+10;
    if (c>='A'&&c<='F') return c-'A'+10;
    return -1;
}
static uint8_t floats(char **state,float *x) {
    for (uint8_t i=0;i<MEGA_INPUTS;++i) {
        char *s=strtok_r(NULL," \t",state);
        if (!s || strlen(s)!=8) return 0;
        uint32_t u=0;
        for (uint8_t k=0;k<8;++k) {int8_t d=hex_digit(s[k]);if (d<0)return 0;u=(u<<4)|(uint8_t)d;}
        memcpy(&x[i],&u,4);if (!isfinite(x[i])) return 0;
    }
    return strtok_r(NULL," \t",state)==NULL;
}
static void hello(void) {
    printf_P(PSTR("{\"kind\":\"hello\",\"protocol\":2,\"chip\":\"ATmega2560\",\"cpu_mhz\":16,"
        "\"sram_bytes\":8192,\"flash_bytes\":262144,\"input_dim\":%u,\"variant\":%u,"
        "\"model\":\"%S\",\"variant_name\":\"%S\",\"artifact\":\"%S\",\"source_artifact\":\"%S\","
        "\"parameter_bytes\":%lu,\"timer_hz\":2000000,\"serial_transport\":\"avr-uart0-v1\",\"energy_protocol\":1,"
        "\"untouched_sram_gap_bytes\":%u,\"reset_flags\":%u}\n"),
        MEGA_INPUTS,WIRE_VARIANT,PSTR(MODEL_ID),PSTR(VARIANT_NAME),PSTR(MODEL_ARTIFACT),PSTR(SOURCE_ARTIFACT),
        (unsigned long)PARAMETER_BYTES,untouched_ram(),mega_reset_flags);
}
static void benchmark(uint32_t seq,uint8_t variant,uint8_t rounds) {
    if (loaded!=UINT32_MAX) {ERROR("batch_incomplete");return;}
    float y[2];uint32_t count=0;uint8_t complete=1;
    const uint32_t start=clock_ticks();
    for (uint8_t r=0;r<rounds;++r) {
        for (uint8_t i=0;i<MEGA_BATCH;++i) {
            if (variant==LOOP_VARIANT) {y[0]=batch[i][0];y[1]=batch[i][1];}
            else if (mega_infer(batch[i],y)) {complete=0;break;}
            sink_bits=bits(y[0]);++count;
        }
        if (!complete || clock_ticks()-start>240000000UL) {complete=0;break;}
    }
    const uint32_t elapsed=clock_ticks()-start;
    printf_P(PSTR("{\"kind\":\"bench\",\"seq\":%lu,\"variant\":%u,\"rounds\":%u,\"n\":%lu,"
        "\"elapsed_ticks\":%lu,\"timer_hz\":2000000,\"complete\":%S,\"sink\":\"%08lx\","
        "\"untouched_sram_gap_bytes\":%u}\n"),(unsigned long)seq,variant,rounds,(unsigned long)count,
        (unsigned long)elapsed,complete?PSTR("true"):PSTR("false"),(unsigned long)sink_bits,untouched_ram());
}
static void energy(uint32_t seq,uint8_t variant,uint32_t duration_ms) {
    if (loaded!=UINT32_MAX) {ERROR("batch_incomplete");return;}
    /* Clear TX-complete before writing; wait for the entire marker to leave UART. */
    UCSR0A|=_BV(TXC0);
    printf_P(PSTR("{\"kind\":\"energy_armed\",\"seq\":%lu,\"variant\":%u,"
        "\"energy_protocol\":1,\"lead_ms\":2000,\"duration_ms\":%lu}\n"),
        (unsigned long)seq,variant,(unsigned long)duration_ms);
    while (!(UCSR0A&_BV(TXC0))) {}
    const uint32_t lead=clock_ticks();
    while (clock_ticks()-lead<4000000UL) {}
    const uint32_t start=clock_ticks(),budget=duration_ms*2000UL;
    uint32_t count=0;float y[2];
    while (clock_ticks()-start<budget) {
        /* Baseline is the powered board polling the timer, not a sleep state. */
        if (variant==LOOP_VARIANT) continue;
        for (uint8_t i=0;i<MEGA_BATCH;++i) {
            if (mega_infer(batch[i],y)) {ERROR("inference_failed");return;}
            sink_bits=bits(y[0]);++count;
        }
    }
    const uint32_t elapsed=clock_ticks()-start;
    printf_P(PSTR("{\"kind\":\"energy_end\",\"seq\":%lu,\"variant\":%u,\"n\":%lu,"
        "\"energy_protocol\":1,\"elapsed_ticks\":%lu,\"timer_hz\":2000000,"
        "\"sink\":\"%08lx\",\"untouched_sram_gap_bytes\":%u,"
        "\"scope\":\"whole_board_polling_loop\"}\n"),
        (unsigned long)seq,variant,(unsigned long)count,(unsigned long)elapsed,
        (unsigned long)sink_bits,untouched_ram());
}
static void command(char *s) {
    if (untouched_ram()<128) {ERROR("sram_guard_failed");return;}
    char *state=NULL;char *cmd=strtok_r(s," \t",&state);
    if (!cmd) return;
    if (!strcmp(cmd,"H")) {if (strtok_r(NULL," \t",&state))ERROR("extra_token");else hello();return;}
    uint32_t seq=0,v=0,n=0;float x[MEGA_INPUTS];
    if (!strcmp(cmd,"P")) {
        if (!number(strtok_r(NULL," \t",&state),&seq,0x7fffffffUL)) {ERROR("invalid_probe");return;}
        char *payload=strtok_r(NULL," \t",&state);
        if (!payload||!*payload||strlen(payload)>384||strtok_r(NULL," \t",&state)) {ERROR("invalid_probe");return;}
        for (const char *p=payload;*p;++p) if (hex_digit(*p)<0) {ERROR("invalid_probe");return;}
        printf_P(PSTR("{\"kind\":\"probe\",\"seq\":%lu,\"payload\":\"%s\"}\n"),(unsigned long)seq,payload);
    } else if (!strcmp(cmd,"V")) {
        if (!number(strtok_r(NULL," \t",&state),&seq,0x7fffffffUL)||
            !number(strtok_r(NULL," \t",&state),&v,10)||v!=WIRE_VARIANT||!floats(&state,x)) {ERROR("invalid_vector");return;}
        float y[2];if (mega_infer(x,y)) {ERROR("inference_failed");return;}
        printf_P(PSTR("{\"kind\":\"pred\",\"seq\":%lu,\"variant\":%u,\"bits\":[\"%08lx\",\"%08lx\"]}\n"),
                 (unsigned long)seq,(unsigned)v,(unsigned long)bits(y[0]),(unsigned long)bits(y[1]));
    } else if (!strcmp(cmd,"L")) {
        if (!number(strtok_r(NULL," \t",&state),&n,31)||!floats(&state,x)) {ERROR("invalid_batch_vector");return;}
        memcpy(batch[n],x,sizeof(x));loaded|=1UL<<n;
        printf_P(PSTR("{\"kind\":\"loaded\",\"slot\":%u}\n"),(unsigned)n);
    } else if (!strcmp(cmd,"B")) {
        if (!number(strtok_r(NULL," \t",&state),&seq,0x7fffffffUL)||
            !number(strtok_r(NULL," \t",&state),&v,11)||(v!=WIRE_VARIANT&&v!=LOOP_VARIANT)||
            !number(strtok_r(NULL," \t",&state),&n,32)||!n||strtok_r(NULL," \t",&state)) {ERROR("invalid_benchmark");return;}
        benchmark(seq,(uint8_t)v,(uint8_t)n);
    } else if (!strcmp(cmd,"E")) {
        if (!number(strtok_r(NULL," \t",&state),&seq,0x7fffffffUL)||
            !number(strtok_r(NULL," \t",&state),&v,2)||(v!=WIRE_VARIANT&&v!=LOOP_VARIANT)||
            !number(strtok_r(NULL," \t",&state),&n,60000)||n<1000||strtok_r(NULL," \t",&state)) {ERROR("invalid_energy_window");return;}
        energy(seq,(uint8_t)v,n);
    } else ERROR("unknown_command");
}
int main(void) {
    UCSR0A=_BV(U2X0);UBRR0H=0;UBRR0L=16;
    UCSR0B=_BV(RXEN0)|_BV(TXEN0);UCSR0C=_BV(UCSZ01)|_BV(UCSZ00);stdout=&uart_stream;
    paint_end=(uint8_t *)(SP-64);
    for (uint8_t *p=&__heap_start;p<paint_end;++p) *p=0xa5;
    TCCR1A=0;TCCR1B=_BV(CS11);TCNT1=0;TIFR1=_BV(TOV1);TIMSK1=_BV(TOIE1);sei();
    uint16_t used=0;uint8_t bad=0;
    for (;;) {
        const int c=uart_get();
        if (c<0) {bad=1;continue;}
        if (c=='\r') continue;
        if (c=='\n') {
            if (bad) ERROR("serial_input_error");
            else {line[used]=0;command(line);}
            used=0;bad=0;
        } else if (!bad && used<sizeof(line)-1) line[used++]=(char)c;
        else bad=1;
    }
}

