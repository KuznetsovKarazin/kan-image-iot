#include <avr/io.h>
#include <avr/wdt.h>
#include <stdint.h>
uint8_t mega_reset_flags __attribute__((section(".noinit")));
void early_reset(void) __attribute__((naked,section(".init3")));
void early_reset(void) {mega_reset_flags=MCUSR;MCUSR=0;wdt_disable();}
