#pragma once
/* Host protocol harness only; these stubs do not emulate an ESP32. */
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstdarg>
#include <cctype>
#include <chrono>
#include <thread>
#include <poll.h>
#include <unistd.h>
struct SerialStub {
    void begin(unsigned) {}
    int available() {pollfd p{0,POLLIN,0};return poll(&p,1,0)>0 && (p.revents&POLLIN);}
    int read() {unsigned char c;return ::read(0,&c,1)==1 ? c : -1;}
    void flush() {fflush(stdout);}
    int printf(const char *f,...) {va_list ap;va_start(ap,f);int n=vprintf(f,ap);va_end(ap);fflush(stdout);return n;}
};
extern SerialStub Serial;
struct EspStub {
    const char *getChipModel(){return "HOST-MOCK";}
    unsigned getChipRevision(){return 0;}
    unsigned getChipCores(){return 0;}
    unsigned getCpuFreqMHz(){return 0;}
    unsigned getFlashChipSize(){return 0;}
    unsigned getPsramSize(){return 0;}
    unsigned getFreeHeap(){return 0;}
    unsigned getMinFreeHeap(){return 0;}
    const char *getSdkVersion(){return "host-protocol-stub";}
};
extern EspStub ESP;
inline void delay(unsigned ms){std::this_thread::sleep_for(std::chrono::milliseconds(ms));}
inline unsigned uxTaskGetStackHighWaterMark(void*){return 0;}
