#pragma once
#include <stdint.h>
#include "esp_timer.h"
// Wall-clock-derived placeholder for protocol tests only, not a hardware counter.
inline uint32_t esp_cpu_get_ccount(){return (uint32_t)(esp_timer_get_time()*240ULL);}
inline int xPortGetCoreID(){return 1;}
inline int xTaskGetAffinity(void*){return 1;}
