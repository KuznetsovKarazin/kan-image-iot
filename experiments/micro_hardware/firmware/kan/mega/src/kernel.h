#pragma once
#include <stdint.h>
#ifndef MEGA_INPUTS
#define MEGA_INPUTS 24
#endif
#define MEGA_BATCH 32
#define LOOP_VARIANT 11
int mega_infer(const float *input, float *output);

