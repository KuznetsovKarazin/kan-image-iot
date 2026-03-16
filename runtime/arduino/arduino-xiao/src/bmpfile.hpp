#pragma once
#include <Arduino.h>
#include <SD.h>

#include "tensorflow/lite/micro/micro_utils.h"

void saveBMP96(const char *filename, uint8_t *rgb);

void saveBMP224(const char *filename, uint8_t *rgb);

void saveBMP320x240(const char *filename, uint8_t *rgb);

void dump_input_tensor(TfLiteTensor* input, const char *filename, int img_size);

void resize_bilinear_rgb888(const uint8_t* src, int sw, int sh, uint8_t* dst, int dw, int dh);

void resizeBilinearRGB(const uint8_t* src, int sw, int sh, uint8_t* dst, int dw, int dh);
