#pragma once
#include <Arduino.h>
#include <TJpg_Decoder.h>

bool loadJPGfromSD(const char* filename, uint8_t** jpgRGB, int label);
bool decodeJPGfromRAM(uint8_t* src_buf, size_t src_len, uint8_t* target_rgb);
