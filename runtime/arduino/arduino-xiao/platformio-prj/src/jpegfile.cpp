#include <Arduino.h>
#include <TJpg_Decoder.h>
#include "jpegfile.hpp"

static File jpgFile;

int jpgW = 0;
int jpgH = 0;

uint8_t* jpgRefRGB = nullptr;

uint8_t* bufferJPG = nullptr;
size_t bufferJPGLen = 0;
uint32_t bufferPos = 0;

size_t jpgRead(JDEC* jd, uint8_t* buf, size_t len) {
    if (buf) {
        // Reads 'len' bytes from the file
        return jpgFile.read(buf, len);
    } else {
        // Skip 'len' bytes
        jpgFile.seek(jpgFile.position() + len);
        return len;
    }
}

int jpgOut(JDEC* jd, void* bitmap, JRECT* rect) {
    uint16_t* src = (uint16_t*)bitmap;

    for (int y = rect->top; y <= rect->bottom; y++) {
        for (int x = rect->left; x <= rect->right; x++) {

            uint16_t c = *src++;

            c = (c >> 8) | (c << 8);  // swap bytes for endianess correction if needed

            // correct conversion RGB565 → RGB888
            uint8_t r = ((c >> 11) & 0x1F) * 255 / 31;
            uint8_t g = ((c >> 5)  & 0x3F) * 255 / 63;
            uint8_t b = ( c        & 0x1F) * 255 / 31;

            int idx = (y * jpgW + x) * 3;

            // Now colors are in RGB888 format and can be stored in the output buffer
            jpgRefRGB[idx + 0] = r;
            jpgRefRGB[idx + 1] = g;
            jpgRefRGB[idx + 2] = b;
        }
    }
    return 1;
}


bool loadJPGfromSD(const char* filename, uint8_t** jpgRGB, int label) {

    char fullpath[128];
    sprintf(fullpath, "/val/%s/%s", (label == 1 ? "person" : "no_person"), filename);
    Serial.printf("[INFO] Complete Path: %s\n", fullpath);

    jpgFile = SD.open(fullpath);
    if (!jpgFile) {
        Serial.println("[ERR] impossible open JPG");
        return false;
    }

    JDEC decoder;
    uint8_t work[4096];   // working buffer for TJpg_Decoder
    Serial.println("[INFO] File opened\n");

    // Prepare the decoder
    if (jd_prepare(&decoder, jpgRead, work, sizeof(work), nullptr) != JDR_OK) {
        Serial.println("[ERR] jd_prepare failed");
        jpgFile.close();
        return false;
    }

    jpgW = decoder.width;
    jpgH = decoder.height;

    Serial.printf("[INFO] JPG size: %d x %d\n", jpgW, jpgH);

    // Allocate RGB888 buffer
    *jpgRGB = (uint8_t*)malloc(jpgW * jpgH * 3);
    if (!jpgRGB) {
        Serial.println("[ERR] malloc failed");
        jpgFile.close();
        return false;
    }

    jpgRefRGB = *jpgRGB; // Setp the global pointer for the output function

    // Decoding of JPEG 
    if (jd_decomp(&decoder, jpgOut, 0) != JDR_OK) {
        Serial.println("[ERR] jd_decomp failed");
        free(jpgRGB);
        jpgFile.close();
        return false;
    }

    jpgFile.close();
    return true;
}

// Read function for the decoder (replaces the one that read from SD)
size_t jpgReadRAM(JDEC* decoder, uint8_t* buf, size_t len) {

    int32_t effectiveLen = (bufferPos + len > bufferJPGLen) ? (bufferJPGLen - bufferPos) : len;
    
    //Serial.printf("[INFO] jpgReadRAM: pos=%u, len=%u, effectiveLen=%u\n", bufferPos, len, effectiveLen);

    if (buf) {
        // Reads 'len' bytes from the file
        memcpy(buf, bufferJPG + bufferPos, effectiveLen);
        bufferPos += effectiveLen;
        return effectiveLen;
    } else {
        // Skip 'len' bytes
        bufferPos += effectiveLen;
        return effectiveLen;
    }
}

bool decodeJPGfromRAM(uint8_t* src_buf, size_t src_len, uint8_t* target_rgb) {
    JDEC decoder;
    uint8_t work[4096]; 
    
    // Setup buffer soruce
    bufferJPG = src_buf;
    bufferJPGLen = src_len;
    bufferPos = 0;

    // Setup destination buffer
    jpgRefRGB = target_rgb; 

    // Prepare the decoder
    if (jd_prepare(&decoder, jpgReadRAM, work, sizeof(work), nullptr) != JDR_OK) {
        Serial.println("[ERR] jd_prepare RAM failed");
        return false;
    }

    jpgW = decoder.width;
    jpgH = decoder.height;

    Serial.printf("[INFO] Decoding in RAM: %d x %d\n", decoder.width, decoder.height);

    // Start decoding
    if (jd_decomp(&decoder, jpgOut, 0) != JDR_OK) {
        Serial.println("[ERR] jd_decomp RAM failed");
        return false;
    }

    return true;
}


