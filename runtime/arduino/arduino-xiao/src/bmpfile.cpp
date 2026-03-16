#include <Arduino.h>
#include <SD.h>

#include "tensorflow/lite/micro/micro_utils.h"

#include "bmpfile.hpp"

void saveBMP96(const char *filename, uint8_t *rgb) {
    File f = SD.open(filename, FILE_WRITE);
    if (!f) {
        Serial.println("[ERR] impossible create BMP");
        return;
    }

    const uint32_t width = 96;
    const uint32_t height = 96;
    const uint32_t imageSize = width * height * 3;
    const uint32_t fileSize = 54 + imageSize;

    uint8_t header[54] = {
        0x42, 0x4D,                         // Signature "BM"
        (uint8_t)(fileSize),                // File size
        (uint8_t)(fileSize >> 8),
        (uint8_t)(fileSize >> 16),
        (uint8_t)(fileSize >> 24),
        0, 0, 0, 0,                          // Reserved
        54, 0, 0, 0,                         // Offset to pixel data
        40, 0, 0, 0,                         // DIB header size
        (uint8_t)(width),
        (uint8_t)(width >> 8),
        (uint8_t)(width >> 16),
        (uint8_t)(width >> 24),
        (uint8_t)(height),
        (uint8_t)(height >> 8),
        (uint8_t)(height >> 16),
        (uint8_t)(height >> 24),
        1, 0,                                // Planes
        24, 0,                               // Bits per pixel (RGB888)
        0, 0, 0, 0,                          // Compression (none)
        (uint8_t)(imageSize),
        (uint8_t)(imageSize >> 8),
        (uint8_t)(imageSize >> 16),
        (uint8_t)(imageSize >> 24),
        0, 0, 0, 0,                          // X pixels per meter
        0, 0, 0, 0,                          // Y pixels per meter
        0, 0, 0, 0,                          // Colors in palette
        0, 0, 0, 0                           // Important colors
    };

    f.write(header, 54);

    // BMP reverse rows (bottom-up)
    for (int y = height - 1; y >= 0; y--) {
        for (int x = 0; x < width; x++) {
            int idx = (y * width + x) * 3;
            f.write(rgb[idx + 2]); // B
            f.write(rgb[idx + 1]); // G
            f.write(rgb[idx + 0]); // R
        }
    }

    f.close();
    Serial.printf("[OK] Salvato BMP: %s\n", filename);
}

void saveBMP224(const char *filename, uint8_t *rgb) {
    File f = SD.open(filename, FILE_WRITE);
    if (!f) {
        Serial.println("[ERR] impossible create BMP");
        return;
    }

    const uint32_t width = 224;
    const uint32_t height = 224;
    const uint32_t imageSize = width * height * 3;
    const uint32_t fileSize = 54 + imageSize;

    uint8_t header[54] = {
        0x42, 0x4D,                         // Signature "BM"
        (uint8_t)(fileSize),                // File size
        (uint8_t)(fileSize >> 8),
        (uint8_t)(fileSize >> 16),
        (uint8_t)(fileSize >> 24),
        0, 0, 0, 0,                          // Reserved
        54, 0, 0, 0,                         // Offset to pixel data
        40, 0, 0, 0,                         // DIB header size
        (uint8_t)(width),
        (uint8_t)(width >> 8),
        (uint8_t)(width >> 16),
        (uint8_t)(width >> 24),
        (uint8_t)(height),
        (uint8_t)(height >> 8),
        (uint8_t)(height >> 16),
        (uint8_t)(height >> 24),
        1, 0,                                // Planes
        24, 0,                               // Bits per pixel (RGB888)
        0, 0, 0, 0,                          // Compression (none)
        (uint8_t)(imageSize),
        (uint8_t)(imageSize >> 8),
        (uint8_t)(imageSize >> 16),
        (uint8_t)(imageSize >> 24),
        0, 0, 0, 0,                          // X pixels per meter
        0, 0, 0, 0,                          // Y pixels per meter
        0, 0, 0, 0,                          // Colors in palette
        0, 0, 0, 0                           // Important colors
    };

    f.write(header, 54);

    // BMP reverse rows (bottom-up)
    for (int y = height - 1; y >= 0; y--) {
        for (int x = 0; x < width; x++) {
            int idx = (y * width + x) * 3;
            f.write(rgb[idx + 2]); // B
            f.write(rgb[idx + 1]); // G
            f.write(rgb[idx + 0]); // R
        }
    }

    f.close();
    Serial.printf("[OK] BMP saved: %s\n", filename);
}

void saveBMP320x240(const char *filename, uint8_t *rgb) {
    File f = SD.open(filename, FILE_WRITE);
    if (!f) {
        Serial.println("[ERR] impossible create BMP");
        return;
    }

    const uint32_t width = 320;
    const uint32_t height = 240;
    const uint32_t imageSize = width * height * 3;
    const uint32_t fileSize = 54 + imageSize;

    uint8_t header[54] = {
        0x42, 0x4D,                         // Signature "BM"
        (uint8_t)(fileSize),                // File size
        (uint8_t)(fileSize >> 8),
        (uint8_t)(fileSize >> 16),
        (uint8_t)(fileSize >> 24),
        0, 0, 0, 0,                          // Reserved
        54, 0, 0, 0,                         // Offset to pixel data
        40, 0, 0, 0,                         // DIB header size
        (uint8_t)(width),
        (uint8_t)(width >> 8),
        (uint8_t)(width >> 16),
        (uint8_t)(width >> 24),
        (uint8_t)(height),
        (uint8_t)(height >> 8),
        (uint8_t)(height >> 16),
        (uint8_t)(height >> 24),
        1, 0,                                // Planes
        24, 0,                               // Bits per pixel (RGB888)
        0, 0, 0, 0,                          // Compression (none)
        (uint8_t)(imageSize),
        (uint8_t)(imageSize >> 8),
        (uint8_t)(imageSize >> 16),
        (uint8_t)(imageSize >> 24),
        0, 0, 0, 0,                          // X pixels per meter
        0, 0, 0, 0,                          // Y pixels per meter
        0, 0, 0, 0,                          // Colors in palette
        0, 0, 0, 0                           // Important colors
    };

    f.write(header, 54);

    // BMP salva le righe al contrario (bottom-up)
    for (int y = height - 1; y >= 0; y--) {
        for (int x = 0; x < width; x++) {
            int idx = (y * width + x) * 3;
            f.write(rgb[idx + 2]); // B
            f.write(rgb[idx + 1]); // G
            f.write(rgb[idx + 0]); // R
        }
    }

    f.close();
    Serial.printf("[OK] BMP saved: %s\n", filename);
}

void dump_input_tensor(TfLiteTensor* input, const char *filename, int img_size) {
  float scale = input->params.scale;
  int zp = input->params.zero_point;

  uint8_t* debug = (uint8_t*)malloc(img_size * img_size * 3);

  for (int i = 0; i < img_size * img_size * 3; i++) {
      float f = (input->data.int8[i] - zp) * scale;  // back tofloat 0–1
      debug[i] = (uint8_t)(f * 255.0f);
  }

  saveBMP96(filename, debug);
  free(debug);
}

void resize_bilinear_rgb888(
    const uint8_t* src, int sw, int sh,
    uint8_t* dst, int dw, int dh
) {
    float x_ratio = (float)(sw - 1) / dw;
    float y_ratio = (float)(sh - 1) / dh;

    for (int j = 0; j < dh; j++) {
        float sy = j * y_ratio;
        int y = (int)sy;
        float y_diff = sy - y;

        for (int i = 0; i < dw; i++) {
            float sx = i * x_ratio;
            int x = (int)sx;
            float x_diff = sx - x;

            int idx = (j * dw + i) * 3;

            int idx00 = (y * sw + x) * 3;
            int idx01 = (y * sw + (x + 1)) * 3;
            int idx10 = ((y + 1) * sw + x) * 3;
            int idx11 = ((y + 1) * sw + (x + 1)) * 3;

            for (int c = 0; c < 3; c++) {
                float p00 = src[idx00 + c];
                float p01 = src[idx01 + c];
                float p10 = src[idx10 + c];
                float p11 = src[idx11 + c];

                float px =
                    p00 * (1 - x_diff) * (1 - y_diff) +
                    p01 * (x_diff)     * (1 - y_diff) +
                    p10 * (1 - x_diff) * (y_diff) +
                    p11 * (x_diff)     * (y_diff);

                dst[idx + c] = (uint8_t)px;
            }
        }
    }
}

void resizeBilinearRGB(
    const uint8_t* src, int sw, int sh,
    uint8_t* dst, int dw, int dh)
{
    float x_ratio = (float)(sw - 1) / dw;
    float y_ratio = (float)(sh - 1) / dh;

    for (int y = 0; y < dh; y++) {
        float sy = y * y_ratio;
        int y0 = (int)sy;
        float dy = sy - y0;

        for (int x = 0; x < dw; x++) {
            float sx = x * x_ratio;
            int x0 = (int)sx;
            float dx = sx - x0;

            int idx = (y * dw + x) * 3;

            for (int c = 0; c < 3; c++) {
                float c00 = src[(y0 * sw + x0) * 3 + c];
                float c01 = src[(y0 * sw + (x0 + 1)) * 3 + c];
                float c10 = src[((y0 + 1) * sw + x0) * 3 + c];
                float c11 = src[((y0 + 1) * sw + (x0 + 1)) * 3 + c];

                float c0 = c00 + dx * (c01 - c00);
                float c1 = c10 + dx * (c11 - c10);
                float c_final = c0 + dy * (c1 - c0);

                dst[idx + c] = (uint8_t)c_final;
            }
        }
    }
}

