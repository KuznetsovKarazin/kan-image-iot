#include <Arduino.h>

#include <Chirale_TensorFlowLite.h>

#include "esp_camera.h"
#include "model_data.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"

#include <SD.h>
#include <SPI.h>
#include "esp_heap_caps.h"

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_utils.h"
#include "tensorflow/lite/c/common.h"

#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"

// For accuracy calculation on device
#include <TJpg_Decoder.h>

#include "jpegfile.hpp"
#include "fileops.hpp"
#include "statistics.hpp"
#include "bmpfile.hpp"

// For accuracy calculation on device with validation dataset
#define PERFORMANCE_TESTING 1

// Inference image size (must be the same as the one used during training and conversion to tflite)
#define IMG_SIZE 96

// -------------------------
// SD CONFIGURATION (SPI)
// -------------------------
#define SD_CS 21   // On XIAO CS is soldered in this way

// Dump the first choosen samples to SD as BMP for debugging
#define DEBUG_SAMPLES 20

// -------------------------
// CAMERA CONFIGURATION
// -------------------------
#define CAM_PIN_PWDN    -1
#define CAM_PIN_RESET   -1
#define CAM_PIN_XCLK    10
#define CAM_PIN_SIOD    40
#define CAM_PIN_SIOC    39

#define CAM_PIN_D7      48
#define CAM_PIN_D6      11
#define CAM_PIN_D5      12
#define CAM_PIN_D4      14
#define CAM_PIN_D3      16
#define CAM_PIN_D2      18
#define CAM_PIN_D1      17
#define CAM_PIN_D0      15

#define CAM_PIN_VSYNC   38
#define CAM_PIN_HREF    47
#define CAM_PIN_PCLK    13

// -------------------------
// JPEG BUFFER
//  ------------------------
uint8_t* jpgRGB = nullptr;

// -------------------------
// TENSOR ARENA
// -------------------------
constexpr int kArenaSize = 303092; // Adapted for KAN 16 4 WM 0.334 lut int8 enforced
uint8_t* tensor_arena = nullptr;

const tflite::Model* model;
tflite::MicroInterpreter* interpreter;
TfLiteTensor* input;
TfLiteTensor* output;

// -------------------------
// CIRCULAR BUFFER SD
// -------------------------
String filenames[10];
int frameIndex = 0;

void saveFrameToSD(uint8_t* buffer, size_t len) {
  String name = "/frame_" + String(frameIndex) + ".jpg";
  File f = SD.open(name, FILE_WRITE);
  if (f) {
    f.write(buffer, len);
    f.close();
    filenames[frameIndex] = name;
    frameIndex = (frameIndex + 1) % 10;
  }
}

// -------------------------
// MEMORY LOGGING
// -------------------------
void logMemory(const char* tag) {
  Serial.printf("\n[%s]\n", tag);
  Serial.printf("  Free DRAM:   %u bytes", heap_caps_get_free_size(MALLOC_CAP_INTERNAL));
  Serial.printf("  Free PSRAM:  %u bytes ", heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
  Serial.printf("  Largest DRAM block:  %u bytes ", heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL));
  Serial.printf("  Largest PSRAM block: %u bytes\n", heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM));
}

// -------------------------
// CAMERA SETUP
// -------------------------
bool initCamera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  config.pin_d0       = CAM_PIN_D0;
  config.pin_d1       = CAM_PIN_D1;
  config.pin_d2       = CAM_PIN_D2;
  config.pin_d3       = CAM_PIN_D3;
  config.pin_d4       = CAM_PIN_D4;
  config.pin_d5       = CAM_PIN_D5;
  config.pin_d6       = CAM_PIN_D6;
  config.pin_d7       = CAM_PIN_D7;
  config.pin_xclk     = CAM_PIN_XCLK;
  config.pin_pclk     = CAM_PIN_PCLK;
  config.pin_vsync    = CAM_PIN_VSYNC;
  config.pin_href     = CAM_PIN_HREF;
  config.pin_sccb_sda = CAM_PIN_SIOD;
  config.pin_sccb_scl = CAM_PIN_SIOC;
  config.pin_pwdn     = CAM_PIN_PWDN;
  config.pin_reset    = CAM_PIN_RESET;
  config.xclk_freq_hz = 20000000;

  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size   = FRAMESIZE_QVGA; // 320x240
  config.jpeg_quality = 10;
  config.fb_count     = 1;
  config.fb_location  = CAMERA_FB_IN_PSRAM; //CAMERA_FB_IN_DRAM;
  config.grab_mode    = CAMERA_GRAB_WHEN_EMPTY;

  return esp_camera_init(&config) == ESP_OK;
}

// -------------------------
// TFLITE SETUP
// -------------------------
void initTFLite() {

  Serial.println("Tensor_arena allocation...");
  
  // External memory
  //tensor_arena = (uint8_t*) heap_caps_malloc(kArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  // Internal memory (faster but more limited)
  tensor_arena = (uint8_t*) heap_caps_malloc(kArenaSize, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  // Internal memory dinamically aligned (allineata)
  //tensor_arena = (uint8_t*) heap_caps_aligned_alloc(16, kArenaSize, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  // Internal memory statically allocated (not viable with large arena)
  //alignas(16) static uint8_t tensor_arena_internal[kArenaSize];
  //tensor_arena = tensor_arena_internal;

  if (!tensor_arena) {
    Serial.println("ERROR: tensor_arena is NULL");
    while (1);
  }
  logMemory("After tensor_arena allocation");

  model = tflite::GetModel(model_tflite);
  if (!model) {
    Serial.println("ERROR: model is NULL");
    while (1);
  }
  Serial.printf("Model version: %d, TFLITE_SCHEMA_VERSION: %d\n", model->version(), TFLITE_SCHEMA_VERSION);

  static tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;

  // operator space: more operators can be added, but this will increase the binary size and memory usage
  constexpr int kOpResolverSize = 40;
  static tflite::MicroMutableOpResolver<kOpResolverSize> resolver;

  // Add operators used in the model. For kan-image-iot classification model, you might need operators like
  // 'Add', 'Conv2D', 'DepthwiseConv2D', 'FullyConnected', 'Mul', 'Reshape', 'Pad', 'Transpose', 'Mean', 'Sub', 'Minimum', 'Sum', 'Abs'
  resolver.AddConv2D();
  resolver.AddAbs();
  resolver.AddDepthwiseConv2D();
  resolver.AddFullyConnected();
  resolver.AddMinimum();
  resolver.AddReshape();
  resolver.AddPad();
  resolver.AddMul();
  resolver.AddSub();
  resolver.AddAdd();
  resolver.AddMean();
  resolver.AddTranspose();
  resolver.AddSum();


  Serial.println("MicroInterpreter building (MutableResolver)...");
  interpreter = new tflite::MicroInterpreter(model, resolver, tensor_arena, kArenaSize);
  if (!interpreter) {
    Serial.println("ERROR: interpreter is NULL after new");
    while (1);
  }

  Serial.println("AllocateTensors...");
  TfLiteStatus alloc_status = interpreter->AllocateTensors();
  if (alloc_status != kTfLiteOk) {
    Serial.printf("ERROR: AllocateTensors returned %d\n", alloc_status);
    while (1);
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.printf("Input tensor: type=%d, bytes=%d\n", input->type, input->bytes);
  for (int i = 0; i < input->dims->size; i++) {
    Serial.printf("  dim[%d] = %d\n", i, input->dims->data[i]);
  }

  Serial.println("initTFLite complete");
  logMemory("After initTFLite");
}


// -------------------------
// SETUP
// -------------------------

#define BOOT_DELAY 15
int cnt = 0;

void setup() {
  Serial.begin(115200);
  delay(2000);

  Serial.println("\n=== BOOT ===");
  logMemory("Boot");

  while(cnt++ < BOOT_DELAY) {
    Serial.printf("Boot delay... %d\n", BOOT_DELAY - cnt);
    delay(1000);
  }

  Serial.println("\nTensorFlow Lite initialization...");
  initTFLite();
  Serial.println("TFLite OK");
  logMemory("Dopo initTFLite");

  Serial.println("\nCamera initialization...");
  if (!initCamera()) {
    Serial.println("ERROR: initCamera()");
    while (1);
  }
  Serial.println("Camera OK");
  logMemory("After initCamera");

  Serial.println("\nInitializing SD...");
  if (!SD.begin(SD_CS)) {
    Serial.println("ERROR: SD.begin()");
    while (1);
  }
  Serial.println("SD OK");
  logMemory("After SD");

#ifdef PERFORMANCE_TESTING
  initDirectories();
#endif

  Serial.println("\n=== System ready ===");

}

// Loop variables
int counter = 0;
int skipped = 0;

static char debugName[64];
static char filename[64];
uint8_t* resized = nullptr;
uint8_t* flipped = nullptr;

uint8_t* jpgBuffer = nullptr;
size_t jpgHeight = 0;
size_t jpgWidth = 0;
size_t jpgLen = 0;

// -------------------------
// MAIN LOOP
// -------------------------
void loop() {
  logMemory("Loop start");

#ifdef PERFORMANCE_TESTING



    int label = -1;
    if (getNextImage(filename, label)) {
        Serial.printf("\n[INFO] Evaluating %s (label: %d)\n", filename, label);

        if (loadJPGfromSD(filename, &jpgRGB, label)) {
            //static uint8_t resized[96 * 96 * 3];
            //if(counter < DEBUG_SAMPLES) {
            //    sprintf(debugName, "/debug_orig_%d.bmp", counter);
            //    saveBMP224(debugName, jpgRGB);
            //}
            if(counter < DEBUG_SAMPLES) {
                sprintf(debugName, "/debug_full_%d.bmp", counter);            
                saveBMP224(debugName, jpgRGB);
            }
            resized = (uint8_t*) heap_caps_malloc(IMG_SIZE * IMG_SIZE * 3, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
            flipped = (uint8_t*) heap_caps_malloc(IMG_SIZE * IMG_SIZE * 3, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
            if (!resized) {
                Serial.println("ERROR: malloc resized");
                free(jpgRGB);
                return;
            }
            Serial.println("malloc resized OK");
            logMemory("After malloc resized");

            Serial.println("Bilinear resize...");
            resizeBilinearRGB(jpgRGB, 224, 224, flipped, IMG_SIZE, IMG_SIZE);
            Serial.println("Resize OK");

            free(jpgRGB);

            Serial.println("[OK] Bilinear resize completed");

            if(counter < DEBUG_SAMPLES) {
              sprintf(debugName, "/debug_%d.bmp", counter++);
              saveBMP96(debugName, resized);
            }

            // Direct 
            memcpy(resized, flipped, IMG_SIZE * IMG_SIZE * 3);

            // Flipped vertically
            //for (int y = 0; y < IMG_SIZE; y++) {
            //    int src_row = y * IMG_SIZE * 3;
            //    int dst_row = (IMG_SIZE - 1 - y) * IMG_SIZE * 3;
            //    memcpy(&resized[dst_row], &flipped[src_row], IMG_SIZE * 3);
            //}

            free(flipped);

            Serial.println("[OK] Flipping complete");
        } else {
            Serial.println("ERROR: loadJPGfromSD");
            skipped++;
            return;
        }
    }
#else 

  // Try to get a frame from the camera
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("ERROR: frame is NULL");
    // In fails wait a bit before retrying    
    delay(1000);    
    return;
  }

  // Buffer copy in RAM
  jpgHeight = fb->height;
  jpgWidth = fb->width; 
  jpgLen = fb->len;
  jpgBuffer = (uint8_t*) heap_caps_aligned_alloc(32, jpgLen, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT); 
  memcpy(jpgBuffer, fb->buf, fb->len);

  Serial.printf("Acquired Frame: %dx%d, len=%u\n", jpgWidth, jpgHeight, jpgLen);

  saveFrameToSD(jpgBuffer, jpgLen);

  logMemory("Before malloc RGB");

  uint8_t* rgb = (uint8_t*) heap_caps_aligned_alloc(32, 320 * 240 * 3, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT); 
  //(uint8_t*) heap_caps_malloc(320 * 240 * 3, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  
  if (!rgb) {
    Serial.println("ERROR: malloc RGB");
    esp_camera_fb_return(fb);
    return;
  }
  Serial.println("malloc rgb OK");
  logMemory("After malloc RGB");
  if(!decodeJPGfromRAM(jpgBuffer, jpgLen, rgb)) {
  //if (!fmt2rgb888(fb->buf, fb->len, fb->format, rgb)) {
    Serial.println("ERROR: decodeJPGfromRAM");
    free(rgb);
    free(jpgBuffer);
    esp_camera_fb_return(fb);
    return;
  }
  free(jpgBuffer);
  Serial.println("decodeJPGfromRAM OK");
  if(counter < DEBUG_SAMPLES) {
    sprintf(debugName, "/cam_f_debug_%d.bmp", counter);
    saveBMP320x240(debugName, rgb);
  }
  logMemory("Before malloc resized");

  uint8_t* resized = (uint8_t*) heap_caps_malloc(IMG_SIZE * IMG_SIZE * 3, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  if (!resized) {
    Serial.println("ERROR: malloc resized");
    delay(1000);
    free(rgb);
    esp_camera_fb_return(fb);
    return;
  }
  Serial.println("malloc resized OK");
  logMemory("After malloc resized");

  Serial.println("Bilinear resize...");
  resize_bilinear_rgb888(rgb, 320, 240, resized, IMG_SIZE, IMG_SIZE);

  if(counter < DEBUG_SAMPLES) {
    sprintf(debugName, "/cam_debug_%d.bmp", counter);
    saveBMP96(debugName, resized);
  }

  Serial.println("Bilinear resize OK");

  free(rgb);
#endif

  Serial.println("Normalizzazione INT8...");

  float scale = input->params.scale;
  int zero_point = input->params.zero_point;
  Serial.println("Params conversion reading");
  Serial.printf("scale = %.8f, zero_point = %d\n", input->params.scale, input->params.zero_point);

  int idx = 0;
  for (int y = 0; y < IMG_SIZE; y++) {
    for (int x = 0; x < IMG_SIZE; x++) {

        int base = (y * IMG_SIZE + x) * 3;

        // Imagenet normalization
        float r = (resized[base+0] / 255.0f - 0.485f) / 0.229f;
        float g = (resized[base+1] / 255.0f - 0.456f) / 0.224f;
        float b = (resized[base+2] / 255.0f - 0.406f) / 0.225f;
        
        float r_c = round(r / scale) + zero_point;
        float g_c = round(g / scale) + zero_point;
        float b_c = round(b / scale) + zero_point;  

        input->data.int8[idx++] = r_c < -128 ? -128 : (r_c > 127 ? 127 : (int8_t)r_c);
        input->data.int8[idx++] = g_c < -128 ? -128 : (g_c > 127 ? 127 : (int8_t)g_c);
        input->data.int8[idx++] = b_c < -128 ? -128 : (b_c > 127 ? 127 : (int8_t)b_c);
      }
  }

  for (int i = 0; i < 10; i++) {
    Serial.printf("%d ", input->data.int8[i]);
  }
  Serial.println();

  free(resized);
  Serial.println("Normalization OK");

//#ifdef PERFORMANCE_TESTING
  if(counter < DEBUG_SAMPLES) {
    sprintf(debugName, "/debug_input_%d.bmp", counter++);
    dump_input_tensor(input, debugName, IMG_SIZE);
  }
//#endif

  logMemory("Before Invoke");

  int64_t start_time = esp_timer_get_time();
  TfLiteStatus status = interpreter->Invoke();
  int64_t end_time = esp_timer_get_time();  
  if (status != kTfLiteOk) {
    Serial.println("ERROR: Invoke()");
  } else {
    Serial.println("Invoke OK");
    int64_t total_time_us = end_time - start_time;
    float total_time_ms = total_time_us / 1000.0f;
    float total_time_sec = total_time_ms / 1000.0f;
    Serial.printf("Execution time: %.2f ms ( %.2f seconds)\n", total_time_ms, total_time_sec);
  }

  float nonperson = (output->data.int8[0] - output->params.zero_point) * output->params.scale;
  float person    = (output->data.int8[1] - output->params.zero_point) * output->params.scale;

  Serial.printf("non-person: %.3f | person: %.3f\n", nonperson, person);
#ifdef PERFORMANCE_TESTING
  Serial.printf("Real Label: %d, Predicted Label: %d\n", label, person > nonperson ? 1 : 0);
  updateConfusionMatrix(label, person > nonperson ? 1 : 0);
  printConfusionMatrix();  
  Serial.printf("Skipped samples: %d\n", skipped);

  // The validation set is composed of 6000 samples, so we can stop after 6000 iterations
  if (checkEndDataset(6000)) {
        Serial.println("\n=== TEST COMPLETED ===");
        while (true); // stop
  }
#endif

#ifndef PERFORMANCE_TESTING
  esp_camera_fb_return(fb);
#endif

  counter++;
  logMemory("Loop end");
}
