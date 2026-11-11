#pragma once
#include <stdint.h>

#define MICRO_MAX_INPUTS 32
#define MICRO_OUTPUTS 2
#define MICRO_VARIANTS 2
#define MICRO_BATCH 32

#ifdef __cplusplus
extern "C" {
#endif
extern const char micro_model_id[];
extern const char micro_artifact_id[];
extern const unsigned micro_input_dim;
extern const char *const micro_variant_names[MICRO_VARIANTS];
extern const unsigned micro_parameter_bytes[MICRO_VARIANTS];
int micro_init_dram(void);
extern unsigned micro_dram_bytes;
extern unsigned micro_dram_fp32_bytes;
extern unsigned micro_dram_int8_bytes;
extern int micro_dram_ok;
int micro_infer(unsigned variant, const float *input, float *output);
#ifdef __cplusplus
}
#endif
