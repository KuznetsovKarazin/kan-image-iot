#include "micro_model.h"
#define MCU_MODEL_CASE MODEL_CASE
#include "mlp_core.inc"
#include "mlp_metadata.h"
const unsigned micro_input_dim=MLP_INPUTS;
const char micro_model_id[]=MLP_MODEL;
const char micro_artifact_id[]=MLP_ARTIFACT;
const char *const micro_variant_names[2]={"mlp_fp32","mlp_int8"};
const unsigned micro_parameter_bytes[2]={MLP_FP32_BYTES,MLP_INT8_BYTES};
unsigned micro_dram_bytes=0,micro_dram_fp32_bytes=0,micro_dram_int8_bytes=0;
int micro_dram_ok=0;
int micro_init_dram(void){return 0;}
int micro_infer(unsigned v,const float*x,float*y){int8_t q[2];if(v==0)return mlp_fp32(MODEL_CASE,x,y);if(v==1)return mlp_int8(MODEL_CASE,x,q,y);return -1;}
