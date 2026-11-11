#include "kernel.h"
#define MCU_MODEL_CASE MODEL_CASE
#define MLP_FORMAT WIRE_VARIANT
#include "mlp_core.inc"
int mega_infer(const float*x,float*y){
#if WIRE_VARIANT==0
return mlp_fp32(MODEL_CASE,x,y);
#else
int8_t q[2];return mlp_int8(MODEL_CASE,x,q,y);
#endif
}
