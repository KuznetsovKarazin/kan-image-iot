#include <Arduino.h>
#include <esp_timer.h>
#include <esp_system.h>
#include <esp_cpu.h>
#include <esp_heap_caps.h>
#include <errno.h>
#include <math.h>
#include <string.h>
#include "micro_model.h"
#include "usb_transport.h"

static float batch[MICRO_BATCH][MICRO_MAX_INPUTS];
static uint32_t loaded=0;
static volatile uint32_t sink_bits=0;
static char line[512];
static size_t used=0;
static bool overflowed=false;

static uint32_t bits(float x) { uint32_t u; memcpy(&u,&x,4); return u; }
static void error(const char *why) { micro_transport_printf("{\"kind\":\"error\",\"reason\":\"%s\"}\n",why); }
static bool number(char *s,uint32_t *out,uint32_t maximum) {
    if (!s || !*s) return false;
    for (const char *p=s;*p;++p) if (*p<'0'||*p>'9') return false;
    errno=0; char *end=nullptr; unsigned long n=strtoul(s,&end,10);
    if (errno || *end || n>maximum) return false;
    *out=(uint32_t)n; return true;
}
static bool floats(char **state,float *x) {
    for (unsigned i=0;i<micro_input_dim;++i) {
        char *s=strtok_r(nullptr," \t",state);
        if (!s || strlen(s)!=8) return false;
        for (char *p=s;*p;++p) if (!isxdigit((unsigned char)*p)) return false;
        char *end=nullptr; unsigned long u=strtoul(s,&end,16);
        if (*end) return false;
        uint32_t word=(uint32_t)u; memcpy(&x[i],&word,4);
        if (!isfinite(x[i])) return false;
    }
    return strtok_r(nullptr," \t",state)==nullptr;
}
static void hello() {
    micro_transport_printf("{\"kind\":\"hello\",\"protocol\":4,\"model\":\"%s\",\"artifact\":\"%s\","
      "\"input_dim\":%u,\"variants\":12,\"chip\":\"%s\",\"revision\":%u,\"cores\":%u,"
      "\"cpu_mhz\":%u,\"flash_bytes\":%u,\"psram_bytes\":%u,\"heap_free\":%u,\"heap_min\":%u,"
      "\"stack_watermark_bytes\":%u,\"idf\":\"%s\",\"radios_started\":false,"
      "\"serial_transport\":\"esp-idf-usb-serial-jtag-v4\",\"rx_buffer_bytes\":%u,\"tx_buffer_bytes\":%u,\"task_core\":%d,\"task_affinity\":%d,"
      "\"dram_copy_ok\":%s,\"dram_copy_bytes\":%u,\"dram_fp32_bytes\":%u,\"dram_int8_bytes\":%u,\"dram_allocation\":\"internal_8bit\"}\n",
      micro_model_id,micro_artifact_id,micro_input_dim,ESP.getChipModel(),ESP.getChipRevision(),
      ESP.getChipCores(),ESP.getCpuFreqMHz(),ESP.getFlashChipSize(),ESP.getPsramSize(),
      ESP.getFreeHeap(),ESP.getMinFreeHeap(),(unsigned)uxTaskGetStackHighWaterMark(nullptr),ESP.getSdkVersion(),
      MICRO_USB_RX_BYTES,MICRO_USB_TX_BYTES,xPortGetCoreID(),(int)xTaskGetAffinity(nullptr),
      micro_dram_ok?"true":"false",micro_dram_bytes,micro_dram_fp32_bytes,micro_dram_int8_bytes);
}
static void benchmark(uint32_t seq,unsigned variant,unsigned rounds) {
    if (loaded!=UINT32_MAX) {error("batch_incomplete"); return;}
    float y[2]; unsigned count=0; bool complete=true;
    const int core_start=xPortGetCoreID();
    int64_t start=esp_timer_get_time();
    const uint32_t cycle_start=esp_cpu_get_ccount();
    for (unsigned r=0;r<rounds;++r) {
        for (unsigned i=0;i<MICRO_BATCH;++i) {
            if (variant==MICRO_VARIANTS) {y[0]=batch[i][0]; y[1]=batch[i][1];}
            else if (micro_infer(variant,batch[i],y)) {error("inference_failed");return;}
            sink_bits=bits(y[0]); ++count;
        }
        if (esp_timer_get_time()-start>2000000) {complete=false; break;}
    }
    const uint32_t elapsed_cycles=esp_cpu_get_ccount()-cycle_start;
    int64_t elapsed=esp_timer_get_time()-start;
    const int core_end=xPortGetCoreID();
    micro_transport_printf("{\"kind\":\"bench\",\"seq\":%u,\"variant\":%u,\"rounds\":%u,\"n\":%u,"
      "\"elapsed_us\":%lld,\"complete\":%s,\"sink\":\"%08lx\",\"heap_free\":%u,"
      "\"heap_min\":%u,\"stack_watermark_bytes\":%u,\"elapsed_cycles\":%lu,\"core_start\":%d,\"core_end\":%d}\n",seq,variant,rounds,count,
      (long long)elapsed,complete?"true":"false",(unsigned long)sink_bits,ESP.getFreeHeap(),
      ESP.getMinFreeHeap(),(unsigned)uxTaskGetStackHighWaterMark(nullptr),
      (unsigned long)elapsed_cycles,core_start,core_end);
}
static void energy(uint32_t seq,unsigned variant,unsigned duration) {
    if (loaded!=UINT32_MAX) {error("batch_incomplete"); return;}
    if (!micro_transport_printf("{\"kind\":\"energy_armed\",\"seq\":%u,\"variant\":%u,\"lead_ms\":2000,\"duration_ms\":%u}\n",seq,variant,duration)) return;
    delay(2000);
    const int64_t start=esp_timer_get_time(); int64_t last_yield=start;
    uint32_t count=0,yields=0; float y[2];
    while (esp_timer_get_time()-start<(int64_t)duration*1000) {
        if (variant==MICRO_VARIANTS) {delay(1); ++yields; continue;}
        for (unsigned i=0;i<MICRO_BATCH;++i) {
            if (micro_infer(variant,batch[i],y)) {error("inference_failed");return;}
            sink_bits=bits(y[0]); ++count;
        }
        if (esp_timer_get_time()-last_yield>=10000) {delay(1);++yields;last_yield=esp_timer_get_time();}
    }
    micro_transport_printf("{\"kind\":\"energy_end\",\"seq\":%u,\"variant\":%u,\"n\":%u,"
      "\"elapsed_us\":%lld,\"scheduler_yields\":%u,\"scope\":\"whole_board_cooperative_loop\"}\n",
      seq,variant,count,(long long)(esp_timer_get_time()-start),yields);
}
static void command(char *s) {
    char *state=nullptr; char *cmd=strtok_r(s," \t",&state);
    if (!cmd) return;
    if (!strcmp(cmd,"H")) {if(strtok_r(nullptr," \t",&state)) error("extra_token");else hello();return;}
    uint32_t seq=0,v=0,n=0; float x[MICRO_MAX_INPUTS];
    if (!strcmp(cmd,"P")) {
        if (!number(strtok_r(nullptr," \t",&state),&seq,0x7fffffff)) {error("invalid_probe");return;}
        char *payload=strtok_r(nullptr," \t",&state);
        if (!payload || !*payload || strlen(payload)>384 || strtok_r(nullptr," \t",&state)) {error("invalid_probe");return;}
        for (const char *p=payload;*p;++p) if (!isxdigit((unsigned char)*p)) {error("invalid_probe");return;}
        micro_transport_printf("{\"kind\":\"probe\",\"seq\":%u,\"payload\":\"%s\"}\n",seq,payload);
    } else if (!strcmp(cmd,"V")) {
        if (!number(strtok_r(nullptr," \t",&state),&seq,0x7fffffff) ||
            !number(strtok_r(nullptr," \t",&state),&v,MICRO_VARIANTS-1) || !floats(&state,x)) {error("invalid_vector");return;}
        float y[2]; int status=micro_infer(v,x,y);
        if (status) {error("inference_failed");return;}
        micro_transport_printf("{\"kind\":\"pred\",\"seq\":%u,\"variant\":%u,\"bits\":[\"%08lx\",\"%08lx\"]}\n",seq,v,(unsigned long)bits(y[0]),(unsigned long)bits(y[1]));
    } else if (!strcmp(cmd,"L")) {
        if (!number(strtok_r(nullptr," \t",&state),&n,MICRO_BATCH-1) || !floats(&state,x)) {error("invalid_batch_vector");return;}
        memcpy(batch[n],x,micro_input_dim*sizeof(float)); loaded|=UINT32_C(1)<<n;
        micro_transport_printf("{\"kind\":\"loaded\",\"slot\":%u}\n",n);
    } else if (!strcmp(cmd,"B") || !strcmp(cmd,"E")) {
        if (!number(strtok_r(nullptr," \t",&state),&seq,0x7fffffff) ||
            !number(strtok_r(nullptr," \t",&state),&v,MICRO_VARIANTS) ||
            !number(strtok_r(nullptr," \t",&state),&n,!strcmp(cmd,"B")?32:60000) ||
            strtok_r(nullptr," \t",&state) || n==0) {error("invalid_benchmark");return;}
        if (!strcmp(cmd,"B")) benchmark(seq,v,n);
        else if (n<1000) error("energy_window_too_short");
        else energy(seq,v,n);
    } else error("unknown_command");
}
void setup() {
    static_assert(sizeof(float)==4,"IEEE754 binary32 required");
#if defined(CONFIG_IDF_TARGET_ESP32C3)
    /* IDF 4.4.7 uses the C3 custom PCCR_MACHINE counter, not the mcycle CSR. */
    cpu_ll_enable_cycle_count();
#endif
    micro_init_dram();
    micro_transport_begin();
    delay(500);
}
void loop() {
    int incoming;
    while ((incoming=micro_transport_read())>=0) {
        char c=(char)incoming;
        if (c=='\r') continue;
        if (c=='\n') {
            if (overflowed) error("line_too_long");
            else {line[used]='\0';command(line);}
            used=0; overflowed=false;
        } else if (used<sizeof(line)-1 && !overflowed) line[used++]=c;
        else overflowed=true;
    }
    delay(1);
}
