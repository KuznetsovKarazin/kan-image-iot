"""Run the actual old and patched SDK drivers against an edge-triggered FIFO model.

This models endpoint completion/status and host short-packet delivery, not an
ESP32 CPU, Windows USB stack, or physical latency measurement.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
API = r'''
#pragma once
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <deque>
#include <string>
#include <iostream>
typedef int esp_err_t;
typedef int BaseType_t;
typedef int portBASE_TYPE;
typedef unsigned UBaseType_t;
typedef unsigned TickType_t;
typedef void *intr_handle_t;
static const int ESP_OK=0, ESP_ERR_INVALID_STATE=-1, ESP_ERR_INVALID_ARG=-2, ESP_ERR_NO_MEM=-3;
static const int pdFALSE=0, pdTRUE=1, RINGBUF_TYPE_BYTEBUF=0;
static const int MALLOC_CAP_INTERNAL=1, MALLOC_CAP_8BIT=2, ETS_USB_SERIAL_JTAG_INTR_SOURCE=3;
static const unsigned USB_SERIAL_JTAG_INTR_SERIAL_IN_EMPTY=8, USB_SERIAL_JTAG_INTR_SERIAL_OUT_RECV_PKT=4;
static int USB_SERIAL_JTAG;
#define ESP_RETURN_ON_FALSE(a,b,...) do {if(!(a)) return (b);} while(0)
#define ESP_LOGE(...) ((void)0)
#define ESP_LOGI(...) ((void)0)
#define portYIELD_FROM_ISR() ((void)0)
struct usb_serial_jtag_driver_config_t {uint32_t tx_buffer_size, rx_buffer_size;};
esp_err_t usb_serial_jtag_driver_uninstall();
struct Ring {std::deque<uint8_t> bytes; size_t capacity; bool empty_item=false;};
typedef Ring *RingbufHandle_t;
static unsigned enabled, raw_status;
static std::string tx_fifo, rx_fifo, host_pending, host_delivered;
static bool packet_pending;
static size_t write_limit=64, zlp_count;
static void (*idle_enqueue_hook)()=nullptr;
static void reset_hardware() {
    enabled=0; raw_status=USB_SERIAL_JTAG_INTR_SERIAL_IN_EMPTY;
    tx_fifo.clear();rx_fifo.clear();host_pending.clear();host_delivered.clear();
    packet_pending=false;write_limit=64;zlp_count=0;idle_enqueue_hook=nullptr;
}
static void *heap_caps_calloc(size_t n,size_t s,int) {return calloc(n,s);}
static void heap_caps_free(void *p) {free(p);}
static RingbufHandle_t xRingbufferCreate(size_t n,int) {return new Ring{{},n,false};}
static void vRingbufferDelete(Ring *p) {delete p;}
static BaseType_t xRingbufferSend(Ring *r,void *p,size_t n,TickType_t) {
    if(r->bytes.size()+n>r->capacity)return pdFALSE;
    auto *b=(uint8_t*)p;r->bytes.insert(r->bytes.end(),b,b+n);return pdTRUE;
}
static BaseType_t xRingbufferSendFromISR(Ring *r,void *p,size_t n,BaseType_t*) {return xRingbufferSend(r,p,n,0);}
static void *receive(Ring *r,size_t *n,size_t cap) {
    if(r->empty_item){r->empty_item=false;*n=0;return malloc(1);}
    *n=std::min(cap,r->bytes.size());if(!*n)return nullptr;
    auto *p=(uint8_t*)malloc(*n);
    for(size_t i=0;i<*n;++i){p[i]=r->bytes.front();r->bytes.pop_front();}
    return p;
}
static void *xRingbufferReceiveUpToFromISR(Ring *r,size_t *n,size_t cap) {return receive(r,n,cap);}
static void *xRingbufferReceiveUpTo(Ring *r,size_t *n,TickType_t,size_t cap) {return receive(r,n,cap);}
static void vRingbufferReturnItemFromISR(Ring*,void *p,BaseType_t*) {free(p);}
static void vRingbufferReturnItem(Ring*,void *p) {free(p);}
static void vRingbufferGetInfo(Ring *r,void*,void*,void*,void*,UBaseType_t *n) {*n=r->bytes.size();}
static uint32_t usb_serial_jtag_ll_get_intsts_mask() {return raw_status&enabled;}
static void usb_serial_jtag_ll_clr_intsts_mask(uint32_t mask) {raw_status&=~mask;}
static void usb_serial_jtag_ll_ena_intr_mask(uint32_t mask) {enabled|=mask;}
static void usb_serial_jtag_ll_disable_intr_mask(uint32_t mask) {
    if(idle_enqueue_hook){auto fn=idle_enqueue_hook;idle_enqueue_hook=nullptr;fn();}
    enabled&=~mask;
}
static int usb_serial_jtag_ll_txfifo_writable() {return !packet_pending&&tx_fifo.size()<64;}
static size_t usb_serial_jtag_ll_write_txfifo(const uint8_t *b,uint32_t n) {
    size_t count=std::min({(size_t)n,64-tx_fifo.size(),write_limit});
    if(packet_pending)return 0;
    tx_fifo.append((const char*)b,count);return count;
}
static void usb_serial_jtag_ll_txfifo_flush() {if(!packet_pending)packet_pending=true;}
static uint32_t usb_serial_jtag_ll_read_rxfifo(uint8_t *b,uint32_t cap) {
    size_t n=std::min((size_t)cap,rx_fifo.size());memcpy(b,rx_fifo.data(),n);rx_fifo.erase(0,n);return n;
}
static void usb_serial_jtag_ll_enable_bus_clock(bool) {}
static void usb_phy_ll_int_jtag_enable(int*) {}
static esp_err_t esp_intr_alloc(int,int,void (*)(void*),void*,intr_handle_t *p) {*p=(void*)1;return ESP_OK;}
static void esp_intr_free(intr_handle_t) {}
'''
MAIN = r'''
#include DRIVER_SOURCE
static bool pump() {
    for(int steps=0;steps<100000;++steps){
        if(raw_status&enabled){usb_serial_jtag_isr_handler_default(nullptr);continue;}
        if(packet_pending){
            const size_t n=tx_fifo.size();
            host_pending+=tx_fifo;tx_fifo.clear();packet_pending=false;
            if(!n)++zlp_count;
            if(n<64){host_delivered+=host_pending;host_pending.clear();}
            raw_status|=USB_SERIAL_JTAG_INTR_SERIAL_IN_EMPTY;
            continue;
        }
        return true;
    }
    return false;
}
static void install(bool bootstrap=true) {
    reset_hardware();usb_serial_jtag_driver_config_t c{2048,2048};
    assert(usb_serial_jtag_driver_install(&c)==ESP_OK);
    // For the negative control, seed the first completion to reproduce an
    // already-live link: the first response succeeds, the next cannot start.
    if(bootstrap)raw_status|=USB_SERIAL_JTAG_INTR_SERIAL_IN_EMPTY;
}
static void send(const std::string &s) {
    assert(usb_serial_jtag_write_bytes(s.data(),s.size(),1)==(int)s.size());
    assert(pump());
}
static void enqueue_during_disable() {
    const char c='R';assert(usb_serial_jtag_write_bytes(&c,1,0)==1);
}
int main() {
    install();const std::string first="{\"kind\":\"hello\"}\n";
    send(first);assert(host_delivered==first);
    send(first);
#ifdef EXPECT_OLD_FAILURE
    assert(host_delivered==first);
    assert(p_usb_serial_jtag_obj->tx_ring_buf->bytes.size()==first.size());
    std::cout<<"old_driver: first reply delivered; second queued but stalled\n";
    usb_serial_jtag_driver_uninstall();return 0;
#else
    assert(host_delivered==first+first);usb_serial_jtag_driver_uninstall();
    install(false);send("initial\n");assert(host_delivered=="initial\n");
    usb_serial_jtag_driver_uninstall();
    install();std::string expected;
    const size_t sizes[]={1,4,63,64,65,127,128,192,255,256,384,512,1023};
    for(unsigned i=0;i<1300;++i){
        std::string frame(sizes[i%13],char('A'+i%26));expected+=frame;
        send(frame);assert(host_delivered==expected);
        assert(p_usb_serial_jtag_obj->tx_ring_buf->bytes.empty());
    }
    assert(zlp_count>0);usb_serial_jtag_driver_uninstall();
    install();write_limit=4;send(std::string(257,'x'));
    assert(host_delivered==std::string(257,'x'));usb_serial_jtag_driver_uninstall();
    install();p_usb_serial_jtag_obj->tx_ring_buf->empty_item=true;
    send("wrap\n");assert(host_delivered=="wrap\n");usb_serial_jtag_driver_uninstall();
    install();send("start\n");host_delivered.clear();
    // Inject an enqueue between the idle decision and interrupt disabling.
    idle_enqueue_hook=enqueue_during_disable;
    send("race\n");assert(host_delivered=="race\nR");usb_serial_jtag_driver_uninstall();
    install();rx_fifo="H\n";raw_status|=USB_SERIAL_JTAG_INTR_SERIAL_OUT_RECV_PKT;
    assert(pump());char data[8]={};assert(usb_serial_jtag_read_bytes(data,8,0)==2);
    assert(std::string(data,2)=="H\n");usb_serial_jtag_driver_uninstall();
    std::cout<<"patched_driver: 1300 idle restarts, packet boundaries, short FIFO writes, wrap, enqueue race, RX PASS\n";
#endif
}
'''


def main():
    upstream=ROOT/'tests/fixtures/usb_serial_jtag_4_4_7.c'
    patched=ROOT/'src/usb_serial_jtag_compat.c'
    outputs={}
    with tempfile.TemporaryDirectory() as temp:
        folder=Path(temp)
        (folder/'api.h').write_text(API)
        for name in ['esp_log.h','hal/usb_serial_jtag_ll.h','hal/usb_phy_ll.h',
                     'freertos/FreeRTOS.h','freertos/semphr.h','freertos/ringbuf.h',
                     'esp_intr_alloc.h','driver/usb_serial_jtag.h','soc/periph_defs.h','esp_check.h']:
            path=folder/name;path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text('#pragma once\n#include "api.h"\n')
        (folder/'main.cpp').write_text(MAIN)
        for name,source in [('old',upstream),('patched',patched)]:
            exe=folder/name
            cmd=['g++','-std=c++17','-O2','-Wall','-Wextra','-Wno-unused-function','-Wno-unused-parameter',
                 '-I'+str(folder),f'-DDRIVER_SOURCE="{source}"',str(folder/'main.cpp'),'-o',str(exe)]
            if name=='old':cmd.append('-DEXPECT_OLD_FAILURE=1')
            subprocess.run(cmd,check=True,capture_output=True,text=True)
            result=subprocess.run([str(exe)],check=True,capture_output=True,text=True)
            outputs[name]=result.stdout.strip();print(outputs[name])
    report={'status':'PASS','physical_execution':False,
            'scope':'Actual upstream and patched driver C code with modeled USB completion edges and short-packet delivery',
            'old_driver_stall_reproduced':True,'patched_idle_restarts':1300,'outputs':outputs,
            'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [upstream,patched]}}
    (ROOT/'results/usb_idle_rearm_checks.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
