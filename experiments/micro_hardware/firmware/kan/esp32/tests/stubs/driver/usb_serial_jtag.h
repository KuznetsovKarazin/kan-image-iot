#pragma once
/* Host-only IDF transport stub; it does not emulate USB hardware or timing. */
#include <cstdint>
#include <cstdlib>
#include <poll.h>
#include <unistd.h>
typedef int esp_err_t;
typedef unsigned TickType_t;
static const esp_err_t ESP_OK = 0;
#define pdMS_TO_TICKS(ms) (ms)
typedef struct {
    uint32_t tx_buffer_size;
    uint32_t rx_buffer_size;
} usb_serial_jtag_driver_config_t;
inline esp_err_t usb_serial_jtag_driver_install(usb_serial_jtag_driver_config_t *config) {
    return config && config->rx_buffer_size>=512 && config->tx_buffer_size>=1024 ? ESP_OK : -1;
}
inline int usb_serial_jtag_read_bytes(void *buf, uint32_t size, TickType_t wait) {
    pollfd p{0,POLLIN,0};
    if (poll(&p,1,(int)wait)<=0 || !(p.revents&POLLIN)) return 0;
    return (int)::read(0,buf,size);
}
inline int usb_serial_jtag_write_bytes(const void *src, size_t size, TickType_t) {
    const char *limit = getenv("MICRO_STUB_WRITE_MAX");
    if (limit) {
        unsigned max = (unsigned)strtoul(limit,nullptr,10);
        if (max && size>max) size=max;
    }
    return (int)::write(1,src,size);
}
