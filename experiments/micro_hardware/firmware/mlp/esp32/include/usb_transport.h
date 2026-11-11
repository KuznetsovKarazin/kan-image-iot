#pragma once
/* Exclusive ESP-IDF USB Serial/JTAG transport. No Arduino HWCDC instance. */
#include <driver/usb_serial_jtag.h>
#include <esp_timer.h>
#include <cstdarg>
#include <cstdio>
#include <cstdint>

static const unsigned MICRO_USB_RX_BYTES = 2048;
static const unsigned MICRO_USB_TX_BYTES = 2048;
static bool micro_transport_ready = false;
static bool micro_transport_failed = false;

static void micro_transport_begin() {
    usb_serial_jtag_driver_config_t config = {};
    config.rx_buffer_size = MICRO_USB_RX_BYTES;
    config.tx_buffer_size = MICRO_USB_TX_BYTES;
    micro_transport_ready = usb_serial_jtag_driver_install(&config) == ESP_OK;
}

static int micro_transport_read() {
    if (!micro_transport_ready || micro_transport_failed) return -1;
    uint8_t c = 0;
    return usb_serial_jtag_read_bytes(&c, 1, 0) == 1 ? c : -1;
}

static bool micro_transport_printf(const char *format, ...)
    __attribute__((format(printf, 1, 2)));

static bool micro_transport_printf(const char *format, ...) {
    if (!micro_transport_ready || micro_transport_failed) return false;
    char frame[1024];
    va_list args;
    va_start(args, format);
    const int length = vsnprintf(frame, sizeof(frame), format, args);
    va_end(args);
    if (length <= 0 || length >= (int)sizeof(frame) || frame[length-1] != '\n') {
        micro_transport_failed = true;
        return false;
    }
    const int64_t deadline = esp_timer_get_time() + 2000000;
    size_t offset = 0;
    while (offset < (size_t)length && esp_timer_get_time() < deadline) {
        const int written = usb_serial_jtag_write_bytes(
            frame + offset, (size_t)length - offset, pdMS_TO_TICKS(100));
        if (written < 0 || (size_t)written > (size_t)length - offset) {
            micro_transport_failed = true;
            return false;
        }
        offset += (size_t)written;
    }
    // Stop accepting requests after a failed/partial response. Never append a
    // new response to a truncated frame and never report a successful run.
    if (offset != (size_t)length) micro_transport_failed = true;
    return !micro_transport_failed;
}
