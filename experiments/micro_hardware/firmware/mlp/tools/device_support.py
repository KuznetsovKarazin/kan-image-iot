"""USB identity tracking and temporary Windows sleep inhibition."""
import re
import sys
import time


class KeepAwake:
    """Thread-scoped idle-sleep request; does not change the Windows power plan.

    Microsoft SetThreadExecutionState documentation:
    https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-setthreadexecutionstate
    Manual sleep, power-button actions and lid closure remain user-controlled.
    """
    def __init__(self, system=None, function=None):
        self.system = sys.platform if system is None else system
        self.function = function
        self.previous = None
        self.status = 'not_available_on_this_os'

    def __enter__(self):
        if self.system == 'win32':
            if self.function is None:
                import ctypes
                self.function = ctypes.windll.kernel32.SetThreadExecutionState
                self.function.argtypes = [ctypes.c_uint32]
                self.function.restype = ctypes.c_uint32
            self.previous = self.function(0x80000001)
            if not self.previous:
                raise RuntimeError('Windows did not accept the idle-sleep inhibition request; no hardware was accessed')
            self.status = 'windows_idle_sleep_inhibited'
        return self

    def __exit__(self, *_):
        if self.previous is not None:
            self.function(self.previous | 0x80000000)

def require(ok,message):
    if not ok:raise ValueError(message)

def normalized_serial(value):
    return re.sub('[^a-z0-9]', '', (value or '').lower())


class PortTracker:
    """Follow this exact USB serial number, never the first available COM port."""
    def __init__(self, requested):
        from serial.tools.list_ports import comports
        self.list_ports = comports
        candidates = [p for p in comports() if p.device.casefold() == requested.casefold()]
        require(len(candidates) == 1, 'Requested COM port is not uniquely present; list ports again')
        p = candidates[0]
        require((p.vid, p.pid) == (0x303A, 0x1001), 'Requires the native Espressif USB Serial/JTAG port (303A:1001)')
        require(bool(normalized_serial(p.serial_number)), 'USB serial number is missing; cannot follow the board safely')
        self.identity = dict(vid=p.vid, pid=p.pid, serial_number=p.serial_number,
                             initial_port=p.device)

    def resolve(self, timeout=20):
        until = time.monotonic() + timeout
        while True:
            found = [p for p in self.list_ports()
                     if (p.vid, p.pid) == (self.identity['vid'], self.identity['pid'])
                     and normalized_serial(p.serial_number) == normalized_serial(self.identity['serial_number'])]
            require(len(found) <= 1, 'Duplicate USB serial number; refusing to choose a board')
            if found:
                return found[0].device
            if time.monotonic() >= until:
                raise TimeoutError('The original USB serial number did not return; no other board was selected')
            time.sleep(.2)

def validate_reset_output(text, target, expected_mac=None):
    chip = 'ESP32-S3' if target == 's3' else 'ESP32-C3'
    require(re.search(r'^Chip is ' + re.escape(chip) + r'(?:\s|\(|$)', text, re.M), 'Unexpected chip in reset log')
    macs = re.findall(r'^MAC:\s*([0-9a-fA-F]{2}(?::[0-9a-fA-F]{2}){5})\s*$', text, re.M)
    require(macs and len({m.lower() for m in macs}) == 1, 'Missing or inconsistent MAC in ROM response')
    mac = macs[0].lower()
    require(expected_mac is None or mac == expected_mac, 'Physical chip MAC changed; batch stopped')
    require('Hard resetting via RTS pin...' in text, 'Missing reset completion message')
    require('Chip was NOT reset' not in text, 'esptool reported a failed reset')
    return mac

