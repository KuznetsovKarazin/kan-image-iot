"""CFN parser reused unchanged from the audited initial pilot; primary V/I/P channels.
Format: https://github.com/didim99/usbmeter-utils/blob/master/cfn2csv.py
"""

import hashlib,math,struct

CHANNEL_NAMES = {0: "voltage_v", 1: "current_a", 2: "dplus_v", 3: "dminus_v", 4: "power_w", 5: "accumulated_capacity_ah", 6: "accumulated_energy_wh"}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def trapezoid(times, values):
    return math.fsum((b - a) * (x + y) / 2
                     for a, b, x, y in zip(times, times[1:], values, values[1:]))

def parse_cfn(path):
    data = path.read_bytes()
    assert len(data) >= 26, "Incomplete CFN header"
    rate, start_ma, stop_ma, stop_s, channel_count = struct.unpack_from("<diiiH", data)
    assert math.isfinite(rate) and 0 < rate <= 1000 and 1 <= channel_count <= 7
    offset = 22
    descriptors = []
    for _ in range(channel_count):
        code, color, flag = struct.unpack_from("<HIB", data, offset)
        offset += 7
        assert code in CHANNEL_NAMES and flag in (0, 1)
        descriptor = dict(code=code, name=CHANNEL_NAMES[code], argb=f"{color:08x}")
        if flag:
            maximum, minimum = struct.unpack_from("<dd", data, offset)
            offset += 16
            descriptor.update(saved_max=maximum, saved_min=minimum)
        descriptors.append(descriptor)
    assert len({d["code"] for d in descriptors}) == channel_count
    count, = struct.unpack_from("<I", data, offset)
    offset += 4
    width = 8 * (channel_count + 1)
    assert 2 <= count <= 10_000_000 and len(data) == offset + count * width
    records = list(struct.iter_unpack("<" + "d" * (channel_count + 1), data[offset:]))
    assert all(math.isfinite(value) for row in records for value in row)
    times = [row[0] for row in records]
    gaps = [b - a for a, b in zip(times, times[1:])]
    assert min(gaps) > 0
    columns = {descriptor["code"]: [row[i + 1] for row in records]
               for i, descriptor in enumerate(descriptors)}
    assert {0, 1, 4}.issubset(columns)
    assert all(0 < v <= 5.5 for v in columns[0]) and min(columns[1]) >= 0
    power = [v * i for v, i in zip(columns[0], columns[1])]
    p_error = max(abs(a - b) for a, b in zip(power, columns[4]))
    assert p_error < 1e-12, "Stored PBUS differs from VBUS * IBUS"
    for descriptor in descriptors:
        if "saved_max" in descriptor:
            column = columns[descriptor["code"]]
            assert math.isclose(max(column), descriptor["saved_max"], abs_tol=1e-12)
            assert math.isclose(min(column), descriptor["saved_min"], abs_tol=1e-12)
    total_j = trapezoid(times, power)
    charge_ah = trapezoid(times, columns[1]) / 3600
    saved_energy_delta_wh = columns[6][-1] - columns[6][0] if 6 in columns else None
    saved_charge_delta_ah = columns[5][-1] - columns[5][0] if 5 in columns else None
    metadata = dict(
        status="CFN_PRIMARY_CHANNELS_VALIDATED", source_sha256=sha(path),
        sample_count=count, data_offset_bytes=offset, record_bytes=width,
        nominal_rate_hz=rate, start_current_ma=start_ma, stop_current_ma=stop_ma,
        stop_time_condition_s=stop_s, channels=descriptors,
        first_time_s=times[0], last_time_s=times[-1],
        max_time_grid_error_s=max(abs(t - (times[0] + i / rate)) for i, t in enumerate(times)),
        min_gap_s=min(gaps), max_gap_s=max(gaps),
        nominal_timestamps_are_not_an_independently_calibrated_clock=True,
        v_times_i_vs_p_max_abs_error_w=p_error,
        integrated_vi_energy_j=total_j, integrated_vi_energy_wh=total_j / 3600,
        saved_channel6_energy_delta_wh=saved_energy_delta_wh,
        integrated_vi_over_saved_energy_ratio=(total_j / 3600) / saved_energy_delta_wh if saved_energy_delta_wh else None,
        integrated_current_charge_ah=charge_ah,
        saved_channel5_charge_delta_ah=saved_charge_delta_ah,
        cumulative_energy_channel_used=False,
        cumulative_energy_review="Accumulated Wh is reported for reference only; in all recordings it equals one quarter of the integrated VBUS*IBUS energy. Energy is computed from the V and I samples.",
        meter_model="FNB58, as specified in the experiment instructions; not encoded by this CFN header",
        meter_serial=None, meter_firmware_version=None, pc_software_version=None,
        independent_calibration_available=False,
        format_reference="https://github.com/didim99/usbmeter-utils/blob/master/cfn2csv.py")
    return records, columns, metadata
