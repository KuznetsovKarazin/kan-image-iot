# KAN and MLP heads on microcontrollers: latency and energy

Firmware, analysis code and measured results for the study
"When Smaller Tables Cost More: KAN Representation on Microcontrollers"
(O. Kuznetsov, E. Frontoni, D. Faggi).

Three trained KAN classification heads (C0, C1: 24-16-2; C2: 32-24-16-2) and
an MLP of the same widths run on frozen MobileNetV3-Small features on three
boards: ESP32-S3, ESP32-C3 and ATmega2560. The CNN is not executed on the
boards; only the head is timed.

## Layout

| Path | Content |
|---|---|
| `firmware/kan/` | KAN firmware (ESP-IDF/PlatformIO for ESP32, avr-gcc for Mega), generated tables (`generated/`), golden reference outputs, host tools for capture and checking |
| `firmware/mlp/` | MLP firmware, quantization and host tools, host training results (`evidence/`) |
| `analysis/` | Meter file reader (`cfn_reader.py`), per-window energy integration, recording analysis, plots |
| `results/all_boards_summary.csv` | All 55 configurations: energy, latency, accuracy, changed decisions, parameter bytes |
| `results/all_boards_results.json` | The same data plus the selected KAN/MLP comparisons |
| `results/mega_energy_results.json` | Mega windows, polling-baseline diagnostics, alignment checks |
| `results/records/` | Per-recording integration results for the Mega |

KAN variants: v0 spline direct, v8 cubic direct, v4 FP32 LUT32, v5 INT8 LUT32,
v9 factored INT8 LUT32 (`s[q0 + u (q1 - q0)]`); on ESP32-S3 also v10 (FP32 LUT
in DRAM) and v11 (factored INT8 LUT in DRAM). MLP variants: v0 FP32, v1 INT8.

## Energy

A FNIRSI FNB-58 USB meter in the board supply line records bus voltage and
current at 100 Hz. Energy per inference is

    E = (1/N) * integral of V_BUS(t) * I_BUS(t) dt

over each active window (trapezoidal rule, interpolated window edges, N =
counted inferences). This is whole-board energy. The meter's accumulated Wh
channel is not used; in every recording it equals one quarter of the
integrated V*I energy, while the accumulated Ah channel equals the integrated
current.

`analysis/integrate_trace.py` imports `audit_energy` from
`firmware/kan/tools`, so run it with

    PYTHONPATH=firmware/kan/tools:analysis python3 analysis/integrate_trace.py --help

## Raw data

The 24 native meter recordings (CFN, about 85 MB) and the complete acquisition
archive with checksums are distributed separately from this repository
(see the article's supplementary material).
