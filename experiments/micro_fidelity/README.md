# IEEE Micro: conversion-fidelity pilot

Start with [REPORT.md](REPORT.md), [PLAN.md](PLAN.md), and
[IMAGE_PROTOCOL.md](IMAGE_PROTOCOL.md). Publication boundaries are recorded in
[PRIOR_WORK.md](PRIOR_WORK.md).

This directory is an isolated experiment. It does not change the production
converter or claim a complete MCU deployment. INT8 means **table storage**;
interpolation, activations and accumulations are FP32.

## Reproduce (Linux, Python 3.12, C99 compiler)

Run from the repository root. The three checkpoints listed in PLAN.md must be
checked out. Do not run an unrestricted `git pull` into a differently configured
working tree; use this branch or a separate checkout.

```bash
python3.12 -m venv .venv-micro
.venv-micro/bin/pip install --extra-index-url https://download.pytorch.org/whl/cpu -r experiments/micro_fidelity/environment.freeze.txt
.venv-micro/bin/python experiments/micro_fidelity/test_fidelity.py
.venv-micro/bin/python experiments/micro_fidelity/run_synthetic.py
.venv-micro/bin/python experiments/micro_fidelity/prepare_images.py --cache /absolute/path/to/coco-pilot-cache
.venv-micro/bin/python experiments/micro_fidelity/run_images.py --cache /absolute/path/to/coco-pilot-cache
.venv-micro/bin/python experiments/micro_fidelity/posthoc_controls.py
.venv-micro/bin/python experiments/micro_fidelity/verify_results.py
.venv-micro/bin/python experiments/micro_fidelity/summarize.py
```

The image preparation downloads the official COCO 2014 annotations and 1,280
original images. Keep that cache outside Git. Images are not redistributed in
this branch. Image IDs, original and processed hashes, labels, licenses and
selection provenance are stored in `results/images/`.

`run_synthetic.py` fixes PyTorch to one CPU thread, deterministic algorithms,
4,096 calibration and 4,096 evaluation vectors for each seed 42/43/44. These
are **input-sampling seeds**, not three training replications.

`run_images.py` uses 256 disjoint calibration and 1,024 evaluation images with
shared FP32 CNN/projector features. It also generates C99 tables in a temporary
folder, compiles them with the recorded flags, and requires matching logits
and decisions against the Python table evaluator. It does not build ESP32
firmware. `native.py` can regenerate those build products from any compiled
`EdgeLUT` instance; no generated headers or shared libraries are committed.

The original converter classes are extracted without modifications from their
AST; importing the full TensorFlow/ONNX converter is unnecessary. The original
checkpoint weights load strictly, with automatic KAN checkpoint-writing disabled.

## Result files

- `results/synthetic/metrics.json`: every variant/resolution/seed/stress result.
- `results/synthetic/*_logits.npz`: per-example outputs; inputs are regenerated
  from the documented generator sequence.
- `results/images/metrics.json`: every image comparison, Wilson intervals and
  paired accuracy changes with stratified bootstrap intervals.
- `results/images/*_features.npz`, `*_logits.npz`: frozen feature vectors, image
  IDs, labels and per-image outputs; no raw images.
- `results/images/native_checks.json`: C-versus-Python numerical checks and
  explicitly scoped CPU head timings.
- `results/*/run.json`: checkpoints, hashes, environment and scope.
- `results/verification.json`: independent recalculation of saved-array metrics.
- `results/anchored_control/`: a separately labelled post hoc control specified
  after seeing the primary results; cheap output re-centering and probability fidelity.

Expected limitations: original test manifests and 16–4–2 checkpoints are absent;
minival may have participated in checkpoint selection; there is no MCU hardware,
no new MLP training, and no fully integer CNN+KAN export in this pilot. Results
cannot replace the hardware experiment requested by the reviewers.

## Prepare a later device check

```bash
.venv-micro/bin/python experiments/micro_fidelity/export_native.py --model small_24_16 --points 256 --out /absolute/path/to/native-build
```

The exporter creates `edge.c`, `edge.h`, little-endian FP32 golden inputs and
expected logits, image IDs and an export manifest. It compiles and checks the
same C implementation on the host before returning. The produced `.so` is a
host check only; compile `edge.c` with the MCU toolchain for the board.
Inputs are projector features, not images. A board run must first replay all
golden feature vectors and save per-example logits before full CNN integration.
