# IEEE Micro conversion-fidelity pilot

Protocol fixed on 2026-11-07 before running comparisons. Base: `v2b`, commit
`330395f7a9406e73d8501a68c9b2daae2747b6af`. No training or checkpoint selection by pilot outcomes.

## Question

Does replacing a composed KAN with the mean of whole-network coordinate-axis
sections introduce an error that increasing LUT resolution cannot remove?
Can edge-wise tabulation that preserves layer composition avoid that error?
This is a conversion audit, not a claim that LUT acceleration itself is new.

## Models and data

Use the saved `kan_person_detector_best.pt` checkpoints for:

1. 24–16–2, MobileNetV3-Small width 0.25, 224 px.
2. 24–16–2, width 1.0, 224 px.
3. 32–24–16–2, width 0.25, 224 px.

All use spline grid 5, degree 3. Load strictly and record SHA-256 and metadata.
These are available repository checkpoints, not the missing 16–4–2 checkpoints
underlying the old article's tables. Do not claim reproduction of those tables.

Stage A: 4,096 calibration and 4,096 separate evaluation feature vectors per
seed (42, 43, 44), uniform in [0, 2], consistent with the projector's nonnegative
output but **not** a model of natural-image features. Add an independent
out-of-calibration-range [0, 4] stress set. Report saturation explicitly.
Calibrate every layer from the original FP32 model only; expand each observed
range by 5%, with a nonzero floor. Never tune ranges using evaluation data.

Stage B, if accessible: a deterministic, disjoint calibration/evaluation image
sample with public COCO annotations and VWW's person-area threshold 0.005.
Record image IDs, hashes, labels, preprocessing, split membership and limitations.
The historical 6,000-image manifest is absent. A new subset is an exploratory
conversion comparison, not an independent reproduction of published accuracy;
original training/model-selection overlap may be unknown. Freeze this stage's
sampling protocol before reading model outputs. No inference of image accuracy
from synthetic features.

## Variants

- Unmodified trained FP32 reference.
- Exact coordinate-axis average, without a LUT: structural-error control.
- Original `FastLutKAN`, extracted unchanged from the base converter, at
  16, 32, 64, 128, 256 points and its original [-2, 2] domain.
- Layer-preserving per-edge LUT at the same five resolutions, FP32 linear
  interpolation and explicit saturation at calibrated bounds. Include the
  SiLU base branch, edge masks/scales and node/subnode affine transforms.
  Unsupported active symbolic or multiplication nodes must fail explicitly.
- The same per-edge LUT with symmetric per-edge INT8 **table storage** and
  FP32 interpolation. This isolates table quantization; it is not fully INT8
  execution, TFLite conversion or an MCU benchmark.
- Audit the existing layerwise converter separately for duplicated offsets
  and omitted affine transforms; do not present simply enabling it as a fix.

## Measurements and acceptance

For every model/seed/variant/resolution: logit RMSE, reference-RMS-normalized
RMSE, maximum absolute error, argmax disagreement, per-layer saturation,
table and metadata bytes. Preserve per-example logits/predictions or a
reproducible deterministic generation route. Image stage additionally reports
paired accuracy changes, disagreements, class recalls and Wilson intervals;
paired bootstrap intervals concern image sampling, not training-seed variance.

Unit tests must cover a nonlinear interaction counterexample, affine offsets,
nonzero edge intercepts, SiLU, masks, clipping, unsupported graph elements,
and resolution convergence. Verify extraction against the source AST.
If a native C evaluator is implemented, compare its logits to Python on the
same inputs and report compiler/flags, CPU and numerical tolerance separately.

Go to a larger study if at 256 points the FP32 edge LUT gives at least a 10x
RMSE reduction versus the original 256-point whole-network LUT on at least
2 of 3 checkpoints for all three synthetic seeds, without hidden fallbacks.
For images seek <1% disagreement with the FP32 reference and report accuracy
without hiding a negative result. These are pilot gates, not journal claims.

## Boundaries and next decision

No MCU hardware is attached. CPU timing cannot establish MCU latency, energy,
RAM or an advantage over an MLP. A publishable study needs exact old artifacts,
a locked independent dataset, a matched MLP comparator and device measurements.
The old Micro operator, accuracy and deployment tables overlap with IEEE Access
(10.1109/ACCESS.2026.3693881) and OJIM (10.1109/OJIM.2026.3712887).
Per-edge LUT compilation overlaps with JSA (10.1016/j.sysarc.2026.103923),
Neurocomputing (10.1016/j.neucom.2026.133774), and the accepted IoT-J LUT-KAN
paper. Cite and delimit all of them. The possible new contribution is a measured
conversion-fidelity failure, its diagnosis, and a validated end-to-end remedy;
novelty and venue suitability remain to be established after the pilot.
