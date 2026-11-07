# Frozen image-stage sampling protocol

Written before any image predictions on 2026-11-07. Use COCO 2014 validation
images whose IDs are in the repository's `scripts/mscoco_minival_ids.txt`.
Download the official annotation archive from the COCO S3 bucket. Define
person=1 exactly as the repository annotation generator: any category_id=1
annotation with `ann['area'] / (image.width * image.height) > 0.005`.
This uses the stored annotation area, not an independently recomputed bbox area.

Within each label, sort candidates by SHA-256 of `micro-pilot-20261107:<image_id>`.
Take the first 128 per label for calibration (256 total), then the next 512
per label for evaluation (1,024 total). Freeze the manifest before downloads
and inference. No substitutions based on predictions. If a file is unavailable,
report a blocked/partial stage; do not silently replace it.

Resize original RGB to 224x224 with Pillow LANCZOS, then JPEG encode/decode
at Pillow default quality (75), matching `prepare_dataset.py`'s intermediate
image save. Convert to tensor and normalize using ImageNet mean/std as in
`src/analyze.py`; its further Resize is a no-op at 224x224. Record library
versions and processed-image hashes. All variants share exactly the same
FP32 CNN/projector features, so this isolates head conversion, not a fully
converted CNN+KAN pipeline. No HardSigmoid or backbone quantization here.

Use the five predeclared LUT resolutions (16, 32, 64, 128, 256), the same 5%
calibration padding, legacy [-2,2], and both FP32 and INT8 stored edge tables.
Add a whole-head LUT with calibration-derived global bounds as a diagnostic
control, to distinguish domain truncation from lost layer composition.
Evaluate the unchanged existing layerwise converter at 32 points.

Report all model/resolution combinations, class recalls, prediction disagreements,
Wilson 95% accuracy intervals and paired accuracy differences. A 10,000-draw
stratified paired bootstrap (seed 20261107) preserves class balance; its interval
measures resampling of these images only. The primary image endpoint is
conversion disagreement at 256 points, not a selected accuracy maximum.

These minival images should be outside training under the repository's split
logic, but their possible use for checkpoint selection cannot be ruled out.
There is no historical per-checkpoint train/validation/test manifest in the
available artifacts. Therefore this is an exploratory paired conversion audit,
not an independent generalization benchmark or reproduction of the paper's
6,000-image test results.
