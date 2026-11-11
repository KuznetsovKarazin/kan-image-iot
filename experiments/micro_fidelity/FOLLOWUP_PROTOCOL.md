# Post hoc diagnostic extension (2026-11-07)

This extension was specified after completing the frozen image stage. The exact
anchored additive control agreed with the FP32 classes on 1024/1024, 1024/1024
and 1022/1024 images, despite large raw-logit errors. This means preserving
layer composition may not be necessary to retain classification decisions on
this particular subset. This cheaper alternative is therefore reported as well.

Using only already saved logits and original f(0), evaluate
`d * legacy_LUT(x) - (d-1) * f(0)` for both the original [-2,2] and the
calibration-derived global domains at every frozen resolution. No fitting or
test-label-dependent parameter choice. Store these post hoc results separately.

Also report binary margin RMSE and probability RMSE for the original controls,
edge LUTs and anchored LUTs. Raw logit RMSE can include a class-common offset
that leaves softmax/argmax unchanged; probability fidelity must be measured
before suggesting that large logit errors necessarily damage confidence.

This is an exploratory diagnostic, not an additional preregistered endpoint.
It does not establish probability calibration to ground truth or prove
noninferiority of an approximate classifier on future data.
