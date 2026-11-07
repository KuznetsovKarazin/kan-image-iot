# Positioning the revised Micro article

The pilot investigates conversion fidelity. It does not introduce LUT-KAN,
claim the first KAN deployment on a microcontroller, or establish superiority
over an MLP. The whole-head converter already describes itself as an
approximation. Its failure to converge to the trained function is a property
of that approximation, not evidence of deliberate misreporting.

## Work to cite and separate from the new study

| Work | Existing contribution | Boundary for a revised Micro paper |
|---|---|---|
| Faggi et al., *Hybrid CNN-KAN Architectures for Ultra-Compact Person Detection on IoT Devices: A Systematic Evaluation of Quantization Strategies*, IEEE Access (2026), [DOI](https://doi.org/10.1109/ACCESS.2026.3693881) | CNN-KAN topology/quantization study, operator conversion, desktop/device measurements | Treat the old operator counts, classification tables and deployment measurements as prior results. Do not reuse them as the new paper's central evidence. |
| Faggi, Kuznetsov and Frontoni, *Sensor-System Characterization of CNN-KAN Person Detection on Dual-Tier IoT Platforms With Uncertainty Quantification*, OJIM (2026), [DOI](https://doi.org/10.1109/OJIM.2026.3712887) | Sensor-system measurements, per-channel runtime issue, uncertainty and deployment characterization | New evidence must isolate conversion stages and preserve paired per-image predictions; aggregate accuracy agreement is insufficient. |
| Kuznetsov, *LUT-KAN: Look-up table acceleration of Kolmogorov–Arnold networks on resource-constrained microcontrollers for embedded edge inference*, Journal of Systems Architecture (2026), [DOI](https://doi.org/10.1016/j.sysarc.2026.103923) | Per-edge tables, interpolation and microcontroller execution | Edge tabulation and code generation are prior methods, not the new contribution. |
| Kuznetsov, *LUT-KAN: An open software toolkit for LUT-compiled and quantized CPU inference of Kolmogorov–Arnold networks*, Neurocomputing 688 (2026), 133774, [DOI](https://doi.org/10.1016/j.neucom.2026.133774) | CPU toolkit, quantization and reproducibility | Do not frame this small C evaluator as a new general LUT toolkit. |
| Kuznetsov, *Look-Up Table Optimization for Kolmogorov-Arnold Networks on Microcontrollers*, IEEE Embedded Systems Letters (2026), [institutional record](https://iris.uniecampus.it/handle/11389/93090) | Integer lookup/interpolation on MCU-class processors | Distinguish the fidelity audit from previously published basis-evaluation optimization. |
| Kuznetsov, *LUT-Compiled Kolmogorov-Arnold Networks for Lightweight DoS Detection on IoT Edge Devices*, IEEE Internet of Things Journal, accepted 12 October 2026; [preprint](https://arxiv.org/abs/2601.08044) | LUT resolution/quantization trade-offs and IDS deployment | CNN-KAN conversion is a different case study; the underlying LUT mechanism is shared. Check publisher metadata before final bibliography export. |
| Hoang, Gupta and Harris, *KANELÉ: Kolmogorov–Arnold Networks for Efficient LUT-based Evaluation*, FPGA '26, pp. 44–55, [DOI](https://doi.org/10.1145/3748173.3779202) | Quantization-aware LUT mapping for FPGA | Include an external comparator in related work; no claim of a first LUT-based KAN design flow. |

## Candidate narrative, conditional on complete validation

Working title: **Preserving KAN Semantics from Training Graph to Embedded Inference**.

1. Show why a converter that produces a runnable small graph can still change
   the learned decision function. Derive the coordinate-section approximation
   and its irreducible interaction error; separate its mean/offset error with
   the anchored additive control.
2. Use the trained checkpoints, locked data and paired predictions to separate
   graph approximation, finite table resolution, clipping, table quantization
   and runtime execution. Report failed controls and memory costs.
3. Reuse established per-edge LUT methods while preserving layer composition,
   edge base functions/masks/scales and node/subnode affine transformations.
4. Carry precisely the same generated parameters and golden vectors to the
   physical MCU. Compare logits/decisions before comparing aggregate accuracy.
5. Report whole-system resource and energy costs and a matched MLP comparator.
   An accuracy result alone cannot establish a KAN deployment advantage.

## Mapping to the prior rejection

- Motivation and architecture: one clear CNN/projector/KAN/conversion figure,
  with a concrete failure and the resulting design rule.
- Novelty: new controlled fidelity evidence; explicitly distinguish all prior
  tables and the established LUT algorithm. A correction to one converter alone
  may still be insufficient for a new Micro article.
- Accuracy/quantization: fixed original checkpoints, common examples at every
  conversion stage, explicit error and saturation measurements.
- Hardware/performance: fresh MCU latency, memory and energy measurements with
  a matched MLP baseline are still required; this pilot provides no such claims.
- Breadth: the two head topologies here are a first gate. Add the exact old
  16–4–2 checkpoints and an independently trained case before generalizing.

A renewed submission should disclose Micro-2026-04-0015 and the earlier
publications, explaining the materially changed contribution. References do
not by themselves make repeated results a new contribution.
