"""Run PLAN.md Stage A. Synthetic disagreement is not image accuracy."""
import argparse
import datetime
import json
import platform
from pathlib import Path
import subprocess
import time

import torch
import numpy as np

from fidelity import (BASE, ROOT, MODEL_SPECS, EdgeLUT, batched, calibrate, evaluate,
                      exact_axis_mean, historical_classes, load_model, metrics,
                      quiet_construct, sha256)


def run(out, n):
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    old, old_layer = historical_classes()
    rows, models = [], {}
    started = time.time()
    for model_id, topology, width in MODEL_SPECS:
        model, meta = load_model(topology, width)
        models[model_id] = meta
        head, d = model.kan, model.feature_dim
        for seed in [42, 43, 44]:
            g = torch.Generator().manual_seed(seed)
            cal = torch.rand(n, d, generator=g) * 2
            x = torch.rand(n, d, generator=g) * 2
            stress = torch.rand(n, d, generator=g) * 4
            bounds = calibrate(head, cal)
            refs = {"in_range": batched(head, x), "stress": batched(head, stress)}
            predictions = {"in_range_reference": refs["in_range"].numpy(),
                           "stress_reference": refs["stress"].numpy()}
            exact = exact_axis_mean(head, x)
            rows.append(dict(model=model_id, seed=seed, dataset="in_range",
                variant="exact_axis_mean", points=0, **metrics(exact, refs["in_range"])))
            predictions["in_range_exact_axis_mean"] = exact.numpy()
            zero = torch.zeros(1, d)
            anchored = d * exact - (d - 1) * head(zero)
            rows.append(dict(model=model_id, seed=seed, dataset="in_range",
                variant="exact_anchored_additive_control", points=0,
                **metrics(anchored, refs["in_range"])))
            for points in [16, 32, 64, 128, 256]:
                variants = {
                    "legacy_whole": quiet_construct(old, head, grid_range=[-2, 2], grid_size=points),
                    "edge_fp32": EdgeLUT(head, bounds, points).eval(),
                    "edge_int8_table": EdgeLUT(head, bounds, points, True).eval(),
                }
                for variant, compiled in variants.items():
                    for dataset, data in [("in_range", x), ("stress", stress)]:
                        y, sat = evaluate(compiled, data)
                        storage = compiled.storage() if isinstance(compiled, EdgeLUT) else {
                            "table_bytes": compiled.lut_table.numel() * 4,
                            "metadata_bytes": 0, "stored_bytes": compiled.lut_table.numel() * 4}
                        rows.append(dict(model=model_id, seed=seed, dataset=dataset,
                            variant=variant, points=points, **metrics(y, refs[dataset]),
                            **sat, **storage))
                        predictions[f"{dataset}_{variant}_{points}"] = y.detach().numpy()
                if points == 32:
                    legacy_deep = torch.nn.Sequential(*[
                        quiet_construct(old_layer, layer, lo, hi, grid_size=points)
                        for layer, (lo, hi) in zip(head.kan.act_fun, bounds)])
                    y = batched(legacy_deep, x)
                    rows.append(dict(model=model_id, seed=seed, dataset="in_range",
                        variant="legacy_layerwise", points=points,
                        **metrics(y, refs["in_range"])))
                    predictions["in_range_legacy_layerwise_32"] = y.numpy()
            np.savez_compressed(out / f"{model_id}_seed{seed}_logits.npz", **predictions)
            (out / f"{model_id}_seed{seed}_ranges.json").write_text(json.dumps([
                {"low": lo.tolist(), "high": hi.tolist()} for lo, hi in bounds], indent=2))
            print(f"Completed {model_id} seed={seed}; {time.time()-started:.1f}s", flush=True)
            # Incremental checkpoint in case a later stage fails.
            (out / "metrics.json").write_text(json.dumps(rows, indent=2))
    metadata = {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "base_commit": BASE, "torch": torch.__version__, "numpy": np.__version__,
        "python": platform.python_version(), "platform": platform.platform(),
        "torch_threads": 1, "n_calibration": n, "n_evaluation": n,
        "seeds": [42, 43, 44], "seeds_are": "synthetic input seeds, not training repetitions",
        "models": models, "converter_sha256": sha256(ROOT / "scripts/convert_to_tflite.py"),
        "fidelity_py_sha256": sha256(Path(__file__).with_name("fidelity.py")),
        "elapsed_seconds": time.time() - started,
        "cpu": subprocess.check_output(["lscpu"], text=True)}
    (out / "run.json").write_text(json.dumps(metadata, indent=2))
    print("Finished:", len(rows), "comparisons", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path(__file__).parent / "results/synthetic")
    p.add_argument("--n", type=int, default=4096)
    args = p.parse_args()
    run(args.out, args.n)
