"""Paired conversion audit on IMAGE_PROTOCOL.md's locked COCO subset."""
import argparse
import datetime
import json
import math
from pathlib import Path
import platform
import tempfile
import time

import numpy as np
from PIL import Image
import torch
import torchvision
from torchvision import transforms

from fidelity import (BASE, MODEL_SPECS, EdgeLUT, batched, calibrate, evaluate,
                      exact_axis_mean, historical_classes, load_model, metrics,
                      quiet_construct, sha256)
from native import validate_native


def wilson(k, n):
    z = 1.959963984540054
    p = k / n
    center = (p + z*z/(2*n)) / (1+z*z/n)
    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [100*(center-half), 100*(center+half)]


def accuracy_metrics(y, ref, labels):
    pred, base = y.argmax(1).numpy(), ref.argmax(1).numpy()
    correct, base_correct = pred == labels, base == labels
    diff = correct.astype(int) - base_correct.astype(int)
    rng = np.random.default_rng(20261107)
    draws = np.zeros(10000)
    for label in [0, 1]:
        values = diff[labels == label]
        counts = np.array([(values == v).sum() for v in [-1, 0, 1]])
        sampled = rng.multinomial(len(values), counts/len(values), size=10000)
        draws += sampled[:, 2] - sampled[:, 0]
    return {"accuracy_pct": float(correct.mean()*100), "correct": int(correct.sum()),
            "accuracy_wilson95_pct": wilson(int(correct.sum()), len(labels)),
            "accuracy_delta_pp": float(diff.mean()*100),
            "delta_paired_stratified_bootstrap95_pp": (np.quantile(draws/len(labels)*100, [.025,.975])).tolist(),
            "reference_wrong_fixed": int((diff==1).sum()),
            "reference_correct_broken": int((diff==-1).sum()),
            "recall_no_person_pct": float((pred[labels==0]==0).mean()*100),
            "recall_person_pct": float((pred[labels==1]==1).mean()*100)}


@torch.inference_mode()
def extract_features(model, records, folder):
    transform = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    outputs = []
    for start in range(0, len(records), 32):
        images = []
        for rec in records[start:start+32]:
            path = folder / rec["filename"]
            if sha256(path) != rec["processed_sha256"]:
                raise RuntimeError("Processed-image hash mismatch: " + str(path))
            with Image.open(path) as im:
                images.append(transform(im.convert("RGB")))
        outputs.append(model.preprocessor(torch.stack(images)))
    return torch.cat(outputs)


def run(cache, out):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    manifest = json.loads((out / "manifest.json").read_text())
    cal_mask = np.array([r["split"] == "calibration" for r in manifest])
    labels = np.array([r["label"] for r in manifest])[~cal_mask]
    assert cal_mask.sum() == 256 and len(labels) == 1024
    assert len({r["image_id"] for r in manifest}) == 1280
    old, old_layer = historical_classes()
    rows, models, native_checks = [], {}, {}
    started = time.time()
    for model_id, topology, width in MODEL_SPECS:
        model, meta = load_model(topology, width)
        models[model_id] = meta
        features = extract_features(model, manifest, cache / "processed224")
        cal, x = features[cal_mask], features[~cal_mask]
        np.savez_compressed(out / f"{model_id}_features.npz", features=features.numpy(),
            image_ids=[r["image_id"] for r in manifest], labels=[r["label"] for r in manifest],
            calibration=cal_mask)
        head = model.kan
        bounds = calibrate(head, cal)
        (out / f"{model_id}_ranges.json").write_text(json.dumps([
            {"low": lo.tolist(), "high": hi.tolist()} for lo, hi in bounds], indent=2))
        reference = batched(head, x)
        logits = {"reference": reference.numpy(), "labels": labels,
                  "image_ids": np.array([r["image_id"] for r in manifest])[~cal_mask]}

        def record(name, points, y, extra=None):
            rows.append(dict(model=model_id, variant=name, points=points,
                **metrics(y, reference), **accuracy_metrics(y, reference, labels), **(extra or {})))
            logits[f"{name}_{points}"] = y.detach().numpy()

        record("reference", 0, reference)
        exact = exact_axis_mean(head, x)
        record("exact_axis_mean", 0, exact)
        with torch.inference_mode():
            anchored = x.shape[1]*exact - (x.shape[1]-1)*head(torch.zeros(1, x.shape[1]))
        record("exact_anchored_additive_control", 0, anchored)
        global_bounds = [float(bounds[0][0].min()), float(bounds[0][1].max())]
        coverage = {"features_outside_legacy_domain_pct":
                    float(((x < -2) | (x > 2)).float().mean()*100),
                    "images_with_feature_outside_legacy_domain_pct":
                    float(((x < -2) | (x > 2)).any(1).float().mean()*100),
                    "calibrated_global_bounds": global_bounds}
        models[model_id]["feature_coverage"] = coverage
        for points in [16, 32, 64, 128, 256]:
            variants = {
                "legacy_whole": quiet_construct(old, head, grid_range=[-2,2], grid_size=points),
                "legacy_calibrated_domain": quiet_construct(old, head, grid_range=global_bounds, grid_size=points),
                "edge_fp32": EdgeLUT(head, bounds, points).eval(),
                "edge_int8_table": EdgeLUT(head, bounds, points, True).eval(),
            }
            for name, compiled in variants.items():
                y, saturation = evaluate(compiled, x)
                if isinstance(compiled, EdgeLUT):
                    storage = compiled.storage()
                else:
                    storage = {"table_bytes": compiled.lut_table.numel()*4,
                               "metadata_bytes":0, "stored_bytes":compiled.lut_table.numel()*4}
                record(name, points, y, dict(**saturation, **storage))
                if name == "edge_int8_table" and points == 256:
                    with tempfile.TemporaryDirectory(prefix="micro-lut-c-") as temp:
                        cy, check = validate_native(compiled, x, y, temp)
                    native_checks[model_id] = check
                    record("native_c_int8_table", points, cy, storage)
            if points == 32:
                legacy_deep = torch.nn.Sequential(*[
                    quiet_construct(old_layer, layer, lo, hi, grid_size=points)
                    for layer, (lo,hi) in zip(head.kan.act_fun, bounds)])
                record("legacy_layerwise", points, batched(legacy_deep, x))
        np.savez_compressed(out / f"{model_id}_logits.npz", **logits)
        (out / "metrics.json").write_text(json.dumps(rows, indent=2))
        (out / "native_checks.json").write_text(json.dumps(native_checks, indent=2))
        print(f"Completed images: {model_id}; {time.time()-started:.1f}s", flush=True)
    run_meta = {"utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "base_commit":BASE, "torch":torch.__version__, "torchvision":torchvision.__version__,
        "numpy":np.__version__, "python":platform.python_version(), "torch_threads":1,
        "manifest_sha256":sha256(out / "manifest.json"), "models":models,
        "elapsed_seconds":time.time()-started,
        "scope":"head conversion on shared FP32 backbone features; exploratory minival subset"}
    (out / "run.json").write_text(json.dumps(run_meta, indent=2))
    print("Image stage complete:",len(rows),"comparisons",flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path(__file__).parent / "results/images")
    a = p.parse_args()
    run(a.cache, a.out)
