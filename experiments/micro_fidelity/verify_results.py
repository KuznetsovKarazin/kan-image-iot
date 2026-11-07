"""Recalculate saved-output metrics independently with NumPy/SciPy."""
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
from scipy.special import expit
import torch

from fidelity import MODEL_SPECS, load_model

HERE = Path(__file__).resolve().parent


def check_metrics(row, y, ref):
    delta = y.astype(np.float64) - ref.astype(np.float64)
    rmse = float(np.sqrt(np.mean(delta**2)))
    rms = float(np.sqrt(np.mean(ref.astype(np.float64)**2)))
    disagreements = int(np.sum(y.argmax(1) != ref.argmax(1)))
    assert np.isclose(row["rmse"], rmse, atol=1e-12, rtol=1e-6)
    assert np.isclose(row["normalized_rmse"], rmse/max(rms,1e-12), atol=1e-12, rtol=1e-6)
    assert np.isclose(row["max_abs_error"], np.max(np.abs(delta)), atol=1e-12, rtol=1e-6)
    assert row["disagreement_count"] == disagreements
    assert np.isclose(row["disagreement_pct"],100*disagreements/len(y),atol=1e-5)
    assert np.isfinite(y).all() and np.isfinite(ref).all()
    return disagreements


def main():
    torch.set_num_threads(1)
    sdir, idir = HERE/"results/synthetic", HERE/"results/images"
    synthetic = json.loads((sdir/"metrics.json").read_text())
    images = json.loads((idir/"metrics.json").read_text())
    checked_synthetic = 0
    for model_id, topology, width in MODEL_SPECS:
        model, _ = load_model(topology,width)
        with torch.inference_mode():
            offset = model.kan(torch.zeros(1, model.feature_dim)).numpy()
        for seed in [42,43,44]:
            arrays = np.load(sdir/f"{model_id}_seed{seed}_logits.npz")
            for row in [r for r in synthetic if r["model"]==model_id and r["seed"]==seed]:
                v, d, g = row["variant"],row["dataset"],row["points"]
                if v=="exact_anchored_additive_control":
                    y=model.feature_dim*arrays["in_range_exact_axis_mean"]-(model.feature_dim-1)*offset
                elif v=="exact_axis_mean":
                    y=arrays["in_range_exact_axis_mean"]
                else:
                    y=arrays[f"{d}_{v}_{g}"]
                check_metrics(row,y,arrays[f"{d}_reference"])
                checked_synthetic+=1
    checked_images=0
    for model_id, _, _ in MODEL_SPECS:
        arrays=np.load(idir/f"{model_id}_logits.npz")
        labels=arrays["labels"]
        ref=arrays["reference"]
        for row in [r for r in images if r["model"]==model_id]:
            y=arrays[f"{row['variant']}_{row['points']}"]
            check_metrics(row,y,ref)
            correct=y.argmax(1)==labels
            baseline=ref.argmax(1)==labels
            assert row["correct"]==int(correct.sum())
            assert row["reference_wrong_fixed"]==int((correct&~baseline).sum())
            assert row["reference_correct_broken"]==int((~correct&baseline).sum())
            assert np.isclose(row["accuracy_delta_pp"],100*(correct.mean()-baseline.mean()))
            ci=binomtest(int(correct.sum()),len(labels)).proportion_ci(method="wilson")
            assert np.allclose(row["accuracy_wilson95_pct"],np.array(ci)*100,atol=1e-10)
            checked_images+=1
        features=np.load(idir/f"{model_id}_features.npz")
        assert features["calibration"].sum()==256
        assert len(set(features["image_ids"].tolist()))==1280
        assert np.array_equal(features["image_ids"][~features["calibration"]],arrays["image_ids"])
    gate=[]
    for model_id, _, _ in MODEL_SPECS:
        ratios=[]
        for seed in [42,43,44]:
            def metric(v):
                return next(r["rmse"] for r in synthetic if r["model"]==model_id and
                    r["seed"]==seed and r["dataset"]=="in_range" and r["points"]==256 and r["variant"]==v)
            ratios.append(metric("legacy_whole")/metric("edge_fp32"))
        gate.append({"model":model_id,"rmse_reduction_ratios":ratios,"passes_all_seeds":min(ratios)>=10})
    result={"status":"PASS","synthetic_rows_recalculated":checked_synthetic,
            "image_rows_recalculated":checked_images,"synthetic_gate":gate,
            "synthetic_go":sum(g["passes_all_seeds"] for g in gate)>=2,
            "scope":"saved-logit fidelity, class decisions, paired counts, Wilson CIs, image-ID separation"}
    posthoc=json.loads((HERE/"results/anchored_control/metrics.json").read_text())
    checked_posthoc=0
    for model_id,_,_ in MODEL_SPECS:
        arrays=np.load(HERE/f"results/anchored_control/{model_id}_logits.npz")
        original=np.load(idir/f"{model_id}_logits.npz")
        ref=original["reference"]
        for row in [r for r in posthoc if r["model"]==model_id]:
            key=f"{row['variant']}_{row['points']}"
            y=arrays[key] if row["post_hoc_derived_variant"] else original[key]
            check_metrics(row,y,ref)
            # Use the same FP32 logits, independent NumPy/SciPy probability arithmetic.
            margin=y[:,1]-y[:,0]
            rm=ref[:,1]-ref[:,0]
            prob_rmse=np.sqrt(np.mean((expit(margin).astype(float)-expit(rm).astype(float))**2))
            assert np.isclose(row["person_probability_rmse"],prob_rmse,rtol=1e-3,atol=2e-8)
            checked_posthoc+=int(row["post_hoc_derived_variant"])
    result["posthoc_derived_rows_recalculated"]=checked_posthoc
    (HERE/"results/verification.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
