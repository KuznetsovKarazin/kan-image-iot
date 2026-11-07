"""Exploratory extension specified in FOLLOWUP_PROTOCOL.md after the primary run."""
import json
from pathlib import Path

import numpy as np
import torch

from fidelity import MODEL_SPECS, load_model, metrics
from run_images import accuracy_metrics

HERE=Path(__file__).resolve().parent


def margin_probability(y,ref):
    margin=y[:,1]-y[:,0]
    rm=ref[:,1]-ref[:,0]
    return {"margin_rmse":float((margin-rm).double().square().mean().sqrt()),
            "person_probability_rmse":float((margin.sigmoid()-rm.sigmoid()).double().square().mean().sqrt()),
            "person_probability_max_abs_error":float((margin.sigmoid()-rm.sigmoid()).abs().max())}


@torch.inference_mode()
def main():
    torch.set_num_threads(1)
    out=HERE/"results/anchored_control"
    out.mkdir(exist_ok=True)
    primary=json.loads((HERE/"results/images/metrics.json").read_text())
    rows=[]
    for model_id,topology,width in MODEL_SPECS:
        model,_=load_model(topology,width)
        d=model.feature_dim
        offset=model.kan(torch.zeros(1,d))
        arrays=np.load(HERE/f"results/images/{model_id}_logits.npz")
        reference=torch.from_numpy(arrays["reference"])
        labels=arrays["labels"]
        logits={"f_zero":offset.numpy()}
        for row in [r for r in primary if r["model"]==model_id]:
            y=torch.from_numpy(arrays[f"{row['variant']}_{row['points']}"])
            rows.append(dict(row,post_hoc_derived_variant=False,**margin_probability(y,reference)))
        for variant in ["legacy_whole","legacy_calibrated_domain"]:
            for points in [16,32,64,128,256]:
                old=torch.from_numpy(arrays[f"{variant}_{points}"])
                y=d*old-(d-1)*offset
                name="anchored_"+variant
                logits[f"{name}_{points}"]=y.numpy()
                rows.append(dict(model=model_id,variant=name,points=points,
                    post_hoc_derived_variant=True,**metrics(y,reference),
                    **accuracy_metrics(y,reference,labels),**margin_probability(y,reference),
                    table_bytes=d*points*2*4,metadata_bytes=8,
                    stored_bytes=d*points*2*4+8))
        np.savez_compressed(out/f"{model_id}_logits.npz",**logits)
    (out/"metrics.json").write_text(json.dumps(rows,indent=2))
    print('Post hoc derived variants:',sum(r['post_hoc_derived_variant'] for r in rows))
    for r in rows:
        if r['points'] in [0,32,256] and r['variant'] in ['exact_anchored_additive_control','anchored_legacy_calibrated_domain','edge_fp32','edge_int8_table']:
            print(r['model'],r['variant'],r['points'],'acc',round(r['accuracy_pct'],3),
                  'disagreement',r['disagreement_count'],'margin RMSE',round(r['margin_rmse'],5),
                  'probability RMSE',round(r['person_probability_rmse'],5))


if __name__=='__main__':
    main()
