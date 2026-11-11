"""Recompute decision quality and descriptive speed comparisons from audited captures."""
import argparse,json,statistics
from pathlib import Path
from collections import defaultdict
from serial_common import decode
from audit_v2 import audit_batch
ROOT=Path(__file__).resolve().parents[1]

def label(bits):
    x=decode(bits);return int(x[1]>x[0])

def metrics(truth,prediction):
    if len(truth)!=len(prediction) or not truth:raise ValueError('Missing/mismatched predictions')
    cm=[[0,0],[0,0]]
    for t,p in zip(truth,prediction):
        if t not in [0,1] or p not in [0,1]:raise ValueError('Nonbinary class')
        cm[t][p]+=1
    recalls=[cm[k][k]/sum(cm[k]) for k in [0,1]]
    f1=[]
    for k in [0,1]:
        tp=cm[k][k];den=2*tp+cm[1-k][k]+cm[k][1-k]
        f1.append(2*tp/den if den else 0.)
    return dict(n=len(truth),correct=cm[0][0]+cm[1][1],accuracy=(cm[0][0]+cm[1][1])/len(truth),
                balanced_accuracy=sum(recalls)/2,macro_f1=sum(f1)/2,recall_class0=recalls[0],recall_class1=recalls[1],
                confusion_rows_true_columns_predicted=cm)

def paired(truth,baseline,candidate):
    return dict(n=len(truth),changed_decisions=sum(a!=b for a,b in zip(baseline,candidate)),
                candidate_only_correct=sum(a!=t and b==t for t,a,b in zip(truth,baseline,candidate)),
                baseline_only_correct=sum(a==t and b!=t for t,a,b in zip(truth,baseline,candidate)),
                both_correct=sum(a==t and b==t for t,a,b in zip(truth,baseline,candidate)),
                both_wrong=sum(a!=t and b!=t for t,a,b in zip(truth,baseline,candidate)))

def quality(g,device=None):
    records=g['records'];truth=[r['label'] for r in records]
    if len(records)!=1024 or len({r['image_id'] for r in records})!=1024:raise ValueError('Evaluation IDs')
    predictions={'kan_original_fp32':[label(r['kan_reference_bits']) for r in records]}
    for v,name in enumerate(['mlp_fp32','mlp_int8']):
        if device is not None and v not in device:continue
        predictions[name]=[label(device[v][r['image_id']] if device is not None else r['expected_bits'][v]) for r in records]
    return dict(case=g['case'],model=g['model'],hardware_seed=101,
                scope='Single predetermined seed on reused 1024-image evaluation set; KAN historical training budget/split unavailable',
                methods={name:metrics(truth,p) for name,p in predictions.items()},
                paired_vs_kan={name:paired(truth,predictions['kan_original_fp32'],p) for name,p in predictions.items() if name.startswith('mlp')},
                int8_vs_fp32=paired(truth,predictions['mlp_fp32'],predictions['mlp_int8']) if len(predictions)==3 else None)

def analyze(folder):
    receipt=audit_batch(folder);state=json.loads((folder/'batch.json').read_text())
    if receipt['status']!='PASS':raise ValueError('Complete audited batch required')
    groups=defaultdict(list);captures=[]
    for slot,entry in state['completed'].items():
        path=folder/entry['directory']/'capture';s=json.loads((path/'summary.json').read_text())
        g=json.loads((ROOT/f'esp32/generated/model{s["case"]}_golden.json').read_text())
        device=defaultdict(dict)
        for line in (path/'predictions.jsonl').read_text().splitlines():
            r=json.loads(line);device[r['variant']][r['image_id']]=r['bits']
        captures.append(dict(slot=slot,quality=quality(g,device)))
        for row in s['latency']:
            if row['variant']<2:groups[(s['case'],row['variant'])].append(row)
    baseline=json.loads((ROOT/'evidence/kan_v2_baseline.json').read_text())
    comparisons=[];timing=[]
    for (case,v),sessions in sorted(groups.items()):
        us=statistics.median(r['median_us'] for r in sessions)
        row=dict(case=case,variant=v,name=['mlp_fp32','mlp_int8'][v],sessions=len(sessions),
                 median_of_session_medians_us=us,min_session_median_us=min(r['median_us'] for r in sessions),
                 max_session_median_us=max(r['median_us'] for r in sessions),throughput_per_s=1e6/us,
                 parameter_array_bytes=json.loads((ROOT/f'esp32/generated/model{case}_golden.json').read_text())['parameter_bytes'][v])
        if all('median_cycles' in r for r in sessions):row['median_of_session_medians_cycles']=statistics.median(r['median_cycles'] for r in sessions)
        timing.append(row)
        for old in baseline['rows']:
            if old['target']!=state['target'] or old['board_id']!=state['board_id'] or old['case']!=case:continue
            for kv,k in old['variants'].items():
                if int(kv) not in [0,4,5,8,9]:continue
                comparisons.append(dict(case=case,kan_variant=int(kv),kan_name=k['name'],mlp_variant=v,
                                        kan_latency_us=k['median_us'],mlp_latency_us=us,kan_over_mlp_latency=k['median_us']/us,
                                        kan_over_mlp_parameter_bytes=k['parameter_bytes']/row['parameter_array_bytes'],
                                        scope='Descriptive ratio of aggregate medians from different acquisition series; no paired significance claim'))
    return dict(status='AUDITED_MLP_INFERENCE',physical_execution=True,target=state['target'],board_id=state['board_id'],
                full_three_session_protocol=state['repetitions']==3,quality_by_capture=captures,latency=timing,
                comparisons_with_earlier_kan=comparisons,
                notes=['Accuracy repeated across resets is not additional independent test data.',
                       'The timing boundary is frozen feature vector to two head logits, including INT8 boundary conversion.',
                       'Parameter arrays exclude firmware/stack; ESP links both MLP formats, Mega one format.',
                       'Whole-board energy is a separate externally measured outcome; throughput does not determine it.'],
                audit=receipt)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batch',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    result=analyze(a.batch) if a.batch else dict(status='HOST_EXPECTED_ONLY',physical_execution=False,
        cases=[quality(json.loads((ROOT/f'esp32/generated/model{c}_golden.json').read_text())) for c in range(3)])
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(result['status'],a.out)

if __name__=='__main__':main()
