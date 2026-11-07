"""Export a host-verified C head and binary golden vectors for later MCU testing."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from fidelity import MODEL_SPECS, EdgeLUT, evaluate, load_model
from native import validate_native

HERE=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',choices=[s[0] for s in MODEL_SPECS],required=True)
    p.add_argument('--points',type=int,choices=[16,32,64,128,256],default=256)
    p.add_argument('--out',type=Path,required=True,help='Build directory outside the Git checkout')
    a=p.parse_args()
    torch.set_num_threads(1)
    _,topology,width=next(s for s in MODEL_SPECS if s[0]==a.model)
    model,meta=load_model(topology,width)
    idir=HERE/'results/images'
    saved=json.loads((idir/'run.json').read_text())['models'][a.model]
    if meta['sha256']!=saved['sha256']:
        raise RuntimeError('Checkpoint differs from frozen pilot')
    features=np.load(idir/f'{a.model}_features.npz')
    xx=np.ascontiguousarray(features['features'][~features['calibration']],dtype=np.float32)
    ranges=json.loads((idir/f'{a.model}_ranges.json').read_text())
    bounds=[(torch.tensor(r['low'],dtype=torch.float32),torch.tensor(r['high'],dtype=torch.float32)) for r in ranges]
    lut=EdgeLUT(model.kan,bounds,a.points,True).eval()
    x=torch.from_numpy(xx)
    expected,_=evaluate(lut,x)
    cy,check=validate_native(lut,x,expected,a.out)
    header=('/* Generated head-only API. INT8 tables; FP32 arithmetic. */\n'
            '#pragma once\n#ifdef __cplusplus\nextern "C" {\n#endif\n'
            'void infer(const float *x, float *y);\n'
            'void infer_batch(const float *x, float *y, int n);\n'
            '#ifdef __cplusplus\n}\n#endif\n')
    (a.out/'edge.h').write_text(header)
    xx.astype('<f4').tofile(a.out/'golden_inputs_f32le.bin')
    expected.numpy().astype('<f4').tofile(a.out/'golden_logits_f32le.bin')
    (a.out/'golden_ids.json').write_text(json.dumps(features['image_ids'][~features['calibration']].tolist()))
    (a.out/'export.json').write_text(json.dumps(dict(model=a.model,points=a.points,
        input_dim=xx.shape[1],output_dim=expected.shape[1],samples=len(xx),
        binary_format='row-major little-endian IEEE754 float32',checkpoint=meta['sha256'],
        storage=lut.storage(),host_validation=check,
        scope='FP32-backbone feature input, head-only C; MCU execution remains untested'),indent=2))
    print('Exported and host-verified:',a.out)


if __name__=='__main__':
    main()
