"""Compile actual ESP kernels on host and check all three frozen cases.
This verifies numerics and RAM-copy equivalence, not device latency.
"""
from pathlib import Path
import ctypes
import hashlib
import json
import re
import subprocess
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
E=ROOT/'esp32';BUILD=ROOT/'host_build';BUILD.mkdir(exist_ok=True)
FLAGS=['-O3','-ffp-contract=off','-fno-fast-math','-shared','-fPIC','-Wall','-Wextra','-Werror']
PTR=ctypes.POINTER(ctypes.c_float)


def decode(words):
    return np.asarray([[int(v,16) for v in row] for row in words],dtype=np.uint32).view(np.float32)


def main():
    results=[]
    for case in range(3):
        golden=json.loads((E/f'generated/model{case}_golden.json').read_text())
        dim=golden['input_dim'];n=len(golden['records'])
        evaluation=decode([r['input_bits'] for r in golden['records']])
        model=(E/f'generated/model{case}.inc').read_text()
        body=re.search(r'static const float d0_grid\[\]=\{(.*?)\};',model,re.S).group(1)
        knots=np.asarray([float(v.strip().removesuffix('f')) for v in body.split(',') if v.strip()][:12],dtype='float32')
        values=np.concatenate([knots,np.nextafter(knots,-np.inf),np.nextafter(knots,np.inf),[-10.,10.]]).astype('float32')
        boundary=np.concatenate([np.repeat(values[:,None],dim,axis=1),
                    np.asarray([np.eye(dim,dtype='float32')[i]*v for v in values for i in range(dim)])])
        stress=np.random.default_rng(20261109+case).uniform(-4,4,(10000,dim)).astype('float32')
        x=np.ascontiguousarray(np.concatenate([evaluation,stress,boundary]))
        so=BUILD/f'case{case}.so'
        subprocess.run(['gcc',*FLAGS,f'-DMODEL_CASE={case}','-I'+str(E/'include'),'-I'+str(E/'generated'),
                        str(E/'src/kernels.c'),'-lm','-o',str(so)],check=True)
        lib=ctypes.CDLL(str(so));lib.micro_batch.argtypes=[ctypes.c_uint,PTR,PTR,ctypes.c_uint]
        assert lib.micro_init_dram()==1 and lib.micro_init_dram()==1
        for field,expected in [('micro_dram_bytes',golden['dram_total_bytes']),
                               ('micro_dram_fp32_bytes',golden['dram_fp32_bytes']),
                               ('micro_dram_int8_bytes',golden['dram_int8_bytes'])]:
            assert ctypes.c_uint.in_dll(lib,field).value==expected
        outputs={};eval_rows=[]
        for v in [0,4,5,8,9,10,11]:
            y=np.zeros((len(x),2),dtype='float32')
            assert lib.micro_batch(v,x.ctypes.data_as(PTR),y.ctypes.data_as(PTR),len(x))==0
            ref=decode([r['expected_bits'][v] for r in golden['records']]);err=np.abs(y[:n]-ref)
            assert np.all(err<=2e-5+2e-5*np.abs(ref)), (case,v,float(err.max()))
            assert np.array_equal(y[:n,1]>y[:n,0],ref[:,1]>ref[:,0]), (case,v,'golden decisions')
            outputs[v]=y
            eval_rows.append(dict(variant=v,n=n,max_abs_logit_error=float(err.max()),decision_changes=0))
        pairs=[]
        for v,base in [(8,0),(9,5),(10,4),(11,9)]:
            y,ref=outputs[v],outputs[base];err=np.abs(y-ref)
            assert np.all(np.isfinite(y)) and np.all(err<=2e-5+2e-5*np.abs(ref)),(case,v,'stress tolerance')
            changed=int(np.count_nonzero((y[:,1]>y[:,0])!=(ref[:,1]>ref[:,0])))
            assert changed==0,(case,v,'stress decisions',changed)
            if v in [10,11]:assert np.array_equal(y.view(np.uint32),ref.view(np.uint32))
            pairs.append(dict(variant=v,reference=base,n=len(x),max_abs_logit_error=float(err.max()),
                              changed_decisions=changed,bit_identical_vectors=int(np.all(y.view(np.uint32)==ref.view(np.uint32),axis=1).sum())))
        mega_rows=[]
        if case in [1,2]:
            for v in [0,4,5,8,9]:
                gen=ROOT/f'mega/generated/case{case}_v{v}';mso=BUILD/f'mega_case{case}_v{v}.so'
                subprocess.run(['gcc',*FLAGS,'-I'+str(ROOT/'mega/src'),'-I'+str(gen),'-include',str(gen/'config.h'),
                                str(ROOT/'mega/src/kernels.c'),'-lm','-o',str(mso)],check=True)
                mlib=ctypes.CDLL(str(mso));mlib.infer_batch.argtypes=[PTR,PTR,ctypes.c_uint32]
                y=np.zeros((len(x),2),dtype='float32')
                assert mlib.infer_batch(x.ctypes.data_as(PTR),y.ctypes.data_as(PTR),len(x))==0
                ref=outputs[v];err=np.abs(y-ref)
                assert np.all(err<=2e-5+2e-5*np.abs(ref))
                assert np.array_equal(y[:,1]>y[:,0],ref[:,1]>ref[:,0])
                mega_rows.append(dict(variant=v,n=len(x),max_abs_logit_error=float(err.max()),changed_decisions=0))
        results.append(dict(case=case,model=golden['model'],evaluation=n,stress=len(stress),boundary=len(boundary),
                            ram_copy_bytes=golden['dram_total_bytes'],golden_checks=eval_rows,pairs=pairs,mega_port_checks=mega_rows))
        print('PASS case',case,'vectors',len(x),'both RAM copies bit-identical to flash',flush=True)
    report=dict(status='PASS_HOST_NUMERICAL',physical_execution=False,compiler=subprocess.check_output(['gcc','--version'],text=True).splitlines()[0],
                 compiler_flags=FLAGS,tolerance=dict(atol=2e-5,rtol=2e-5),cases=results,
                 source_sha256=hashlib.sha256((E/'src/kernels.c').read_bytes()).hexdigest())
    (ROOT/'evidence/v2/host_numeric.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
