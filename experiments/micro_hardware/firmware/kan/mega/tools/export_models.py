"""Export existing constants to split PROGMEM arrays, without training or requantization."""
import gzip
import hashlib
import json
from pathlib import Path
import re
import struct

ROOT=Path(__file__).resolve().parents[1]
VARIANTS=[0,4,5,8,9]
CHUNK=2048

def digest(raw): return hashlib.sha256(raw).hexdigest()
def dump(p,d):p.write_text(json.dumps(d,indent=2)+'\n')

def export(case,variant):
    golden_raw=(ROOT/f'reference/model{case}_golden.json').read_bytes()
    g=json.loads(golden_raw)
    original=gzip.decompress((ROOT.parent/f'reference/model{case}.inc.gz').read_bytes()).decode()
    selected={8:0,9:5}.get(variant,variant)
    arrays={name:(kind,[v.strip() for v in values.split(',') if v.strip()])
            for kind,name,values in re.findall(r'static const (float|int8_t) (\w+)\[\]=\{\s*(.*?)\s*\};',original,re.S)}
    direct=re.search(r'static const DirectLayer direct_layers\[\]=\{(.*?)\};',original).group(1)
    rows=[x.split(',') for x in re.findall(r'\{([^{}]+)\}',direct)]
    assert [list(map(int,r[:2])) for r in rows]==([[32,24],[24,16],[16,2]] if case==2 else [[24,16],[16,2]])
    needed=[];tail='';model_depth=len(rows)
    if selected==0:
        needed=[n for r in rows for n in r[4:]]
        converted=['{'+','.join(r[:4]+['read_'+n for n in r[4:]])+'}' for r in rows]
        tail='static const DirectLayer direct_layers[]={'+','.join(converted)+'};\n'
    elif selected in [2,3]:
        name='whole256' if variant==2 else 'whole32cal'
        needed=[name]+(['f_zero'] if variant==3 else [])
        step=re.search(r'static const float '+name+r'_step=([^;]+);',original).group(1)
        low='-2.0f' if variant==2 else re.search(r'static const float whole32cal_low=([^;]+);',original).group(1)
        tail=f'#define WHOLE_POINTS {256 if variant==2 else 32}\n#define WHOLE_LOW ({low})\n#define WHOLE_STEP ({step})\n#define WHOLE_READ read_{name}\n'
    else:
        body=re.search(r'static const TableLayer table_variants\[4\]\[\d+\]=\{(.*?)\};',original).group(1)
        trows=[r.split(',') for r in re.findall(r'\{([^{}]+)\}',body)]
        rows=trows[(selected-4)*model_depth:(selected-4+1)*model_depth]
        needed=[n for r in rows for n in r[3:] if n!='NULL']
        converted=['{'+','.join(r[:3]+[('read_'+n if n!='NULL' else n) for n in r[3:]])+'}' for r in rows]
        tail='static const TableLayer table_layers[]={'+','.join(converted)+'};\n'
    needed=list(dict.fromkeys(needed));out='/* Frozen constants, split into bounded-size AVR flash objects. */\n'
    entries=[]
    for name in needed:
        kind,values=arrays[name]
        raw=b''.join(struct.pack('<f',float(x.removesuffix('f'))) if kind=='float' else struct.pack('b',int(x)) for x in values)
        entries.append(dict(name=name,type=kind,elements=len(values),bytes=len(raw),binary_sha256=digest(raw)))
        chunks=[values[i:i+CHUNK] for i in range(0,len(values),CHUNK)]
        for i,chunk in enumerate(chunks):
            out+=f'static const {kind} {name}_chunk{i}[] FLASH_STORAGE={{\n'
            out+='\n'.join(','.join(chunk[k:k+16])+(',' if k+16<len(chunk) else '') for k in range(0,len(chunk),16))+'\n};\n'
        out+=f'static {kind} read_{name}(uint32_t i) {{\n'
        if len(chunks)==1:
            out+=f'    return i<{len(values)}UL ? FLASH_{"FLOAT" if kind=="float" else "INT8"}({name}_chunk0,i) : '+('NAN' if kind=='float' else '0')+';\n'
        else:
            out+=f'    switch(i/{CHUNK}UL) {{\n'
            for i,chunk in enumerate(chunks):
                out+=f'    case {i}: return i%{CHUNK}UL<{len(chunk)}UL ? FLASH_{"FLOAT" if kind=="float" else "INT8"}({name}_chunk{i},i%{CHUNK}UL) : '+('NAN' if kind=='float' else '0')+';\n'
            out+='    default: return '+('NAN' if kind=='float' else '0')+';\n    }\n'
        out+='}\n'
    out+=tail
    target=ROOT/f'generated/case{case}_v{variant}';target.mkdir(parents=True,exist_ok=True)
    source_names=['src/kernel.h','src/kernels.c','src/main.c','src/startup.c','tools/export_models.py','tools/build.py']
    source_hashes={n:digest((ROOT/n).read_bytes()) for n in source_names}
    export=dict(case=case,variant=variant,model=g['model'],variant_name=g['variants'][variant],
                source_artifact=g['artifact'],original_source_sha256=digest(original.encode()),
                golden_sha256=digest(golden_raw),parameter_bytes=g['parameter_bytes'][variant],
                actual_exported_constant_bytes=sum(e['bytes'] for e in entries),arrays=entries,
                source_sha256=source_hashes,model_inc_sha256=digest(out.encode()))
    export['input_dim']=g['input_dim']
    if variant==8:
        cubic=(ROOT.parent/'esp32/generated/cubic.inc').read_text()
        cubic=cubic.replace('[]={','[] FLASH_STORAGE={')
        (target/'cubic.inc').write_text(cubic)
        export['cubic_inc_sha256']=digest(cubic.encode())
        export['actual_exported_constant_bytes']+=748
    assert export['actual_exported_constant_bytes']==export['parameter_bytes']
    export['artifact']=digest(json.dumps(export,sort_keys=True).encode())
    (target/'model.inc').write_text(out)
    config=(f'#define MODEL_CASE {case}\n#define MEGA_INPUTS {g["input_dim"]}\n#define SELECTED_VARIANT {selected}\n#define WIRE_VARIANT {variant}\n#define MODEL_DEPTH {model_depth}\n'
            f'#define MODEL_ID "{g["model"]}"\n#define VARIANT_NAME "{g["variants"][variant]}"\n'
            f'#define MODEL_ARTIFACT "{export["artifact"]}"\n#define SOURCE_ARTIFACT "{g["artifact"]}"\n'
            f'#define PARAMETER_BYTES {export["parameter_bytes"]}UL\n')
    if variant==8:config+='#define OPT_CUBIC 1\n'
    if variant==9:config+='#define OPT_Q_FACTOR 1\n'
    (target/'config.h').write_text(config)
    dump(target/'export.json',export)
    return export

def main():
    rows=[export(c,v) for c in [1,2] for v in VARIANTS]
    dump(ROOT/'results/export_checks.json',dict(status='PASS',scope='Exact selected constants, no retraining or requantization',exports=rows))
    print('Exported 10 single-variant C1/C2 images from frozen constants.')

if __name__=='__main__':main()
