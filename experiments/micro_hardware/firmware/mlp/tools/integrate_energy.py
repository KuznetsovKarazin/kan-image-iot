"""Integrate an explicitly aligned, converted power trace; never infer power from latency.

Analysis CSV columns: time_s,voltage_v,current_a (SI units). This is an internal
interchange schema, not a claim that FNIRSI software exports this CSV format.
Retain the native meter file and conversion/alignment evidence; see ENERGY_RU.txt.
"""
import argparse,bisect,csv,json,math
from pathlib import Path
from audit_energy import audit
from run_mlp import must,sha

def read_power(path):
    rows=[]
    with path.open(newline='',encoding='utf-8-sig') as f:
        reader=csv.DictReader(f)
        must(reader.fieldnames==['time_s','voltage_v','current_a'],'Explicit SI CSV columns required')
        for r in reader:
            t,v,i=[float(r[k]) for k in reader.fieldnames]
            must(all(math.isfinite(x) for x in [t,v,i]) and 0<v<=5.5 and i>=0,'Invalid/non-5V power sample')
            must(not rows or t>rows[-1][0],'Power timestamps not strictly increasing')
            rows.append((t,v*i))
    must(len(rows)>=2,'Empty power trace');return rows

def integrate(rows,start,end,max_gap):
    times=[r[0] for r in rows]
    must(rows[0][0]<=start<end<=rows[-1][0],'Power trace does not cover window')
    left=max(0,bisect.bisect_right(times,start)-1);right=bisect.bisect_left(times,end)
    selected=rows[left:right+1]
    must(len(selected)>=2 and max(b[0]-a[0] for a,b in zip(selected,selected[1:]))<=max_gap,'Power sampling gap too large')
    def value(t):
        j=min(len(rows)-2,max(0,bisect.bisect_right(times,t)-1));a,b=rows[j:j+2]
        return a[1]+(b[1]-a[1])*(t-a[0])/(b[0]-a[0])
    points=[(start,value(start))]+[(t,p) for t,p in selected if start<t<end]+[(end,value(end))]
    joules=sum((b[0]-a[0])*(a[1]+b[1])/2 for a,b in zip(points,points[1:]))
    return dict(joules=joules,mean_power_w=joules/(end-start),max_power_w=max(p for _,p in points),duration_s=end-start)

def analyze(capture,power,alignment):
    receipt=audit(capture);a=json.loads(alignment.read_text());rows=read_power(power)
    must(a.get('physical_measurement') is True,'Physical power trace declaration required')
    for k in ['meter_model','meter_serial','conversion_description','alignment_method','calibration_note']:
        must(isinstance(a.get(k),str) and bool(a[k].strip()),'Missing provenance: '+k)
    original=(alignment.parent/a['source_recording_path']).resolve()
    must(original.is_file() and sha(original)==a['source_recording_sha256'],'Native recording SHA mismatch')
    must(sha(power)==a['power_csv_sha256'],'Converted power CSV SHA mismatch')
    gap=float(a['max_sample_gap_s']);unc=float(a['boundary_uncertainty_s'])
    must(math.isfinite(gap) and 0<gap<=.1 and math.isfinite(unc) and 0<unc<=.1,'Explicit gap/boundary bounds required; unresolved alignment is not usable')
    mapping=a['windows'];must(len(mapping)==len(receipt['windows']),'Alignment window count')
    result=[];previous_end=None
    for w,position in zip(receipt['windows'],mapping):
        must(position['seq']==w['seq'],'Alignment sequence identity')
        start,end=float(position['start_s']),float(position['end_s'])
        must(previous_end is None or start>=previous_end,'Overlapping/out-of-order power windows');previous_end=end
        x=integrate(rows,start,end,gap)
        expanded=integrate(rows,start-unc,end+unc,gap)
        delta=abs(x['duration_s']-w['device_elapsed_s'])
        must(delta<=2*unc+.005*w['device_elapsed_s'],'Power/device durations do not align')
        result.append(dict(**w,**x,start_s=start,end_s=end,device_trace_duration_difference_s=delta,
                           boundary_error_bound_j=2*unc*expanded['max_power_w']))
    active=[]
    for i,w in enumerate(result):
        if w['role']!='active':continue
        before,after=result[i-1],result[i+1]
        must(before['role']=='idle_before' and after['role']=='idle_after' and before['active_variant']==after['active_variant']==w['variant'],'Unpaired idle baselines')
        idle=(before['mean_power_w']+after['mean_power_w'])/2
        drift=abs(before['mean_power_w']-after['mean_power_w'])/idle if idle else None
        incremental=w['joules']-idle*w['duration_s']
        active.append(dict(seq=w['seq'],repetition=w['repetition'],variant=w['variant'],inferences=w['inferences'],
                           whole_board_j_per_inference=w['joules']/w['inferences'],
                           incremental_j_per_inference=incremental/w['inferences'],
                           active_mean_power_w=w['mean_power_w'],idle_mean_power_w=idle,idle_relative_drift=drift,
                           boundary_only_error_bound_j_per_inference=w['boundary_error_bound_j']/w['inferences'],
                           review_required=drift is None or drift>.05 or incremental<=0))
    return dict(status='POWER_TRACE_INTEGRATED_REVIEW_REQUIRED',physical_execution=True,windows=result,active=active,
                alignment_provenance=a,power_csv_sha256=sha(power),capture_sha256=sha(capture/'energy_summary.json'),
                limitations=['Boundary-only bound excludes meter accuracy, conversion error and clock calibration.',
                             'Incremental energy is relative to this board/firmware idle policy; retain signed estimates.',
                             'No architecture-wide confidence claim: one physical board per architecture.',
                             'Original power recording and time alignment require scientific review before publication.'])

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture',type=Path,required=True);p.add_argument('--power-csv',type=Path,required=True)
    p.add_argument('--alignment',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();r=analyze(a.capture,a.power_csv,a.alignment);a.out.write_text(json.dumps(r,indent=2)+'\n');print(r['status'],a.out)
