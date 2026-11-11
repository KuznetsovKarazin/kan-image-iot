"""Audit one physical Mega recording; raw V*I energy, signed polling contrast.

The waveform alignment rule was chosen after inspection and is common to all
14 recordings. It does not optimize energy. All threshold diagnostics are kept.
"""
import argparse,csv,hashlib,json,os,statistics,sys
from pathlib import Path
from cfn_reader import parse_cfn

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def describe(a):return dict(n=len(a),mean=statistics.mean(a),sample_sd=statistics.stdev(a),minimum=min(a),maximum=max(a))

def intervals(t,p,threshold):
    groups=[]
    for i,value in enumerate(p):
        if value>=threshold:continue
        if not groups or t[i]-t[groups[-1][-1]]>.25:groups.append([])
        groups[-1].append(i)
    # Exclude recording-edge standby tails. No amplitude floor: transient dips
    # inside an active interval must not truncate it or be mistaken for onset.
    groups=[g for g in groups if t[g[-1]]-t[g[0]]>30 and g[0]>0 and g[-1]+1<len(t)]
    if len(groups)!=3:return None
    return [[(t[g[0]-1]+t[g[0]])/2,(t[g[-1]]+t[g[-1]+1])/2] for g in groups]

def analyze(root,family,case,variant,capture,native,out):
    out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(root/'release/packages'/family/'tools'))
    from audit_energy import audit
    from integrate_trace import analyze as integrate_trace,integrate
    receipt=audit(capture);s=read(capture/'energy_summary.json')
    assert s['target']=='mega' and s['board_id']=='MEGA-A' and s['physical_execution'] is True
    assert case in [1,2] and variant in ([0,4,5,8,9] if family=='kan' else [0,1])
    assert s['case']==case and s['variants']==[variant] and s['repetitions']==3 and s['seconds']==60 and s['baseline_seconds']==15
    assert receipt['prediction_checks']==1024 and len(receipt['windows'])==9
    save(out/'device_audit.json',receipt)
    records,c,m=parse_cfn(native);instrument=read(root/'reference/instrument_metadata.json')
    m.update(meter_model=instrument['meter_model'],meter_serial=instrument['meter_serial'],meter_identity_scope='Same user-reported FNB58 setup; CFN does not contain a serial-number field')
    assert m['nominal_rate_hz']==100 and m['start_current_ma']==0 and m['stop_current_ma']==0
    save(out/'cfn_metadata.json',m)
    t=[x[0] for x in records];power=[v*i for v,i in zip(c[0],c[1])];sortedp=sorted(power)
    low=sortedp[len(sortedp)//2];high=sortedp[int(.9*(len(sortedp)-1))]
    threshold=(low+high)/2
    ref=intervals(t,power,threshold);assert ref is not None
    active=[w for w in s['windows'] if w['role']=='active']
    host=[w[k] for w in active for k in ['host_estimated_start_rel_s','host_estimated_end_rel_s']]
    offsets=[x-h for x,h in zip([x for pair in ref for x in pair],host)]
    offset=statistics.median(offsets);residual=max(abs(x-offset) for x in offsets)
    assert residual<1.0
    diagnostics=[]
    for fraction in [.25,.35,.45,.5,.55,.65,.75]:
        th=low+fraction*(high-low);edges=intervals(t,power,th)
        row=dict(fraction_of_p50_to_p90=fraction,threshold_w=th,all_three_intervals_found=edges is not None)
        if edges:
            diffs=[x-h for x,h in zip([x for pair in edges for x in pair],host)]
            row.update(intervals_s=edges,median_offset_s=statistics.median(diffs),median_offset_change_s=statistics.median(diffs)-offset,
                       maximum_residual_to_reference_offset_s=max(abs(x-offset) for x in diffs),
                       maximum_edge_change_s=max(abs(x-y) for pair,r in zip(edges,ref) for x,y in zip(pair,r)))
        diagnostics.append(row)
    csv_path=out/'power_si.csv'
    with csv_path.open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f,lineterminator='\n');w.writerow(['time_s','voltage_v','current_a']);w.writerows(zip(t,c[0],c[1]))
    alignment=dict(physical_measurement=True,meter_model=instrument['meter_model'],meter_serial=instrument['meter_serial'],
        source_recording_path=Path(os.path.relpath(native,out)).as_posix(),source_recording_sha256=sha(native),power_csv_sha256=sha(csv_path),
        conversion_description='Primary native CFN VBUS and IBUS doubles; exact PBUS=V*I check; accumulated Wh excluded.',
        alignment_method='One median offset fitted to six negative-going/positive-going active edges. Common rule: threshold=(whole-record P50+P90)/2; active regions below threshold, gaps <=0.25 s merged, duration>30 s, recording-edge standby tails excluded. No filtering or clipping of integration samples.',
        analysis_rule_preregistered=False,calibration_note='No independent meter gain, time-base or response-delay calibration supplied.',
        max_sample_gap_s=.011,boundary_uncertainty_s=1.0,boundary_uncertainty_scope='Numerical sensitivity allowance only, not a certified metrological bound',
        cfn_minus_runner_origin_s=offset,edge_offsets_s=offsets,max_edge_fit_residual_s=residual,
        reference_threshold_w=threshold,whole_record_p50_w=low,whole_record_p90_w=high,threshold_diagnostics=diagnostics,
        hardware_synchronization=False,windows=[dict(seq=w['seq'],start_s=w['host_estimated_start_rel_s']+offset,end_s=w['host_estimated_end_rel_s']+offset) for w in s['windows']])
    save(out/'alignment.json',alignment)
    result=integrate_trace(capture,csv_path,out/'alignment.json');save(out/'integrated_energy.json',result)
    assert len(result['windows'])==9 and len(result['active'])==3
    rows=[];power_rows=list(zip(t,power))
    for x in result['active']:
        i=next(i for i,w in enumerate(result['windows']) if w['seq']==x['seq'])
        trio=result['windows'][i-1:i+2];before,window,after=trio
        shifts=[]
        for delta in [-1.,-.5,-.1,0.,.1,.5,1.]:
            es=[integrate(power_rows,w['start_s']+delta,w['end_s']+delta,.011) for w in trio]
            idle=(es[0]['mean_power_w']+es[2]['mean_power_w'])/2
            shifts.append(dict(offset_shift_s=delta,whole_board_uj=es[1]['joules']*1e6/x['inferences'],
                               signed_polling_difference_uj=(es[1]['joules']-idle*window['duration_s'])*1e6/x['inferences']))
        rows.append(dict(**x,active_duration_s=window['duration_s'],total_j=window['joules'],
            whole_board_uj_per_inference=x['whole_board_j_per_inference']*1e6,
            signed_polling_difference_uj_per_inference=x['incremental_j_per_inference']*1e6,
            cycle_us_per_inference=window['duration_s']*1e6/x['inferences'],common_offset_sensitivity=shifts,
            review_interpretation='Original integrator review flag retained. Negative difference is relative to an active polling baseline; it is not negative inference energy.',
            whole_board_review_required=(x['idle_relative_drift'] is None or x['idle_relative_drift']>.05),
            whole_board_boundary_only_allowance_uj=x['boundary_only_error_bound_j_per_inference']*1e6))
    golden=read(root/f'release/packages/{family}/mega/reference/model{case}_golden.json')
    fields=['whole_board_uj_per_inference','signed_polling_difference_uj_per_inference','active_mean_power_w','idle_mean_power_w','cycle_us_per_inference']
    aggregate=dict(family=family,case=case,variant=variant,name=golden['variants'][variant],
                   total_inferences=sum(w['inferences'] for w in rows),**{f:describe([w[f] for w in rows]) for f in fields})
    before=alignment['windows'][0]['start_s']-m['first_time_s'];after=m['last_time_s']-alignment['windows'][-1]['end_s']
    assert min(before,after)>1
    outdata=dict(status='WHOLE_BOARD_ENERGY_INTEGRATED_SIGNED_POLLING_DIFFERENCE_RETAINED',target='mega',board_id='MEGA-A',
        family=family,case=case,variant=variant,windows=9,active_windows_count=3,prediction_checks=1024,
        continuous_recordings=1,reset_sessions=1,temporal_windows_per_variant=3,aggregate=aggregate,active_windows=rows,
        recording_samples=m['sample_count'],recording_seconds=m['last_time_s']-m['first_time_s'],
        cfn_before_first_window_s=before,cfn_after_last_window_s=after,requested_5s_padding_met=(min(before,after)>=5),
        max_idle_pair_relative_drift=max(w['idle_relative_drift'] for w in rows),
        original_integrator_review_flagged_windows=[w['seq'] for w in rows if w['review_required']],
        whole_board_review_flagged_windows=[w['seq'] for w in rows if w['whole_board_review_required']],
        signed_polling_difference_interpretation='Descriptive active-minus-polling contrast only; not isolated incremental inference energy',
        max_alignment_residual_s=residual,accumulated_vi_over_wh_ratio=m['integrated_vi_over_saved_energy_ratio'],
        maximum_whole_board_offset_sensitivity_percent=max(abs(x['whole_board_uj']/w['whole_board_uj_per_inference']-1)*100 for w in rows for x in w['common_offset_sensitivity']),
        minimum_signed_difference_across_offsets_uj=min(x['signed_polling_difference_uj'] for w in rows for x in w['common_offset_sensitivity']),
        maximum_signed_difference_across_offsets_uj=max(x['signed_polling_difference_uj'] for w in rows for x in w['common_offset_sensitivity']),
        source_summary_sha256=sha(capture/'energy_summary.json'),source_native_cfn_sha256=sha(native),
        publication_metrology_complete=False,instrument_metadata=instrument)
    save(out/'energy_results.json',outdata)
    print(f'{family}_C{case}_v{variant}: {aggregate[fields[0]]["mean"]:.3f} uJ; polling contrast {aggregate[fields[1]]["mean"]:.3f} uJ; max residual {residual:.3f} s',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--family',choices=['kan','mlp'],required=True)
    p.add_argument('--case',type=int,required=True);p.add_argument('--variant',type=int,required=True);p.add_argument('--capture',type=Path,required=True)
    p.add_argument('--cfn',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    analyze(a.root.resolve(),a.family,a.case,a.variant,a.capture.resolve(),a.cfn.resolve(),a.out.resolve())
