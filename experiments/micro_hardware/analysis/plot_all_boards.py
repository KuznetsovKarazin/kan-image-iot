"""Scientific comparison; separate latency and energy experiments, explicit units."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator,FuncFormatter
root=Path(__file__).resolve().parent
d=json.loads((root/'results/all_boards_results.json').read_text());rows=d['best_observed_methods']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False,'svg.fonttype':'none'})
fig,axes=plt.subplots(1,2,figsize=(12.3,7.2),sharey=True)
fig.subplots_adjust(left=.115,right=.98,top=.78,bottom=.285,wspace=.12)
fig.text(.025,.95,'Three boards: head latency and measured energy',fontsize=17,weight='bold')
fig.text(.025,.901,'Lowest observed energy variant within each family; latency comes from separate audited timing sessions.',fontsize=10.5)
for ax,field,title,units in zip(axes,['latency_us','energy_uj'],['Isolated head latency','Whole-board energy'],['µs/inference','µJ/inference']):
    ax.set_xscale('log');ax.set_xlim(5,500000);ax.set_ylim(6.65,-.65)
    ax.set_yticks(range(len(rows)),[f"{r['target'].upper()} / C{r['case']}" for r in rows]);ax.tick_params(axis='y',length=0,pad=8)
    ax.xaxis.set_major_locator(FixedLocator([10,100,1000,10000,100000]));ax.xaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{v:,.0f}'))
    ax.grid(axis='x',color='#e2e8f0',lw=.8)
    for y in [2.5,4.5]:ax.axhline(y,color='#cbd5e1',lw=.7)
    for y,r in enumerate(rows):
        m,k=r['mlp_'+field],r['kan_'+field]
        ax.plot([m,k],[y,y],c='#cbd5e1',lw=1.6,zorder=2)
        for value,color in [(m,'#2563eb'),(k,'#0f766e')]:
            ax.scatter(value,y,color=color,s=35,zorder=3)
            text=f'{value:.2f}' if value<100 else f'{value:,.1f}'
            ax.annotate(text,(value,y),xytext=(0,10 if color=='#0f766e' else -15),ha='center',textcoords='offset points',fontsize=8.5,color=color)
    ax.set_title(title,fontsize=12,weight='bold',pad=15);ax.set_xlabel(units+' — log scale; lower is better',fontsize=9,labelpad=10)
fig.legend(handles=[Line2D([],[],marker='o',ls='',color='#0f766e',label='Best observed KAN'),Line2D([],[],marker='o',ls='',color='#2563eb',label='Best observed MLP')],loc='lower center',bbox_to_anchor=(.5,.183),ncol=2,frameon=False)
fig.text(.025,.17,'S3: KAN FP32 LUT / DRAM; MLP FP32.  C3: KAN factored INT8 LUT / flash; MLP INT8.',fontsize=9,color='#334155')
fig.text(.025,.14,'Mega: KAN FP32 LUT / flash; MLP INT8. Methods were selected from the tested variants, separately for each family and platform.',fontsize=9,color='#334155')
fig.text(.025,.095,'Quality: C2 KAN INT8 has 851/1024 correct; original KAN FP32 has 856/1024. C2 FP32 LUT keeps 856 correct but changes two decisions.',fontsize=8.5,color='#475569')
fig.text(.025,.063,'Energy: mean of three temporal windows per configuration; one board per platform. Latency: median of three session medians.',fontsize=8.5,color='#475569')
fig.text(.025,.03,'Head only. No independent meter calibration. Mega active-minus-polling differences are negative; only total powered-board energy is compared here.',fontsize=8.5,color='#475569')
out=root/'figures';out.mkdir(exist_ok=True)
for ext in ['png','svg','pdf']:fig.savefig(out/f'All_Boards_Latency_Energy_2026-11-10.{ext}',dpi=180,metadata={'Creator':'Reproducible three-board energy analysis'})
plt.close(fig)

# Retain a raw-trace illustration of the signed Mega background contrast.
from cfn_reader import parse_cfn
key='kan_C1_v0';base=root/'results/records'/key
a=json.loads((base/'alignment.json').read_text());e=json.loads((base/'energy_results.json').read_text())
recordings,channels,_=parse_cfn((base/a['source_recording_path']).resolve())
t=[x[0] for x in recordings];p=[v*i for v,i in zip(channels[0],channels[1])]
fig,ax=plt.subplots(figsize=(11.5,4.6));fig.subplots_adjust(left=.085,right=.985,top=.78,bottom=.22)
ax.plot(t,p,color='#334155',lw=.65)
for w in e['active_windows']:
    position=next(x for x in a['windows'] if x['seq']==w['seq'])
    ax.axvspan(position['start_s'],position['end_s'],color='#0f766e',alpha=.13)
ax.set_xlabel('Native CFN time (s)');ax.set_ylabel('VBUS × IBUS (W)');ax.grid(alpha=.2)
fig.text(.025,.92,'Mega baseline example: polling power exceeds inference power',fontsize=14,weight='bold')
fig.text(.025,.84,'C1, original KAN; shaded regions are the three measured active windows. Samples are not filtered or clipped.',fontsize=9.5)
fig.text(.025,.075,'Idle references execute a busy polling loop. A negative active-minus-idle difference is not negative energy consumed by inference.',fontsize=9)
fig.text(.025,.025,'The primary metric integrates positive total VBUS × IBUS over each active window and divides by the verified inference count.',fontsize=9)
for ext in ['png','svg','pdf']:fig.savefig(out/f'Mega_Polling_Baseline_2026-11-10.{ext}',dpi=180,metadata={'Creator':'Reproducible Mega raw trace illustration'})
plt.close(fig)
