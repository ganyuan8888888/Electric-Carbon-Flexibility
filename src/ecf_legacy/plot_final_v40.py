"""Publication figures from complete saved observations and frozen benchmarks."""
import plot_evidence_v40 as style
from plot_evidence_v40 import *
FIG=ROOT/'revision_v40/figures';FIG.mkdir(exist_ok=True);style.FIG=FIG

def ac_data():
 d=json.loads((ROOT/'revision_v39/experiments/ac_complete_summary.json').read_text());assert d['all_checks_passed']
 old=json.loads((ROOT/'revision_v37/experiments/ac_final_summary.json').read_text())
 return d,old

def ac_network():
 data,old=ac_data();fig=plt.figure(figsize=(7.25,7.1));gs=fig.add_gridspec(3,2,wspace=.31,hspace=.30);t=(np.arange(48)+.5)/2
 plotdata=[]
 for j,date in enumerate(DATES):
  aa=[r for r in old if r['date']==date];bb=[r for r in data['rows'] if r['date']==date]
  a={k:np.array([[next(r[k] for r in aa if r['scenario']==s and r['period']==p) for p in range(48)] for s in range(5)]) for k in ['vmin','vmax','max_line_loading','loss_mw']}
  b={k:np.array([[next(r[k] for r in bb if r['scenario']==s and r['period']==p) for p in range(48)] for s in range(5)]) for k in ['vmin','vmax','max_loading','loss_mw']}
  detail_grid=gs[j,1].subgridspec(2,1,height_ratios=[3,1.25],hspace=.58)
  left=fig.add_subplot(gs[j,0]);right=fig.add_subplot(detail_grid[0])
  for vals,color,ls,label in [(a,'#ff0000','--','Initial AC mapping'),(b,'#0000ff','-','Verified recourse')]:
   vmin=vals['vmin'].min(axis=0);vmax=vals['vmax'].max(axis=0)
   left.fill_between(t,vmin,vmax,color=color,alpha=.08,lw=0)
   left.plot(t,vmin,c=color,ls=ls,lw=.8);left.plot(t,vmax,c=color,ls=ls,lw=.8,label=label)
   left.plot(t[::4],vals['vmin'][0,::4],c=color,ls='none',marker='s' if color=='#ff0000' else 'o',mfc='white',ms=2.4,mew=.55)
  left.axhline(.95,c='black',ls=':',lw=.75);left.axhline(1.05,c='black',ls=':',lw=.75)
  left.set(xlim=(0,24),ylim=(.825,1.195),ylabel='Voltage envelope (p.u.)');left.set_yticks([.85,.95,1.05,1.15]);left.set_xticks([0,4,8,12,16,20,24]);axes_style(left)
  left.set_title(f'({chr(97+2*j)}) {date}',loc='left',fontsize=8,pad=4)
  for vals,key,color,ls,marker,label in [(a,'max_line_loading','#ff0000','--','s','Initial AC mapping'),(b,'max_loading','#0000ff','-','o','Verified recourse')]:
   ar=100*vals[key];right.fill_between(t,ar.min(axis=0),ar.max(axis=0),fc=color,alpha=.09,lw=0)
   right.plot(t,ar[0],c=color,ls=ls,lw=.85,marker=marker,markevery=4,mfc='white',ms=2.6,mew=.55,label=label)
  right.axhline(100,c='black',ls=':',lw=.85);right.set(xlim=(0,24),ylim=(78,114),ylabel='Branch loading (%)');right.set_xticks([0,4,8,12,16,20,24]);right.set_yticks([80,90,100,110]);axes_style(right)
  right.set_title(f'({chr(98+2*j)}) Thermal-limit verification',loc='left',fontsize=7.7,pad=4)
  inset=fig.add_subplot(detail_grid[1]);sel=(t>=17)&(t<=22)
  for vals,key,c,ls in [(a,'max_line_loading','#ff0000','--'),(b,'max_loading','#0000ff','-')]:inset.plot(t[sel],100*vals[key][0,sel],c=c,ls=ls,lw=.6)
  inset.axhline(100,c='black',ls=':',lw=.55);inset.set_xlim(17,22);inset.xaxis.set_major_locator(MaxNLocator(3));inset.yaxis.set_major_locator(MaxNLocator(2));inset.tick_params(direction='in',top=True,right=True,labelsize=5.5,pad=1,length=2)
  inset.set_title('Evening loading detail',fontsize=6.8,pad=3)
  if j==0:ac_handles,ac_labels=left.get_legend_handles_labels()
  right.tick_params(labelbottom=False)
  if j<2:left.tick_params(labelbottom=False)
  else:left.set_xlabel('Time (h)');inset.set_xlabel('Time (h)',fontsize=8)
  plotdata.append(dict(date=date,initial=a,verified=b))
 fig.legend(ac_handles,ac_labels,loc='upper center',bbox_to_anchor=(.5,1),ncol=2,fontsize=7.3,framealpha=1,edgecolor='black',fancybox=False)
 fig.subplots_adjust(left=.085,right=.98,top=.935,bottom=.075);save(fig,'fig_ac_network_v39',plotdata)

def ac_carbon():
 data,_=ac_data();fig=plt.figure(figsize=(7.25,6.15));gs=fig.add_gridspec(3,2,wspace=.39,hspace=.23);plotdata=[]
 for j,date in enumerate(DATES):
  before=[];after=[];cost=[];emissions=[]
  for s in range(5):
   raw=json.loads((ROOT/f'revision_v39/experiments/ac_restore/day_{date}_{s}.json').read_text());r=next(x for x in data['days'] if x['date']==date and x['scenario']==s);actual=json.loads((ROOT/r['source']).read_text())
   cap=np.array(raw['initial_budget_t'])+raw['original_quota_t'];alloc=actual.get('allocation_t',actual.get('restored_industrial_allocation_t'))
   before.append(100*np.array(raw['restored_industrial_allocation_t'])/cap);after.append(100*np.array(alloc)/cap)
   p=load(ROOT/f'results/uncertainty_v26/{date}',f'scenario_{s:02}');plan=load(ROOT/f'results/robust_v28/{date}/scenario_{s:02}','M4_proposed')
   curtail=150*p['dt_h']*np.maximum(p['available_renewable_mw']-plan['pr'],0).sum()
   cost.append(100*(r['restored_generation_cost_usd']-r['original_generation_cost_usd'])/(r['original_generation_cost_usd']+curtail))
   emissions.append(100*(r['restored_source_emissions_t']-r['original_source_emissions_t'])/r['original_source_emissions_t'])
  before=np.array(before);after=np.array(after);ax=fig.add_subplot(gs[j,0]);work=fig.add_subplot(gs[j,1]);right=work.twinx();parks=np.arange(1,7)
  for a,c,off,label in [(before,'#0000ff',-.18,'AC correction only'),(after,'#ff8000',.18,'Quota-feasible recourse')]:
   ax.boxplot([a[:,i] for i in range(6)],positions=parks+off,widths=.29,patch_artist=True,manage_ticks=False,
    boxprops=dict(fc=c,ec='black',lw=.45),medianprops=dict(c='black',lw=.6),whiskerprops=dict(c='black',lw=.4),capprops=dict(c='black',lw=.4),
    flierprops=dict(marker='D',ms=1.8,mfc=c,mec='black',mew=.3))
   ax.plot(parks+off,a[0],color=c,ls='none',marker='o',mfc='white',mew=.6,ms=2.8)
  assert min(before.min(),after.min())>=0 and max(before.max(),after.max())<135
  ax.axhline(100,c='black',lw=.8,ls=':');ax.set(xlim=(.45,6.55),ylim=(0,150),ylabel='Daily quota utilization (%)');ax.set_xticks(parks);ax.set_yticks([0,25,50,75,100,125]);axes_style(ax)
  ax.set_title(f'({chr(97+2*j)}) {date}',loc='left',fontsize=8,pad=4)
  if j==0:legend(ax,handles=[Patch(fc='#0000ff',ec='black',lw=.4),Patch(fc='#ff8000',ec='black',lw=.4)],labels=['AC correction only','Quota-feasible recourse'],loc='lower left',fontsize=6.4)
  x=np.arange(1,6);work.bar(x,cost,width=.50,fc='#ff8000',ec='black',lw=.45)
  right.plot(x,emissions,c='#0000ff',ls='--',marker='s',mfc='white',ms=3.3,mew=.65,lw=.9)
  work.set(xlim=(.45,5.55),ylim=(0,7.8),ylabel='Operating-cost increase (%)');right.set(ylim=(0,7.8),ylabel='Source-emission increase (%)');work.set_xticks(x);work.set_yticks([0,2,4,6]);right.set_yticks([0,2,4,6]);axes_style(work);axes_style(right,False);dual_axis(work,right)
  work.set_title(f'({chr(98+2*j)}) AC correction impact',loc='left',fontsize=7.7,pad=4)
  if j==0:legend(work,handles=[Patch(fc='#ff8000',ec='black',lw=.4),Line2D([0],[0],c='#0000ff',ls='--',marker='s',mfc='white',ms=3,lw=.8)],labels=['Cost','Emissions'],ncol=2,loc='upper right',fontsize=6.4,bbox_to_anchor=(1,.82))
  if j<2:ax.tick_params(labelbottom=False);work.tick_params(labelbottom=False)
  else:ax.set_xlabel('Industrial park');work.set_xlabel('Joint source–load scenario')
  plotdata.append(dict(date=date,before_quota_utilization=before,verified_quota_utilization=after,cost_increase_percent=cost,source_emission_increase_percent=emissions))
 fig.subplots_adjust(left=.09,right=.905,top=.965,bottom=.075);save(fig,'fig_ac_carbon_v39',plotdata)

def algorithm():
 rows=[json.loads(p.read_text()) for p in sorted((ROOT/'revision_v39/experiments/final_benchmark').glob('*.json'))];assert len(rows)==36
 fig=plt.figure(figsize=(7.25,6.1));gs=fig.add_gridspec(3,2,wspace=.43,hspace=.28);sizes=[5,10,20,40];pos=np.arange(4)
 methods=['full','no_reuse','reuse'];colors=['black','#0000ff','#ff8000'];marks=['o','s','^'];styles=['-','--','-.'];labels=['Full MILP','NCCG without reuse','Proposed NCCG']
 maxt=max(x['elapsed_s']/60 for x in rows);mint=min(x['elapsed_s']/60 for x in rows)
 for j,date in enumerate(DATES):
  ax=fig.add_subplot(gs[j,0]);work=fig.add_subplot(gs[j,1]);right=work.twinx()
  for k,(method,c,mark,ls) in enumerate(zip(methods,colors,marks,styles)):
   rr=[next(r for r in rows if r['date']==date and r['scenarios']==s and r['method']==method) for s in sizes]
   ax.plot(sizes,[r['elapsed_s']/60 for r in rr],c=c,ls=ls,marker=mark,mfc='white',ms=3.4,mew=.65,lw=.95)
   for s,r in zip(sizes,rr):
    if r['status']!='optimal':
     assert r['status'] in ['time_limited','user_limit']
     ax.annotate('TL',(s,r['elapsed_s']/60),xytext=(-3,4),textcoords='offset points',ha='right',fontsize=6,color=c)
   work.bar(pos+(k-1)*.23,[r['MILP_calls'] for r in rr],width=.21,fc=c,ec='black',lw=.4)
   right.plot(pos,[r['LP_calls'] for r in rr],c=c,ls=ls,marker=mark,mfc='white',ms=2.9,mew=.6,lw=.85)
  ax.set_yscale('log');ax.set(xlim=(3,42),ylim=(mint*.7,max(20,maxt*3)),ylabel='Optimization time (min)');ax.set_xticks(sizes);ax.axhline(1,c='.4',ls=':',lw=.65);axes_style(ax)
  local=[r for r in rows if r['date']==date];work.set_ylim(0,1.28*max(r['MILP_calls'] for r in local));lpmax=max(1,max(r['LP_calls'] for r in local));right.set_ylim(-.04*lpmax,1.28*lpmax)
  work.set(xlim=(-.55,3.55),ylabel='MILP calls');right.set_ylabel('LP evaluations');work.set_xticks(pos,sizes);work.yaxis.set_major_locator(MaxNLocator(4,integer=True));right.yaxis.set_major_locator(MaxNLocator(4,integer=True));axes_style(work);axes_style(right,False);dual_axis(work,right)
  ax.set_title(f'({chr(97+2*j)}) {date}',loc='left',fontsize=8,pad=4);work.set_title(f'({chr(98+2*j)}) Optimization workload',loc='left',fontsize=7.7,pad=4)
  if j<2:ax.tick_params(labelbottom=False);work.tick_params(labelbottom=False)
  else:ax.set_xlabel('Number of joint scenarios');work.set_xlabel('Number of joint scenarios')
 handles=[Line2D([0],[0],c=c,ls=ls,marker=m,mfc='white',ms=3.4,lw=.95) for c,ls,m in zip(colors,styles,marks)]
 fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.51,1),ncol=3,fontsize=7.2,framealpha=1,edgecolor='black',fancybox=False,borderpad=.25)
 fig.subplots_adjust(left=.085,right=.92,top=.925,bottom=.075);save(fig,'fig_scaling_v39',dict(rows=rows,time_unit='minutes',formal_repeats=1))

if __name__=='__main__':globals()[sys.argv[1]]()
