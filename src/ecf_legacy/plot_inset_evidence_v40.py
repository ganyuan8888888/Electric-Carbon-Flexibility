"""Reference-matched multi-panel scientific plots; unchanged audited observations."""
from evidence_core_v37 import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator,AutoMinorLocator
from matplotlib.legend_handler import HandlerTuple
V37=ROOT/'revision_v37/experiments';FIG=ROOT/'revision_v40/figures';FIG.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman'],'mathtext.fontset':'stix','font.size':8.5,
 'axes.labelsize':9,'axes.linewidth':.65,'legend.fontsize':7.1,'pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white'})
METHODS=['M1','M4-N','M4'];COLORS=['#ff0000','#0000ff','#ff8000'];MARKS=['s','o','^'];STYLES=['-', '--','-.']

def axes_style(ax,grid=True):
 ax.tick_params(which='both',direction='in',top=True,right=True,pad=2,length=3,width=.65)
 ax.tick_params(which='minor',length=1.7,width=.5);ax.set_axisbelow(True)
 if grid:ax.grid(axis='y',color='.7',lw=.45)

def legend(ax,**kwargs):
 return ax.legend(framealpha=1,fancybox=False,edgecolor='black',borderpad=.2,labelspacing=.15,handlelength=2.1,handletextpad=.4,columnspacing=.8,**kwargs)

def dual_axis(left,right):
 # Both major AND minor y ticks are exclusive to their own side.
 left.tick_params(axis='y',which='both',right=False,labelright=False,left=True,labelleft=True)
 right.tick_params(axis='y',which='both',left=False,labelleft=False,right=True,labelright=True)
 right.tick_params(axis='x',which='both',top=False,bottom=False,labeltop=False,labelbottom=False)
 left.spines['right'].set_visible(False);right.spines['left'].set_visible(False)
 assert not any(t.tick2line.get_visible() or t.label2.get_visible() for t in left.yaxis.get_major_ticks()+left.yaxis.get_minor_ticks())
 assert not any(t.tick1line.get_visible() or t.label1.get_visible() for t in right.yaxis.get_major_ticks()+right.yaxis.get_minor_ticks())
 pairs=getattr(left.figure,'_dual_pairs',[]);pairs.append((left,right));left.figure._dual_pairs=pairs

def save(fig,key,data):
 fig.canvas.draw();axis_checks=[]
 for left,right in getattr(fig,'_dual_pairs',[]):
  assert not any(t.tick2line.get_visible() or t.label2.get_visible() for t in left.yaxis.get_major_ticks()+left.yaxis.get_minor_ticks())
  assert not any(t.tick1line.get_visible() or t.label1.get_visible() for t in right.yaxis.get_major_ticks()+right.yaxis.get_minor_ticks())
  axis_checks.append({'left_label':left.get_ylabel(),'right_label':right.get_ylabel(),'left_major_and_minor_ticks_exclusive':True,'right_major_and_minor_ticks_exclusive':True})
 fig.savefig(FIG/(key+'.png'),dpi=450,bbox_inches='tight',pad_inches=.025)
 fig.savefig(FIG/(key+'.pdf'),bbox_inches='tight',pad_inches=.025)
 dump(FIG/(key+'_data.json'),data);plt.close(fig)
 dump(FIG/(key+'_axes_audit.json'),axis_checks)

def region():
 coarse=json.loads((V37/'region_refined_summary.json').read_text());dense=json.loads((V37/'region_dense_summary.json').read_text())
 fig=plt.figure(figsize=(7.25,6.45));gs=fig.add_gridspec(3,2,width_ratios=[1.1,1],wspace=.36,hspace=.18)
 depth_colors=['#00cc00','#ff0000','#0000ff'];depths=[5,7,9]
 for row,date in enumerate(DATES):
  left=fig.add_subplot(gs[row,0]);right=fig.add_subplot(gs[row,1]);r=next(x for x in coarse if x['date']==date and x['chart']==0);dr=next(x for x in dense if x['date']==date and x['chart']==0)
  x=np.array(r['alpha']);truth=np.array(r['original_boundary']);fine=np.array(dr['alpha']);bt=np.array(dr['original_boundary']);bc=np.array(dr['certified_boundary'])
  left.plot(fine,bt,'k-',lw=1.1,label='Original boundary',zorder=4)
  for depth,color,style in zip(depths,depth_colors,[':', '--','-.']):
   b=np.array(next(c['boundary'] for c in r['levels'] if c['depth']==depth));left.plot(x,np.where(np.isfinite(b),b,np.nan),color=color,ls=style,lw=.95,label=f'Certified, depth {depth}')
  valid=np.isfinite(bc);left.fill_between(fine,bt,bc,where=valid,color='#ff8000',alpha=.28,lw=0)
  left.axhline(1.2,color='.5',lw=.65,ls=':');left.set_xlim(0,1)
  # Preserve the full original boundary, including portions beyond the tested cap.
  spread=max(.005,np.ptp(bt));left.set_ylim(bt.min()-.08*spread,bt.max()+.78*spread)
  left.yaxis.set_major_locator(MaxNLocator(4));left.ticklabel_format(axis='y',style='plain',useOffset=False);left.set_xticks(np.linspace(0,1,6));left.xaxis.set_minor_locator(AutoMinorLocator(2));axes_style(left)
  left.set_ylabel('Budget multiplier');left.text(.02,.94,f'({chr(97+row*2)}) {date}',transform=left.transAxes,va='top',fontsize=8)
  if row==0:legend(left,ncol=2,loc='lower right',fontsize=6.5)
  # Original inset retained in the upper-left clear area, separated from the legend and boundary.
  inset=left.inset_axes([.17,.62,.49,.25]);mask=(fine>=.55)&(fine<=.70);inset.plot(fine[mask],bt[mask],'k-',lw=.75)
  inset.plot(fine[mask],np.where(valid,bc,np.nan)[mask],color='#0000ff',ls='-.',lw=.75)
  inset.fill_between(fine[mask],bt[mask],bc[mask],where=valid[mask],fc='#ff8000',alpha=.35,lw=0)
  inset.set_xlim(.55,.70);inset.yaxis.set_major_locator(MaxNLocator(2));inset.xaxis.set_major_locator(MaxNLocator(2));inset.ticklabel_format(axis='y',style='plain',useOffset=False)
  inset.tick_params(labelsize=5.8,direction='in',top=True,right=True,pad=1,length=2);inset.grid(axis='y',color='.75',lw=.35)
  inset.text(.03,.92,'Boundary detail',transform=inset.transAxes,fontsize=5.8,va='top')
  charts=[z for z in coarse if z['date']==date];twin=right.twinx()
  for k,c in enumerate(charts):
   levels=c['levels'];d=[v['depth'] for v in levels];cov=[100*v['coverage'] for v in levels]
   color=['#ff0000','#00cc00','#0000ff'][k];right.plot(d,cov,color=color,marker=MARKS[k],ls=STYLES[k],ms=3,mfc='white',mew=.6,lw=1,label=f'Chart {k+1}')
   gaps=[100*v['max_budget_gap'] if v.get('max_budget_gap') is not None else np.nan for v in levels]
   twin.plot(d,gaps,color=color,ls=':',lw=.6,alpha=.55)
  right.set(xlim=(0,9),ylim=(0,108));right.set_xticks(range(10));right.set_yticks([0,25,50,75,100]);right.set_ylabel('Grid coverage (%)')
  twin.set_yscale('log');twin.set_ylim(.008,80);twin.set_yticks([.01,.1,1,10]);twin.set_yticklabels(['0.01','0.1','1','10']);twin.set_ylabel('Max. budget gap (%)',fontsize=8)
  axes_style(right);axes_style(twin,False);twin.tick_params(labelsize=7);dual_axis(right,twin)
  right.text(.02,.94,f'({chr(98+row*2)}) Interval refinement',transform=right.transAxes,va='top',fontsize=8)
  legend(right,ncol=len(charts),loc='lower right',fontsize=6.7)
  right.text(.97,.20,'Dotted: boundary gap',transform=right.transAxes,ha='right',fontsize=6.4,color='.25')
  if row==2:left.set_xlabel('Chart coordinate');right.set_xlabel('Bisection depth')
  else:left.tick_params(labelbottom=False);right.tick_params(labelbottom=False)
 fig.subplots_adjust(left=.085,right=.925,top=.985,bottom=.075)
 save(fig,'fig_certification_v38',{'original_data_version':37,'refinement':coarse,'dense':dense,'inset_coordinate_window':[.55,.70]})

def oos():
 fig=plt.figure(figsize=(7.25,6.25));outer=fig.add_gridspec(3,1,hspace=.16);all_data=[]
 for j,date in enumerate(DATES):
  gs=outer[j].subgridspec(2,1,height_ratios=[3.3,1],hspace=.035);ax=fig.add_subplot(gs[0]);heat=fig.add_subplot(gs[1],sharex=ax);right=ax.twinx()
  base=V37/'oos_final'/date;draws=json.loads((base/'draws.json').read_text())['pressure_draws'];order=np.argsort([d[1] for d in draws],kind='stable')
  B=load(ROOT/f'results/robust_v28/{date}','common')['budget'];records=[]
  for i in order:
   f=base/f'pressure_{i:03}.json';retry=V37/'oos_numerical_retry'/date/f.name;records.append(json.loads((retry if retry.exists() else f).read_text()))
  status=np.zeros((3,200));margin=np.full((3,200),np.nan);image_data=np.ones((3,200,4))
  holdout=np.zeros((3,12));hold_image=np.ones((3,12,4))
  for i in range(12):
   f=base/f'held_out_{i:03}.json';retry=V37/'oos_numerical_retry'/date/f.name;record=json.loads((retry if retry.exists() else f).read_text())
   for k,method in enumerate(METHODS):
    holdout[k,i]=record['results'][method].get('witness_found',False)
    hold_image[k,i]=matplotlib.colors.to_rgba(COLORS[k]) if holdout[k,i] else [.9,.9,.9,1]
  centers=np.arange(10)*20+10.5;bins=np.array_split(np.arange(200),10);rates=[]
  for k,(method,color) in enumerate(zip(METHODS,COLORS)):
   for i,r in enumerate(records):
    v=r['results'][method];status[k,i]=v.get('witness_found',False)
    if 'allocation_t' in v:margin[k,i]=100*np.max((np.array(v['allocation_t'])-B-np.array(v['quota_t']))/B)
    if status[k,i]:image_data[k,i]=matplotlib.colors.to_rgba(color)
    else:image_data[k,i]=[.9,.9,.9,1]
   groups=[margin[k,b][np.isfinite(margin[k,b])] for b in bins]
   bp=ax.boxplot(groups,positions=centers+(k-1)*4.2,widths=3.6,patch_artist=True,manage_ticks=False,
      boxprops=dict(facecolor=color,edgecolor='black',linewidth=.45),medianprops=dict(color='black',linewidth=.6),
      whiskerprops=dict(color='black',linewidth=.4),capprops=dict(color='black',linewidth=.4),flierprops=dict(marker='.',markersize=1.2,markerfacecolor=color,markeredgecolor=color))
   rate=[100*status[k,b].mean() for b in bins];rates.append(rate)
   right.plot(centers,rate,color=color,ls=STYLES[k],marker=MARKS[k],mfc='white',mew=.6,ms=3,lw=1.)
  ax.set_yscale('symlog',linthresh=.02);ax.set_ylim(-80,180);ax.set_yticks([-10,0,10,100]);ax.set_yticklabels(['−10','0','10','100']);ax.axhline(0,color='.25',lw=.6,ls=':')
  ax.set_ylabel('Budget residual (%)');right.set_ylim(-5,120);right.set_yticks([0,25,50,75,100]);right.set_ylabel('Verified plans (%)')
  axes_style(ax);axes_style(right,False);ax.tick_params(labelbottom=False);dual_axis(ax,right)
  ax.text(.015,.96,f'({chr(97+j)}) {date}',transform=ax.transAxes,va='top',fontsize=8)
  if j==0:
   handles=[(Patch(fc=c,ec='black',lw=.4),Line2D([0],[0],c=c,ls=ls,marker=m,mfc='white',ms=3,lw=.9)) for c,ls,m in zip(COLORS,STYLES,MARKS)]
   legend(ax,handles=handles,labels=['M1','M4-N','M4 (proposed)'],handler_map={tuple:HandlerTuple(ndivide=None)},ncol=3,loc='upper right',fontsize=7.)
  heat.imshow(image_data,origin='upper',aspect='auto',extent=[.5,200.5,2.5,-.5],interpolation='nearest')
  heat.imshow(hold_image,origin='upper',aspect='auto',extent=[-28.5,-4.5,2.5,-.5],interpolation='nearest')
  heat.axvline(-2,color='black',lw=.6);heat.set_yticks([0,1,2],METHODS);heat.set_ylim(2.5,-.5)
  heat.set_xlim(-30,200.5);heat.set_xticks([-16.5,1,40,80,120,160,200],['Holdout\n(n = 12)','1','40','80','120','160','200']);axes_style(heat,False);heat.tick_params(axis='y',length=0,labelsize=6.8)
  if j<2:heat.tick_params(labelbottom=False)
  else:heat.set_xlabel('Stress-trajectory rank by input error-amplitude multiplier')
  heat.set_ylabel('Status',fontsize=7.5,labelpad=3)
  all_data.append({'date':date,'ranked_case_indices':order,'error_amplitude_multipliers':[draws[i][1] for i in order],
    'verified_status':status,'heldout_status':holdout,'best_candidate_carbon_excess_percent':margin,'twenty_case_bin_rates':rates,'missing_margin_counts':np.sum(~np.isfinite(margin),axis=1),
    'heatmap':'Method color: original-constraint witness; gray: unresolved; all 12 held-out and all 200 generated cases per date retained.'})
  assert np.nanmin(margin)>-80 and np.nanmax(margin)<180,'Do not clip recorded residuals.'
 fig.subplots_adjust(left=.105,right=.91,top=.985,bottom=.08)
 save(fig,'fig_oos_v38',all_data)

def scaling():
 data=json.loads((ROOT/'revision_v38/experiments/final_evidence_summary.json').read_text(encoding='utf8'))
 rows=data['regional'];methods=['full','no_reuse','reuse'];colors=['#000000','#0000ff','#ff8000'];labels=['Full MILP','NCCG without reuse','Proposed NCCG'];marks=['o','s','^'];styles=['-','--','-.']
 fig=plt.figure(figsize=(7.25,6.05));gs=fig.add_gridspec(3,2,width_ratios=[1,1],hspace=.20,wspace=.39)
 alltimes=[r['elapsed_s']/60 for r in rows];yrange=(min(alltimes)*.7,max(10.,max(alltimes)*1.65))
 for j,date in enumerate(DATES):
  ax=fig.add_subplot(gs[j,0]);work=fig.add_subplot(gs[j,1]);right=work.twinx();parks=[6,12,24,48];pos=np.arange(4)
  for m,(method,color,mark,style) in enumerate(zip(methods,colors,marks,styles)):
   rr=[next(r for r in rows if r['date']==date and r['parks']==n and r['method']==method) for n in parks]
   times=np.array([r['elapsed_s']/60 for r in rr]);ax.plot(parks,times,c=color,ls=style,marker=mark,mfc='white',ms=3.6,mew=.75,lw=1.,label=labels[m])
   for n,t,r in zip(parks,times,rr):
    if r['status']!='optimal':ax.annotate('limit',(n,t),xytext=(-3,5),textcoords='offset points',fontsize=6,color=color,ha='right')
   work.bar(pos+(m-1)*.23,[r['MILP_calls'] for r in rr],width=.21,fc=color,ec='black',lw=.35,alpha=.88)
   right.plot(pos,[r['LP_calls'] for r in rr],c=color,ls=style,marker=mark,mfc='white',ms=3.2,mew=.6,lw=.9)
  ax.set_yscale('log');ax.set_ylim(*yrange);ax.set_xlim(3,51);ax.set_xticks(parks);ax.set_yticks([.01,.1,1,10]);ax.set_yticklabels(['0.01','0.1','1','10']);axes_style(ax)
  ax.axhline(1,c='.45',lw=.7,ls=':');ax.text(.98,.97,f'({chr(97+2*j)}) {date}',transform=ax.transAxes,ha='right',va='top',fontsize=7.8)
  ax.set_ylabel('Optimization time (min)');work.set_ylabel('MILP calls');right.set_ylabel('Fixed-mode LP calls',fontsize=8)
  work.set_xlim(-.55,3.55);work.set_xticks(pos,parks);work.set_ylim(0,max(r['MILP_calls'] for r in rows if r['date']==date)*1.35+1)
  right.set_ylim(-.5,max(r['LP_calls'] for r in rows if r['date']==date)*1.35+1);work.yaxis.set_major_locator(MaxNLocator(4,integer=True));right.yaxis.set_major_locator(MaxNLocator(4,integer=True))
  axes_style(work);axes_style(right,False);dual_axis(work,right)
  work.text(.03,.97,f'({chr(98+2*j)}) Solver workload',transform=work.transAxes,va='top',fontsize=7.8)
  work.text(.98,.84,'Bars: MILP   Lines: LP',transform=work.transAxes,ha='right',fontsize=6.4)
  if j<2:ax.tick_params(labelbottom=False);work.tick_params(labelbottom=False)
  else:ax.set_xlabel('Number of industrial parks');work.set_xlabel('Number of industrial parks')
 handles=[Line2D([0],[0],c=c,ls=ls,marker=m,mfc='white',ms=3.5,lw=1) for c,ls,m in zip(colors,styles,marks)]
 fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.51,1),ncol=3,fontsize=7.4,framealpha=1,edgecolor='black',fancybox=False,borderpad=.25,columnspacing=1.2)
 fig.subplots_adjust(left=.09,right=.917,top=.945,bottom=.075)
 save(fig,'fig_scaling_v38',{'formal_rows':rows,'time_unit':'minutes','repeats':1,'runtime_uncertainty':'No error bars from a single formal run. All predetermined dates and sizes shown.'})

if __name__=='__main__':
 if len(sys.argv)==1:region();oos();scaling()
 else:globals()[sys.argv[1]]()
