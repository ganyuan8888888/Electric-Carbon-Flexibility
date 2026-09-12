"""Actual outer bounds and inner updates, combined in three reference-style rows."""
from pathlib import Path
from collections import Counter
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter,MaxNLocator,AutoMinorLocator
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
ROOT=__import__("repository_paths").ROOT
OUT=ROOT/'revision_v32/figures';OUT.mkdir(exist_ok=True,parents=True)
DATES=['2020-04-07','2020-10-17','2020-11-29']
plt.rcParams.update({'font.family':'Times New Roman','font.size':10,'mathtext.fontset':'stix',
 'axes.linewidth':.8,'pdf.fonttype':42,'hatch.linewidth':.25})
fig,axes=plt.subplots(3,1,figsize=(7.8,7.2))
audit={}
for i,(ax,date) in enumerate(zip(axes,DATES)):
 p=ROOT/'revision_v32/algorithm_results'/date/'recovery_reuse.json'
 r=json.loads(p.read_text());trace=r['outer_trace'];x=np.array([v['iteration'] for v in trace])
 scale=r['upper_usd']/r['upper_ratio']
 lb=np.array([v['lower_ratio'] for v in trace])*scale
 ub=np.array([v['upper_ratio'] for v in trace])*scale
 calls=r['inner_call_audit'];counts=Counter()
 for call in calls: counts[call['outer_iteration']]+=call['checks']
 inner=[counts[v] for v in x];assert all(n>=1 for n in inner)
 assert np.all(np.diff(lb)>=-1e-5) and np.all(np.diff(ub)<=1e-5)
 assert np.all(lb<=ub+1e-5) and abs(lb[-1]-ub[-1])<=1e-5
 right=ax.twinx();right.bar(x,inner,width=.19,color='#f6d4c0',edgecolor='#8e8e8e',linewidth=.55,hatch='++++++++',zorder=1)
 right.set_ylim(0,5.4);right.set_yticks(range(6));right.set_ylabel('Inner checks',fontsize=10.5,labelpad=8)
 right.tick_params(direction='in',length=4,width=.7)
 for xi,n in zip(x,inner):
  if not n: ax.text(xi,.045,'0',transform=ax.get_xaxis_transform(),ha='center',va='bottom',fontsize=9,color='#595959',bbox=dict(fc='white',ec='none',pad=.5),zorder=9)
 ax.set_zorder(right.get_zorder()+1);ax.patch.set_visible(False)
 ax.plot(x,ub,c='black',lw=1.25,marker='s',ms=4.2,mfc='white',mec='black',mew=.9)
 ax.plot(x,lb,c='#ff3030',ls=(0,(3,2)),lw=1.25,marker='o',ms=4.2,mfc='white',mec='#ff3030',mew=.9)
 yspan=max(ub.max()-lb.min(),.04*ub.max());ax.set_ylim(lb.min()-.11*yspan,ub.max()+.29*yspan)
 ax.set_xlim(x[0]-.3,x[-1]+.3);ax.set_xticks(x)
 fmt=ScalarFormatter(useMathText=True);fmt.set_powerlimits((0,0));ax.yaxis.set_major_formatter(fmt)
 ax.yaxis.set_major_locator(MaxNLocator(nbins=4));ax.yaxis.set_minor_locator(AutoMinorLocator(2))
 ax.grid(axis='y',color='.8',lw=.4);ax.tick_params(direction='in',top=True,right=False,length=4,width=.7)
 ax.tick_params(which='minor',direction='in',length=2,width=.6)
 ax.set_xlabel('Outer iterations',fontsize=11,labelpad=4);ax.set_ylabel('Objective bound (USD)',fontsize=10.5,labelpad=7)
 ax.text(.025,.88,f'({chr(97+i)}) {date}',transform=ax.transAxes,fontsize=10)
 audit[date]={'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'outer_iterations':x.tolist(),
 'upper_usd':ub.tolist(),'lower_usd':lb.tolist(),'inner_checks':inner,'inner_call_audit':calls,
 'zero_meaning':'Outer bounds already closed; no inner call. Screening-only infeasibility checks are counted once, separately identified in the call log.'}
handles=[Line2D([0],[0],c='black',lw=1.25,marker='s',mfc='white',ms=4.2,label='Upper Bound'),
 Line2D([0],[0],c='#ff3030',lw=1.25,ls=(0,(3,2)),marker='o',mfc='white',ms=4.2,label='Lower Bound'),
 Patch(facecolor='#f6d4c0',edgecolor='#8e8e8e',hatch='++++++++',label='Inner checks')]
fig.legend(handles=handles,ncol=3,loc='upper center',bbox_to_anchor=(.51,1),framealpha=1,edgecolor='black',
 fancybox=False,borderpad=.3,columnspacing=1.5,handlelength=2.5,fontsize=10)
fig.subplots_adjust(left=.11,right=.9,top=.925,bottom=.072,hspace=.49)
for ext in ['png','pdf','svg']:
 fig.savefig(OUT/f'fig_public_algorithm_convergence.{ext}',dpi=450,bbox_inches='tight',pad_inches=.035)
(OUT/'convergence_data_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8')
print(OUT/'fig_public_algorithm_convergence.png')
