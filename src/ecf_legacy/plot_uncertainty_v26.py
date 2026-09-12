"""Frozen-scenario figures in the existing reference layouts and palette."""
from pathlib import Path
import sys,json
ROOT=__import__("repository_paths").ROOT
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from plot_revision_v12 import panel,legend,COLORS,STYLES
from dispatch_result_io import load
from green_direct_v18 import make_case

DATES=['2020-04-07','2020-10-17','2020-11-29']
OUT=ROOT/'revision_v26/figures';OUT.mkdir(parents=True,exist_ok=True)
LABELS=['Constant control','Multi-time-scale dispatch','C-OPF','Proposed']

def save(fig,stem):
    assert not any(any('\u4e00'<=c<='\u9fff' for c in o.get_text()) for o in fig.findobj(plt.Text))
    for ext in ['png','pdf','svg']:fig.savefig(OUT/(stem+'.'+ext),dpi=400,bbox_inches='tight',pad_inches=.04)
    plt.close(fig)

def _obsolete_chronology():
    fig,axes=plt.subplots(3,1,figsize=(7.4,4.9),sharex=True,sharey=True,gridspec_kw={'hspace':.065})
    audit={}
    for j,(date,ax) in enumerate(zip(DATES,axes)):
        net,_=make_case(date);types=net.renewable['Unit Type'].to_numpy()
        ps=[load(ROOT/f'results/uncertainty_v26/{date}',f'scenario_{s:02}') for s in range(5)]
        base=load(ROOT/f'results/robust_v26/{date}','forecast_common_reference')['base']
        values=np.array([[p['background_mw'].sum(axis=0)+base.sum(axis=0),
            p['available_renewable_mw'][types=='WIND'].sum(axis=0),
            p['available_renewable_mw'][np.isin(types,['PV','RTPV'])].sum(axis=0)] for p in ps])/1000
        values=np.concatenate([values,(values[:,0]-values[:,1]-values[:,2])[:,None,:]],axis=1)
        t=ps[0]['time_h']
        for m,(c,ls,mark) in enumerate(zip(COLORS,STYLES,['s','o','^','D'])):
            # Each fine line is an actual frozen scenario, never a visual offset.
            for sid in range(1,5):
                ax.plot(t,values[sid,m],color=c,ls=['-','--',':','-.'][sid-1],lw=.48,alpha=.42,zorder=2)
            ax.plot(t,values[0,m],color=c,ls=ls,lw=1.2,marker=mark,ms=2.6,mfc='white',mew=.65,markevery=4,zorder=4)
        panel(ax);ax.set_ylabel('Power (GW)',fontsize=10)
        ax.tick_params(labelsize=9,length=2.8,width=.65);ax.grid(axis='y',color='.72',lw=.4)
        for spine in ax.spines.values():spine.set_linewidth(.65)
        ax.text(.014,.87,f'({chr(97+j)}) {date}',transform=ax.transAxes,fontsize=9,bbox=dict(fc='white',ec='none',pad=.2))
        audit[date]=dict(scenarios=5,curves=20,main_curves=4,scenario_curves=16,
            net_demand_definition='System demand minus available wind and all PV; not scheduled conventional generation.',
            source='Same frozen source inputs; no rescaling or synthetic visual variation.',data=values.tolist())
    axes[0].set_ylim(-.10,6.35);axes[0].set_yticks([0,2,4,6])
    hands=[Line2D([0],[0],c=c,ls=ls,marker=mark,ms=2.6,mfc='white',lw=1.2,label=l) for c,ls,mark,l in zip(COLORS,STYLES,['s','o','^','D'],['System demand','Available wind','Available PV','Net demand'])]
    hands.append(Line2D([0],[0],c='.35',lw=.5,label='Scenarios S1–S4'))
    fig.legend(handles=hands,ncol=5,loc='upper center',bbox_to_anchor=(.516,.947),fontsize=8,framealpha=1,edgecolor='black',fancybox=False,borderpad=.25,handlelength=2.2,columnspacing=.9)
    fig.subplots_adjust(top=.885,bottom=.10,left=.085,right=.99)
    axes[-1].set_xlabel('Time (h)',fontsize=10)
    save(fig,'fig_public_chronology');(OUT/'chronology_data.json').write_text(json.dumps(audit,indent=2))

def chronology():
    # User approved the faceted uncertainty envelope; do not restore the
    # rejected overlaid scenario-line chart on a subsequent batch redraw.
    from plot_source_uncertainty_panels_v26 import draw
    draw()

def _obsolete_convergence():
    fig,axes=plt.subplots(3,2,figsize=(7.25,5.6),gridspec_kw={'wspace':.26,'hspace':.32})
    for row,date in enumerate(DATES):
        r=load(ROOT/f'results/robust_v26/{date}','NCCG');assert r is not None
        rows=r['outer_trace'];x=[v['iteration'] for v in rows]
        ax=axes[row,0];ax.plot(x,[v['lower_ratio'] for v in rows],'b-s',ms=3.4,mfc='white',lw=1.15,label='Lower bound')
        ax.plot(x,[np.nan if v['upper_ratio'] is None else v['upper_ratio'] for v in rows],'r-o',ms=3.4,mfc='white',lw=1.15,label='Upper bound')
        ax.set_ylabel('Normalized objective');ax.set_xlabel('Outer iteration')
        if row==0:legend(ax,ncol=2,fontsize=8,loc='best')
        ax=axes[row,1]
        for k in sorted(set(v['outer_iteration'] for v in r['inner_trace'])):
            rr=[v for v in r['inner_trace'] if v['outer_iteration']==k]
            x=[v['inner_iteration'] for v in rr];gap=[np.nan if v['upper_ratio'] is None else max(0,v['upper_ratio']-v['lower_ratio']) for v in rr]
            ax.plot(x,gap,c=COLORS[(k-1)%4],ls=STYLES[(k-1)%4],marker=['o','s','^','D'][(k-1)%4],ms=3,mfc='white',lw=1,label=f'Outer {k}')
        ax.set_yscale('symlog',linthresh=1e-8);ax.set_ylabel('Inner bound gap');ax.set_xlabel('Inner iteration')
        legend(ax,ncol=2,fontsize=7.2,loc='best')
        for col in range(2):
            ax=axes[row,col];ax.tick_params(top=True,right=True,direction='in',pad=2);ax.grid(axis='y',c='.7',lw=.45)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True));ax.text(.03,1.025,f'({chr(97+2*row+col)}) {date}',transform=ax.transAxes,fontsize=9)
    save(fig,'fig_public_algorithm_convergence')

def convergence():
    from plot_revision_v26 import formatting,convergence as draw
    formatting();draw()

def _obsolete_allocations():
    fig,axes=plt.subplots(3,1,figsize=(7.25,5.0),sharex=True,gridspec_kw={'hspace':.045})
    colors=['#ff0000','#00cc00','#0000ff','#ff8000','#00d5dd','#ee00ee']
    for row,(date,ax) in enumerate(zip(DATES,axes)):
        root=ROOT/f'results/robust_v26/{date}';common=load(root,'common');nccg=load(root,'NCCG')
        vals=np.array([load(root/f'scenario_{s:02}','M4_proposed')['allocation_t'] for s in range(5)])/1000
        bottom=np.zeros(5);x=np.arange(5)
        for k,c in enumerate(colors):ax.bar(x,vals[:,k],bottom=bottom,width=.55,color=c,ec='black',lw=.35,label=f'Park {k+1}');bottom+=vals[:,k]
        ax.axhline((common['budget']+nccg['q']).sum()/1000,c='k',ls='--',lw=1.1,label='Total contracted allowance')
        ax.tick_params(top=True,right=True,direction='in',pad=2);ax.grid(axis='y',c='.6',lw=.4);ax.set_axisbelow(True)
        ax.set_ylim(0,max(bottom.max(),(common['budget']+nccg['q']).sum()/1000)*1.27);ax.text(.02,.86,f'({chr(97+row)}) {date}',transform=ax.transAxes,fontsize=9)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=4,loc='upper center',bbox_to_anchor=(.53,1.01),fontsize=8.0,framealpha=1,edgecolor='black',fancybox=False,borderpad=.23)
    fig.text(.015,.50,'Attributed CO$_2$ ($10^3$ t)',rotation=90,va='center',fontsize=10)
    fig.subplots_adjust(left=.095,right=.99,top=.88,bottom=.10)
    axes[-1].set_xticks(range(5),['Nominal','S1','S2','S3','S4']);axes[-1].set_xlabel('Joint source-load scenario')
    save(fig,'fig_dispatch_stacked_v12')

def allocations():
    from plot_revision_v26 import hourly,formatting
    formatting();hourly()

if __name__=='__main__':
    for name in (sys.argv[1:] or ['chronology','convergence','allocations']):globals()[name]()
