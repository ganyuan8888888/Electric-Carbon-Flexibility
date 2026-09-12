"""Separate quantities so finite-scenario uncertainty is visible, not spaghetti."""
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
from green_direct_v18 import make_case
from dispatch_result_io import load
from plot_uncertainty_v26 import save,OUT,DATES

def draw():
    plt.rcParams.update({'font.family':'Times New Roman','mathtext.fontset':'stix','font.size':10,
        'axes.linewidth':.7,'pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(3,3,figsize=(8.15,5.55),sharex=True,sharey='col',
        gridspec_kw={'left':.073,'right':.985,'bottom':.09,'top':.855,'wspace':.22,'hspace':.47})
    colors=['#e60000','#00a832','#003dff'];allvalues=[];audit={}
    for row,date in enumerate(DATES):
        net,_=make_case(date);types=net.renewable['Unit Type'].to_numpy()
        ps=[load(ROOT/f'results/uncertainty_v26/{date}',f'scenario_{s:02}') for s in range(5)]
        base=load(ROOT/f'results/robust_v26/{date}','forecast_common_reference')['base']
        vals=np.array([[p['background_mw'].sum(0)+base.sum(0),
            p['available_renewable_mw'][types=='WIND'].sum(0),
            p['available_renewable_mw'][np.isin(types,['PV','RTPV'])].sum(0)] for p in ps])/1000
        allvalues.append(vals);t=ps[0]['time_h'];bounds=[]
        for col,c in enumerate(colors):
            ax=axes[row,col];lo=vals[:,col].min(0);hi=vals[:,col].max(0);nominal=vals[0,col]
            ax.fill_between(t,lo,hi,facecolor=c,alpha=.22,linewidth=0,zorder=1)
            ax.plot(t,lo,color=c,ls='--',lw=.75,zorder=2)
            ax.plot(t,hi,color=c,ls='--',lw=.75,zorder=2)
            ax.plot(t,nominal,color=c,lw=1.2,marker=['s','o','^'][col],mfc='white',mew=.7,ms=2.7,markevery=6,zorder=3)
            ax.set_xlim(0,24);ax.set_xticks([0,6,12,18,24]);ax.set_xticks(np.arange(0,25,3),minor=True)
            ax.tick_params(which='both',top=True,right=True,direction='in',labelsize=9,pad=2,length=3)
            ax.tick_params(which='minor',length=1.7);ax.grid(axis='y',color='.78',lw=.4);ax.set_axisbelow(True)
            ax.yaxis.set_major_locator(MaxNLocator(3));ax.text(.025,.89,f'({chr(97+row*3+col)})',transform=ax.transAxes,fontsize=9)
            if col==0:ax.set_ylabel('Power (GW)',fontsize=10)
            if row==2:ax.set_xlabel('Time (h)',fontsize=10)
            bounds.append(dict(nominal=nominal.tolist(),lower=lo.tolist(),upper=hi.tolist(),max_span_gw=float((hi-lo).max())))
        y=axes[row,0].get_position().y0-.038 if row<2 else .002
        fig.text(.525,y,date,ha='center',va='top',fontsize=10)
        audit[date]=dict(scenarios=5,panels=bounds,scope='Pointwise projection of five whole-day joint trajectories; not independent timewise uncertainty choices or a confidence interval.')
    data=np.stack(allvalues)
    for col in range(3):
        lo=float(data[:,:,col,:].min());hi=float(data[:,:,col,:].max());span=hi-lo
        axes[0,col].set_ylim(max(0,lo-.10*span) if col==0 else -.025*span,hi+.17*span)
        axes[0,col].set_title(['System demand','Available wind','Available PV'][col],fontsize=11,pad=8)
    # Labels distinguish the meaning of line and fill without listing every
    # individual scenario or using a probability label unsupported by data.
    hands=[Line2D([0],[0],c='.15',lw=1.2,marker='o',mfc='white',ms=3,label='Nominal profile'),
        Line2D([0],[0],c='.15',ls='--',lw=.8,label='Lower / upper bound'),
        Patch(fc='.65',alpha=.35,ec='none',label='Five-scenario envelope')]
    fig.legend(handles=hands,ncol=3,loc='upper center',bbox_to_anchor=(.53,.98),fontsize=9,
        framealpha=1,edgecolor='black',fancybox=False,borderpad=.25,handlelength=2.4,columnspacing=1.4)
    save(fig,'fig_public_chronology')
    (OUT/'chronology_uncertainty_panels_data.json').write_text(json.dumps(audit,indent=2),encoding='utf8')

if __name__=='__main__':draw()
