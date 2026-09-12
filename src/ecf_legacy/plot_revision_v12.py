"""Reference-style scientific plots from the four actual computed methods."""
from pathlib import Path
import json,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from plot_research_figures import ROOT,save as save_base
from figure_axes_audit_v12 import audit as audit_axes
from rts_network import make_rts_network,rts_profiles

DATES=['2020-04-07','2020-10-17','2020-11-29']
METHODS=['M1_constant','M2_EP2024','M3_COPF2025','M4_proposed']
COLORS=['#ff0000','#00cc00','#0000ff','#ff8000'];STYLES=['-',':','--','-.']
LABELS=['M1','M2 (2024)','M3 (2025)','M4 (proposed)']
net=make_rts_network();DATA={}
plt.rcParams.update({'font.size':10,'axes.labelsize':11,'legend.fontsize':9,'axes.linewidth':.8})

def save(fig,name):
    audit_axes(fig,name);save_base(fig,name)

def result(date,name):
    p=ROOT/f'results/comparison_v12_{date}'
    return {**json.loads((p/f'{name}.json').read_text()),**dict(np.load(p/f'{name}.npz'))}

def panel(ax,label=None):
    ax.set_xlim(0,24);ax.set_xticks(np.arange(0,25,2));ax.set_xticks(np.arange(0,24.01,.5),minor=True)
    ax.tick_params(which='both',top=True,right=True,labelright=False,direction='in',pad=2,length=3)
    ax.tick_params(which='minor',length=2);ax.grid(axis='y',color='.55',lw=.5);ax.set_axisbelow(True)
    if label:ax.text(.015,.86,label,transform=ax.transAxes,va='top',fontsize=10,bbox=dict(facecolor='white',edgecolor='none',pad=.3))

def legend(ax,handles=None,ncol=4,loc='upper left',**kw):
    return ax.legend(handles=handles,ncol=ncol,loc=loc,framealpha=1,edgecolor='black',fancybox=False,
        borderpad=.22,labelspacing=.18,handlelength=2.2,handletextpad=.35,columnspacing=.8,**kw)

def grid_accuracy():
    steps=[2,5,15,30,60,120,240];colors=['#0000ff','#00cc00','#000000','#ff0000','#ff8000','#00d5dd','#ee00ee'];markers=['>','<','s','o','^','v','d']
    fig,axes=plt.subplots(2,1,figsize=(7.25,4.25),sharex=True,gridspec_kw={'hspace':.035})
    ins=[axes[0].inset_axes([.70,.49,.27,.30]),axes[1].inset_axes([.075,.10,.25,.35])]
    zoom=[(18,22),(2.5,6)];ranges=[[],[]]
    for step,color,mark in zip(steps,colors,markers):
        d=np.load(ROOT/f'results/time_grid/step_{step}min.npz');t=d['time_h'];temp=d['states'][:,:12].mean(axis=1)
        tp=(d['edges_h'][1:]+d['edges_h'][:-1])/2;p=d['power_w']/1e6
        label=f'{step} min' if step<60 else f'{step//60} h'
        for j,(x,y) in enumerate([(t,temp),(tp,p)]):
            style=dict(color=color,lw=1.05,ls='--' if step==15 else '-',marker=mark,ms=3.1,mfc='white',mew=.65,markevery=max(1,len(x)//15))
            axes[j].plot(x,y,label=label,**style);ins[j].plot(x,y,**style)
            sample=np.interp(np.linspace(*zoom[j],81),x,y);ranges[j].extend(sample)
    for j,ax in enumerate(axes):
        panel(ax);lo,hi=ax.get_ylim();ax.set_ylim(lo,hi+(hi-lo)*(.19 if j==0 else .07))
        ax.text(.016,.84,f'({chr(97+j)})',transform=ax.transAxes)
        ins[j].set_xlim(*zoom[j]);a,b=min(ranges[j]),max(ranges[j]);ins[j].set_ylim(a-.12*(b-a),b+.12*(b-a))
        ins[j].tick_params(which='both',top=True,right=True,direction='in',labelsize=8,pad=1,length=2)
        ins[j].set_xticks(np.arange(np.ceil(zoom[j][0]),zoom[j][1]+.01,1))
        patch,c1,c2=mark_inset(ax,ins[j],loc1=2 if j==0 else 1,loc2=4 if j==0 else 3,fc='none',ec='.25',ls='--',lw=.65)
        for artist in [patch,c1,c2]:artist.set_clip_path(ax.patch)
    legend(axes[0],ncol=7,fontsize=8,loc='upper center')
    axes[0].set_ylabel(r'$\overline{T}$ (°C)');axes[1].set_ylabel(r'$P^{\rm cell}$ (MW)');axes[1].set_xlabel('Time (h)')
    save(fig,'fig_time_grid_accuracy')

def stacked_dispatch():
    fig,axes=plt.subplots(3,1,figsize=(7.25,5.65),sharex=True,gridspec_kw={'hspace':.025})
    fuels=net.fossil['Fuel'].to_numpy();types=net.renewable['Unit Type'].to_numpy()
    handles=[Patch(facecolor=c,edgecolor='black',lw=.4,label=l) for c,l in zip(COLORS,['Coal','Gas + oil','Hydro + nuclear','Wind + solar'])]
    handles += [Line2D([0],[0],color='black',lw=1.2,marker='o',ms=3,mfc='white',label='Total demand'),Line2D([0],[0],color='.25',ls='--',lw=1.15,label='Curtailment (right)')]
    for j,(date,ax) in enumerate(zip(DATES,axes)):
        r=result(date,'M4_proposed');p=rts_profiles(net,date);t=p['time_h'];right=ax.twinx()
        components=np.array([r['pg'][fuels=='Coal'].sum(axis=0),r['pg'][fuels!='Coal'].sum(axis=0),r['pr'][np.isin(types,['HYDRO','ROR','NUCLEAR'])].sum(axis=0),r['pr'][np.isin(types,['WIND','PV','RTPV'])].sum(axis=0)])
        demand=p['background_mw'].sum(axis=0)+r['pl'].sum(axis=0)
        assert abs(components.sum(axis=0)-demand).max()<1e-5
        base=np.zeros(48)
        for y,c in zip(components,COLORS):
            ax.bar(t,y/1000,bottom=base/1000,width=.40,color=c,edgecolor='black',linewidth=.3);base+=y
        ax.plot(t,demand/1000,'k-',lw=1.15,marker='o',ms=2.3,mfc='white',markevery=3)
        curt=(p['available_renewable_mw']-r['pr']).sum(axis=0)
        right.plot(t,curt/1000,c='.25',ls='--',lw=1.15)
        ax.set_ylim(0,demand.max()/1000*(1.4 if j==0 else 1.16));right.set_ylim(0,max(.1,curt.max()/1000*1.2))
        right.set_ylabel('Curtailment (GW)');right.tick_params(direction='in',pad=2);right.yaxis.set_major_locator(MaxNLocator(nbins=4,prune='upper'))
        panel(ax);ax.tick_params(axis='y',which='both',right=False,labelright=False);right.tick_params(axis='y',which='both',left=False,labelleft=False,right=True,labelright=True);ax.set_ylabel('Supply (GW)')
        ax.text(.016,.66 if j==0 else .94,f'({chr(97+j)}) {date}',transform=ax.transAxes,va='top',fontsize=9)
        DATA[f'stacked_{date}_source_mw']=components;DATA[f'stacked_{date}_demand_mw']=demand;DATA[f'stacked_{date}_curt_mw']=curt
    legend(axes[0],handles,ncol=3,loc='upper left',fontsize=8.4)
    axes[-1].set_xlabel('Time (h)');save(fig,'fig_dispatch_stacked_v12')

def dispatch_curves():
    date=DATES[-1];prof=rts_profiles(net,date);t=prof['time_h']
    fig,axes=plt.subplots(3,1,figsize=(7.25,4.75),sharex=True,gridspec_kw={'hspace':0})
    for name,label,c,ls in zip(METHODS,LABELS,COLORS,STYLES):
        r=result(date,name);series=[r['pg'].sum(axis=0)/1000,r['pl'].sum(axis=0), (prof['available_renewable_mw']-r['pr']).sum(axis=0)/1000]
        for ax,y in zip(axes,series):ax.plot(t,y,color=c,ls=ls,lw=1.25,label=label)
        for j,y in enumerate(series):DATA[f'dispatch_{name}_{j}']=y
    for j,ax in enumerate(axes):panel(ax);ax.text(.018,.90,f'({chr(97+j)})',transform=ax.transAxes,va='top')
    lo,hi=axes[0].get_ylim();axes[0].set_ylim(lo,hi+.3*(hi-lo));legend(axes[0],ncol=4,loc='upper right',fontsize=8.6)
    axes[0].set_ylabel(r'$P^{\rm G}$ (GW)');axes[1].set_ylabel(r'$P^{\rm ind}$ (MW)');axes[2].set_ylabel(r'$P^{\rm curt}$ (GW)');axes[-1].set_xlabel('Time (h)')
    save(fig,'fig_public_dispatch_power')

def costs():
    fig,axes=plt.subplots(3,1,figsize=(7.25,5.4),sharex=True,gridspec_kw={'hspace':.055})
    handles=[Patch(facecolor=c,edgecolor='black',lw=.35,label=f'M{m+1} cost') for m,c in enumerate(COLORS)]
    handles += [Line2D([0],[0],color=c,ls=ls,lw=1.1,label=f'M{m+1} carbon') for m,(c,ls) in enumerate(zip(COLORS,STYLES))]
    for j,(date,ax) in enumerate(zip(DATES,axes)):
        prof=rts_profiles(net,date);t=prof['time_h'];right=ax.twinx();values=[];carbon=[]
        def hourly(r):return np.array([net.generation_cost(r['pg'][:,q]).sum() for q in range(48)])+150*(prof['available_renewable_mw']-r['pr']).sum(axis=0)
        reference=hourly(result(date,'economic_lower_bound'))
        for m,(name,c,ls) in enumerate(zip(METHODS,COLORS,STYLES)):
            r=result(date,name);cost=(hourly(r)-reference)*.5/1000;allocated=r['carbon_rate'].sum(axis=0)
            assert abs(cost.sum()*1000-(r['objective_actual']-result(date,'economic_lower_bound')['objective_actual']))<1e-5
            assert abs(allocated.sum()*.5-r['allocation_t'].sum())<1e-6
            ax.bar(t+(m-1.5)*.1,cost,width=.095,color=c,edgecolor='black',lw=.28)
            right.plot(t,allocated,color=c,ls=ls,lw=1.1);values.extend(cost);carbon.extend(allocated)
            DATA[f'cost_{date}_{name}']=cost;DATA[f'carbon_{date}_{name}']=allocated
        lo,hi=min(values),max(values);span=max(.1,hi-lo);ax.set_ylim(min(0,lo)-.13*span,hi+(.8 if j==0 else .2)*span)
        lo,hi=min(carbon),max(carbon);span=max(1,hi-lo);right.set_ylim(lo-.18*span,hi+(.8 if j==0 else .25)*span)
        ax.axhline(0,color='black',lw=.5);panel(ax);ax.yaxis.set_major_locator(MaxNLocator(nbins=4,prune='upper'));ax.tick_params(axis='y',which='both',right=False,labelright=False);right.tick_params(axis='y',which='both',left=False,labelleft=False,right=True,labelright=True,direction='in',pad=2)
        right.yaxis.set_major_locator(MaxNLocator(nbins=4,prune='upper'))
        ax.set_ylabel(r'$\Delta C_t$ ($10^3$ USD)');right.set_ylabel(r'Allocated CO$_2$ (t/h)')
        ax.text(.015,.61 if j==0 else .9,f'({chr(97+j)}) {date}',transform=ax.transAxes,fontsize=9,bbox=dict(facecolor='white',edgecolor='none',pad=.3))
    legend(axes[0],handles,ncol=4,fontsize=8.2)
    axes[-1].set_xlabel('Time (h)');save(fig,'fig_public_dispatch_cost_carbon')

def temperatures():
    fig,axes=plt.subplots(3,1,figsize=(7.25,5.15),sharex=True,sharey=True,gridspec_kw={'hspace':.035})
    for j,(date,ax) in enumerate(zip(DATES,axes)):
        for m,(name,color) in enumerate(zip(METHODS,COLORS)):
            r=result(date,name);samples=[]
            for k,b in enumerate(r['branches']):
                s=np.load(ROOT/f'results/execution/public_v12/site{k}_candidate{b}.npz')['states']
                samples.append(s[:,12::12,:12].transpose(0,2,1).reshape(-1,24))
            values=np.vstack(samples)
            ax.boxplot([values[:,q] for q in range(24)],positions=np.arange(1,25)+(m-1.5)*.18,widths=.16,patch_artist=True,manage_ticks=False,whis=1.5,showfliers=True,
                boxprops=dict(facecolor=color,edgecolor='black',lw=.4),medianprops=dict(color='black',lw=.6),whiskerprops=dict(color='black',lw=.4),capprops=dict(color='black',lw=.4),flierprops=dict(marker='d',ms=.9,mfc='black',mec='black',mew=0))
            DATA[f'temp_{date}_{name}']=values
        ax.set_ylim(939,981);ax.set_yticks([940,950,960,970,980]);ax.set_ylabel(r'$T$ (°C)');panel(ax);ax.set_xlim(.35,24.65)
        ax.text(.03,.095,f'({chr(97+j)}) {date}',transform=ax.transAxes,fontsize=9)
    legend(axes[0],[Patch(facecolor=c,edgecolor='black',label=l) for c,l in zip(COLORS,LABELS)],ncol=4,loc='upper center',fontsize=8.4)
    axes[-1].set_xticks(np.arange(1,25));axes[-1].set_xlabel('Time (h)');save(fig,'fig_public_temperature_distribution')

def carbon_heatmap():
    date=DATES[-1];fig=plt.figure(figsize=(7.25,5.55));gs=fig.add_gridspec(4,2,width_ratios=[1,.032],height_ratios=[1,1.85,1,1.85],hspace=.04,wspace=.025)
    nodeids=[113,118,213,218,313];indices=[int(np.flatnonzero(net.busdata['Bus ID'].to_numpy()==x)[0]) for x in nodeids]
    colors=['#ff8000','#ee00ee','#0000ff','#ff0000','#00cc00'];marks=['v','^','o','s','D'];t=(np.arange(48)+.5)/2
    upper=max(result(date,n)['rho'].max() for n in ['M2_EP2024','M4_proposed'])
    for j,name in enumerate(['M2_EP2024','M4_proposed']):
        r=result(date,name);ax=fig.add_subplot(gs[2*j,0]);heat=fig.add_subplot(gs[2*j+1,0]);cax=fig.add_subplot(gs[2*j+1,1])
        for i,c,m,label in zip(indices,colors,marks,nodeids):ax.plot(t,r['rho'][i],color=c,marker=m,mfc=c,ms=3,lw=1,markevery=2,label=f'Node {label}')
        panel(ax);ax.set_ylabel(r'$\rho_n$ (t/MWh)');ax.tick_params(labelbottom=False)
        a,b=ax.get_ylim();ax.set_ylim(a,b+(b-a)*(.37 if j==0 else .12));ax.text(.03,.45,'(a) M2 (2024)' if j==0 else '(b) M4 (proposed)',transform=ax.transAxes,fontsize=9,bbox=dict(facecolor='white',edgecolor='none',pad=.3))
        im=heat.imshow(r['rho'],origin='lower',aspect='auto',extent=[0,24,.5,73.5],cmap='jet',vmin=0,vmax=upper,interpolation='nearest')
        heat.set_ylabel('Bus index');heat.set_yticks([1,20,40,60,73]);panel(heat);heat.grid(False)
        cb=fig.colorbar(im,cax=cax);cb.set_label('Carbon intensity\n(t/MWh)',fontsize=9);cb.ax.tick_params(labelsize=8,direction='in',pad=2)
        if j==0:legend(ax,ncol=5,loc='upper center',fontsize=8)
        if j==1:heat.set_xlabel('Time (h)')
        else:heat.tick_params(labelbottom=False)
        DATA[f'heatmap_{name}']=r['rho']
    save(fig,'fig_nodal_carbon_v12')

def convergence():
    fig,axes=plt.subplots(3,2,figsize=(7.25,5.25),sharex='col',gridspec_kw={'hspace':.06,'wspace':.20})
    for j,date in enumerate(DATES):
        lower=result(date,'economic_lower_bound')['relaxation_lower_bound']
        stages=[('M4_economic_start','Economic start','#ff8000','-'),('M4_constant_start','Constant start','#0000ff','--')]
        for name,label,color,ls in stages:
            h=result(date,name)['history'];x=np.array([r['iteration'] for r in h]);c=np.array([r['cost']-lower for r in h])/1000;v=np.array([r['violation_t'] for r in h]);ok=v<=1e-4
            axes[j,0].plot(x,c,color=color,ls=ls,lw=1.1,label=label)
            axes[j,0].plot(x[ok],c[ok],'o',color=color,ms=2.5)
            axes[j,1].semilogy(x,np.maximum(v,1e-5),color=color,ls=ls,lw=1.1)
            DATA[f'convergence_{date}_{name}']=np.column_stack([x,c,v])
        for ax in axes[j]:
            ax.tick_params(which='both',top=True,right=True,labelright=False,direction='in',pad=2,length=3);ax.grid(axis='y',color='.6',lw=.4);ax.set_xlim(0,32)
            ax.set_xticks(np.arange(0,33,8));ax.set_xticks(np.arange(0,33,2),minor=True)
        axes[j,0].text(.04 if j==0 else .60,.12 if j==0 else .84,date,transform=axes[j,0].transAxes,fontsize=8.5)
        axes[j,0].set_ylabel(r'$C-\underline{C}$ ($10^3$ USD)')
        axes[j,1].set_ylabel('Budget excess (t)');axes[j,1].set_ylim(4e-6,1e4)
    legend(axes[0,0],ncol=1,loc='upper right',fontsize=8.2)
    axes[-1,0].set_xlabel('Iteration');axes[-1,1].set_xlabel('Iteration');save(fig,'fig_public_algorithm_convergence')

def main():
    grid_accuracy()
    if len(sys.argv)>1 and sys.argv[1]=='grid':return
    if len(sys.argv)>1 and sys.argv[1]=='stack':stacked_dispatch();return
    stacked_dispatch();dispatch_curves();costs();temperatures();carbon_heatmap();convergence()
    np.savez_compressed(ROOT/'figures/revision_v12_plot_data.npz',**DATA)
    (ROOT/'figures/revision_v12_figure_scope.json').write_text(json.dumps(dict(methods=METHODS,method_count=4,
        stacks='Additive physical source components of M4; independent power balance checked.',
        cost_carbon='Process illustration: period cost difference from the same-day no-carbon-budget economic reference, with industrial allocated carbon rate. Main performance comparisons use actual total operating cost, total curtailed energy and total source emissions.',
        boxes='216 actual spatial states per method and hour; not uncertainty samples.',heatmap='Actual DC proportional-sharing nodal carbon intensities; not voltage or inundation.'),indent=2))

if __name__=='__main__':main()
