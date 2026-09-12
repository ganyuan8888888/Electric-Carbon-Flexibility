"""Use approved/reference figure forms with the frozen v26 experiment data."""
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
from dispatch_result_io import load
from green_direct_v18 import make_case as original_case
from proposed_library_v21 import library
from plot_revision_v12 import panel,legend,COLORS,STYLES
from plot_uncertainty_v26 import DATES,LABELS
OUT=ROOT/'revision_v27/figures';OUT.mkdir(parents=True,exist_ok=True)
def save(fig,stem):
    assert not any(any('\u4e00'<=c<='\u9fff' for c in o.get_text()) for o in fig.findobj(plt.Text))
    for ext in ['png','pdf','svg']:fig.savefig(OUT/(stem+'.'+ext),dpi=400,bbox_inches='tight',pad_inches=.04)
    plt.close(fig)
METHODS=['M1_constant','M2_native','M3_native','M4_proposed']

def root(date):return ROOT/f'results/robust_v27/{date}'
def case(date):
    net,_=original_case(date)
    return net,load(ROOT/f'results/uncertainty_v26/{date}','scenario_00')
def result(date,name):
    folder=root(date)/('baselines/scenario_00' if name in ['M2_native','M3_native','Liu2025_execution'] else 'scenario_00')
    r=load(folder,name);assert r is not None,(date,name)
    return r
def formatting():
    plt.rcParams.update({'font.family':'Times New Roman','mathtext.fontset':'stix','font.size':9,'axes.linewidth':.65,'pdf.fonttype':42})
def label(ax,j,date=''):
    panel(ax);ax.tick_params(labelsize=8.5,length=2.7,width=.6)
    ax.grid(axis='y',c='.73',lw=.38)
    ax.text(.02,.86,f'({chr(97+j)}) '+date,transform=ax.transAxes,fontsize=9,bbox=dict(fc='white',ec='none',pad=.2))
def method_legend(fig,top=.992):
    h=[Line2D([0],[0],c=c,ls=ls,lw=1.2,label=l) for c,ls,l in zip(COLORS,STYLES,LABELS)]
    fig.legend(handles=h,ncol=4,loc='upper center',bbox_to_anchor=(.535,top),fontsize=8,
        framealpha=1,edgecolor='black',fancybox=False,borderpad=.25,columnspacing=1.2,handlelength=2.5)

def reference():
    colors=['#ff0000','#00aa00','#0000ff'];fills=['#ffb9bd','#b4cddd','#ffe9be'];marks=['s','o','^']
    fig,ax=plt.subplots(figsize=(7.25,2.55));right=ax.twinx();price=[];green=[];hands=[]
    for d in DATES:
        net,p=case(d);r=load(root(d),'forecast_price_duals');assert r is not None
        t=p['time_h'];price.append(r['economic_nodal_price_per_mwh'][net.industrial_buses[0]])
        green.append(p['available_renewable_mw'][net.direct_source_indices].sum(0)/1000)
    right.set_ylim(.62,0)
    for j in np.argsort([v.sum() for v in green])[::-1]:right.fill_between(t,0,green[j],fc=fills[j],ec=colors[j],lw=.55,alpha=.62)
    ax.set_zorder(right.get_zorder()+1);ax.patch.set_visible(False)
    for j,(d,c,m) in enumerate(zip(DATES,colors,marks)):
        ax.plot(t,price[j],c=c,marker=m,mfc='white',ms=3.1,lw=1.05,markevery=3)
        hands.append(Line2D([0],[0],c=c,marker=m,mfc='white',ms=3,lw=1,label=d[5:]+' price'))
    hands.extend(Patch(fc=fc,ec=c,lw=.6,label=d[5:]+' PV') for d,c,fc in zip(DATES,colors,fills))
    panel(ax);ax.tick_params(right=False);right.tick_params(direction='in',pad=2)
    ax.set_ylabel('Price (USD/MWh)');right.set_ylabel('Available direct PV (GW)');ax.set_xlabel('Time (h)')
    ax.set_ylim(min(np.min(price)-15,-165),max(np.max(price)+60,200));ax.set_yticks([-150,-75,0,75,150])
    legend(ax,hands,ncol=2,fontsize=7.2,loc='upper left');save(fig,'fig_reference_price_background')

def dispatch():
    date=DATES[-1];net,p=case(date);t=p['time_h']
    fig,axes=plt.subplots(3,1,figsize=(7.25,4.7),sharex=True,gridspec_kw={'hspace':.035})
    for name,l,c,ls in zip(METHODS,LABELS,COLORS,STYLES):
        r=result(date,name)
        if name=='M3_native':assert r['checked_feasible'],'Do not present unresolved C-OPF as a feasible schedule'
        vals=[r['pg'].sum(0)/1000,r['pl'].sum(0),(p['available_renewable_mw']-r['pr']).sum(0)/1000]
        for ax,y in zip(axes,vals):ax.plot(t,y,c=c,ls=ls,lw=1.15,label=l)
    for j,(ax,y) in enumerate(zip(axes,['Conventional (GW)','Factory load (MW)','Curtailment (GW)'])):
        label(ax,j);ax.axvspan(20,24,fc='.93',zorder=-2);ax.set_ylabel(y)
    method_legend(fig);fig.subplots_adjust(top=.89,left=.10,right=.985,bottom=.10)
    axes[-1].set_xlabel('Time (h)');save(fig,'fig_public_dispatch_power')

def costs():
    fig,axes=plt.subplots(3,1,figsize=(7.25,5.0),sharex=True,gridspec_kw={'hspace':.05})
    for j,(date,ax) in enumerate(zip(DATES,axes)):
        net,p=case(date);t=p['time_h'];right=ax.twinx();vs=[];es=[]
        def rate(r):return np.array([net.generation_cost(r['pg'][:,q]).sum() for q in range(48)])+150*(p['available_renewable_mw']-r['pr']).sum(0)
        base=rate(load(root(date),'forecast_reference_price25'))
        for m,(name,c,ls) in enumerate(zip(METHODS,COLORS,STYLES)):
            r=result(date,name)
            if name=='M3_native' and not r['checked_feasible']:continue
            v=(rate(r)-base)*.5/1000;e=np.array([net.generation_emissions(r['pg'][:,q]).sum() for q in range(48)])
            assert abs(e.sum()*.5-r['generation_emission_t'])<1e-5
            ax.bar(t+(m-1.5)*.10,v,width=.095,fc=c,ec='black',lw=.25)
            right.plot(t,e,c=c,ls=ls,lw=1);vs.extend(v);es.extend(e)
        lo,hi=min(vs),max(vs);span=max(1,hi-lo);ax.set_ylim(lo-.15*span,hi+.2*span)
        lo,hi=min(es),max(es);span=max(1,hi-lo);right.set_ylim(lo-.15*span,hi+.2*span)
        label(ax,j,date);ax.axhline(0,c='k',lw=.45);ax.yaxis.set_major_locator(MaxNLocator(4));ax.tick_params(right=False)
        right.tick_params(direction='in',pad=2,labelsize=8);right.yaxis.set_major_locator(MaxNLocator(4))
        ax.set_ylabel(r'$\Delta C_t$ ($10^3$ USD)');right.set_ylabel('Source CO$_2$ (t/h)')
        if not result(date,'M3_native')['checked_feasible']:ax.text(.985,.08,'C-OPF: no verified point',ha='right',transform=ax.transAxes,fontsize=7.5)
    method_legend(fig);fig.text(.54,.922,'Bars: operating cost difference     Lines: source emission rate',ha='center',fontsize=8)
    fig.subplots_adjust(top=.885,bottom=.095,left=.10,right=.905);axes[-1].set_xlabel('Time (h)')
    save(fig,'fig_public_dispatch_cost_carbon')

def temperatures():
    fig,axes=plt.subplots(3,1,figsize=(7.25,4.25),sharex=True,sharey=True,gridspec_kw={'hspace':.025})
    allvalues=[];data={}
    for j,(date,ax) in enumerate(zip(DATES,axes)):
        net,p=case(date)
        from native_industrial_v14 import parameters
        models,_=parameters(net)
        fixed=np.repeat(np.concatenate([m.initial()[:12] for m in models])[:,None],48,axis=1)
        native=result(date,'Liu2025_execution')['states'][:,1:,:12].transpose(0,2,1).reshape(-1,48)
        proposed=result(date,'M4_proposed_execution')['states'][:,:,6::6,:12].transpose(0,1,3,2).reshape(-1,48)
        # The two constant-input methods have identical temperatures; draw once.
        for m,(values,c) in enumerate(zip([fixed,native,proposed],[COLORS[0],COLORS[1],COLORS[3]])):
            ax.boxplot([values[:,q] for q in range(48)],positions=np.arange(.5,24.01,.5)+(m-1)*.10,
                widths=.09,patch_artist=True,manage_ticks=False,showfliers=True,
                boxprops=dict(fc=c,ec='black',lw=.3),medianprops=dict(c='black',lw=.4),
                whiskerprops=dict(c='black',lw=.28),capprops=dict(c='black',lw=.28),
                flierprops=dict(marker='d',ms=.7,mfc='black',mec='black',mew=.25))
            allvalues.extend([values.min(),values.max()]);data[f'{date}_{m}']=values
        panel(ax);ax.set_xlim(.2,24.3);ax.set_ylabel(r'$T$ (°C)');ax.grid(axis='y',c='.68',lw=.35)
        ax.text(.025,.035,f'({chr(97+j)}) {date}',transform=ax.transAxes,fontsize=8.5,bbox=dict(fc='white',ec='none',pad=.15),zorder=10)
        ax.tick_params(labelsize=8,length=2.5);ax.axhline(940,c='.4',ls='--',lw=.6);ax.axhline(980,c='.4',ls='--',lw=.6)
    axes[0].set_ylim(5*np.floor(min(allvalues)/5)-2,5*np.ceil(max(allvalues)/5)+2)
    handles=[Patch(fc=c,ec='black',lw=.4,label=l) for c,l in zip([COLORS[0],COLORS[1],COLORS[3]],['Constant control / C-OPF','Multi-time-scale dispatch','Proposed'])]
    fig.legend(handles=handles,ncol=3,loc='upper center',bbox_to_anchor=(.54,.99),fontsize=8,framealpha=1,edgecolor='black',fancybox=False,borderpad=.22)
    fig.subplots_adjust(top=.91,left=.08,right=.985,bottom=.09);axes[-1].set_xticks(np.arange(0,25,2));axes[-1].set_xlabel('Time (h)')
    save(fig,'fig_public_temperature_distribution');np.savez_compressed(OUT/'temperature_data.npz',**data)

def heatmap():
    date=DATES[-1];net,p=case(date);t=p['time_h'];rows=[result(date,n) for n in ['M2_native','M4_proposed']]
    upper=max(float(r['rho'].max()) for r in rows)
    fig=plt.figure(figsize=(7.25,4.65));outer=fig.add_gridspec(2,1,hspace=.25)
    colors=['#ff8000','#ee00ee','#0000ff','#ff0000','#00bb00'];marks=['v','^','o','s','D']
    for j,r in enumerate(rows):
        gs=outer[j].subgridspec(2,2,height_ratios=[1,1.7],width_ratios=[1,.032],hspace=.04,wspace=.045)
        ax=fig.add_subplot(gs[0,0]);heat=fig.add_subplot(gs[1,0],sharex=ax);cax=fig.add_subplot(gs[1,1])
        for k,c,m in zip(range(5),colors,marks):ax.plot(t,r['rho'][net.industrial_buses[k]],c=c,marker=m,ms=2.7,mfc=c,mew=.3,lw=1,markevery=2,label=f'Factory {k+1}')
        panel(ax);ax.tick_params(labelbottom=False);ax.set_ylim(-.06,upper*1.45);ax.set_yticks([.5,1]);ax.set_ylabel(r'$\rho_k$ (t/MWh)',fontsize=9)
        ax.text(.985,.81,'(a) Multi-time-scale dispatch' if j==0 else '(b) Proposed',ha='right',transform=ax.transAxes,fontsize=8,bbox=dict(fc='white',ec='none',pad=.3))
        if j==0:legend(ax,ncol=5,loc='upper left',fontsize=7.2)
        im=heat.imshow(r['rho'],origin='lower',aspect='auto',extent=[0,24,.5,79.5],cmap='jet',vmin=0,vmax=upper,interpolation='nearest')
        panel(heat);heat.grid(False);heat.set_ylim(.5,79.5);heat.set_yticks([1,20,40,60,79]);heat.set_ylabel('Node index');heat.set_xlabel('Time (h)')
        cb=fig.colorbar(im,cax=cax);cb.set_ticks(np.linspace(0,upper,5));cb.ax.set_yticklabels([f'{v:.2f}' for v in np.linspace(0,upper,5)]);cb.ax.tick_params(direction='in',labelsize=8,pad=2)
        cax.set_title('Carbon\n(t/MWh)',fontsize=7.1,pad=5)
    save(fig,'fig_nodal_carbon_v12')

def envelopes():
    net,p=case(DATES[-1]);lib=library(net);P=lib['P'];before=lib['admissible'].copy();base=P[:,0]
    plus=np.maximum(P-base[:,None,:],0)[:,:,p['time_h']>=20].sum(2)*p['dt_h']
    after=before&(plus<=(.10*base[:,0]*4)[:,None]+1e-8);tight=before&(plus<=(.01*base[:,0]*4)[:,None]+1e-8)
    fig,axes=plt.subplots(3,2,figsize=(7.25,4.7),sharex=True,sharey=True,gridspec_kw={'hspace':.06,'wspace':.12})
    for k,ax in enumerate(axes.flat):
        curves=[(after,'#ff8000','-','Process + modal / Recovery 10%'),(tight,'#00aa00',':','Recovery 1%')] if np.array_equal(before,after) else [(before,'#0000ff','--','Process + modal'),(after,'#ff8000','-','Recovery 10%'),(tight,'#00aa00',':','Recovery 1%')]
        for mask,c,ls,l in curves:
            values=P[k,mask[k]];ax.plot(p['time_h'],values.max(0),c=c,ls=ls,lw=1.1,label=l);ax.plot(p['time_h'],values.min(0),c=c,ls=ls,lw=1.1)
        panel(ax);ax.axvspan(20,24,fc='.93',zorder=-2);ax.text(.03,.82,f'({chr(97+k)}) Factory {k+1}',transform=ax.transAxes,fontsize=8.7)
        if k%2==0:ax.set_ylabel('Factory load (MW)')
    lo,hi=P[before].min(),P[before].max();axes[0,0].set_ylim(lo-8,hi+25)
    h,l=axes[0,1].get_legend_handles_labels();fig.legend(h,l,ncol=3,fontsize=8,loc='upper center',bbox_to_anchor=(.52,.995),framealpha=1,edgecolor='black',fancybox=False,borderpad=.22)
    axes[-1,0].set_xlabel('Time (h)');axes[-1,1].set_xlabel('Time (h)');save(fig,'fig_public_candidate_envelopes')
    (OUT/'library_geometry.json').write_text(json.dumps({'before':before.sum(1).tolist(),'after':after.sum(1).tolist(),'tight':tight.sum(1).tolist(),'scope':'Full certified industrial trajectories; pointwise plots do not permit free recombination.'},indent=2))

def hourly():
    """Hourly grouped attribution bars + PCC power lines, as the supplied reference."""
    from matplotlib.ticker import AutoMinorLocator
    fig,axes=plt.subplots(3,1,figsize=(6.8,5.8),sharex=True,gridspec_kw={'hspace':.025})
    x=np.arange(24)+.5; data={}
    for row,(date,ax) in enumerate(zip(DATES,axes)):
        net,p=case(date);right=ax.twinx();co2rows=[];pccrows=[];data[date]={}
        for j,name in enumerate(METHODS):
            r=result(date,name)
            attribution=(r['rho'][net.industrial_buses]*r['pl']).sum(0)
            hourly_co2=(attribution*p['dt_h']).reshape(24,2).sum(1)
            pcc=(r['pl']-r['pr'][net.direct_source_indices]).sum(0)
            hourly_pcc=pcc.reshape(24,2).mean(1)
            assert np.isclose(hourly_co2.sum(),np.sum(r['allocation_t']),atol=1e-6)
            assert np.isclose(hourly_pcc.sum(),pcc.sum()*p['dt_h'],atol=1e-6)
            co2rows.append(hourly_co2);pccrows.append(hourly_pcc)
            ax.bar(x+(j-1.5)*.135,hourly_co2,width=.115,color=COLORS[j],ec='black',lw=.35)
            right.plot(x,hourly_pcc,c=COLORS[j],ls=STYLES[j],lw=1.0)
            data[date][name]=dict(hourly_attributed_co2_t=hourly_co2.tolist(),hourly_pcc_mw=hourly_pcc.tolist(),
                daily_attributed_co2_t=float(hourly_co2.sum()),pcc_energy_mwh=float(hourly_pcc.sum()))
        panel(ax);ax.grid(axis='y',c='.72',lw=.4);right.grid(False)
        ax.set_ylim(0,np.max(co2rows)*1.18);right.set_ylim(min(0,np.min(pccrows)*1.05),np.max(pccrows)*(1.35 if row==0 else 1.12))
        ax.yaxis.set_major_locator(MaxNLocator(4));right.yaxis.set_major_locator(MaxNLocator(4))
        ax.tick_params(which='both',direction='in',top=True,right=False,pad=2,labelsize=8)
        right.tick_params(which='both',direction='in',pad=2,labelsize=8)
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.text(.37 if row==0 else .02,.91,f'({chr(97+row)}) {date}',transform=ax.transAxes,fontsize=8,bbox=dict(fc='white',ec='none',pad=.1))
        ax.set_ylabel('Industrial CO$_2$ (t)',fontsize=8.5,labelpad=4)
        right.set_ylabel('PCC power (MW)',fontsize=8.5,labelpad=4)
    from matplotlib.legend_handler import HandlerTuple
    handles=[(Patch(fc=c,ec='black',lw=.35),Line2D([0],[0],c=c,ls=ls,lw=1.0)) for c,ls in zip(COLORS,STYLES)]
    fig.legend(handles,LABELS,ncol=1,loc='upper left',bbox_to_anchor=(.002,.998),bbox_transform=axes[0].transAxes,
        handler_map={tuple:HandlerTuple(ndivide=None,pad=.35)},
        fontsize=6.6,framealpha=1,edgecolor='black',fancybox=False,borderpad=.18,
        labelspacing=.15,handlelength=3.1,handletextpad=.45,borderaxespad=.2)
    axes[-1].set_xlim(0,24);axes[-1].set_xticks(np.arange(0,25,2));axes[-1].set_xlabel('Time (h)',fontsize=9)
    fig.subplots_adjust(left=.10,right=.90,top=.99,bottom=.075)
    save(fig,'fig_dispatch_stacked_v12')
    (OUT/'hourly_carbon_power_data.json').write_text(json.dumps(dict(scope='Nominal scenario, each typical day. Emissions integrated, power averaged over each complete clock hour; no interpolation.',days=data),indent=2),encoding='utf8')

def _obsolete_convergence():
    fig,axes=plt.subplots(3,2,figsize=(7.25,5.5),gridspec_kw={'wspace':.28,'hspace':.43})
    for row,date in enumerate(DATES):
        r=load(root(date),'NCCG');rr=r['outer_trace'];ax=axes[row,0]
        ax.plot([v['iteration'] for v in rr],[v['lower_ratio'] for v in rr],'b-s',ms=3.5,mfc='white',lw=1.1,label='Lower bound')
        finite=[v for v in rr if v.get('upper_ratio') is not None]
        ax.plot([v['iteration'] for v in finite],[v['upper_ratio'] for v in finite],'r-o',ms=4.5,mfc='white',lw=1.1,label='Upper bound')
        ax.set_xlabel('Outer iteration');ax.set_ylabel('Normalized objective')
        if row==0:legend(ax,ncol=1,fontsize=7.3,loc='best')
        if len(finite)<len(rr):ax.text(.98,.06,r'Initial $UB=+\infty$',ha='right',transform=ax.transAxes,fontsize=7.5)
        ax=axes[row,1]
        for k in sorted(set(v['outer_iteration'] for v in r['inner_trace'])):
            inner=[v for v in r['inner_trace'] if v['outer_iteration']==k]
            c=COLORS[(k-1)%4];ls=STYLES[(k-1)%4];mark=['s','o','^','D'][(k-1)%4]
            ax.plot([v['inner_iteration'] for v in inner],[v['lower_ratio'] for v in inner],c=c,ls=ls,marker=mark,ms=3.3,mfc='white',lw=1,label=f'Outer {k}')
            ff=[v for v in inner if v.get('upper_ratio') is not None]
            if ff:ax.plot([v['inner_iteration'] for v in ff],[v['upper_ratio'] for v in ff],ls='none',marker='x',c=c,ms=5,mew=.8)
        ax.set_xlabel('Inner iteration');ax.set_ylabel('Inner bound')
        legend(ax,ncol=2,fontsize=7,loc='best')
        for col in range(2):
            ax=axes[row,col];ax.tick_params(top=True,right=True,direction='in',pad=2,labelsize=8);ax.grid(axis='y',c='.72',lw=.4)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True));ax.text(.02,1.035,f'({chr(97+2*row+col)}) {date}',transform=ax.transAxes,fontsize=8.5)
            lo,hi=ax.get_ylim();ax.set_ylim(lo,hi+(hi-lo)*.35)
    fig.text(.52,.01,r'Inner curves: lower bounds; crosses: finite upper bounds',ha='center',fontsize=8)
    fig.subplots_adjust(bottom=.115,top=.94);save(fig,'fig_public_algorithm_convergence')

def _outer_only_convergence():
    """User reference: black solid squares UB, red dashed circles LB."""
    from matplotlib.ticker import ScalarFormatter,AutoMinorLocator
    fig,axes=plt.subplots(3,1,figsize=(7.25,4.9),gridspec_kw={'hspace':.34})
    audit={}
    for j,(date,ax) in enumerate(zip(DATES,axes)):
        r=load(root(date),'NCCG');assert 'initialization' in r
        rr=r['outer_trace'];ref=load(root(date),'common')['refs'][0]
        x=np.array([v['iteration'] for v in rr]);lb=np.array([v['lower_ratio'] for v in rr])*ref
        ub=np.array([v['upper_ratio'] for v in rr])*ref
        assert np.isfinite(ub).all() and (np.diff(lb)>=-1e-5).all() and (np.diff(ub)<=1e-5).all()
        assert (ub>=lb-1e-5).all()
        ax.plot(x,ub,c='black',ls='-',marker='s',mfc='white',mec='black',ms=3.5,mew=.8,lw=1.15,label='Upper Bound')
        ax.plot(x,lb,c='#ff3030',ls=(0,(2.5,2.5)),marker='o',mfc='white',mec='#ff3030',ms=3.5,mew=.8,lw=1.0,label='Lower Bound')
        span=max(float(ub.max()-lb.min()),.01*float(ub.max()))
        ax.set_ylim(float(lb.min())-.14*span,float(ub.max())+.25*span)
        ax.set_xlim(x[0],x[-1]);ax.set_xticks(x);ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_major_locator(MaxNLocator(4));ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        f=ScalarFormatter(useMathText=True);f.set_powerlimits((0,0));ax.yaxis.set_major_formatter(f)
        ax.tick_params(which='both',direction='in',top=True,right=True,labelsize=8.5,pad=2)
        ax.tick_params(which='major',length=3.3,width=.7);ax.tick_params(which='minor',length=1.8,width=.65)
        for spine in ax.spines.values():spine.set_linewidth(.8)
        ax.set_ylabel('Objective bound (USD)',fontsize=9);ax.set_xlabel('Iterations',fontsize=9)
        ax.text(.03,.86,f'({chr(97+j)}) {date}',transform=ax.transAxes,fontsize=8.5,bbox=dict(fc='white',ec='none',pad=.2))
        if j==0:legend(ax,ncol=1,loc='upper right',fontsize=7.5)
        audit[date]=dict(iterations=x.tolist(),upper_usd=ub.tolist(),lower_usd=lb.tolist(),initial_upper_ratio=r['initialization']['upper'],
            source='Actual finite-initial-incumbent NCCG rerun, no interpolated or added iterations.')
    fig.subplots_adjust(left=.13,right=.985,top=.96,bottom=.085)
    save(fig,'fig_public_algorithm_convergence')
    (OUT/'convergence_reference_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8')

def convergence():
    """Three dates, outer/inner columns, the user's exact bound line styles."""
    from matplotlib.ticker import ScalarFormatter,AutoMinorLocator
    fig,axes=plt.subplots(3,2,figsize=(7.25,5.65),gridspec_kw={'hspace':.39,'wspace':.30})
    audit={}
    for row,date in enumerate(DATES):
        n=load(root(date),'NCCG');iv=load(root(date),'fixed_quota_inner_convergence');assert iv is not None
        ref=load(root(date),'common')['refs'][0];audit[date]={}
        for col,records in enumerate([n['outer_trace'],iv['trace']]):
            ax=axes[row,col];x=np.array([v['iteration' if col==0 else 'inner_iteration'] for v in records])
            lb=np.array([v['lower_ratio'] for v in records])*ref;ub=np.array([v['upper_ratio'] for v in records])*ref
            assert np.isfinite(ub).all() and (np.diff(lb)>=-1e-5).all() and (np.diff(ub)<=1e-5).all() and (ub>=lb-1e-5).all()
            ax.plot(x,ub,c='black',ls='-',marker='s',mfc='white',mec='black',ms=4.1,mew=.8,lw=1.05,label='Upper Bound')
            ax.plot(x,lb,c='#ff3030',ls=(0,(2.5,2.5)),marker='o',mfc='white',mec='#ff3030',ms=3.3,mew=.8,lw=1.,label='Lower Bound')
            span=max(float(ub.max()-lb.min()),.01*float(ub.max()))
            ax.set_ylim(float(lb.min())-.14*span,float(ub.max())+.36*span)
            ax.set_xlim(x[0]-.05,x[-1]+.05);ax.set_xticks(x);ax.xaxis.set_minor_locator(AutoMinorLocator(2))
            ax.yaxis.set_major_locator(MaxNLocator(4));ax.yaxis.set_minor_locator(AutoMinorLocator(2))
            f=ScalarFormatter(useMathText=True);f.set_powerlimits((0,0));ax.yaxis.set_major_formatter(f);ax.yaxis.get_offset_text().set_fontsize(8)
            ax.tick_params(which='both',direction='in',top=True,right=True,labelsize=8,pad=2)
            ax.tick_params(which='major',length=3.,width=.65);ax.tick_params(which='minor',length=1.7,width=.6)
            for spine in ax.spines.values():spine.set_linewidth(.75)
            ax.set_ylabel('Scaled objective bound',fontsize=8.5);ax.set_xlabel('Outer iterations' if col==0 else 'Inner iterations (fixed quota)',fontsize=8.5)
            ax.text(.03,.88,f'({chr(97+2*row+col)}) {date}',transform=ax.transAxes,fontsize=8,bbox=dict(fc='white',ec='none',pad=.15))
            audit[date]['outer' if col==0 else 'inner']=dict(iterations=x.tolist(),upper_usd=ub.tolist(),lower_usd=lb.tolist())
        audit[date]['inner_scope']=iv['scope']
    h,l=axes[0,0].get_legend_handles_labels()
    fig.legend(h,l,ncol=2,loc='upper center',bbox_to_anchor=(.55,1.005),fontsize=8.5,framealpha=1,edgecolor='black',fancybox=False,borderpad=.24,handlelength=2.5)
    fig.subplots_adjust(left=.10,right=.995,top=.92,bottom=.085)
    save(fig,'fig_public_algorithm_convergence')
    (OUT/'convergence_reference_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8')

def _complete_inner_diagnostic():
    """Actual outer history and every inner oracle, with quota resets separated."""
    from matplotlib.ticker import ScalarFormatter, AutoMinorLocator
    fig,axes=plt.subplots(3,2,figsize=(7.25,5.65),gridspec_kw={'hspace':.46,'wspace':.31})
    exported={}
    for row,date in enumerate(DATES):
        a=json.loads((ROOT/f'revision_v27/iteration_audit/{date}.json').read_text())
        ref=a['refs'][0];exported[date]=a
        for col,ax in enumerate(axes[row]):
            if col==0:
                groups=[a['outer_trace']]; xs=[np.arange(1,len(groups[0])+1)]
            else:
                groups=[];xs=[];offset=0
                for k in range(1,a['outer_iterations']+1):
                    rows=[r for r in a['inner_trace'] if r['outer_iteration']==k]
                    if not rows:continue
                    groups.append(rows);xs.append(np.arange(offset+1,offset+len(rows)+1));offset+=len(rows)
            values=[]
            for records,x in zip(groups,xs):
                lb=np.array([r['lower_ratio'] if r['lower_ratio'] is not None else np.nan for r in records])*ref
                ub=np.array([r['upper_ratio'] if r['upper_ratio'] is not None else np.nan for r in records])*ref
                values.extend(lb[np.isfinite(lb)]);values.extend(ub[np.isfinite(ub)])
                ax.plot(x,ub,c='black',ls='-',marker='s',mfc='white',mec='black',ms=3.8,mew=.8,lw=1.05,label='Upper Bound')
                ax.plot(x,lb,c='#ff3030',ls=(0,(2.5,2.5)),marker='o',mfc='white',mec='#ff3030',ms=3.3,mew=.8,lw=1.,label='Lower Bound')
                if col:
                    k=records[0]['outer_iteration']
                    ax.text((x[0]+x[-1])/2,.85,f'Outer {k}',transform=ax.get_xaxis_transform(),ha='center',fontsize=7.1)
                    if x[0]>1:ax.axvline(x[0]-.5,c='.73',ls=':',lw=.65)
                    infx=[v for v,r in zip(x,records) if r.get('upper_is_infinite') and r['status']!='infeasible']
                    if infx:ax.text((x[0]+x[-1])/2,.73,r'$UB=+\infty$',transform=ax.get_xaxis_transform(),ha='center',fontsize=6.6)
                    cuts=[v for v,r in zip(x,records) if r['status']=='infeasible']
                    for v in cuts:
                        ax.text(v,.035,'Cut',transform=ax.get_xaxis_transform(),ha='center',fontsize=6.8,color='.25')
            lo,hi=min(values),max(values);span=max(hi-lo,.04*abs(hi))
            ax.set_ylim(lo-.12*span,hi+.45*span)
            xmax=int(xs[-1][-1]);ax.set_xlim(.8,xmax+.2)
            ax.set_xticks(np.arange(1,xmax+1));ax.xaxis.set_minor_locator(AutoMinorLocator(2))
            ax.yaxis.set_major_locator(MaxNLocator(3));ax.yaxis.set_minor_locator(AutoMinorLocator(2))
            f=ScalarFormatter(useMathText=True);f.set_powerlimits((0,0));ax.yaxis.set_major_formatter(f);ax.yaxis.get_offset_text().set_fontsize(8)
            ax.tick_params(which='both',direction='in',top=True,right=True,labelsize=8,pad=2)
            for spine in ax.spines.values():spine.set_linewidth(.75)
            ax.set_ylabel('Scaled objective bound',fontsize=8.3)
            ax.set_xlabel('Outer iterations' if col==0 else 'Cumulative inner iterations',fontsize=8.5,labelpad=4)
            ax.text(.03,.94,f'({chr(97+2*row+col)}) {date}',transform=ax.transAxes,fontsize=8,bbox=dict(fc='white',ec='none',pad=.15))
            if col and a['inner_iterations_by_outer'].get(str(a['outer_iterations']))==0:
                ax.text(.97,.43,'Outer 2: no inner call\nOuter bounds already closed',ha='right',transform=ax.transAxes,fontsize=7.1)
    h,l=axes[0,0].get_legend_handles_labels()
    fig.legend(h[:2],l[:2],ncol=2,loc='upper center',bbox_to_anchor=(.55,1.005),fontsize=8.5,framealpha=1,edgecolor='black',fancybox=False,borderpad=.24,handlelength=2.5)
    fig.subplots_adjust(left=.10,right=.995,top=.89,bottom=.085)
    save(fig,'fig_public_algorithm_convergence')
    (OUT/'convergence_reference_audit.json').write_text(json.dumps(exported,indent=2),encoding='utf8')

if __name__=='__main__':
    formatting()
    for f in (sys.argv[1:] or ['reference','dispatch','costs','temperatures','heatmap','envelopes','convergence']):
        globals()[f]();print(f,'complete',flush=True)
