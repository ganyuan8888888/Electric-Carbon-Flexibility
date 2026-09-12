from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes,mark_inset
from plot_research_figures import ROOT,OUT,save,COLORS,STYLES

def axes_style(ax):
    ax.tick_params(top=True,right=True,direction='in',length=3)
    ax.grid(axis='y',color='.65',lw=.45)

def thermal_profiles():
    cases=['constant','daily_5pct','daily_10pct','two_hour_pulse']
    labels=['Constant','Daily ±5%','Daily ±10%','Two-hour pulses']
    fig,axs=plt.subplots(3,1,figsize=(7.25,5.1),sharex=True,gridspec_kw={'hspace':0})
    for name,label,c,ls in zip(cases,labels,COLORS,STYLES):
        d=np.load(ROOT/f'results/thermal/{name}.npz');x=d['time_h'];s=d['states'];n=s.shape[1]//3
        axs[0].plot(x,s[:,:n].mean(axis=1),color=c,ls=ls,lw=1.2,label=label)
        axs[1].plot(x,s[:,2*n:].mean(axis=1)*100,color=c,ls=ls,lw=1.2)
        axs[2].step(np.r_[0,np.arange(1,49)/2],np.r_[d['power_w']/1e6,d['power_w'][-1]/1e6],where='post',color=c,ls=ls,lw=1.2)
    for j,ax in enumerate(axs):
        axes_style(ax);ax.set_xlim(0,24);ax.set_xticks(np.arange(0,25,2));ax.text(.01,.92,f'({chr(97+j)})',transform=ax.transAxes,va='top')
    axs[0].set_ylabel('Bath temperature (°C)');axs[1].set_ylabel('Ledge thickness (cm)');axs[2].set_ylabel('Cell power (MW)')
    axs[0].legend(loc='lower center',bbox_to_anchor=(.5,1.02),ncol=4,fontsize=9,framealpha=1,fancybox=False,edgecolor='black',handlelength=2.8,labelspacing=.2)
    axs[-1].set_xlabel('Time (h)')
    small=inset_axes(axs[1],width='30%',height='50%',loc='lower right',borderpad=1.2)
    for name,c,ls in zip(cases,COLORS,STYLES):
        d=np.load(ROOT/f'results/thermal/{name}.npz');n=d['states'].shape[1]//3
        small.plot(d['time_h'],d['states'][:,2*n:].mean(axis=1)*100,color=c,ls=ls,lw=.8)
    values=[]
    for name in cases:
        d=np.load(ROOT/f'results/thermal/{name}.npz');n=d['states'].shape[1]//3
        values.extend((d['states'][d['time_h']>=20,2*n:].mean(axis=1)*100).tolist())
    pad=.08*(max(values)-min(values))
    small.set_xlim(20,24);small.set_ylim(min(values)-pad,max(values)+pad);small.tick_params(labelsize=8,direction='in',pad=1)
    save(fig,'fig_thermal_profiles')

def power_retraction():
    d=np.load(ROOT/'results/process/power_retraction.npz');t=d['time_s'];I=d['current_ka']
    if I.shape[0]!=len(t):I=I.T
    fig,axs=plt.subplots(3,1,figsize=(7.25,4.9),sharex=True,gridspec_kw={'hspace':0})
    for j in range(I.shape[1]):axs[0].plot(t,I[:,j],color=COLORS[j],ls=STYLES[j],lw=1.2,label=f'Series {j+1}')
    from process_model import power
    command=d['command_mw'];actual=command+d['residual_mw']
    v=np.array([1.75,1.78]);res=np.array([.0135,.0133]);nc=np.array([180,180])
    steady=sum(power(180,v[j],res[j],nc[j]) for j in range(2));derivative=nc*(v+2*res*180)/1000
    linear_i2=180+(command-steady-derivative[0]*(I[:,0]-180))/derivative[1]
    linear_power=power(I[:,0],v[0],res[0],nc[0])+power(linear_i2,v[1],res[1],nc[1])
    np.savez_compressed(ROOT/'results/process/power_retraction_fair_comparison.npz',time_s=t,command_mw=command,
                        first_order_power_mw=linear_power,first_order_current2_ka=linear_i2,retracted_power_mw=actual)
    axs[1].plot(t,command,'k-',lw=1.3,label='Command')
    axs[1].plot(t,linear_power,color=COLORS[2],ls='--',lw=1.1,label='First-order compensation')
    axs[1].plot(t,actual,color=COLORS[0],ls=':',lw=1,label='Nonlinear retraction')
    axs[2].plot(t,linear_power-command,color=COLORS[2],ls='--',lw=1.1,label='First-order residual')
    axs[2].plot(t,d['residual_mw'],color=COLORS[0],ls='-',lw=1.1,label='Retraction residual')
    for j,ax in enumerate(axs):axes_style(ax);ax.set_xlim(t[0],t[-1]);ax.text(.01,.9,f'({chr(97+j)})',transform=ax.transAxes,va='top')
    axs[0].set_ylabel('Current (kA)');axs[1].set_ylabel('Power (MW)');axs[2].set_ylabel('Residual (MW)');axs[2].set_xlabel('Time (s)')
    axs[0].legend(loc='lower center',bbox_to_anchor=(.5,1.02),ncol=2,fontsize=9,framealpha=1,fancybox=False)
    axs[1].set_ylim(267,278)
    axs[1].legend(loc='upper right',ncol=3,fontsize=8,framealpha=1,fancybox=False)
    small=inset_axes(axs[2],width='29%',height='43%',loc='upper right',borderpad=1.15)
    small.plot(t,linear_power-command,color=COLORS[2],ls='--',lw=.8)
    small.plot(t,d['residual_mw'],color=COLORS[0],lw=.8)
    small.set_xlim(30,45);small.set_ylim(-.03,.2);small.tick_params(labelsize=7,direction='in',pad=1);small.set_title('Residual near zero (MW)',fontsize=8,pad=2)
    axs[2].set_ylim(-.1,2.9)
    save(fig,'fig_power_retraction')

def spectral_geometry():
    rng=np.random.default_rng(20260908);fig,axs=plt.subplots(2,2,figsize=(7.3,6.0),gridspec_kw={'hspace':.33,'wspace':.28})
    stored={}
    from matplotlib.colors import Normalize
    norm=Normalize(1,2.75)
    for n,col in [(2,0),(3,1)]:
        a=rng.uniform(1,1.65,(1800,n));theta=rng.uniform(-np.pi,np.pi,(1800,n));eta=a*np.exp(1j*theta)
        # Fixed total cell count: d_total=n*d and b_total=n*b. Dividing
        # harmonic coefficients and loss by these totals avoids capacity bias.
        s=eta.mean(axis=1);q=(eta**2).mean(axis=1);loss=(abs(eta)**2).mean(axis=1)
        ax=axs[0,col];ax.scatter(s.real,s.imag,c=loss,cmap='viridis',norm=norm,s=5,alpha=.60,linewidths=0)
        ax.set(xlabel=r'Re($c_1/d_\Sigma$)',ylabel=r'Im($c_1/d_\Sigma$)',xlim=(-1.8,1.8),ylim=(-1.8,1.8))
        ax.text(.03,.96,f'({chr(97+col)}) {n} independent series',transform=ax.transAxes,va='top',fontsize=10)
        if n==2:
            z=rng.uniform(1,1.65,1200)*np.exp(1j*rng.uniform(-np.pi,np.pi,1200));eta=np.column_stack([z,-z])
        else:
            z=rng.uniform(1,1.65,(12000,2))*np.exp(1j*rng.uniform(-np.pi,np.pi,(12000,2)))
            third=-z.sum(axis=1);keep=(abs(third)>=1)&(abs(third)<=1.65)
            eta=np.column_stack([z[keep],third[keep]])[:1800]
            eta=np.vstack([eta,np.exp(2j*np.pi*np.arange(3)/3)])
        q=(eta**2).mean(axis=1);loss=(abs(eta)**2).mean(axis=1)
        sc=axs[1,col].scatter(q.real,q.imag,c=loss,cmap='viridis',norm=norm,s=6,alpha=.65,linewidths=0)
        if n==2:
            th=np.linspace(0,2*np.pi,500);axs[1,col].plot(np.cos(th),np.sin(th),'r--',lw=1.1,label='Excluded radius')
            axs[1,col].fill(np.cos(th),np.sin(th),facecolor='none',edgecolor='#ffaaaa',hatch='///',lw=0)
            axs[1,col].legend(loc='upper right',fontsize=8,fancybox=False)
        else:axs[1,col].plot(0,0,'r*',markersize=9,label='Balanced phases');axs[1,col].legend(loc='upper right',fontsize=8,fancybox=False)
        axs[1,col].set(xlabel=r'Re($c_2/b_\Sigma$)',ylabel=r'Im($c_2/b_\Sigma$)',xlim=(-2.9,2.9),ylim=(-2.9,2.9))
        axs[1,col].text(.03,.96,f'({chr(99+col)}) $c_1=0$',transform=axs[1,col].transAxes,va='top',fontsize=10)
        stored[f'eta_{n}']=eta
    for ax in axs.flat:axes_style(ax);ax.set_aspect('equal');ax.grid(False);ax.axhline(0,c='.8',lw=.4,zorder=0);ax.axvline(0,c='.8',lw=.4,zorder=0)
    np.savez_compressed(OUT/'spectral_geometry_samples.npz',**stored)
    from matplotlib.cm import ScalarMappable
    fig.subplots_adjust(bottom=.17)
    cax=fig.add_axes([.27,.035,.50,.016]);cb=fig.colorbar(ScalarMappable(norm=norm,cmap='viridis'),cax=cax,orientation='horizontal')
    cb.set_label(r'Normalized excitation energy, $n^{-1}\sum_i|\eta_i|^2$',fontsize=10);cb.ax.tick_params(labelsize=9)
    save(fig,'fig_spectral_geometry')

def library_diagnostics():
    d=json.loads((ROOT/'results/library/candidate_library_smooth_v2.json').read_text());records=d['records'][1:]
    fig,axs=plt.subplots(3,1,figsize=(7.25,4.8),sharex=True,gridspec_kw={'hspace':0})
    for delta,color,mark in [(9,COLORS[1],'o'),(18,COLORS[0],'s')]:
        rr=[r for r in records if r['delta_ka']==delta];x=[r['up_start_h'] for r in rr]
        v1=[max(s['maximum_temperature_c'] for s in r['phase_summaries']) for r in rr]
        v2=[max(s['terminal_ledge_error_m'] for s in r['phase_summaries'])*100 for r in rr]
        v3=[r['execution']['reachability_upper_bound'] for r in rr]
        for ax,y in zip(axs,[v1,v2,v3]):ax.scatter(x,y,s=24,facecolors='none',edgecolors=color,marker=mark,lw=.8,label=rf'$\Delta I={delta}$ kA')
    for ax in axs:axes_style(ax);ax.set_xlim(0,22);ax.set_xticks(np.arange(0,23,2))
    axs[2].axhline(.05,color='black',ls='--',lw=1,label='Admissible radius')
    axs[0].set_ylabel('Peak temperature (°C)');axs[1].set_ylabel('Terminal ledge\nerror (cm)');axs[2].set_ylabel('State-radius bound')
    axs[0].legend(loc='lower center',bbox_to_anchor=(.5,1.02),ncol=2,fontsize=9,fancybox=False,framealpha=1)
    axs[2].set_xlabel('Start of increased-current interval (h)')
    for j,ax in enumerate(axs):ax.text(.015,.9,f'({chr(97+j)})',transform=ax.transAxes,va='top')
    save(fig,'fig_library_certification')

if __name__=='__main__':
    thermal_profiles();power_retraction();spectral_geometry()
    # The sparse library_diagnostics figure was explicitly rejected by the user.
    # Replacement: redraw_temperature_distribution.py, based on dense boxes.
