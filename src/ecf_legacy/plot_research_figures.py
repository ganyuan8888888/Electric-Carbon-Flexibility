"""Scientific figures built from stored numerical results, with reference layout.

No curve is duplicated as another method. Operating-point sweeps are labeled as
operating points; they are not presented as algorithm comparisons.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

ROOT=__import__("repository_paths").ROOT
OUT=ROOT/'figures';OUT.mkdir(exist_ok=True,parents=True)
plt.rcParams.update({'font.family':'Times New Roman','mathtext.fontset':'stix','font.size':11,
                     'axes.linewidth':.8,'xtick.direction':'in','ytick.direction':'in','savefig.dpi':400,
                     'pdf.fonttype':42,'ps.fonttype':42,'axes.unicode_minus':True})
COLORS=['#ff0000','#00b51a','#003bff','#e87800']
STYLES=['-',':','--','-.']

def save(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=400,bbox_inches='tight',pad_inches=.04)
    fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight',pad_inches=.04)
    fig.savefig(OUT/(name+'.svg'),bbox_inches='tight',pad_inches=.04)
    plt.close(fig)

def public_profiles():
    raw=np.load(ROOT/'results/public_data/chronology.npz')
    audit=json.loads((ROOT/'results/public_data/data_audit.json').read_text())
    fig,axes=plt.subplots(3,1,figsize=(7.25,4.65),sharex=True,sharey=True,gridspec_kw={'hspace':0})
    t=(np.arange(288)+.5)/12;lines=[]
    for j,(ax,day) in enumerate(zip(axes,audit['main_case_days'])):
        sl=slice(day['day_index']*288,(day['day_index']+1)*288)
        series=[raw['load'][sl],raw['wind'][sl],raw['pv'][sl]+raw['rtpv'][sl],raw['net_load'][sl]]
        for x,c,style,label in zip(series,COLORS,STYLES,['Demand','Wind','PV','Net demand']):
            line,=ax.plot(t,x/1000,color=c,ls=style,lw=1.4,label=label)
            if j==0:lines.append(line)
        ax.set_xlim(0,24);ax.set_ylim(-1,7)
        ax.set_yticks([0,2,4,6]);ax.set_xticks(np.arange(0,25,2))
        ax.tick_params(top=True,right=True,length=4,pad=3)
        ax.grid(axis='y',color='#909090',lw=.45,alpha=.7)
        ax.set_ylabel('Power (GW)',labelpad=5)
        ax.text(.015,.75 if j==0 else .90,f'({chr(97+j)}) {day["date"]}   RE share = {day["renewable_share"]:.1%}',transform=ax.transAxes,
                fontsize=10,va='top',bbox={'facecolor':'white','alpha':.82,'edgecolor':'none','pad':1})
    axes[0].legend(handles=lines,loc='upper right',ncol=4,framealpha=1,edgecolor='black',fancybox=False,
                   columnspacing=1.0,handlelength=2.7,borderpad=.25,labelspacing=.2,fontsize=10)
    axes[-1].set_xlabel('Time (h)')
    save(fig,'fig_public_chronology')

def stability_surfaces():
    from harmonic_process import cycle_matrices
    currents=[180.,189.,198.];acds=[.038,.040,.043]
    nu=np.linspace(1.6,2.65,71);amp=np.linspace(0,40,61);aa,nn=np.meshgrid(amp,nu,indexing='ij')
    results=[]
    for acd in acds:
        row=[]
        for mu in currents:
            M,_=cycle_matrices(lambda theta:mu+aa*np.cos(theta),acd,nn)
            ex=np.log(np.max(abs(np.linalg.eigvals(M)),axis=-1))*nn/(2*np.pi)
            row.append(ex)
        results.append(row)
    z=np.array(results)
    np.savez_compressed(OUT/'stability_surface_data.npz',mean_current_ka=currents,acd_m=acds,
                        frequency_normalized=nu,amplitude_ka=amp,exponent=z)
    fig=plt.figure(figsize=(9.25,8.25))
    cmap=plt.get_cmap('jet')
    for r,acd in enumerate(acds):
        # Every column in a row uses the same z range and color scale.
        upper=max(.03,float(np.ceil(np.max(z[r])*20)/20));lower=-.04
        norm=Normalize(vmin=-.005,vmax=upper)
        for c,mu in enumerate(currents):
            ax=fig.add_axes([.035+c*.300,.740-r*.325,.273,.230],projection='3d')
            ax.plot_surface(aa,nn,z[r,c],rstride=2,cstride=2,cmap=cmap,norm=norm,linewidth=.14,
                            edgecolor=(.15,.15,.15,.2),antialiased=True,alpha=.94)
            ax.contourf(aa,nn,z[r,c],zdir='z',offset=lower,levels=np.linspace(-.005,upper,30),cmap=cmap,norm=norm)
            ax.plot_surface(aa[[0,-1]][:,[0,-1]],nn[[0,-1]][:,[0,-1]],np.zeros((2,2)),color='red',alpha=.22,shade=False)
            ax.set(xlim=(0,40),ylim=(1.6,2.65),zlim=(lower,upper))
            ax.set_xticks([0,10,20,30,40]);ax.set_yticks([1.6,2.,2.4,2.6])
            ax.set_zticks(np.linspace(0,upper,4))
            from matplotlib.ticker import FormatStrFormatter
            ax.zaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax.set_xlabel('Amplitude (kA)',labelpad=-1,fontsize=9)
            ax.set_ylabel('Frequency',labelpad=-2,fontsize=9)
            ax.set_zlabel(r'$\alpha_{\max}$',labelpad=-2,fontsize=10)
            ax.tick_params(pad=0,labelsize=8)
            ax.view_init(elev=23,azim=-58);ax.set_box_aspect((1.05,1,.70))
            ax.text2D(.30,.98,rf'$\mu={mu:.0f}\ \mathrm{{kA}}$',transform=ax.transAxes,fontsize=11)
            for axis in [ax.xaxis,ax.yaxis,ax.zaxis]:
                axis.pane.fill=False;axis._axinfo['grid'].update(color=(.55,.55,.55,.5),linewidth=.4)
            ax.text2D(.06,.31,r'$\alpha^{\mathrm{lim}}=0$',fontsize=8,color='#7c0000',zorder=100,
                      transform=ax.transAxes,bbox=dict(facecolor='white',alpha=.92,edgecolor='none',pad=.2))
        cbax=fig.add_axes([.946,.775-r*.325,.012,.16])
        cb=fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),cax=cbax);cb.ax.tick_params(labelsize=9,length=2)
        cb.set_ticks(np.linspace(0,upper,4))
        cb.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        fig.text(.50,.683-r*.325,rf'({chr(97+r)}) ACD = {acd*100:.1f} cm',ha='center',fontsize=12)
    save(fig,'fig_stability_surfaces')

if __name__=='__main__':
    import sys
    if '--surfaces' in sys.argv:stability_surfaces()
    else:public_profiles()
