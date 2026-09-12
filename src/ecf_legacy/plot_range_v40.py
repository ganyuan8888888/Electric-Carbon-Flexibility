"""Closed, filled chronological electric/carbon range sections, all English.

The depicted halfspace sections are explicitly labelled outer approximations.
There is no change-from-reference coordinate and no fabricated smooth boundary.
"""
from pathlib import Path
import sys,json
ROOT=__import__("repository_paths").ROOT
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm,ticker,colors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from dispatch_result_io import load
from revise_comparators_v20 import DATES

plt.rcParams.update({'font.family':'Times New Roman','mathtext.fontset':'stix','font.size':9,
                    'axes.linewidth':.65,'pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white'})

def draw(partial=False,version='v24'):
    dates=[d for d in DATES if all((ROOT/f'results/electric_carbon_outer_{version}_{d}/section_{t:02d}.json').exists() for t in range(0,48,2))]
    if not partial:assert len(dates)==3,dates
    rows=len(dates);fig=plt.figure(figsize=(10.7,3.06*rows));audit=[]
    for row,date in enumerate(dates):
        source=ROOT/f'results/electric_carbon_outer_{version}_{date}'
        data=[load(source,f'section_{t:02d}') for t in range(0,48,2)]
        for col,k in enumerate([1,3,4]):
            ax=fig.add_axes([.020+col*.323,1-(row+1)/rows+.13/rows,.304,.82/rows],projection='3d')
            ys=[];zs=[]
            for ix,item in enumerate(data):
                section=next(s for s in item['sections'] if s['site']==k)
                polygon=np.asarray(section['vertices']);hour=float(item['time_h'])
                ys.extend(polygon[:,0]);zs.extend(polygon[:,1])
                front=np.column_stack([np.full(len(polygon),max(0,hour-.25)),polygon])
                back=np.column_stack([np.full(len(polygon),min(24,hour+.70)),polygon])
                palette=['#ff0000','#ff8000','#ffc000','#ffff00','#bfff00','#60ff00','#00ff00','#00ff80',
                         '#00ffbf','#00ffff','#00bfff','#0080ff','#0040ff','#0000ff','#8000ff','#bf00ff',
                         '#ff00ff','#ff00bf','#ff0040','#ff0000','#ff8000','#ffc000','#ffff00','#bfff00']
                color=colors.to_rgba(palette[ix])
                faces=[front,back]+[np.array([front[i],front[(i+1)%len(front)],back[(i+1)%len(front)],back[i]]) for i in range(len(front))]
                poly=Poly3DCollection(faces,facecolors=[color]*2+[tuple(np.r_[np.array(color[:3])*.85,1])]*(len(faces)-2),
                                     edgecolors=(0,0,0,.95),linewidths=.38,antialiased=True)
                ax.add_collection3d(poly)
            ylo=min(ys);yhi=max(ys);zlo=min(zs);zhi=max(zs)
            ax.set(xlim=(0,24),ylim=(ylo-.02*(yhi-ylo),yhi+.02*(yhi-ylo)),zlim=(0,zhi*1.035))
            ax.set_xticks([0,6,12,18,24]);ax.yaxis.set_major_locator(ticker.MaxNLocator(3));ax.zaxis.set_major_locator(ticker.MaxNLocator(4))
            ax.set_xlabel('Time (h)',fontsize=9,labelpad=0)
            ax.set_ylabel(r'$P^{\mathrm{PCC}}$ (MW)',fontsize=9,labelpad=0)
            ax.zaxis.set_rotate_label(False)
            ax.set_zlabel(r'$\dot C^{\mathrm{ind}}$ (t/h)',fontsize=9,labelpad=1,rotation=90)
            ax.text2D(.50,1.00,f'Park {k}',transform=ax.transAxes,fontsize=10,ha='center')
            ax.view_init(elev=24,azim=-128);ax.set_box_aspect((1.65,1.0,.72));ax.tick_params(labelsize=8,pad=0)
            for axis in [ax.xaxis,ax.yaxis,ax.zaxis]:
                axis.pane.fill=False;axis.pane.set_edgecolor((.7,.7,.7,.3));axis._axinfo['grid'].update(color=(.55,.55,.55,.36),linewidth=.35)
            audit.append(dict(date=date,site=k,sections=24,power_range=[ylo,yhi],carbon_rate_range=[zlo,zhi],
                region_type='McCormick supporting-halfspace outer approximation',
                anchor_residual=max(v['exact_anchor_constraint_residual'] for v in data)))
        fig.text(.5,1-(row+1)/rows+.030/rows,f'({chr(97+row)}) {date}',ha='center',fontsize=11)
    texts=[obj.get_text() for obj in fig.findobj(plt.Text)]
    assert not any(any('\u4e00'<=c<='\u9fff' for c in s) for s in texts)
    stem=ROOT/'revision_v40/figures/fig_electric_carbon_region_v22'
    fig.canvas.draw()
    assert all(ax.zaxis.label.get_rotation()==90 for ax in fig.axes)
    for ext in ['png','pdf','svg']:fig.savefig(stem.with_suffix('.'+ext),dpi=400,bbox_inches='tight',pad_inches=.03)
    (ROOT/'revision_v40/range_figure_audit.json').write_text(json.dumps(dict(english_only=True,title_removed=True,zlabel_rotation=90,panels=audit),indent=2))
    plt.close(fig);print(stem)

if __name__=='__main__':draw(version='v28')
