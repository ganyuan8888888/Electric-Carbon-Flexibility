from pathlib import Path
import json
ROOT=__import__("repository_paths").ROOT

def audit(fig,name):
    fig.canvas.draw();groups={}
    for ax in fig.axes:
        position=tuple(round(x,8) for x in ax.get_position().bounds)
        groups.setdefault(position,[]).append(ax)
    pairs=[]
    for group in groups.values():
        if len(group)!=2:continue
        left,right=group
        # The primary axis must label only the left side; the twin only right.
        duplicated_left_labels=any(t.label2.get_visible() for t in left.yaxis.get_major_ticks()+left.yaxis.get_minor_ticks())
        duplicated_right_labels=any(t.label1.get_visible() for t in right.yaxis.get_major_ticks()+right.yaxis.get_minor_ticks())
        assert not duplicated_left_labels and not duplicated_right_labels,name
        pairs.append(dict(left_label=left.get_ylabel(),right_label=right.get_ylabel(),duplicate_right_numbers=False,duplicate_left_numbers=False))
    path=ROOT/'figures/axes_audit_v12.json'
    data=json.loads(path.read_text()) if path.exists() else {}
    data[name]=dict(axes=len(fig.axes),dual_axis_pairs=pairs,passed=True)
    path.write_text(json.dumps(data,indent=2))
