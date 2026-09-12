"""Original continuous thermal and modal checks for new full-cycle candidates."""
from pathlib import Path
import sys,json,time
import numpy as np
from scipy.integrate import solve_ivp
from validate_library_continuous_v12 import rhs_fast,params,check_rhs
from thermal_process import ThermalCell
from execution_validation import HZ,PHASES
from check_oriented_reachability import evaluate
ROOT=__import__("repository_paths").ROOT;OUT=ROOT/'results/full_cycle_v13';CACHE={}

def retune(site,date,label='cycle_spectral',source_label='cost'):
    from optimize_full_cycle_v13 import scan
    from execution_validation import moment_average
    src=OUT/f'{date}_{source_label}_site{site}';dst=OUT/f'{date}_{label}_site{site}'
    if dst.with_suffix('.json').exists():return
    raw=np.load(src.with_suffix('.npz'));mapping=scan();current=raw['current_ka'];amp=np.interp(current,mapping['means'],mapping['amplitude_values'])
    meta=json.loads((ROOT/'results/library/public_v3/candidate_library.json').read_text());m=ThermalCell(**meta['parameters'][site])
    mu,ae=moment_average(current,amp);avg=m.simulate(mu,ae,dt=300.)
    np.savez_compressed(dst.with_suffix('.npz'),current_ka=current,amplitude_ka=amp,states=avg['states'],power_w=avg['power_w'],signal=raw['signal'])
    dst.with_suffix('.json').write_text(json.dumps(dict(site=site,date=date,label=label,source=src.name,
        scope='Spectrally reconstructed candidate; averaged thermal state recomputed; not admitted until independent actual-carrier and modal validation.'),indent=2))

def validate(site,date='2020-11-29',label='cycle_spectral'):
    source=OUT/f'{date}_{label}_site{site}';dst=OUT/f'{date}_{label}_site{site}_execution'
    if dst.with_suffix('.json').exists():return json.loads(dst.with_suffix('.json').read_text())
    start=time.perf_counter();raw=np.load(source.with_suffix('.npz'));current=raw['current_ka'];amp=raw['amplitude_ka']
    meta=json.loads((ROOT/'results/library/public_v3/candidate_library.json').read_text())
    m=ThermalCell(**meta['parameters'][site]);p=params(m);err=check_rhs(m,p);n=m.n;dim=3*n+2
    y=np.tile(np.r_[m.initial(),0.,0.],3);states=[np.tile(m.initial(),(3,1))];powers=[];production=[]
    minT=1e6;maxT=-1e6;minL=1e6;maxL=-1e6;minI=1e6;maxI=-1e6
    for t in range(48):
        old=180. if t==0 else current[t-1];ao=0. if t==0 else amp[t-1]
        minI=min(minI,old-ao,current[t]-amp[t]);maxI=max(maxI,old+ao,current[t]+amp[t])
        args=(old,current[t],ao,amp[t],p,m.area,m.top_conductance,m.laplacian)
        before=y.reshape(3,dim).copy()
        sol=solve_ivp(rhs_fast,[0,1800],y,args=args,method='DOP853',rtol=2e-9,atol=1e-10,max_step=1/HZ/8,dense_output=True)
        assert sol.success,sol.message
        y=sol.y[:,-1];after=y.reshape(3,dim)
        powers.append((after[:,-2]-before[:,-2])/1800);production.append(after[:,-1]-before[:,-1])
        dense=sol.y.reshape(3,dim,-1)
        minT=min(minT,float(dense[:,:n].min()));maxT=max(maxT,float(dense[:,:n].max()))
        minL=min(minL,float(dense[:,2*n:3*n].min()));maxL=max(maxL,float(dense[:,2*n:3*n].max()))
        for delta in range(300,1801,300):states.append(sol.sol(delta).reshape(3,dim)[:,:3*n])
    states=np.asarray(states).transpose(1,0,2);powers=np.asarray(powers).T;production=np.asarray(production).T
    expected=48*180*1800*1000*.0269815385*m.current_efficiency/(3*96485.33212)
    report=dict(site=site,date=date,label=label,rhs_relative_error=err,
        minimum_temperature_c=minT,maximum_temperature_c=maxT,minimum_ledge_m=minL,maximum_ledge_m=maxL,
        minimum_current_ka=float(minI),maximum_current_ka=float(maxI),
        terminal_temperature_error_c=float(abs(states[:,-1,:n]-states[:,0,:n]).max()),
        terminal_ledge_error_m=float(abs(states[:,-1,2*n:]-states[:,0,2*n:]).max()),
        maximum_temperature_difference_c=float(abs(states[:,:,:n]-raw['states'][None,:,:n]).max()),
        maximum_ledge_difference_m=float(abs(states[:,:,2*n:]-raw['states'][None,:,2*n:]).max()),
        maximum_cell_mean_power_difference_w=float(abs(powers.mean(axis=0)-raw['power_w']).max()),
        production_kg=production.sum(axis=1).tolist(),production_error_kg=float(abs(production.sum(axis=1)-expected).max()))
    report['thermal_passed']=bool(minT>=940 and maxT<=980 and minL>=.04 and maxL<=.25 and report['terminal_temperature_error_c']<=2 and report['terminal_ledge_error_m']<=.015)
    print('actual carrier',report,flush=True)
    modal=evaluate(current,amp,CACHE);report['modal']=modal
    report['certified']=bool(report['thermal_passed'] and modal['sufficient_check_passed'] and minI>=130-1e-5 and maxI<=230+1e-5 and report['production_error_kg']<=1e-5)
    report['runtime_s']=time.perf_counter()-start
    np.savez_compressed(dst.with_suffix('.npz'),states=states,power_w=powers,time_h=np.arange(289)/12,production_kg=production,current_ka=current,amplitude_ka=amp)
    dst.with_suffix('.json').write_text(json.dumps(report,indent=2));print('candidate validation',report,flush=True)
    return report

if __name__=='__main__':
    from optimize_full_cycle_v13 import solve
    from threadpoolctl import threadpool_limits
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('date',nargs='?',default='2020-11-29');parser.add_argument('sites',nargs='*',type=int);parser.add_argument('--label',default='cycle_spectral')
    args=parser.parse_args();date=args.date;label=args.label
    with threadpool_limits(limits=1):
        reports=[]
        for site in args.sites or range(6):
            if not (OUT/f'{date}_cost_site{site}.json').exists():solve(site,date,label='cost')
            if label!='cost':retune(site,date,label)
            reports.append(validate(site,date,label))
    (OUT/f'{date}_{label}_validation_summary.json').write_text(json.dumps(reports,indent=2))
