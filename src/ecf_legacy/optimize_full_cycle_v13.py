"""Generate full-day industrial trajectories using the unchanged thermal model.

Research candidates only: averaged thermal optimization is followed by actual
carrier integration and the original modal certificate before admission. The
validated library is used by the proposed method. Source-native comparators
do not receive it; old shared-library results remain supplementary diagnostics.
"""
from pathlib import Path
import os,sys,json,time
ROOT=__import__("repository_paths").ROOT
_dll_path=__import__("pathlib").Path(__import__("casadi").__file__).resolve().parent
_dll=os.add_dll_directory(str(_dll_path)) if os.name=='nt' else None
os.environ['PATH']=str(_dll_path)+os.pathsep+os.environ.get('PATH','')
import numpy as np
import casadi as ca
from thermal_process import ThermalCell
from harmonic_process import cycle_matrices
from execution_validation import NU,moment_average
OUT=ROOT/'results/full_cycle_v13'
OUT.mkdir(exist_ok=True)

def scan():
    dst=OUT/'excitation_scan_refined.npz'
    if dst.exists():return dict(np.load(dst))
    means=np.r_[np.arange(130.,195.),np.arange(195.,211.001,.05)];amps=np.arange(0.,40.01,.5)
    mu,am=np.meshgrid(means,amps,indexing='ij')
    M,_=cycle_matrices(lambda s:mu+am*np.cos(s),.043,NU,steps=512)
    exponent=np.log(np.max(abs(np.linalg.eigvals(M)),axis=-1))*NU/(2*np.pi)
    exponent[(mu-am<130)|(mu+am>230)]=1e6
    best=amps[np.argmin(exponent,axis=1)]
    fine=np.maximum(0,best[:,None]+np.linspace(-.5,.5,201)[None,:]);mu=means[:,None]
    M,_=cycle_matrices(lambda s:mu+fine*np.cos(s),.043,NU,steps=512)
    ef=np.log(np.max(abs(np.linalg.eigvals(M)),axis=-1))*NU/(2*np.pi)
    valid=(ef<=-.004)&(mu-fine>=130)&(mu+fine<=230)
    chosen=np.full(len(means),np.nan)
    for j,v in enumerate(means):
        if valid[j].any():
            # Center of the detected stable component provides interpolation
            # margin. Every resulting whole trajectory is rechecked directly.
            ids=np.flatnonzero(valid[j]);groups=np.split(ids,np.flatnonzero(np.diff(ids)>1)+1)
            group=min(groups,key=lambda v:fine[j,v[0]])
            chosen[j]=float(fine[j,group[len(group)//2]])
        if v<195 and ef[j,0]<=-.004:chosen[j]=0.
    keep=np.isfinite(chosen);grid=means[keep];aa=chosen[keep]
    trial=np.sort(np.r_[grid,*(grid[:-1]+q*np.diff(grid) for q in [.25,.5,.75])]);at=np.interp(trial,grid,aa)
    M,_=cycle_matrices(lambda s:trial+at*np.cos(s),.043,NU,steps=512)
    et=np.log(np.max(abs(np.linalg.eigvals(M)),axis=-1))*NU/(2*np.pi)
    assert et.max()<-.0012,('interpolation validation',trial[et.argmax()],at[et.argmax()],et.max())
    z=dict(means=grid,amplitude_values=aa,checked_means=trial,checked_exponents=et,refined_scan_means=means,refined_amplitudes=fine,refined_exponents=ef)
    np.savez_compressed(dst,**z)
    print('refined excitation scan',dict(min_mean=float(grid.min()),max_mean=float(grid.max()),maximum_checked_exponent=float(et.max()),points=len(grid)),flush=True)
    return z

def solve(site,date,signal=None,label='cost',steps_per_period=6,execution_target=None,passive=False,command_groups=None,refined_spectrum=False,midpoint=False,mean_ceiling=None,warm_start_label=None):
    start=time.perf_counter();meta=json.loads((ROOT/'results/library/public_v3/candidate_library.json').read_text())
    m=ThermalCell(**meta['parameters'][site]);scan_data=dict(np.load(OUT/'excitation_scan.npz'))
    valid=scan_data['selected']>=0;grid=scan_data['means'][valid];amps=scan_data['amplitudes'][scan_data['selected'][valid]]
    if refined_spectrum:
        exact=scan();grid=exact['means'];amps=exact['amplitude_values']
    if mean_ceiling is not None:
        endpoint=float(np.interp(mean_ceiling,grid,amps));keep=grid<mean_ceiling
        grid=np.r_[grid[keep],mean_ceiling];amps=np.r_[amps[keep],endpoint]
    if passive:
        from scipy.optimize import brentq
        from harmonic_process import matrix
        limit=brentq(lambda v:np.real(np.linalg.eigvals(matrix(v,.043))).max()+.0012,180,210)
        grid=np.array([130.,limit]);amps=np.zeros(2)
    # This inexpensive amplitude surrogate is used for candidate generation.
    # The refined spectral map subsequently reconstructs the actual waveform;
    # no interpolated surrogate is admitted without direct validation.
    if signal is None:
        prices=np.load(ROOT/f'results/reference_prices/{date}.npz')
        signal=prices['industrial_price'][site]
    signal=np.asarray(signal,dtype=float);assert signal.shape==(48,),signal.shape
    amplitude=ca.interpolant('amp','linear',[grid],amps)
    J=48*steps_per_period;dt=1800/steps_per_period;n=m.n
    op=ca.Opti();mu=op.variable(48)
    x=op.variable(3*n,J+1)
    center=np.r_[np.full(n,960.),np.full(n,300.),np.full(n,m.ledge0)]
    scale=np.r_[np.full(n,20.),np.full(n,100.),np.full(n,.1)]
    state=ca.repmat(ca.DM(center),1,J+1)+ca.repmat(ca.DM(scale),1,J+1)*x
    op.subject_to(op.bounded(grid[0],mu,grid[-1]));op.subject_to(mu[-1]==180.)
    if execution_target is not None:
        # This is a cost-blind external execution projection, not a published
        # baseline optimizer. It receives no price, carbon or trajectory data.
        op.subject_to(op.bounded(-18.,ca.vertcat(mu[0]-180,mu[1:]-mu[:-1]),18.))
        if command_groups is not None:
            same=np.flatnonzero(np.asarray(command_groups)[1:]==np.asarray(command_groups)[:-1])+1
            if len(same):op.subject_to(mu[same]==mu[same-1])
    op.subject_to(state[:,0]==m.initial())
    op.subject_to(op.bounded(940.5,state[:n,:],979.5))
    op.subject_to(op.bounded(.041,state[2*n:,:],.249))
    op.subject_to(op.bounded(m.initial()[:n]-1.5,state[:n,-1],m.initial()[:n]+1.5))
    op.subject_to(op.bounded(m.initial()[2*n:]-.013,state[2*n:,-1],m.initial()[2*n:]+.013))
    currents=[];seconds=[];powers=[]
    area=ca.DM(m.area);top=ca.DM(m.top_conductance);lap=ca.DM(m.laplacian)
    def energy(z):
        T,S,l=z[:n],z[n:2*n],z[2*n:]
        mass=m.bath_mass0-m.ledge_density*area*(l-m.ledge0)
        return ca.vertcat((mass*m.bath_cp+m.metal_mass/n*m.metal_cp)*(T-m.nominal_liquidus),
            m.shell_capacity_total/n*(S-m.ambient),-m.ledge_density*area*m.fusion_heat*l)
    e_scale=ca.DM(np.r_[np.full(n,3e6),np.full(n,m.shell_capacity_total/n),m.ledge_density*m.area*m.fusion_heat])
    for t in range(48):
        old=180. if t==0 else mu[t-1];ao=0. if t==0 else amplitude(mu[t-1]);a=amplitude(mu[t]);d=mu[t]-old;da=a-ao
        # Three-series aggregate moments, including the smooth first cycle.
        mean=mu[t]-d/(2*81)
        second=mu[t]**2-(old*d+.625*d*d)/81+.5*(a*a-(ao*da+.625*da*da)/81)
        currents.append(mean);seconds.append(second);period_power=0
        for j in range(t*steps_per_period,(t+1)*steps_per_period):
            z=(state[:,j+1]+state[:,j])/2 if midpoint else state[:,j+1]
            T,S,l=z[:n],z[n:2*n],z[2*n:]
            conductance=ca.exp(m.temperature_resistivity*(T-960))/(n*m.resistance_mohm)
            req=1/ca.sum1(conductance);fraction=conductance*req
            power=1000*(m.v_reversible*mean+req*second)
            prod=mean*1000*.0269815385*m.current_efficiency/(3*96485.33212)
            heat=fraction*(power-1000*m.external_resistance_mohm*second-prod*6.60*3.6e6)
            mass=m.bath_mass0-m.ledge_density*area*(l-m.ledge0)
            liq=m.nominal_liquidus-4.5*(10.5*m.bath_mass0/mass-10.5)
            bath=area*m.bath_ledge_h*(T-liq);shell=area*(liq-S)/(l/m.ledge_conductivity+m.wall_resistance)
            rate=ca.vertcat(heat-lap@T-bath-top*(T-m.ambient),shell-area*m.shell_h*(S-m.ambient),bath-shell)
            op.subject_to((energy(state[:,j+1])-energy(state[:,j])-dt*rate)/e_scale==0)
            period_power+=power/steps_per_period
        powers.append(period_power)
    op.subject_to(ca.sum1(ca.vertcat(*currents))==48*180)
    pp=ca.vertcat(*powers);norm=max(float(np.max(abs(signal))),1.)
    if execution_target is None:
        op.minimize(ca.dot(ca.DM(signal/norm),pp)/1e6+1e-7*ca.sumsqr(mu[1:]-mu[:-1]))
    else:
        target=np.asarray(execution_target);assert target.shape==(48,)
        op.minimize(ca.sumsqr((mu-ca.DM(target))/18))
    op.set_initial(mu,180);op.set_initial(x,np.tile(((m.initial()-center)/scale)[:,None],(1,J+1)))
    if warm_start_label is not None:
        seed=np.load(OUT/f'{date}_{warm_start_label}_site{site}.npz')
        if seed['states'].shape[1]==3*n:
            warm_states=np.vstack([np.interp(np.linspace(0,1,J+1),np.linspace(0,1,len(seed['states'])),seed['states'][:,j]) for j in range(3*n)]).T
            op.set_initial(mu,np.clip(seed['current_ka'],grid[0],grid[-1]))
            op.set_initial(x,((warm_states-center)/scale).T)
    op.solver('ipopt',{'expand':True,'print_time':False},{'print_level':0,'max_iter':700,'tol':1e-8,'acceptable_tol':1e-7,'max_cpu_time':240.,'linear_solver':'mumps'})
    try:sol=op.solve();status=sol.stats()['return_status'];get=sol.value
    except RuntimeError as exc:
        try:status=op.stats()['return_status'];get=op.debug.value;get(mu)
        except RuntimeError:
            report=dict(site=site,date=date,label=label,status='solver_initialization_failed',error=str(exc))
            (OUT/f'{date}_{label}_site{site}_failure.json').write_text(json.dumps(report,indent=2))
            return report
    row=np.asarray(get(mu));amp=np.interp(row,grid,amps);states=np.asarray(get(state)).T;power=np.asarray(get(pp))
    g=np.asarray(get(op.g)).ravel();lb=np.asarray(get(op.lbg)).ravel();ub=np.asarray(get(op.ubg)).ravel()
    residual=float(np.maximum(np.maximum(lb-g,g-ub),0).max())
    dst=OUT/f'{date}_{label}_site{site}'
    np.savez_compressed(dst.with_suffix('.npz'),current_ka=row,amplitude_ka=amp,states=states,power_w=power,signal=signal)
    report=dict(site=site,date=date,label=label,status=status,scaled_constraint_violation=residual,runtime_s=time.perf_counter()-start,
        min_current=float(row.min()),max_current=float(row.max()),minimum_temperature_c=float(states[:,:n].min()),maximum_temperature_c=float(states[:,:n].max()),
        minimum_ledge_m=float(states[:,2*n:].min()),maximum_ledge_m=float(states[:,2*n:].max()),
        charge_error_ka_intervals=float(moment_average(row,amp)[0].sum()-48*180),
        scope='Unadmitted averaged thermal candidate; actual carrier and original modal checks required.')
    report.update(refined_spectrum=refined_spectrum,energy_time_discretization='implicit midpoint' if midpoint else 'backward Euler',mean_ceiling=mean_ceiling,warm_start_label=warm_start_label)
    if execution_target is not None:
        report.update(scope='Cost-blind external plant execution projection. Not part of the source-native scheduling algorithm.',
            passive=passive,current_target_squared_error=float(np.sum((row-target)**2)),
            command_groups_preserved=command_groups is not None)
    dst.with_suffix('.json').write_text(json.dumps(report,indent=2));print(report,flush=True)
    return report

if __name__=='__main__':
    from rts_network import make_rts_network
    from threadpoolctl import threadpool_limits
    net=make_rts_network();date=sys.argv[1] if len(sys.argv)>1 else '2020-11-29'
    raw=np.load(ROOT/f'results/reference_prices/{date}.npz')
    print('price fields',raw.files,flush=True)
    with threadpool_limits(limits=1):
        for site in list(map(int,sys.argv[2:])) or range(6):
            # Explicit key/shape checked against the existing saved price file.
            price=raw['industrial_price'][site]
            solve(site,date,price)
