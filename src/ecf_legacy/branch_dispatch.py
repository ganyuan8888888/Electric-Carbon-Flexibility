"""Finite-trajectory generation/load dispatch with exact carbon-flow validation.

The MILP omitting carbon constraints provides a relaxation lower bound. Local
carbon cuts guide trial schedules, but are not called global valid cuts. Every
accepted feasible schedule is independently evaluated with the exact nonlinear
carbon equations. The reported bound is valid for the finite trajectory model;
it is NOT a bound for the unrestricted continuous industrial control problem.
"""
from pathlib import Path
import json,time
import numpy as np
import cvxpy as cp
from network_model import make_network,profiles

ROOT=__import__("repository_paths").ROOT

def load_library(net,only_static=False,version=None):
    if version=='public_v3':
        location=ROOT/'results/library/public_v3'
        raw=np.load(location/'candidate_library.npz')
        meta=json.loads((location/'candidate_library.json').read_text(encoding='utf8'))
        keep=raw['certified'].copy()
        if only_static:keep&=~raw['uses_excitation']
        assert keep.shape[0]==len(net.industrial_buses)
        assert keep[:,0].all(),'Nominal production branch must be certified at every site'
        baseline=raw['power_w'][:,0].mean(axis=-1)
        cells=np.maximum(1,np.round(net.industrial_base*1e6/baseline/3)).astype(int)
        P=cells[:,None,None]*3*raw['power_w']/1e6
        return {'P':P,'ids':np.arange(P.shape[1]),'cells_per_series':cells,'metadata':meta,
                'admissible':keep}
    raw=np.load(ROOT/'results/library/candidate_library.npz')
    keep=raw['thermal_feasible'].copy()
    if only_static:keep&=~raw['uses_excitation']
    ids=np.flatnonzero(keep);power=raw['power_w'][keep]
    baseline=raw['power_w'][0].mean()
    # Exactly three independently supplied series, with an integer number of
    # series-connected cells per series. Carrier phases are 0, 2pi/3 and 4pi/3.
    cells=np.maximum(1,np.round(net.industrial_base*1e6/baseline/3)).astype(int)
    P=cells[:,None,None]*3*power[None,:,:]/1e6
    return {'P':P,'ids':ids,'cells_per_series':cells,'metadata':json.loads((ROOT/'results/library/candidate_library.json').read_text())}

def evaluate(net,prof,pg,pr,pl):
    rho=[];rates=[];res=[];cons=[]
    for t in range(pl.shape[1]):
        c=net.carbon(pg[:,t],pr[:,t],pl[:,t],prof['background_mw'][:,t])
        rho.append(c['rho']);rates.append(c['industrial_carbon_rate'])
        res.append(c['carbon_balance_residual']);cons.append(c['attribution_conservation_residual'])
    dt=prof['dt_h'];curt=prof['available_renewable_mw']-pr
    economic=dt*(sum(np.sum(net.generation_cost(pg[:,t])) for t in range(pl.shape[1]))+150*np.sum(curt))
    return {'rho':np.array(rho).T,'carbon_rate':np.array(rates).T,
            'allocation_t':np.sum(rates,axis=0)*dt,'generation_emission_t':float(sum(net.generation_emissions(pg[:,t]).sum() for t in range(pl.shape[1]))*dt),
            'curtailment_mwh':float(np.sum(curt)*dt),'objective_actual':float(economic),
            'carbon_balance_residual':float(max(res)),'attribution_conservation_residual':float(max(abs(np.asarray(cons))))}

def master(net,prof,lib,center=None,budgets=None,trust=.08,relax=False,fixed_branch=None,time_limit=30.,choose_commitment=False,fixed_rho=None,budget_buffer=.001,mip_gap=2e-5,branch_reference=None,max_branch_changes=None,exact_heat_cut=False):
    start=time.perf_counter();K,B,T=lib['P'].shape;ng=len(net.gen);nr=len(net.ren_buses)
    w=cp.Variable((K,B),boolean=not relax);pg=cp.Variable((ng,T));pr=cp.Variable((nr,T));epi=cp.Variable((ng,T))
    pl=cp.vstack([w[k]@lib['P'][k] for k in range(K)])
    inj=net.genmap@pg+net.renmap@pr-net.loadmap@pl-prof['background_mw']
    flow=net.ptdf@inj+net.flow_offset[:,None]
    commitment=cp.Variable(ng,boolean=True) if choose_commitment else getattr(net,'fixed_commitment',np.ones(ng))
    power_balance=cp.sum(inj,axis=0)==0
    line_upper=flow<=net.capacity[:,None];line_lower=flow>=-net.capacity[:,None]
    constraints=[cp.sum(w,axis=1)==1,w>=0,w<=1,pg>=cp.multiply(net.pmin,commitment)[:,None],pg<=cp.multiply(net.pmax,commitment)[:,None],
                 pr>=prof.get('minimum_renewable_mw',0),pr<=prof['available_renewable_mw'],power_balance,line_upper,line_lower]
    if 'admissible' in lib:constraints.append(w<=lib['admissible'].astype(float))
    ramp=getattr(net,'ramp_mw_per_hour',net.pmax*.12)[:,None]*prof['dt_h']
    constraints += [pg[:,1:]-pg[:,:-1]<=ramp,pg[:,:-1]-pg[:,1:]<=ramp]
    # Tangents are global lower supports of the true convex quadratic fuel cost.
    # The independent feasible objective always uses the original quadratic.
    if hasattr(net,'fuel_power_points'):
        constraints.append(epi>=0)
        for g,(xp,hp) in enumerate(zip(net.fuel_power_points,net.fuel_heat_points)):
            for j in range(len(xp)-1):
                slope=(hp[j+1]-hp[j])/(xp[j+1]-xp[j]);offset=hp[j]-slope*xp[j]
                constraints.append(epi[g]>=net.fuel_price[g]*(slope*pg[g]+offset*commitment[g])+net.vom[g]*pg[g])
    else:
        for s in np.linspace(0,1,17):
            anchor=(net.pmin+s*(net.pmax-net.pmin))[:,None]
            constraints.append(epi>=.006*cp.multiply(anchor,pg)-.003*anchor**2)
    if fixed_branch is not None:
        for k,b in enumerate(fixed_branch):constraints.append(w[k,b]==1)
    if branch_reference is not None and max_branch_changes is not None:
        constraints.append(cp.sum(cp.hstack([w[k,int(b)] for k,b in enumerate(branch_reference)]))>=K-max_branch_changes)
    if fixed_rho is not None:
        assert budgets is not None and fixed_rho.shape==(K,T)
        constraints.append(prof['dt_h']*cp.sum(cp.multiply(fixed_rho,pl),axis=1)<=budgets)
    slack=None
    if center is not None and budgets is not None:
        constraints += [cp.abs(pg-center['pg'])<=trust*net.pmax[:,None],
                        cp.abs(pr-center['pr'])<=trust*np.maximum(prof['available_renewable_mw'],1),
                        cp.abs(pl-center['pl'])<=trust*2*np.maximum(center['pl'],1)]
        base=np.zeros(K);gradient=np.empty((K,ng+nr+K,T))
        if exact_heat_cut:
            from scipy.linalg import solve as dense_solve
            heat_gradient=np.zeros((K,ng,T));heat_ref=np.zeros((ng,T));heat_slope=np.zeros((ng,T))
        for t in range(T):
            c=net.carbon(center['pg'][:,t],center['pr'][:,t],center['pl'][:,t],prof['background_mw'][:,t],derivatives='fast')
            base+=c['industrial_carbon_rate']*prof['dt_h'];gradient[:,:,t]=c['dcarbon']*prof['dt_h']
            if exact_heat_cut:
                active=c['active'];selection=net.loadmap.T[:,active]
                adj=dense_solve(c['D'][np.ix_(active,active)].T,selection.T).T
                heat_gradient[:,:,t]=center['pl'][:,t,None]*(adj@net.genmap[active])*net.fuel_carbon_t_mmbtu[None,:]*prof['dt_h']
                heat_ref[:,t]=net.fuel_heat(center['pg'][:,t]);heat_slope[:,t]=net.fuel_heat(center['pg'][:,t],True)
        approximations=[base[k]+cp.sum(cp.multiply(gradient[k,:ng],pg-center['pg']))+
                        cp.sum(cp.multiply(gradient[k,ng:ng+nr],pr-center['pr']))+
                        cp.sum(cp.multiply(gradient[k,ng+nr:],pl-center['pl'])) for k in range(K)]
        if exact_heat_cut:
            heat_epi=cp.multiply(1/net.fuel_price[:,None],epi-cp.multiply(net.vom[:,None],pg))
            remainder=heat_epi-heat_ref-cp.multiply(heat_slope,pg-center['pg'])
            approximations=[a+cp.sum(cp.multiply(heat_gradient[k],remainder)) for k,a in enumerate(approximations)]
        slack=cp.Variable(K,nonneg=True)
        constraints.append(cp.hstack(approximations)<=budgets*(1-budget_buffer)+slack)
    dt=prof['dt_h'];curt=prof['available_renewable_mw']-pr
    objective=dt*(cp.sum(epi)+150*cp.sum(curt))
    if not hasattr(net,'fuel_power_points'):objective+=dt*cp.sum(cp.multiply(net.cost[:,None],pg))
    if slack is not None:objective+=2e4*cp.sum(slack)
    problem=cp.Problem(cp.Minimize(objective/1000),constraints)
    problem.solve(solver='HIGHS',highs_options={'time_limit':time_limit,'mip_rel_gap':mip_gap,'log_to_console':False})
    if pg.value is None:return {'status':problem.status,'runtime_s':time.perf_counter()-start}
    if not all(np.isfinite(v.value).all() for v in [pg,pr,pl,w]):
        return {'status':'nonfinite_solver_candidate','runtime_s':time.perf_counter()-start}
    result={'pg':pg.value,'pr':pr.value,'pl':pl.value,'weights':w.value,'flow':flow.value,
            'status':problem.status,'runtime_s':time.perf_counter()-start,'master_objective':float(problem.value*1000),
            'branches':np.argmax(w.value,axis=1),'carbon_linear_slack_t':None if slack is None else slack.value}
    stats=problem.solver_stats.extra_stats
    canonical_primal=float(getattr(stats,'objective_function_value',np.nan))
    canonical_dual=float(getattr(stats,'mip_dual_bound',np.nan))
    # HiGHS receives CVXPY's variable-dependent objective. Restore the canonical
    # constant (notably 150 * available renewable energy) in BOTH primal/bound.
    objective_offset=float(problem.value)-canonical_primal
    result['canonical_objective_offset']=objective_offset*1000
    result['relaxation_lower_bound']=(canonical_dual+objective_offset)*1000 if not relax else float(problem.value)*1000
    if result['status']=='optimal' and not np.isfinite(result['relaxation_lower_bound']):result['relaxation_lower_bound']=float(problem.value*1000)
    result.update(evaluate(net,prof,result['pg'],result['pr'],result['pl']))
    result['commitment']=np.rint(commitment.value).astype(int) if choose_commitment else np.asarray(commitment)
    result['library_scope']=lib['metadata'].get('version','step_v1')
    if 'admissible' in lib:
        from audit_dispatch_feasibility import primal_audit
        check=primal_audit(net,prof,lib,result,require_integral=not relax)
        if not check['passed']:
            return dict(status='failed_independent_primal_audit',runtime_s=time.perf_counter()-start,primal_audit=check)
        result['primal_audit']=check
    if relax and center is None and fixed_rho is None and not choose_commitment:
        result['economic_nodal_price_per_mwh']=-(power_balance.dual_value[None,:]+net.ptdf.T@(line_upper.dual_value-line_lower.dual_value))*1000/dt
    return result

def coordinated_dispatch(net,prof,lib,budget_ratio=.97,max_iterations=24):
    start=time.perf_counter();K=lib['P'].shape[0]
    # Fixed-production baseline defines a transparent allocation budget. The
    # ratio is a declared scenario input, not tuned separately for each method.
    constant=master(net,prof,lib,fixed_branch=np.zeros(K,dtype=int),time_limit=30)
    budget=budget_ratio*constant['allocation_t']
    current=master(net,prof,lib,time_limit=30)
    lower=current['relaxation_lower_bound'];best=None;history=[];trust=.10
    def violation(r):return float(np.maximum(0,r['allocation_t']-budget).sum())
    def merit(r):return r['objective_actual']+2e4*violation(r)
    for iteration in range(max_iterations+1):
        v=violation(current)
        if v<=1e-4 and (best is None or current['objective_actual']<best['objective_actual']):best=current.copy()
        history.append({'iteration':iteration,'objective':current['objective_actual'],'violation_t':v,
                        'curtailment_mwh':current['curtailment_mwh'],'trust':trust,
                        'branches':current['branches'].tolist(),'elapsed_s':time.perf_counter()-start})
        print(net.name,'iteration',iteration,history[-1],flush=True)
        if iteration==max_iterations:break
        trial=master(net,prof,lib,center=current,budgets=budget,trust=trust,time_limit=30)
        if 'pg' not in trial:trust*=.5;continue
        if merit(trial)<merit(current)-1e-3:
            change=abs(trial['objective_actual']-current['objective_actual'])
            current=trial;trust=min(.2,trust*1.3)
            if violation(current)<1e-4 and change<1: # Stationarity heuristic, not a global optimality declaration.
                if best is None or current['objective_actual']<best['objective_actual']:best=current.copy()
                break
        else:
            trust*=.5
            if trust<.001:break
    result=best if best is not None else current
    result.update(budgets_t=budget,carbon_feasible=bool(violation(result)<=1e-4),
                  global_finite_model_lower_bound=lower,
                  finite_model_gap_percent=100*(result['objective_actual']-lower)/max(1,abs(result['objective_actual'])) if best is not None else None,
                  history=history,total_runtime_s=time.perf_counter()-start,
                  note='Local carbon-guided MILP iteration; exact carbon verified. Process switching certification is a separate pending gate.')
    return result

def save_result(name,result):
    out=ROOT/'results/dispatch';out.mkdir(exist_ok=True,parents=True)
    np.savez_compressed(out/f'{name}.npz',**{k:v for k,v in result.items() if isinstance(v,np.ndarray)})
    report={k:v for k,v in result.items() if not isinstance(v,np.ndarray)}
    for k in ['budgets_t','allocation_t','branches']:report[k]=result[k].tolist()
    (out/f'{name}.json').write_text(json.dumps(report,indent=2),encoding='utf8')

if __name__=='__main__':
    import sys
    name=sys.argv[1] if len(sys.argv)>1 else 'case39'
    if name=='rts_gmlc':
        from rts_network import make_rts_network,rts_profiles
        net=make_rts_network();prof=rts_profiles(net);lib=load_library(net)
        pre=master(net,prof,lib,fixed_branch=np.zeros(len(net.industrial_buses),dtype=int),choose_commitment=True,time_limit=45)
        if 'pg' not in pre:raise RuntimeError(('Common commitment infeasible',pre))
        net.fixed_commitment=pre['commitment'];print('common commitment',net.fixed_commitment.tolist(),flush=True)
    else:
        net=make_network(name);prof=profiles(net);lib=load_library(net)
    result=coordinated_dispatch(net,prof,lib)
    save_result(name+'_initial',result)
    print('complete',name,'carbon feasible',result['carbon_feasible'],'gap',result['finite_model_gap_percent'],flush=True)
