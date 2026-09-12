"""Published algorithm specializations for the common industrial DC benchmark.

PVE: Tan et al., TPWRS 2024, DOI 10.1109/TPWRS.2023.3257033, Eq. (5)-(6)
and Algorithm 1. Here the internal polytope is the simplex of admitted complete
trajectories. An orthonormal affine coordinate change retains all its power
coordinates. Support LPs and progressive facet-normal searches are performed.

C-OPF: Chen et al., TPWRS 2025, DOI 10.1109/TPWRS.2024.3514516.
The dual nonnegative line flows, complementarity and carbon balance are solved
by IPOPT. This is the paper's DC specialization, with daily load-carbon budgets
and industrial trajectory weights replacing the example's carbon-intensity
caps. No storage or AC claim is made. Discrete execution recovery is explicit.
"""
from pathlib import Path
import sys,json,time,os
ROOT=__import__("repository_paths").ROOT
_casadi_dll_path=__import__("pathlib").Path(__import__("casadi").__file__).resolve().parent
_dll_handle=os.add_dll_directory(str(_casadi_dll_path)) if os.name=='nt' else None
os.environ['PATH']=str(_casadi_dll_path)+os.pathsep+os.environ.get('PATH','')
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog
from scipy.spatial import ConvexHull
import casadi as ca
from branch_dispatch import evaluate
from audit_dispatch_feasibility import primal_audit

def projected_vertices(P,tolerance=1e-8):
    """Exact simplex support oracle, PVE in the full affine trajectory span."""
    start=time.perf_counter();origin=P[0];_,s,vh=np.linalg.svd(P-origin,full_matrices=False)
    rank=int(np.sum(s>max(1,s[0])*1e-10));basis=vh[:rank].T;X=(P-origin)@basis
    ids=set();calls=0;history=[]
    def support(direction):
        nonlocal calls
        r=linprog(-(X@direction),A_eq=np.ones((1,len(P))),b_eq=[1],bounds=(0,1),method='highs')
        assert r.success,r.message
        calls+=1;return int(np.argmax(r.x))
    if rank==0:return [0],dict(rank=0,calls=0,history=[],runtime_s=time.perf_counter()-start)
    for d in np.eye(rank):ids.update([support(d),support(-d)])
    # Axial extrema may be affinely dependent. Find a separating direction
    # orthogonal to their affine span, preserving the paper's support oracle.
    while len(ids)<rank+1 or np.linalg.matrix_rank(X[sorted(ids)]-X[min(ids)],tol=1e-8)<rank:
        pts=X[sorted(ids)]-X[min(ids)];_,sv,v=np.linalg.svd(pts,full_matrices=True)
        r=int(np.sum(sv>1e-8));before=len(ids)
        for d in v[r:]:ids.update([support(d),support(-d)])
        if len(ids)==before:raise RuntimeError('Failed to initialize full-dimensional PVE hull')
    for iteration in range(100):
        ordered=sorted(ids)
        if rank==1:
            facets=np.array([[1.,-X[ordered,0].max()],[-1.,X[ordered,0].min()]])
        else:facets=ConvexHull(X[ordered]).equations
        extra=set();gap=0.
        for facet in facets:
            normal,offset=facet[:-1],facet[-1];i=support(normal)
            improvement=float((normal@X[i]+offset)/np.linalg.norm(normal))
            gap=max(gap,improvement)
            if improvement>tolerance:extra.add(i)
        history.append(dict(iteration=iteration,vertices=len(ids),facets=len(facets),maximum_facet_distance=gap))
        if gap<=tolerance:break
        assert extra-ids,'PVE failed to improve';ids.update(extra)
    # Independent vertex-to-hull LP checks: zero residual establishes equality
    # here; do not equate positive facet distances with Hausdorff distance.
    residual=0.
    for x in X:
        r=linprog(np.zeros(len(ids)),A_eq=np.vstack([np.ones(len(ids)),X[sorted(ids)].T]),b_eq=np.r_[1,x],bounds=(0,None),method='highs')
        assert r.success,'Incomplete projection'
        residual=max(residual,float(abs(X[sorted(ids)].T@r.x-x).max()))
    return sorted(ids),dict(rank=rank,calls=calls,history=history,equality_residual=residual,runtime_s=time.perf_counter()-start)

def pve_library(lib):
    result={**lib,'admissible':np.zeros_like(lib['admissible'])};reports=[]
    for k in range(len(lib['P'])):
        available=np.flatnonzero(lib['admissible'][k]);vertices,report=projected_vertices(lib['P'][k,available])
        result['admissible'][k,available[vertices]]=True
        report.update(site=k,vertices_original=available[vertices].tolist());reports.append(report)
    return result,reports

def copf(net,prof,lib,start,budget,relax=True,fixed_branch=None,max_seconds=300,max_iter=600,eliminate_dual=True,native_rows=None,source_price=0.,native_solver_profile='legacy',metric_caps=None,curtailment_price=0.):
    """Sparse dual-flow C-OPF, with exact independent checks on the output."""
    begin=time.perf_counter()
    if native_rows is None:K,B,T=lib['P'].shape
    else:K,B,T=len(native_rows),1,prof['background_mw'].shape[1]
    n=len(net.bus);L=len(net.branch)
    # Eliminate quantities fixed by the common commitment and must-take data.
    activeg=np.flatnonzero(net.fixed_commitment);activer=np.flatnonzero(net.curtailable)
    ng=len(activeg);nr=len(activer);gb=net.gen[activeg,0].astype(int);rb=net.ren_buses[activer]
    op=ca.Opti();pg=op.variable(ng,T);pr=op.variable(nr,T);theta=op.variable(n-1,T)
    rho=op.variable(n,T);heat=op.variable(ng,T)
    if native_rows is not None:
        assert lib is None and fixed_branch is None
        native_current=op.variable(K,T);native_state=op.variable(2*K,T+1)
        op.subject_to(op.bounded(144.,ca.vec(native_current),216.))
        op.subject_to(ca.sum2(native_current)==180*T)
        delta=ca.horzcat(native_current[:,0]-180,native_current[:,1:]-native_current[:,:-1])
        op.subject_to(op.bounded(-18.,ca.vec(delta),18.))
        for k,row in enumerate(native_rows):
            z=native_state[2*k:2*k+2,:]
            op.subject_to(z[:,0]==0)
            op.subject_to(z[:,1:]==ca.DM(row['Ad'])@z[:,:-1]+ca.DM(row['Bd'])@(native_current[k,:]-180))
            op.subject_to(op.bounded(940-row['initial_temperature_c'],z[0,:],980-row['initial_temperature_c']))
            if row.get('case_execution_bounds',False):
                from harmonic_process import matrix
                from scipy.optimize import brentq
                bound=brentq(lambda v:np.real(np.linalg.eigvals(matrix(v,.043))).max()+.0012,180,210)
                op.subject_to(native_current[k,:]<=bound)
                if row.get('terminal_recovery',False):op.subject_to(op.bounded(-2.,z[:,-1],2.))
        p0=np.array([r['nominal_power_mw'] for r in native_rows])
        pl=ca.repmat(ca.DM(p0/180),1,T)*native_current
        w=ca.DM.ones(K,1)
        op.set_initial(native_current,start.get('current_ka',np.full((K,T),180.)))
        op.set_initial(native_state,start.get('linear_temperature_deviation_c',np.zeros((2*K,T+1))))
    elif fixed_branch is None:
        w=op.variable(K,B);op.subject_to(ca.sum2(w)==1)
        op.subject_to(op.bounded(0,ca.vec(w),1));op.subject_to(ca.vec(w*(~lib['admissible']))==0)
        pl=ca.vertcat(*[w[k,:]@ca.DM(lib['P'][k]) for k in range(K)])
        op.set_initial(w,start['weights'])
    else:
        ww=np.zeros((K,B));ww[np.arange(K),fixed_branch]=1
        w=ca.DM(ww);pl=ca.DM(lib['P'][np.arange(K),fixed_branch])
    gm=ca.DM(sp.csc_matrix(net.genmap[:,activeg]));rm=ca.DM(sp.csc_matrix(net.renmap[:,activer]));lm=ca.DM(sp.csc_matrix(net.loadmap))
    inc=ca.DM(sp.csc_matrix(net.incidence));send=ca.DM(sp.csc_matrix(np.maximum(net.incidence,0)));recv=ca.DM(sp.csc_matrix(np.maximum(-net.incidence,0)))
    tap=np.where(net.branch[:,8]==0,1,net.branch[:,8]);sus=100/(net.branch[:,3]*tap)
    angles=ca.vertcat(ca.DM.zeros(1,T),theta)
    flow=ca.repmat(ca.DM(sus),1,T)*(inc.T@angles)
    fp=ca.fmax(flow,0) if eliminate_dual else op.variable(L,T)
    fm=ca.fmax(-flow,0) if eliminate_dual else op.variable(L,T)
    fixedren=net.renmap[:,net.must_take]@prof['available_renewable_mw'][net.must_take]
    local=gm@pg+rm@pr+fixedren;demand=prof['background_mw']+lm@pl
    op.subject_to(local-demand==inc@flow)
    cap=np.repeat(net.capacity[:,None],T,axis=1).ravel(order='F')
    op.subject_to(op.bounded(-cap,ca.vec(flow),cap))
    if not eliminate_dual:
        op.subject_to(flow==fp-fm)
        op.subject_to(op.bounded(0,ca.vec(fp),cap))
        op.subject_to(op.bounded(0,ca.vec(fm),cap))
    # Scholtes relaxation approaching exact complementarity. Every accepted
    # result is checked by the direction selected by its signed DC flow.
    eps=op.parameter()
    if not eliminate_dual:op.subject_to(ca.vec(fp*fm)<=eps)
    op.subject_to(op.bounded(ca.vec(ca.repmat(ca.DM(net.pmin[activeg]),1,T)),ca.vec(pg),ca.vec(ca.repmat(ca.DM(net.pmax[activeg]),1,T))))
    op.subject_to(op.bounded(0,ca.vec(pr),prof['available_renewable_mw'][activer].ravel(order='F')))
    ramp=net.ramp_mw_per_hour[activeg,None]*prof['dt_h']
    op.subject_to(op.bounded(ca.vec(ca.repmat(ca.DM(-ramp),1,T-1)),ca.vec(pg[:,1:]-pg[:,:-1]),ca.vec(ca.repmat(ca.DM(ramp),1,T-1))))
    for j,g in enumerate(activeg):
        xp=net.fuel_power_points[g];hp=net.fuel_heat_points[g]
        # Exact at an optimum: reducing excess heat decreases positive fuel
        # cost and every carbon-budget LHS (D^{-1} >= 0). The output is still
        # checked with the original fuel curve, never the epigraph variable.
        for z in range(len(xp)-1):
            slope=(hp[z+1]-hp[z])/(xp[z+1]-xp[z]);line=slope*pg[j,:]+hp[z]-slope*xp[z]
            op.subject_to(ca.vec(heat[j,:]-line)>=0)
    sources=gm@(ca.repmat(ca.DM(net.fuel_carbon_t_mmbtu[activeg]),1,T)*heat)
    incoming=recv@fp+send@fm
    carbon_in=recv@((send.T@rho)*fp)+send@((recv.T@rho)*fm)
    op.subject_to((rho*(local+incoming)-sources-carbon_in)/1000==0)
    op.subject_to(op.bounded(0,ca.vec(rho),3.))
    allocation=prof['dt_h']*ca.sum2(rho[net.industrial_buses.tolist(),:]*pl)
    op.subject_to(allocation<=budget-2e-6)
    fuel=ca.sum1(ca.DM(net.fuel_price[activeg]).T@heat+ca.DM(net.vom[activeg]).T@pg)
    curt=ca.sum1(ca.sum2(prof['available_renewable_mw'][activer]-pr))
    source_emission=prof['dt_h']*ca.sum1(ca.sum2(ca.repmat(ca.DM(net.fuel_carbon_t_mmbtu[activeg]),1,T)*heat))
    objective=prof['dt_h']*(ca.sum2(fuel)+150*curt)+source_price*source_emission
    if metric_caps is not None:
        # Optional proposed-method polishing only. Defaults leave the source
        # comparator formulation unchanged. Caps use physical totals, and the
        # caller independently checks the original fuel curves after solving.
        op.subject_to(prof['dt_h']*(ca.sum2(fuel)+150*curt)<=metric_caps[0])
        op.subject_to(source_emission<=metric_caps[1])
        op.subject_to(prof['dt_h']*curt<=metric_caps[2])
    objective += curtailment_price*prof['dt_h']*curt
    op.minimize(objective/1e4)
    op.set_initial(pg,start['pg'][activeg]);op.set_initial(pr,start['pr'][activer])
    f=start['flow']
    if not eliminate_dual:op.set_initial(fp,np.maximum(f,0));op.set_initial(fm,np.maximum(-f,0))
    th=np.linalg.lstsq(np.diag(sus)@net.incidence[1:].T,f,rcond=None)[0]
    op.set_initial(theta,th);op.set_initial(rho,np.clip(start['rho'],0,3))
    hh=np.array([net.fuel_heat(start['pg'][:,t]) for t in range(T)]).T[activeg];op.set_initial(heat,hh)
    opts={'max_iter':max_iter,'max_wall_time':max_seconds,'tol':1e-8,'acceptable_tol':1e-7,'print_level':0,'sb':'yes',
        'mu_strategy':'monotone','mu_init':1e-7,'bound_push':1e-8,'bound_frac':1e-8,'slack_bound_push':1e-8,'slack_bound_frac':1e-8,
        'bound_relax_factor':0,'honor_original_bounds':'yes'}
    if native_solver_profile in ['adaptive','lowmemory']:
        # The old near-feasible-library initialization used an unusually small
        # barrier. Native continuous industrial states need a normal interior
        # start; retain the unsuccessful attempt for the solver audit.
        opts.update(mu_strategy='adaptive',mu_init=.1,bound_push=.01,bound_frac=.01,
            slack_bound_push=.01,slack_bound_frac=.01)
    lowmemory=native_solver_profile=='lowmemory'
    if lowmemory:opts.update(hessian_approximation='limited-memory',limited_memory_max_history=6,max_cpu_time=max_seconds)
    op.solver('ipopt',{'expand':not lowmemory,'print_time':False},opts)
    stages=[];solution=None
    for epsilon in ([0.] if eliminate_dual else [1e-3,1e-7]):
        op.set_value(eps,epsilon)
        try:
            solution=op.solve();status=solution.stats()['return_status'];getter=solution.value
        except RuntimeError as exc:
            try:status=op.stats().get('return_status','failed');getter=op.debug.value
            except RuntimeError:return dict(status='solver_initialization_failed',error=str(exc),runtime_s=time.perf_counter()-begin)
        stages.append(dict(epsilon=epsilon,status=status,iterations=op.stats().get('iter_count'),elapsed_s=time.perf_counter()-begin))
        print('C-OPF',stages[-1],flush=True)
        try:op.set_initial(op.x,getter(op.x))
        except RuntimeError:break
        if status in ['Maximum_WallTime_Exceeded','Maximum_CpuTime_Exceeded']:break
    try:
        pgval=np.zeros_like(start['pg']);pgval[activeg]=getter(pg)
        prval=prof['available_renewable_mw'].copy();prval[activer]=getter(pr)
        if native_rows is not None:
            wval=np.ones((K,1));plval=np.asarray(getter(pl))
            auditlib={'P':plval[:,None,:],'admissible':np.ones((K,1),bool)}
        else:
            wval=np.asarray(getter(w)) if fixed_branch is None else ww
            plval=np.einsum('kb,kbt->kt',wval,lib['P']);auditlib=lib
        result=dict(pg=pgval,pr=prval,pl=plval,weights=wval,commitment=net.fixed_commitment,
            branches=np.argmax(wval,axis=1),flow=net.ptdf@(net.genmap@pgval+net.renmap@prval-net.loadmap@plval-prof['background_mw'])+net.flow_offset[:,None],
            status=stages[-1]['status'],stages=stages,runtime_s=time.perf_counter()-begin)
        result.update(evaluate(net,prof,pgval,prval,plval))
        result['primal_audit']=primal_audit(net,prof,auditlib,result,require_integral=fixed_branch is not None)
        g=np.asarray(getter(op.g)).ravel();lb=np.asarray(getter(op.lbg)).ravel();ub=np.asarray(getter(op.ubg)).ravel()
        result['native_nlp_constraint_violation']=float(np.maximum(np.maximum(lb-g,g-ub),0).max())
        if native_rows is not None:
            result.update(current_ka=np.asarray(getter(native_current)),linear_temperature_deviation_c=np.asarray(getter(native_state)),
                native_industrial_interface='Liu-2025 two-state constant-voltage case adaptation; continuous commands; no proposed trajectory library',
                adaptation_parameters=native_rows,source_emission_price_usd_t=source_price)
            result['native_solver_profile']=native_solver_profile
            g=np.asarray(getter(op.g)).ravel();lb=np.asarray(getter(op.lbg)).ravel();ub=np.asarray(getter(op.ubg)).ravel()
            result['native_nlp_constraint_violation']=float(np.maximum(np.maximum(lb-g,g-ub),0).max())
        result['budget_violation_t']=float(np.maximum(result['allocation_t']-budget,0).sum())
        result['carbon_feasible']=result['budget_violation_t']<=1e-4 and result['primal_audit']['passed']
        result['maximum_complementarity']=float(np.max(getter(fp)*getter(fm)))
        result['dual_flow_implementation']='exact analytical elimination' if eliminate_dual else 'explicit variables and homotopy'
        return result
    except (RuntimeError,ValueError) as exc:
        return dict(status='no_primal_result',error=str(exc),stages=stages,runtime_s=time.perf_counter()-begin)
