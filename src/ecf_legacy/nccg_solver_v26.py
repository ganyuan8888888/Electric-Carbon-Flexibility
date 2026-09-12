"""Actual nested CCG for a frozen, finite certified recourse MILP.

Outer bounds belong ONLY to this conservative library model. Fixed-pattern
LPs are solved directly; full recourse is an MILP. All-scenario extensive-form
MILP is an independent stopping/solution check. No hand-made iteration data.
"""
from pathlib import Path
import sys,json,time,math,hashlib
ROOT=__import__("repository_paths").ROOT
import numpy as np
import cvxpy as cp
from scipy.optimize import linprog
from dispatch_result_io import load,save

def signature(c):return (c['pattern'],*c['endpoints'],round(c['left'],12),round(c['right'],12))

class FiniteRecourse:
    def __init__(self,date,allowed=None,result_root=None):
        self.date=date;self.path=Path(result_root)/date if result_root is not None else ROOT/f'results/robust_v26/{date}'
        common=load(self.path,'common');self.B=common['budget'];self.refs=common['refs'];self.qbar=common['qbar']/self.B
        self.cells=[];self.maps=[];self.hashes=[]
        for s in range(5):
            f=self.path/f'scenario_{s:02}/cells.json';self.hashes.append(hashlib.sha256(f.read_bytes()).hexdigest())
            data=json.loads(f.read_text());cells=data['cells']
            if allowed is not None:cells=[c for c in cells if c['pattern'] in allowed]
            if allowed is None:assert cells,(date,s,'Candidate completion still required; do not omit this scenario.')
            self.cells.append(cells);self.maps.append({signature(c):i for i,c in enumerate(cells)})
        self.cache={};self.lp_calls=0;self.mip_calls=0

    def trade(self,u):return float(np.maximum(25*self.B*u,20*self.B*u).sum()/self.refs[0])

    def fixed_lp(self,s,u,key):
        """Value of a fixed binary pattern; exact LP infeasibility -> infinity."""
        self.lp_calls+=1
        if key not in self.maps[s]:return np.inf
        c=self.cells[s][self.maps[s][key]]
        if np.max(np.array(c['allocation_upper'])/self.B-1-u)>1e-7:return np.inf
        a=np.array(c['metrics_left'])/self.refs;d=(np.array(c['metrics_right'])-c['metrics_left'])/self.refs
        a[0]+=self.trade(u)
        r=linprog([0.,1.],A_ub=np.column_stack([d,-np.ones(3)]),b_ub=-a,bounds=[(0,1),(0,None)],method='highs')
        if not r.success:
            if r.status==2:return np.inf
            raise RuntimeError(r.message)
        return float(r.fun)

    def solve(self,scenarios,u_fixed=None,robust_cap=None):
        start=time.perf_counter();S=list(scenarios);u=cp.Variable(6);trade=cp.Variable(6);eta=cp.Variable(nonneg=True)
        if any(not self.cells[s] for s in S):return dict(status='infeasible_finite_recourse_set',value=np.inf,bound=np.inf,runtime_s=0.)
        cons=[u>=-1,u<=self.qbar,trade>=cp.multiply(25*self.B/self.refs[0],u),trade>=cp.multiply(20*self.B/self.refs[0],u)]
        if u_fixed is not None:cons.append(u==u_fixed)
        variables={};local_values=[]
        for s in S:
            cells=self.cells[s];J=len(cells);z=cp.Variable(J,boolean=True);a=cp.Variable(J,nonneg=True)
            upper=np.array([c['allocation_upper'] for c in cells])/self.B[None,:]
            lo=np.array([c['metrics_left'] for c in cells])/self.refs[None,:]
            diff=(np.array([c['metrics_right'] for c in cells])-np.array([c['metrics_left'] for c in cells]))/self.refs[None,:]
            metric=lo.T@z+diff.T@a
            local=eta if robust_cap is None else cp.Variable(nonneg=True)
            cons += [cp.sum(z)==1,a<=z,upper.T@z<=1+u,metric[0]+cp.sum(trade)<=local,metric[1]<=local,metric[2]<=local]
            if robust_cap is not None:cons.append(local<=robust_cap)
            local_values.append(local)
            variables[s]=(z,a)
        if robust_cap is not None:cons.append(eta==robust_cap)
        model=cp.Problem(cp.Minimize(eta if robust_cap is None else cp.sum(cp.hstack(local_values))),cons)
        model.solve(solver='HIGHS',highs_options={'threads':1,'time_limit':180,'mip_rel_gap':1e-9,'mip_abs_gap':1e-10,
            'mip_feasibility_tolerance':1e-9,'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9,'log_to_console':False})
        self.mip_calls+=1
        if eta.value is None:return dict(status=model.status,value=np.inf,bound=np.inf,runtime_s=time.perf_counter()-start)
        stats=model.solver_stats.extra_stats
        bound=float(getattr(stats,'mip_dual_bound',model.value))
        if not np.isfinite(bound):
            assert model.status=='optimal';bound=float(model.value)
        chosen={}
        for s,(z,a) in variables.items():
            j=int(np.argmax(z.value));assert abs(z.value[j]-1)<1e-6
            chosen[s]=dict(index=j,key=signature(self.cells[s][j]),alpha=float(a.value[j]),cell_id=self.cells[s][j]['cell_id'])
        result=dict(status=model.status,value=float(model.value) if robust_cap is None else max(float(v.value) for v in local_values),bound=bound,u=np.asarray(u.value),selected=chosen,
            max_constraint_violation=max(float(np.max(c.violation())) for c in cons),runtime_s=time.perf_counter()-start)
        if robust_cap is not None:result.update(secondary_objective=float(model.value),secondary_bound=bound,robust_cap=robust_cap)
        assert result['max_constraint_violation']<1e-6,result
        model._solver_cache.clear()
        return result

    def inner(self,u,outer_iteration,trace,tol=1e-8,feasible_pattern_start=False):
        patterns=[];seen={};lower=-np.inf;upper=np.inf;incumbent=None;start=time.perf_counter()
        if feasible_pattern_start:
            for s,cells in enumerate(self.cells):
                good=[c for c in cells if np.max(np.array(c['allocation_upper'])/self.B-1-u)<=1e-7]
                if not good:return dict(infeasible=s,lower=np.inf,upper=np.inf)
                key=signature(good[0])
                if key not in patterns:patterns.append(key)
        # One unrestricted MILP seed; subsequent subproblem-1 values are actual
        # restricted-pattern LPs. The uncertainty set is explicitly finite.
        scenario=0
        for j in range(sum(map(len,self.cells))+5):
            solved_scenario=scenario
            result=self.solve([scenario],u);seen[scenario]=result
            if not np.isfinite(result['value']):return dict(infeasible=scenario,lower=np.inf,upper=np.inf)
            if result['bound']>lower:lower=result['bound'];incumbent=scenario
            key=tuple(result['selected'][scenario]['key'])
            duplicate=key in patterns
            if not duplicate:patterns.append(key)
            values=np.array([min(self.fixed_lp(s,u,k) for k in patterns) for s in range(5)])
            upper=float(np.max(values));scenario=int(np.argmax(values))
            row=dict(outer_iteration=outer_iteration,inner_iteration=j+1,lower_ratio=float(lower),
                upper_ratio=upper if np.isfinite(upper) else None,upper_is_infinite=not np.isfinite(upper),
                worst_scenario=scenario,patterns=len(patterns),elapsed_s=time.perf_counter()-start)
            trace.append(row)
            if upper-lower<=tol*max(1,abs(upper)) and np.isfinite(upper):
                return dict(infeasible=None,lower=float(lower),upper=upper,worst_scenario=incumbent,patterns=len(patterns))
            if duplicate and scenario==solved_scenario:
                (self.path/'inner_duplicate_diagnostic.json').write_text(json.dumps(dict(trace=trace,result=result,values=values.tolist()),default=lambda v:v.tolist() if isinstance(v,np.ndarray) else str(v),indent=2))
                raise RuntimeError('Repeated integer pattern with unresolved bound gap; see inner_duplicate_diagnostic.json')
        raise RuntimeError('Finite pattern generation did not terminate.')

    def feasible_initialization(self):
        """First admitted cell per scenario, at the common purchase cap.

        This deterministic rule precedes NCCG and does not use its optimum.
        Each complete interval is already certified against B + qbar.
        """
        u=self.qbar.copy();plans={};start=time.perf_counter()
        for s,cells in enumerate(self.cells):
            c=cells[0];assert np.max(np.array(c['allocation_upper'])/self.B-1-u)<1e-7
            a=np.array(c['metrics_left'])/self.refs;d=(np.array(c['metrics_right'])-c['metrics_left'])/self.refs
            a[0]+=self.trade(u)
            r=linprog([0.,1.],A_ub=np.column_stack([d,-np.ones(3)]),b_ub=-a,bounds=[(0,1),(0,None)],method='highs')
            self.lp_calls+=1;assert r.success
            plans[s]=dict(cell_id=c['cell_id'],key=signature(c),alpha=float(r.x[0]),value=float(r.fun))
        return dict(u=u,upper=max(r['value'] for r in plans.values()),plans=plans,
            rule='First admitted certified cell in each scenario at q=qbar; independent LP along each cell. No use of the robust optimum.',elapsed_s=time.perf_counter()-start)

    def nccg(self):
        start=time.perf_counter();initial=self.feasible_initialization()
        active=[0];lower=-np.inf;upper=initial['upper'];incumbent=initial['u'].copy();trace=[];inner_trace=[]
        for iteration in range(1,12):
            master=self.solve(active)
            if not np.isfinite(master['value']):raise RuntimeError('Frozen certified robust model infeasible.')
            lower=max(lower,master['bound'])
            if upper-lower<=1e-7*max(1,abs(upper)):
                trace.append(dict(iteration=iteration,lower_ratio=float(lower),upper_ratio=float(upper),
                    relative_gap=float(max(0,upper-lower)/max(1,abs(upper))),active_scenarios=list(active),
                    stopping_reason='Existing all-scenario feasible incumbent and new master bound close the gap.',elapsed_s=time.perf_counter()-start))
                break
            inner=self.inner(master['u'],iteration,inner_trace)
            if inner['infeasible'] is not None:
                s=inner['infeasible']
                trace.append(dict(iteration=iteration,lower_ratio=float(lower),upper_ratio=None if not np.isfinite(upper) else upper,
                    feasibility_scenario=s,active_scenarios=list(active),elapsed_s=time.perf_counter()-start))
                assert s not in active;active.append(s);continue
            candidate=max(master['value'],inner['upper'])
            if candidate<upper:upper=candidate;incumbent=master['u'].copy()
            trace.append(dict(iteration=iteration,lower_ratio=float(lower),upper_ratio=float(upper),
                relative_gap=float(max(0,upper-lower)/max(1,abs(upper))),active_scenarios=list(active),
                worst_scenario=inner['worst_scenario'],inner_patterns=inner['patterns'],elapsed_s=time.perf_counter()-start))
            if upper-lower<=1e-7*max(1,abs(upper)):break
            s=inner['worst_scenario'];assert s not in active,(trace,inner);active.append(s)
        else:raise RuntimeError('Outer loop did not close certified-model gap.')
        extensive=self.solve(range(5));assert abs(extensive['value']-upper)<max(2e-6,2e-8*abs(upper)),(upper,extensive['value'])
        realized={s:self.solve([s],incumbent) for s in range(5)}
        return dict(date=self.date,lower_ratio=lower,upper_ratio=upper,q=incumbent*self.B,u=incumbent,
            lower_usd=lower*self.refs[0],upper_usd=upper*self.refs[0],outer_trace=trace,inner_trace=inner_trace,
            extensive_objective=extensive['value'],extensive_gap_to_nccg=abs(extensive['value']-upper),
            scenarios=realized,elapsed_s=time.perf_counter()-start,LP_calls=self.lp_calls,MIP_calls=self.mip_calls,
            cells_sha256=self.hashes,initialization=initial,bound_scope='Frozen finite certified polyhedral recourse model, not unrestricted nonlinear optimum.')

def run(date):
    model=FiniteRecourse(date);result=model.nccg();save(model.path,'NCCG',result)
    print(date,'NCCG',result['lower_ratio'],result['upper_ratio'],'outer',len(result['outer_trace']),
        'inner',len(result['inner_trace']),'extensive error',result['extensive_gap_to_nccg'],flush=True)
    fixed=FiniteRecourse(date,allowed={'constant'})
    if any(not c for c in fixed.cells):
        r=dict(status='infeasible_finite_recourse_set',empty_scenarios=[s for s,c in enumerate(fixed.cells) if not c],
            scope='No constant-control certified candidate in these scenarios. This is not a proof that the unrestricted nonlinear fixed-load problem is infeasible.')
    else:
        r=fixed.nccg()
        assert result['upper_ratio']<=r['upper_ratio']+1e-6,'Constant-control feasible set must be nested in proposed set.'
    save(model.path,'constant_robust',r)

if __name__=='__main__':run(sys.argv[1])
