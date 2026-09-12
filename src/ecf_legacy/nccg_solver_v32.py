"""Carbon-quota screening and recovery-pattern reuse for certified finite NCCG.

The certified cells and objective are unchanged. Reused modes are screened and
their values recomputed at the CURRENT quota. No bound is carried across quotas.
"""
from pathlib import Path
import time
import numpy as np
from nccg_solver_v26 import FiniteRecourse, signature

ROOT=__import__("repository_paths").ROOT

class RecoveryReuseNCCG(FiniteRecourse):
    def __init__(self, date, result_root=None):
        super().__init__(date, result_root=result_root or ROOT/'results/robust_v28')
        self.pool = []
        self.value_cache = {}
        self.cache_hits = 0
        self.screened_cell_instances = 0
        self.reused_pattern_instances = 0
        self.solver_audit = []
        self.inner_call_audit = []

    def eligible(self, s, u):
        return [c for c in self.cells[s]
                if np.max(np.asarray(c['allocation_upper'])/self.B-1-u) <= 1e-7]

    def fixed_lp(self, s, u, key):
        # Exact float bytes avoid accidental reuse of a value at another quota.
        token = (s, np.asarray(u, dtype=np.float64).tobytes(), tuple(key))
        if token in self.value_cache:
            self.cache_hits += 1
            return self.value_cache[token]
        value = super().fixed_lp(s, u, key)
        self.value_cache[token] = value
        return value

    def solve(self, scenarios, u_fixed=None, robust_cap=None):
        scenarios = list(scenarios)
        if u_fixed is None:
            return super().solve(scenarios, u_fixed, robust_cap)
        original = self.cells
        filtered = list(original)
        for s in scenarios:
            filtered[s] = self.eligible(s, u_fixed)
            self.screened_cell_instances += len(original[s])-len(filtered[s])
        self.cells = filtered
        try:
            result = super().solve(scenarios, u_fixed, robust_cap)
        finally:
            self.cells = original
        for s, selected in result.get('selected', {}).items():
            selected['index'] = self.maps[s][tuple(selected['key'])]
        self.solver_audit.append({
            'scenarios': scenarios, 'retained_cells': [len(filtered[s]) for s in scenarios],
            'original_cells': [len(original[s]) for s in scenarios],
            'max_constraint_violation': result.get('max_constraint_violation'),
        })
        return result

    def feasible_initialization(self):
        result = super().feasible_initialization()
        # The initial feasible plans supply the OUTER upper bound. The inner
        # mode pool is cold-started and is populated only by recourse solves.
        return result

    def inner(self, u, outer_iteration, trace, tol=1e-9, feasible_pattern_start=False):
        start = time.perf_counter()
        call = dict(outer_iteration=outer_iteration, checks=0, quota_screening=True, bound_updates=0)
        self.inner_call_audit.append(call)
        current = [self.eligible(s, u) for s in range(5)]
        for s, cells in enumerate(current):
            if not cells:
                call.update(checks=1, status='screening_infeasible', scenario=s)
                return dict(infeasible=s, lower=np.inf, upper=np.inf)
        valid_keys = [{signature(c) for c in cells} for cells in current]
        patterns = [k for k in self.pool if any(k in keys for keys in valid_keys)]
        reused = len(patterns)
        self.reused_pattern_instances += reused
        def values():
            return np.array([min((self.fixed_lp(s,u,k) for k in patterns if k in valid_keys[s]),default=np.inf)
                             for s in range(5)])
        vals = values()
        upper = float(vals.max())
        initial_upper = upper
        lower = -np.inf
        worst = None
        for j in range(sum(map(len,current))+5):
            call['checks'] = j+1
            scenario = int(vals.argmax())
            result = self.solve([scenario], u)
            if not np.isfinite(result['value']):
                return dict(infeasible=scenario,lower=np.inf,upper=np.inf)
            if result['bound'] > lower:
                lower = result['bound']
                worst = scenario
            key = tuple(result['selected'][scenario]['key'])
            duplicate = key in patterns
            if not duplicate:
                patterns.append(key)
            vals = values()
            upper = min(upper,float(vals.max()))
            row = dict(outer_iteration=outer_iteration,inner_iteration=j+1,
                lower_ratio=float(lower),upper_ratio=upper,upper_is_infinite=not np.isfinite(upper),
                solved_scenario=scenario,worst_scenario=int(vals.argmax()),
                patterns=len(patterns),reused_patterns=reused,initial_upper=initial_upper,
                eligible_cells=[len(c) for c in current],elapsed_s=time.perf_counter()-start)
            trace.append(row)
            call['bound_updates'] = j+1
            if np.isfinite(upper) and upper-lower <= tol*max(1,abs(upper)):
                call['status'] = 'bounds_closed'
                for k in patterns:
                    if k not in self.pool:
                        self.pool.append(k)
                return dict(infeasible=None,lower=float(lower),upper=upper,
                            worst_scenario=worst,patterns=len(patterns))
            if duplicate and int(vals.argmax()) == scenario:
                raise RuntimeError('Repeated certified mode with an unresolved gap.')
        raise RuntimeError('Recovery-mode generation did not converge.')

    def nccg(self):
        # Each outer round completes its inner feasibility/bound check before
        # applying the stopping test (including a final verification round).
        start=time.perf_counter();initial=self.feasible_initialization()
        active=[0];lower=-np.inf;upper=initial['upper'];incumbent=initial['u'].copy();trace=[];inner_trace=[]
        for iteration in range(1,12):
            master=self.solve(active)
            if not np.isfinite(master['value']):raise RuntimeError('Frozen certified robust model infeasible.')
            lower=max(lower,master['bound'])
            inner=self.inner(master['u'],iteration,inner_trace)
            if inner['infeasible'] is not None:
                scenario=inner['infeasible']
                trace.append(dict(iteration=iteration,lower_ratio=float(lower),upper_ratio=float(upper),
                    feasibility_scenario=scenario,active_scenarios=list(active),elapsed_s=time.perf_counter()-start))
                assert scenario not in active;active.append(scenario);continue
            candidate=max(master['value'],inner['upper'])
            if candidate<upper:upper=candidate;incumbent=master['u'].copy()
            trace.append(dict(iteration=iteration,lower_ratio=float(lower),upper_ratio=float(upper),
                relative_gap=float(max(0,upper-lower)/max(1,abs(upper))),active_scenarios=list(active),
                worst_scenario=inner['worst_scenario'],inner_patterns=inner['patterns'],elapsed_s=time.perf_counter()-start))
            if upper-lower<=1e-8*max(1,abs(upper)):break
            scenario=inner['worst_scenario'];assert scenario not in active,(trace,inner);active.append(scenario)
        else:raise RuntimeError('Outer loop did not close certified-model gap.')
        assert len(self.inner_call_audit)==len(trace) and all(r['checks']>=1 for r in self.inner_call_audit)
        extensive=self.solve(range(5));assert abs(extensive['value']-upper)<max(2e-6,2e-8*abs(upper))
        realized={s:self.solve([s],incumbent) for s in range(5)}
        result=dict(date=self.date,lower_ratio=lower,upper_ratio=upper,q=incumbent*self.B,u=incumbent,
            lower_usd=lower*self.refs[0],upper_usd=upper*self.refs[0],outer_trace=trace,inner_trace=inner_trace,
            extensive_objective=extensive['value'],extensive_gap_to_nccg=abs(extensive['value']-upper),
            scenarios=realized,elapsed_s=time.perf_counter()-start,LP_calls=self.lp_calls,MIP_calls=self.mip_calls,
            cells_sha256=self.hashes,initialization=initial,bound_scope='Frozen finite certified polyhedral recourse model, not unrestricted nonlinear optimum.')
        result.update(cache_hits=self.cache_hits,
            screened_cell_instances=self.screened_cell_instances,
            reused_pattern_instances=self.reused_pattern_instances,
            mode_pool_size=len(self.pool),solver_audit=self.solver_audit,
            inner_call_audit=self.inner_call_audit,
            algorithm='Interval-certified NCCG with carbon-quota screening and recovery-mode reuse')
        return result
