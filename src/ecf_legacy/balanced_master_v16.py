"""Minimize the largest normalized total, preserving exact-carbon checks.

Normalization references are saved comparison results. They do not change
loads, fuel curves, device resources, or any comparator's formulation.
"""
import numpy as np
import cvxpy as cp
from parametric_master_v13 import Model as Base

CACHE={}

class Model(Base):
    def __init__(self,net,prof,lib,relax,reference_totals):
        super().__init__(net,prof,lib,relax)
        self.reference_totals=np.asarray(reference_totals,dtype=float)
        assert self.reference_totals.shape==(3,) and (self.reference_totals>0).all()
        self.ratio=cp.Variable(nonneg=True)
        totals=[self.cost,self.emissions,self.curtailment]
        cons=self.problem.constraints+[x/ref<=self.ratio for x,ref in zip(totals,self.reference_totals)]
        self.problem=cp.Problem(cp.Minimize((self.reference_totals[0]*self.ratio+2e4*cp.sum(self.slack))/1000),cons)
        assert self.problem.is_dpp()

    def run(self,**kwargs):
        result=super().run(**kwargs)
        if 'pg' in result:
            totals=np.array([result[k] for k in ['objective_actual','generation_emission_t','curtailment_mwh']])
            result.update(normalization_reference_totals=self.reference_totals.copy(),
                worst_total_ratio=float(np.max(totals/self.reference_totals)),
                relative_improvement_percent=100*(1-totals/self.reference_totals))
            if 'relaxation_lower_bound' in result:
                result['normalized_minimax_master_bound']=result.pop('relaxation_lower_bound')
        return result

def master(net,prof,lib,reference_totals,relax=False,**kwargs):
    key=(id(net),id(prof),id(lib),bool(relax),tuple(reference_totals))
    if key not in CACHE:CACHE[key]=Model(net,prof,lib,relax,reference_totals)
    return CACHE[key].run(**kwargs)
