"""Optional total-metric epsilon constraints for Pareto diagnosis.

Original carbon-flow master and independent feasibility checks are unchanged.
All caps reset at every call. Economic cost retains the original definition;
the optional emission price is a scalarization parameter, not reported revenue.
"""
import numpy as np
import cvxpy as cp
from parametric_master_v12 import Model as Base
CACHE={}

class Model(Base):
    def __init__(self,net,prof,lib,relax):
        super().__init__(net,prof,lib,relax)
        self.carbon_constraint=self.problem.constraints[-1]
        dt=prof['dt_h']
        self.cost=dt*(cp.sum(self.epi)+150*cp.sum(prof['available_renewable_mw']-self.pr))
        self.emissions=dt*cp.sum(cp.multiply(net.fuel_carbon_t_mmbtu[:,None]/net.fuel_price[:,None],self.epi-cp.multiply(net.vom[:,None],self.pg)))
        self.curtailment=dt*cp.sum(prof['available_renewable_mw']-self.pr)
        self.ccap=cp.Parameter(nonneg=True);self.ecap=cp.Parameter(nonneg=True);self.rcap=cp.Parameter(nonneg=True);self.eprice=cp.Parameter(nonneg=True)
        self.problem=cp.Problem(cp.Minimize(self.problem.objective.expr+self.eprice*self.emissions/1000),
            self.problem.constraints+[self.cost<=self.ccap,self.emissions<=self.ecap,self.curtailment<=self.rcap])
        assert self.problem.is_dpp()

    def run(self,total_cost_cap=None,source_emission_cap=None,curtailment_cap=None,emission_price=0.,**kwargs):
        self.ccap.value=1e12 if total_cost_cap is None else total_cost_cap
        self.ecap.value=1e12 if source_emission_cap is None else source_emission_cap
        self.rcap.value=1e12 if curtailment_cap is None else curtailment_cap
        self.eprice.value=emission_price
        r=super().run(**kwargs)
        if 'pg' in r:
            r['total_metric_caps']=dict(cost=total_cost_cap,source_emissions=source_emission_cap,curtailment=curtailment_cap)
            r['total_metric_excess']={key:max(0.,r[key]-bound) for key,bound in [('objective_actual',self.ccap.value),('generation_emission_t',self.ecap.value),('curtailment_mwh',self.rcap.value)]}
            r['total_metric_feasible']=max(r['total_metric_excess'].values())<=1e-4
            r['emission_scalarization_price']=emission_price
            if self.relax:
                nodal=-(self.power_balance.dual_value[None,:]+self.net.ptdf.T@(self.line_upper.dual_value-self.line_lower.dual_value))*1000/self.prof['dt_h']
                multiplier=np.asarray(self.carbon_constraint.dual_value)*1000
                r['source_carbon_nodal_price_per_mwh']=nodal
                r['industrial_column_price_per_mwh']=self.net.loadmap.T@nodal+np.einsum('k,kmt->mt',multiplier,self.clvals)/self.prof['dt_h']
                r['industrial_carbon_multiplier_usd_t']=multiplier
            if emission_price:
                r['scalarized_lower_bound']=r.pop('relaxation_lower_bound')
        return r

def master(net,prof,lib,relax=False,**kwargs):
    key=(id(net),id(prof),id(lib),bool(relax))
    if key not in CACHE:CACHE[key]=Model(net,prof,lib,relax)
    return CACHE[key].run(**kwargs)
