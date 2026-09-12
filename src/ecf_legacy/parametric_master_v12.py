"""Reusable DPP carbon master; all changing numerical inputs are parameters.

Cache is process-local and keyed by identities of the complete network, profile
and library objects and by integrality. Carbon budgets, every coefficient,
centers, radii, branch bounds and neighborhood constraints are reset per call.
"""
import time,gc
import numpy as np
import cvxpy as cp
from scipy.linalg import solve

CACHE={}

class Model:
    def __init__(self,net,prof,lib,relax):
        self.net,self.prof,self.lib,self.relax=net,prof,lib,relax
        K,B,T=lib['P'].shape;ng=len(net.gen);nr=len(net.ren_buses)
        self.w=cp.Variable((K,B),boolean=not relax);self.pg=cp.Variable((ng,T));self.pr=cp.Variable((nr,T));self.epi=cp.Variable((ng,T));self.slack=cp.Variable(K,nonneg=True)
        w,pg,pr,epi=self.w,self.pg,self.pr,self.epi
        self.pl=cp.vstack([w[k]@lib['P'][k] for k in range(K)]);pl=self.pl
        inj=net.genmap@pg+net.renmap@pr-net.loadmap@pl-prof['background_mw']
        self.flow=net.ptdf@inj+net.flow_offset[:,None]
        self.power_balance=cp.sum(inj,axis=0)==0;self.line_upper=self.flow<=net.capacity[:,None];self.line_lower=self.flow>=-net.capacity[:,None]
        self.lowerw=cp.Parameter((K,B));self.upperw=cp.Parameter((K,B));self.neighbor=cp.Parameter((K,B));self.neighbor_rhs=cp.Parameter()
        self.centerpg=cp.Parameter((ng,T));self.centerpr=cp.Parameter((nr,T));self.centerpl=cp.Parameter((K,T))
        self.radpg=cp.Parameter((ng,T),nonneg=True);self.radpr=cp.Parameter((nr,T),nonneg=True);self.radpl=cp.Parameter((K,T),nonneg=True)
        self.cg=[cp.Parameter((ng,T)) for _ in range(K)];self.cr=[cp.Parameter((nr,T)) for _ in range(K)]
        self.cl=[cp.Parameter((K,T)) for _ in range(K)];self.ce=[cp.Parameter((ng,T)) for _ in range(K)];self.rhs=cp.Parameter(K)
        on=net.fixed_commitment
        cons=[cp.sum(w,axis=1)==1,w>=self.lowerw,w<=self.upperw,
            cp.sum(cp.multiply(self.neighbor,w))>=self.neighbor_rhs,
            pg>=net.pmin[:,None]*on[:,None],pg<=net.pmax[:,None]*on[:,None],pr>=prof['minimum_renewable_mw'],pr<=prof['available_renewable_mw'],
            self.power_balance,self.line_upper,self.line_lower,
            cp.abs(pg-self.centerpg)<=self.radpg,cp.abs(pr-self.centerpr)<=self.radpr,cp.abs(pl-self.centerpl)<=self.radpl,epi>=0]
        ramp=net.ramp_mw_per_hour[:,None]*prof['dt_h'];cons += [pg[:,1:]-pg[:,:-1]<=ramp,pg[:,:-1]-pg[:,1:]<=ramp]
        for g,(xp,hp) in enumerate(zip(net.fuel_power_points,net.fuel_heat_points)):
            for j in range(len(xp)-1):
                slope=(hp[j+1]-hp[j])/(xp[j+1]-xp[j]);offset=hp[j]-slope*xp[j]
                cons.append(epi[g]>=net.fuel_price[g]*(slope*pg[g]+offset*on[g])+net.vom[g]*pg[g])
        linear=[]
        for k in range(K):linear.append(cp.sum(cp.multiply(self.cg[k],pg))+cp.sum(cp.multiply(self.cr[k],pr))+cp.sum(cp.multiply(self.cl[k],pl))+cp.sum(cp.multiply(self.ce[k],epi)))
        cons.append(cp.hstack(linear)<=self.rhs+self.slack)
        objective=prof['dt_h']*(cp.sum(epi)+150*cp.sum(prof['available_renewable_mw']-pr))+2e4*cp.sum(self.slack)
        self.problem=cp.Problem(cp.Minimize(objective/1000),cons)
        assert self.problem.is_dpp()
        self.compiled_calls=0

    def run(self,center=None,budgets=None,trust=.08,fixed_branch=None,time_limit=45.,budget_buffer=.001,mip_gap=2e-7,branch_reference=None,max_branch_changes=None):
        from branch_dispatch import evaluate
        from audit_dispatch_feasibility import primal_audit
        start=time.perf_counter();net,prof,lib=self.net,self.prof,self.lib
        K,B,T=lib['P'].shape;ng=len(net.gen);nr=len(net.ren_buses)
        lowerw=np.zeros((K,B));upperw=lib['admissible'].astype(float).copy()
        if fixed_branch is not None:lowerw[np.arange(K),fixed_branch]=1
        self.lowerw.value=lowerw;self.upperw.value=upperw
        selector=np.zeros((K,B));rhs=0.
        if branch_reference is not None and max_branch_changes is not None:
            selector[np.arange(K),branch_reference]=1;rhs=K-max_branch_changes
        self.neighbor.value=selector;self.neighbor_rhs.value=rhs
        self.cgvals=np.zeros((K,ng,T));self.crvals=np.zeros((K,nr,T));self.clvals=np.zeros((K,K,T));self.cevals=np.zeros((K,ng,T))
        if center is None:
            for param,shape in [(self.centerpg,(ng,T)),(self.centerpr,(nr,T)),(self.centerpl,(K,T))]:param.value=np.zeros(shape)
            for param,shape in [(self.radpg,(ng,T)),(self.radpr,(nr,T)),(self.radpl,(K,T))]:param.value=np.full(shape,1e6)
            self.rhs.value=np.full(K,1e9)
        else:
            assert budgets is not None
            self.centerpg.value=center['pg'];self.centerpr.value=center['pr'];self.centerpl.value=center['pl']
            self.radpg.value=np.broadcast_to(trust*net.pmax[:,None],(ng,T));self.radpr.value=trust*np.maximum(prof['available_renewable_mw'],1);self.radpl.value=2*trust*np.maximum(center['pl'],1)
            intercept=np.zeros(K)
            for t in range(T):
                pg0,pr0,pl0=center['pg'][:,t],center['pr'][:,t],center['pl'][:,t]
                c=net.carbon(pg0,pr0,pl0,prof['background_mw'][:,t],derivatives='fast');active=c['active']
                adj=solve(c['D'][np.ix_(active,active)].T,net.loadmap.T[:,active].T).T
                hg=pl0[:,None]*(adj@net.genmap[active])*net.fuel_carbon_t_mmbtu[None,:]*prof['dt_h']
                assert hg.min()>-1e-8,hg.min()
                gr=c['dcarbon']*prof['dt_h'];h0=net.fuel_heat(pg0);slope=net.fuel_heat(pg0,True)
                self.cgvals[:,:,t]=gr[:,:ng]-hg*slope[None,:]-hg*(net.vom/net.fuel_price)[None,:]
                self.crvals[:,:,t]=gr[:,ng:ng+nr];self.clvals[:,:,t]=gr[:,ng+nr:];self.cevals[:,:,t]=hg/net.fuel_price[None,:]
                intercept+=c['industrial_carbon_rate']*prof['dt_h']-gr[:,:ng]@pg0-gr[:,ng:ng+nr]@pr0-gr[:,ng+nr:]@pl0-hg@h0+(hg*slope[None,:])@pg0
            self.rhs.value=budgets*(1-budget_buffer)-intercept
            # Center substitution must recover the exact carbon allocation.
            heat0=np.array([net.fuel_heat(center['pg'][:,t]) for t in range(T)]).T
            epi0=net.fuel_price[:,None]*heat0+net.vom[:,None]*center['pg']
            substitute=intercept+np.sum(self.cgvals*center['pg'][None,:,:],axis=(1,2))+np.sum(self.crvals*center['pr'][None,:,:],axis=(1,2))+np.sum(self.clvals*center['pl'][None,:,:],axis=(1,2))+np.sum(self.cevals*epi0[None,:,:],axis=(1,2))
            assert abs(substitute-center['allocation_t']).max()<1e-7
        for k in range(K):
            self.cg[k].value=self.cgvals[k];self.cr[k].value=self.crvals[k];self.cl[k].value=self.clvals[k];self.ce[k].value=self.cevals[k]
        # Reuse CVXPY's parameter-to-matrix map, but release each native solver.
        # Concurrent default-thread HiGHS instances exhausted this 16-GB host.
        self.problem._solver_cache.clear();gc.collect()
        self.problem.solve(solver='HIGHS',warm_start=False,highs_options={'time_limit':time_limit,'mip_rel_gap':mip_gap,'log_to_console':False,'threads':1})
        self.compiled_calls+=1
        if self.pg.value is None:return dict(status=self.problem.status,runtime_s=time.perf_counter()-start)
        pg,pr,pl,w=[v.value for v in [self.pg,self.pr,self.pl,self.w]]
        if not all(np.isfinite(x).all() for x in [pg,pr,pl,w]):return dict(status='nonfinite_candidate',runtime_s=time.perf_counter()-start)
        r=dict(pg=pg,pr=pr,pl=pl,weights=w,flow=self.flow.value,status=self.problem.status,commitment=np.asarray(net.fixed_commitment),
            branches=np.argmax(w,axis=1),runtime_s=time.perf_counter()-start,master_objective=float(self.problem.value*1000),carbon_linear_slack_t=self.slack.value,
            master_reuse_calls=self.compiled_calls,library_scope=lib['metadata']['version'])
        stats=self.problem.solver_stats.extra_stats;primal=float(getattr(stats,'objective_function_value',np.nan));dual=float(getattr(stats,'mip_dual_bound',np.nan))
        offset=float(self.problem.value)-primal;r['canonical_objective_offset']=1000*offset
        r['relaxation_lower_bound']=1000*(dual+offset) if not self.relax else 1000*float(self.problem.value)
        if r['status']=='optimal' and not np.isfinite(r['relaxation_lower_bound']):r['relaxation_lower_bound']=1000*float(self.problem.value)
        r.update(evaluate(net,prof,pg,pr,pl));r['primal_audit']=primal_audit(net,prof,lib,r,require_integral=not self.relax)
        if not r['primal_audit']['passed']:return dict(status='failed_independent_primal_audit',runtime_s=r['runtime_s'],primal_audit=r['primal_audit'])
        if self.relax and center is None:r['economic_nodal_price_per_mwh']=-(self.power_balance.dual_value[None,:]+net.ptdf.T@(self.line_upper.dual_value-self.line_lower.dual_value))*1000/prof['dt_h']
        self.problem._solver_cache.clear();gc.collect()
        return r

def master(net,prof,lib,relax=False,exact_heat_cut=True,**kwargs):
    assert exact_heat_cut
    key=(id(net),id(prof),id(lib),bool(relax))
    if key not in CACHE:CACHE[key]=Model(net,prof,lib,relax)
    return CACHE[key].run(**kwargs)
