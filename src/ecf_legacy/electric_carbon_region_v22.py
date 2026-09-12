"""Recovery-labelled, interval-certified electric/carbon projection prototype.

Each chart interpolates NETWORK dispatches having exactly the same certified
full-horizon industrial trajectory. No interpolation between industrial
branches is admitted. The componentwise interval enclosure covers carbon
matrix changes at flow reversals and heat-curve knots. Carbon-quota purchases
are a modelled park contract, not negative physical emissions or ETS offsets.
"""
from pathlib import Path
import sys,json,time
ROOT=__import__("repository_paths").ROOT
import numpy as np
import cvxpy as cp
from threadpoolctl import threadpool_limits
from revise_comparators_v20 import setup,DATES
from parametric_master_v13 import Model
from dispatch_result_io import load,save
from branch_dispatch import evaluate

class DirectionModel(Model):
    def __init__(self,net,p,lib,kind):
        super().__init__(net,p,lib,True)
        direct=cp.sum(self.pr[net.direct_source_indices,:])
        # Boundary probes, not comparator settings or reported dispatch optima.
        obj={'import':direct+1e-6*self.cost,
             'export':-direct+1e-6*self.cost,
             'carbon':self.emissions,
             'curtailment':self.curtailment+1e-5*self.cost}[kind]
        self.problem=cp.Problem(cp.Minimize(obj/1000),self.problem.constraints)

def batched_carbon(net,p,pg,pr,pl):
    f=net.ptdf@(net.genmap@pg+net.renmap@pr-net.loadmap@pl-p['background_mw'])+net.flow_offset[:,None]
    n=len(net.bus);T=pl.shape[1];u=net.branch[:,0].astype(int);v=net.branch[:,1].astype(int)
    D=np.zeros((T,n,n));pin=(net.genmap@pg+net.renmap@pr).T.copy()
    for t in range(T):
        sender=np.where(f[:,t]>=0,u,v);receiver=np.where(f[:,t]>=0,v,u)
        np.add.at(pin[t],receiver,abs(f[:,t]));D[t,np.arange(n),np.arange(n)]=pin[t]
        np.add.at(D[t],(receiver,sender),-abs(f[:,t]))
    # The case has nonzero throughput at every node; verify, never invert a
    # zero-throughput node by adding an arbitrary numerical regularizer.
    assert pin.min()>1e-8
    emission=np.array([net.generation_emissions(pg[:,t]) for t in range(T)]).T
    e=(net.genmap@emission).T
    inv=np.linalg.inv(D);rho=np.einsum('tij,tj->ti',inv,e)
    return D,e,inv,rho,f

def enclose(net,p,a,b,left,right):
    mid=(left+right)/2;r=(right-left)/2
    pg=a['pg']+mid*(b['pg']-a['pg']);pr=a['pr']+mid*(b['pr']-a['pr']);pl=a['pl']
    D,e,inv,rho,f=batched_carbon(net,p,pg,pr,pl)
    dpg=b['pg']-a['pg'];dpr=b['pr']-a['pr'];df=net.ptdf@(net.genmap@dpg+net.renmap@dpr)
    T,n=rho.shape;u=net.branch[:,0].astype(int);v=net.branch[:,1].astype(int)
    # Positive and negative directed flows are each 1-Lipschitz in signed flow.
    # Where no reversal is possible only the active direction is needed.
    Dbar=np.zeros_like(D);pinbar=(r*abs(net.genmap@dpg+net.renmap@dpr)).T
    for t in range(T):
        delta=r*abs(df[:,t]);positive=f[:,t]+delta>0;negative=f[:,t]-delta<0
        np.add.at(pinbar[t],v,delta*positive);np.add.at(pinbar[t],u,delta*negative)
        Dbar[t,np.arange(n),np.arange(n)]=pinbar[t]
        np.add.at(Dbar[t],(v,u),delta*positive);np.add.at(Dbar[t],(u,v),delta*negative)
    pglo=pg-r*abs(dpg);pghi=pg+r*abs(dpg)
    eslope=np.array([np.maximum(abs(net.generation_emission_derivatives(pglo[:,t])),
                               abs(net.generation_emission_derivatives(pghi[:,t]))) for t in range(T)]).T
    ebar=(net.genmap@(r*abs(dpg)*eslope)).T
    B=np.einsum('tij,tjk->tik',abs(inv),Dbar)
    q=B.sum(axis=2).max(axis=1)
    if q.max()>=.95:return None
    vbar=np.einsum('tij,tj->ti',abs(inv),ebar+np.einsum('tij,tj->ti',Dbar,abs(rho)))
    # |Delta rho| <= (I-|D0^-1| Dbar)^-1 |D0^-1|(ebar+Dbar|rho0|).
    bound=np.linalg.solve(np.eye(n)[None,:,:]-B,vbar[:,:,None])[:,:,0]
    assert bound.min()>-1e-8
    rate0=(rho[:,net.industrial_buses].T*pl)
    ratebar=bound[:,net.industrial_buses].T*pl
    upper=(rate0+ratebar).sum(axis=1)*p['dt_h']
    error=ratebar.sum(axis=1)*p['dt_h']
    lower=(np.maximum(rate0-ratebar,0)).sum(axis=1)*p['dt_h']
    return dict(upper=upper,lower=lower,error=error,q=float(q.max()),mid=mid)

def certify(net,p,a,b,budget,purchase_ratio,max_depth=11):
    assert np.max(abs(a['pl']-b['pl']))<1e-7
    accepted=[];unresolved=[];queue=[(0.,1.,0)]
    cap=budget*(1+purchase_ratio)
    while queue:
        left,right,depth=queue.pop();cert=enclose(net,p,a,b,left,right)
        if cert is not None and np.all(cert['upper']<=cap+1e-7):
            accepted.append(dict(left=left,right=right,depth=depth,**cert));continue
        if cert is not None and np.any(cert['lower']>cap+1e-7):
            continue
        mid=(left+right)/2
        # A point outside the cap is not itself a certificate for the whole
        # interval. Keep bisecting, recording all discarded terminal cells.
        if depth>=max_depth:
            unresolved.append([left,right]);continue
        # Bound run time without pretending unresolved cells are infeasible.
        if len(accepted)+len(unresolved)+len(queue)>450:
            unresolved.append([left,right]);continue
        queue.extend([(mid,right,depth+1),(left,mid,depth+1)])
    accepted.sort(key=lambda v:v['left'])
    return accepted,unresolved

def settlement(allocation,allowance,buy_price=25.,sell_price=20.):
    net_quantity=np.asarray(allocation)-np.asarray(allowance)
    return dict(purchase_t=np.maximum(net_quantity,0),sale_t=np.maximum(-net_quantity,0),
                settlement_usd=buy_price*np.maximum(net_quantity,0)-sell_price*np.maximum(-net_quantity,0))

def run(date):
    began=time.perf_counter();net,p,old,prev,base,lib,refs,budget=setup(date)
    out=ROOT/f'results/electric_carbon_region_v22_{date}';out.mkdir(exist_ok=True)
    a=load(ROOT/f'results/integrated_v21_direct_{date}','M4_exact_carbon_selected') or load(prev,'M4_proposed')
    fixed=np.argmax(a['weights'],axis=1)
    # Exact same plant power and hence the same actual industrial execution.
    chartlib={'P':a['pl'][:,None,:],'admissible':np.ones((6,1),bool),'metadata':{'version':'recovery_labelled_projection_v22'}}
    save(out,'anchor',a);save(out,'case',dict(budget=budget,time_h=p['time_h'],branches=fixed,
          contract_buy_price_usd_t=25.,contract_sell_price_usd_t=20.,purchase_ratios=[0.,.02,.05],
          market_scope='Modelled park electricity-carbon contract; physical source emissions unchanged by transfers.'))
    summary=[]
    # One independent direction per chart; domain is the union of images, not
    # the convex hull of incompatible production or recovery branches.
    for kind in ['import','carbon','curtailment']:
        b=load(out,'direction_'+kind)
        if b is None:
            model=DirectionModel(net,p,chartlib,kind)
            b=model.run(fixed_branch=np.zeros(6,int),total_cost_cap=1.10*a['objective_actual'],time_limit=80,mip_gap=1e-8)
            assert b.get('primal_audit',{}).get('passed'),b.get('status')
            assert np.max(abs(b['pl']-a['pl']))<1e-7
            save(out,'direction_'+kind,b)
        for ratio in [0.,.02,.05]:
            name=f'{kind}_quota{int(ratio*100):02d}'
            oldcert=load(out,name)
            if oldcert is not None and oldcert.get('certificate_version')==2:continue
            if oldcert is not None:save(out,name+'_prior_traversal',oldcert)
            cells,unresolved=certify(net,p,a,b,budget,ratio)
            # Check enclosure against exact original carbon, including cell
            # endpoints, midpoint and deterministic off-centre interior points.
            max_excess=-np.inf;max_equation_residual=0.;samples=[]
            for cell in cells:
                for fraction in [0.,.2113248654,.5,.7886751346,1.]:
                    alpha=cell['left']+fraction*(cell['right']-cell['left'])
                    pg=a['pg']+alpha*(b['pg']-a['pg']);pr=a['pr']+alpha*(b['pr']-a['pr'])
                    D,e,inv,rho,f=batched_carbon(net,p,pg,pr,a['pl'])
                    rate=rho[:,net.industrial_buses].T*a['pl'];total=rate.sum(axis=1)*p['dt_h']
                    max_excess=max(max_excess,float((total-cell['upper']).max()))
                    max_equation_residual=max(max_equation_residual,float(abs(np.einsum('tij,tj->ti',D,rho)-e).max()))
                    samples.append(dict(alpha=alpha,grid_power=a['pl']-pr[net.direct_source_indices],
                       remaining_carbon_t=np.flip(np.cumsum(np.flip(rate,axis=1),axis=1),axis=1)*p['dt_h'],
                       allocation_t=total,**settlement(total,budget)))
            assert not cells or max_excess<1e-6
            result=dict(certificate_version=2,cells=cells,unresolved=unresolved,samples=samples,
                        certificate_sample_excess_t=None if not cells else max_excess,
                        maximum_carbon_balance_residual=max_equation_residual,
                        covered_parameter_length=sum(c['right']-c['left'] for c in cells),
                        direction=kind,purchase_ratio=ratio,branch_vector=fixed,
                        contract=settlement(a['allocation_t'],budget))
            save(out,name,result)
            row={k:result[k] for k in ['covered_parameter_length','certificate_sample_excess_t','maximum_carbon_balance_residual','direction','purchase_ratio']}
            row.update(cells=len(cells),unresolved=len(unresolved));summary.append(row)
            print('REGION',date,row,flush=True)
    (out/'run_summary.json').write_text(json.dumps(dict(elapsed_s=time.perf_counter()-began,rows=summary),indent=2))

if __name__=='__main__':
    with threadpool_limits(limits=1):
        for date in DATES[::-1]:run(date)
