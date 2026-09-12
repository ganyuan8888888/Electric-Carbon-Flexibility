"""Case adaptation of Liu et al., Processes 13 (2025), 1411.

Native structure: two thermal states, constant-voltage current/power interface,
linear production, command ramp and command-count limits (Eqs. 11, 14, 34-39).
Use the temperature-model Case 2, not its alternative four-hour energy model.
No proposed waveform, Floquet map, certified library or branch search is used.

Case parameters are obtained from the same nominal equipment operating point.
The lumped network preserves nominal net heat and sensible heat capacities;
this is an explicitly calibrated two-state adaptation, not an exact numerical
reproduction of the paper's 400-kA cell. External plant tests remain separate.
"""
from pathlib import Path
import json,sys,time
import numpy as np
import cvxpy as cp
from scipy.linalg import expm
from thermal_process import ThermalCell
from rts_network import make_rts_network,rts_profiles
from branch_dispatch import evaluate
from dispatch_result_io import load,save
from audit_dispatch_feasibility import primal_audit
ROOT=__import__("repository_paths").ROOT

def parameters(net):
    # This file contains common equipment parameters only. No proposed
    # candidate, waveform, branch or certificate is loaded by the baseline.
    raw=json.loads((ROOT/'data/industrial_case_parameters_v15.json').read_text())
    cells=[];rows=[]
    for k,p in enumerate(raw['parameters']):
        m=ThermalCell(**p);cells.append(m);state=m.initial();T,S,l=np.split(state,3)
        aux=m.electrical(T,180.,0.,.043)
        N=int(np.maximum(1,np.round(net.industrial_base[k]*1e6/aux['power_w']/3)))*3
        heat=float(aux['heat_w'].sum());t0=float(T.mean());s0=float(S.mean())
        G=heat/(t0-s0);H=heat/(s0-m.ambient)
        C=float(np.sum(m.mass(l)*m.bath_cp)+m.metal_mass*m.metal_cp);Cs=m.shell_capacity_total
        A=np.array([[-G/C,G/C],[G/Cs,-(G+H)/Cs]])
        # Linear input coefficient is calibrated to the shared nominal heat.
        # A constant term restores the same equilibrium exactly.
        b=np.array([heat/180/C,0.]);aug=np.zeros((3,3));aug[:2,:2]=A;aug[:2,2]=b
        disc=expm(aug*1800);Ad=disc[:2,:2];Bd=disc[:2,2]
        rows.append(dict(site=k,N=N,nominal_power_mw=N*aux['power_w']/1e6,
            nominal_voltage_v=aux['power_w']/180000,initial_temperature_c=t0,
            initial_solid_temperature_c=s0,liquid_capacity_j_k=C,solid_capacity_j_k=Cs,
            liquid_solid_conductance_w_k=G,solid_ambient_conductance_w_k=H,
            net_heat_w=heat,Ad=Ad.tolist(),Bd=Bd.tolist()))
    return cells,rows

def constraints(cp_current,rows,change_limit=12,relax_commands=False,case_execution_bounds=False,terminal_recovery=False):
    # Source Eq. (39) is a daily energy interval. The equality below is the
    # explicitly declared fixed-output case adaptation under its U=constant
    # interface; it is not claimed to be the literal published constraint.
    K,T=cp_current.shape;x=cp.Variable((2*K,T+1))
    cons=[cp_current>=144.,cp_current<=216.,cp.sum(cp_current,axis=1)==180*T]
    if case_execution_bounds:
        # Unexcited operation of the same device: no periodic stabilization is
        # supplied to the baseline. Derive its DC boundary from the plant,
        # using the same 0.0012 decay requirement used in the case.
        from harmonic_process import matrix
        from scipy.optimize import brentq
        bound=brentq(lambda v:np.real(np.linalg.eigvals(matrix(v,.043))).max()+.0012,180,210)
        cons.append(cp_current<=bound)
    # The +/-20% interface and 20% per-hour command ramp are from Table 1.
    # The shared case's actual 130--230 kA equipment bounds are wider.
    delta=cp.hstack([cp_current[:,0:1]-180,cp_current[:,1:]-cp_current[:,:-1]])
    cons += [cp.abs(delta)<=18.]
    up=down=None
    if change_limit is not None:
        up=cp.Variable((K,T),boolean=not relax_commands);down=cp.Variable((K,T),boolean=not relax_commands)
        cons += [up>=0,down>=0,up+down<=1,delta<=18*up,delta>=-18*down,
            cp.sum(up+down,axis=1)<=change_limit]
    for k,row in enumerate(rows):
        block=x[2*k:2*k+2];Ad=np.array(row['Ad']);Bd=np.array(row['Bd'])
        cons += [block[:,0]==0,
            block[:,1:]==Ad@block[:,:-1]+Bd[:,None]@cp.reshape(cp_current[k]-180,(1,T),order='C'),
            block[0]>=940-row['initial_temperature_c'],block[0]<=980-row['initial_temperature_c']]
        if case_execution_bounds and terminal_recovery:cons += [cp.abs(block[:,-1])<=2.]
    p0=np.array([r['nominal_power_mw'] for r in rows])
    return cp.multiply(p0[:,None]/180,cp_current),x,cons,up,down

def run(date,carbon_price=25.,change_limit=12,time_limit=240,case_execution_bounds=False,terminal_recovery=False,net=None,prof=None,output=None,use_provided_commitment=False):
    net=make_rts_network() if net is None else net
    prof=rts_profiles(net,date) if prof is None else prof
    models,rows=parameters(net)
    if use_provided_commitment:
        on=np.asarray(net.fixed_commitment).copy()
    else:
        constant=load(ROOT/f'results/comparison_v12_{date}','M1_constant');on=constant['commitment'];net.fixed_commitment=on
    K=6;T=48;ng=len(net.gen);nr=len(net.ren_buses);dt=prof['dt_h']
    current=cp.Variable((K,T));P,x,cons,up,down=constraints(current,rows,change_limit,case_execution_bounds=case_execution_bounds,terminal_recovery=terminal_recovery)
    pg=cp.Variable((ng,T));pr=cp.Variable((nr,T));epi=cp.Variable((ng,T))
    inj=net.genmap@pg+net.renmap@pr-net.loadmap@P-prof['background_mw'];flow=net.ptdf@inj+net.flow_offset[:,None]
    cons += [pg>=net.pmin[:,None]*on[:,None],pg<=net.pmax[:,None]*on[:,None],
        pr>=prof['minimum_renewable_mw'],pr<=prof['available_renewable_mw'],cp.sum(inj,axis=0)==0,
        flow<=net.capacity[:,None],flow>=-net.capacity[:,None],epi>=0,
        cp.abs(pg[:,1:]-pg[:,:-1])<=net.ramp_mw_per_hour[:,None]*dt]
    for g,(xp,hp) in enumerate(zip(net.fuel_power_points,net.fuel_heat_points)):
        for j in range(len(xp)-1):
            a=(hp[j+1]-hp[j])/(xp[j+1]-xp[j]);b=hp[j]-a*xp[j]
            cons.append(epi[g]>=net.fuel_price[g]*(a*pg[g]+b*on[g])+net.vom[g]*pg[g])
    cost=dt*(cp.sum(epi)+150*cp.sum(prof['available_renewable_mw']-pr))
    emission=dt*cp.sum(cp.multiply(net.fuel_carbon_t_mmbtu[:,None]/net.fuel_price[:,None],epi-cp.multiply(net.vom[:,None],pg)))
    problem=cp.Problem(cp.Minimize((cost+carbon_price*emission)/1000),cons);begin=time.perf_counter()
    problem.solve(solver='HIGHS',highs_options={'threads':1,'time_limit':time_limit,'mip_rel_gap':1e-5,'log_to_console':False})
    assert pg.value is not None,problem.status
    r=dict(pg=pg.value,pr=pr.value,pl=P.value,flow=flow.value,current_ka=current.value,
        linear_temperature_deviation_c=x.value,commitment=on,weights=np.ones((K,1)),branches=np.zeros(K,dtype=int),
        status=problem.status,runtime_s=time.perf_counter()-begin,carbon_price_usd_t=carbon_price,
        command_limit=change_limit,adaptation_parameters=rows,search_objective=float(problem.value*1000))
    if up is not None:r.update(command_up=up.value,command_down=down.value)
    r.update(evaluate(net,prof,r['pg'],r['pr'],r['pl']))
    stats=problem.solver_stats.extra_stats
    r.update(mip_gap=float(getattr(stats,'mip_gap',np.nan)),mip_node_count=int(getattr(stats,'mip_node_count',0)),
        case_execution_bounds=case_execution_bounds,added_terminal_recovery=bool(case_execution_bounds and terminal_recovery))
    # A one-point wrapper is used ONLY by the independent network audit after
    # solving. There is no trajectory library in this optimization problem.
    checklib={'P':r['pl'][:,None,:],'admissible':np.ones((K,1),bool)}
    r['primal_audit']=primal_audit(net,prof,checklib,r)
    r['linear_constraint_violation']=max(float(np.max(c.violation(),initial=0)) for c in cons)
    r['scope']='Liu-2025 temperature-model adaptation; planned quantities pending independent nonlinear execution.'
    out=ROOT/f'results/native_industrial_v14_{date}' if output is None else Path(output);out.mkdir(exist_ok=True)
    name=f'Liu2025_price{carbon_price:g}_commands{change_limit}'+('_casebounds' if case_execution_bounds else '')
    if not terminal_recovery:name+='_native_terminal_free'
    save(out,name,r)
    return r
    print({k:r[k] for k in ['status','runtime_s','objective_actual','generation_emission_t','curtailment_mwh','linear_constraint_violation']},flush=True)
    return r

if __name__=='__main__':
    pass # Dependencies are installed through requirements.txt.
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):run(sys.argv[1] if len(sys.argv)>1 else '2020-11-29',
        change_limit=None if '--continuous' in sys.argv else 12,case_execution_bounds='--dc-envelope' in sys.argv,
        terminal_recovery='--added-terminal-recovery' in sys.argv)
