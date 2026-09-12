"""Native HiGHS MIP start from an independently feasible CVXPY witness."""
from evidence_core_v37 import *
import cvxpy.settings as settings
import highspy as hp

def solve_with_start(problem, options, use_start):
 data,chain,inverse=problem.get_problem_data('HIGHS')
 A=data[settings.A].tocsc();b=data[settings.B];c=data[settings.C];n=len(c)
 model=hp.HighsModel();p=model.lp_;p.num_col_=n;p.num_row_=len(b);p.col_cost_=c
 p.row_upper_=b;p.row_lower_=np.r_[b[:data[settings.DIMS].zero],np.full(len(b)-data[settings.DIMS].zero,-np.inf)]
 lb=data[settings.LOWER_BOUNDS];ub=data[settings.UPPER_BOUNDS]
 lower=np.full(n,-np.inf) if lb is None else lb.copy();upper=np.full(n,np.inf) if ub is None else ub.copy()
 integrality=[hp.HighsVarType.kContinuous]*n
 for i in data[settings.BOOL_IDX]:integrality[i]=hp.HighsVarType.kInteger;lower[i]=max(lower[i],0.);upper[i]=min(upper[i],1.)
 for i in data[settings.INT_IDX]:integrality[i]=hp.HighsVarType.kInteger
 p.col_lower_=lower;p.col_upper_=upper;p.integrality_=integrality
 p.a_matrix_.format_=hp.MatrixFormat.kColwise;p.a_matrix_.start_=A.indptr;p.a_matrix_.index_=A.indices;p.a_matrix_.value_=A.data
 solver=hp.Highs()
 for k,v in options.items():assert solver.setOptionValue(k,v)!=hp.HighsStatus.kError,(k,v)
 solver.passModel(model)
 audit=dict(supplied=False)
 if use_start:
  columns=data[settings.PARAM_PROB].var_id_to_col;x=np.full(n,np.nan)
  for variable in problem.variables():
   if variable.id in columns and variable.value is not None:
    first=columns[variable.id];x[first:first+variable.size]=np.asarray(variable.value).ravel(order='F')
  assert np.isfinite(x).all()
  ax=A@x;violation=max(float(np.max(np.maximum(ax-b,0))),float(np.max(np.abs(ax[:data[settings.DIMS].zero]-b[:data[settings.DIMS].zero]))),float(np.max(np.maximum(lower-x,0))),float(np.max(np.maximum(x-upper,0))))
  assert violation<2e-7,violation
  solution=hp.HighsSolution();solution.col_value=x;solution.value_valid=True
  assert solver.setSolution(solution)!=hp.HighsStatus.kError
  audit=dict(supplied=True,max_constraint_violation=violation,objective=float(c@x))
 solver.run()
 results=dict(solution=solver.getSolution(),basis=solver.getBasis(),info=solver.getInfo(),model_status=solver.getModelStatus().name,run_time=solver.getRunTime())
 if results['model_status']=='kInfeasible':results['dual_ray']=solver.getDualRay()
 problem.unpack_results(results,chain,inverse)
 return audit
