"""Common valid quota-support cuts, applied to all three solver formulations."""
import benchmark_clusters_v38 as engine
from benchmark_clusters_v38 import *
from scipy import sparse
Base=engine.ClusterModel

class PresolvedClusterModel(Base):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs)
  weights=np.array([[(i>>k)&1 for k in range(6)] for i in range(1,64)],float)
  rows=[];rhs=[]
  for g in range(self.groups):
   required=np.max([np.min(self.arr[s][g][2]@weights.T,axis=0)-weights.sum(axis=1) for s in range(self.S)],axis=0)
   a=np.zeros((len(weights),len(self.B)));a[:,6*g:6*(g+1)]=weights;rows.append(a);rhs.append(required)
  self.quota_cuts=(sparse.csr_matrix(np.vstack(rows)),np.concatenate(rhs))
  # Every inequality is a lower support value of the existing discrete
  # allocation set; none changes an original all-scenario feasible solution.
  A,b=self.quota_cuts;assert np.min(A@self.qbar-b)>=-1e-7

def protocol():
 path=ROOT/'revision_v38/experiments/cluster_benchmark_protocol.json';path.parent.mkdir(parents=True,exist_ok=True)
 obj={'regions':[1,2,4,8],'parks':[6,12,24,48],'dates':DATES,'scenarios':5,'network_per_region':[79,126],
 'regional_error_profile':'Original five source-load scenarios, cyclic permutation (s+g) modulo 5 for region g.',
 'regional_endowment':'Original allowance multiplied by 1+0.005*g, g=0,...,G-1. Original process library, generator capacities and network physics retained in every region.',
 'coupling':'Region-specific first-stage allowances and dispatches; one common worst-case objective aggregates cost, emissions and curtailment over all regions. No inter-region electrical tie is introduced or claimed.',
 'presolve':'For each region and every nonempty subset of its six parks, use the minimum summed normalized cell allocation per scenario as a necessary quota support bound, then take the maximum across scenarios. Same exact redundant cuts for full MILP and both NCCG variants.',
 'stopping':'600 seconds per method, one HiGHS thread, MIP relative gap 1e-8 and absolute gap 1e-9; common NCCG outer/inner gap 2e-8.',
 'retention':'All calibration runs retained separately; no outcome-based size or day exclusion. Report all timeouts and incumbent/bound pairs. Original six-park method outcomes remain unchanged.'}
 if path.exists():assert json.loads(path.read_text(encoding='utf8'))==obj
 else:dump(path,obj)

if __name__=='__main__':
 protocol();engine.ClusterModel=PresolvedClusterModel;engine.RESULT=ROOT/'revision_v38/experiments/clusters_presolved'
 with threadpool_limits(limits=1):engine.run(sys.argv[1],int(sys.argv[2]),sys.argv[3],int(sys.argv[4]) if len(sys.argv)>4 else 0)
