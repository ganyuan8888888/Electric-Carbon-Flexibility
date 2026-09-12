"""Exact local carbon-flow derivative and resolvent remainder certificate.

This is a building block of the joint branch certificate, not a claim that
Neumann series or matrix perturbation theory are new. Flow directions and the
active nodal set must remain fixed for the affine D/e representation used here.
"""
import numpy as np
from scipy.linalg import solve

def certificate(reference,delta,actual=None):
    D=reference['D'];rho=reference['rho'];active=reference['active']
    dD=np.einsum('j,jnm->nm',delta,reference['dD'])
    de=reference['de']@delta
    idx=np.ix_(active,active)
    L=solve(D[idx],dD[idx]);v=solve(D[idx],(de-dD@rho)[active])
    q=float(np.linalg.norm(L,np.inf))
    rem=q*np.linalg.norm(v,np.inf)/(1-q) if q<1 else np.inf
    prediction=rho.copy();prediction[active]+=v
    unchanged=bool(np.all(reference['flow']*(reference['flow']+reference['df']@delta)>=-1e-10))
    result={'q':q,'rho_linear':prediction,'remainder_bound':float(rem),'same_flow_directions':unchanged,
        'valid':bool(q<1 and unchanged)}
    if actual is not None:
        result['actual_remainder']=float(np.max(abs(actual['rho'][active]-prediction[active])))
        result['active_set_unchanged']=bool(np.array_equal(actual['active'],active))
        result['valid']&=result['active_set_unchanged']
        result['emission_affine_residual']=float(np.max(abs(actual['e']-reference['e']-de)))
        result['valid']&=result['emission_affine_residual']<1e-7
    return result
