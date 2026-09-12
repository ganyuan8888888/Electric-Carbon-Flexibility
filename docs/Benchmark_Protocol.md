# Benchmark protocol

## Scientific instances

The nominal study has six industrial parks, 79 buses, 126 branches, 48 scheduling periods, and five complete source–load scenarios per representative day. The scheduling interval is 30 minutes. Process integration uses a finer grid, with settings stored in the parameter and execution records.

The scaling study combines four physical regional models with 24 parks. Each region retains its network and industrial constraints. Distinct regional error combinations are generated with seed `2026091139`; the first five combinations are cyclic and larger sets are nested. The common worst-case objective couples regional operation. No inter-region electrical lines are introduced.

## Implementations

| Label | Implementation in this repository |
|---|---|
| Full MILP | `benchmark_joint_v39.JointModel.solve`, using the finalized v38 extensive-form solver |
| NCCG without reuse | v38 `FinalClusterModel.nccg` with the same distinct regional combinations |
| Proposed NCCG | `benchmark_candidate_v39.CandidateModel`, including valid LP lower bounds, candidate primal initialization, feasible MIP starts, recovery-mode reuse, and bound closure |

The proposed implementation includes optimizations beyond recovery-mode reuse. Consequently, its comparison with the archived v38 implementation measures the combined implementation, not an isolated single-feature ablation. The finite cells, objective, scenario identities, tolerances, and source network parameters remain common. The earlier calibration and validation records are retained in the archive.

Historical five-scenario comparator observations were retained from v38, with source hashes. New 10/20/40-scenario observations were measured using the frozen comparator code. `reproduce.py solve` runs every requested method freshly and sequentially. It never copies historical time into a new observation or pads elapsed time. Both historical and fresh records retain solver status, incumbent, bound, and traces.

## Numerical settings

HiGHS uses one thread. The manuscript scaling protocol sets a 600-second method limit, MIP relative gap `1e-8`, MIP absolute gap `1e-9`, and NCCG bound tolerance `2e-8`. Source routines retain their original per-solve and initialization settings; the command-line limit controls the overall deadline and main solve limit. Preprocessing is included in the wrapper wall time. A solver limit can retain an incumbent and bound without establishing optimality.

Only compare solution quality as certified when the relevant feasibility and bound checks pass. Fresh timing depends on hardware, solver version, compilation, and system activity. The repository checks equality of optimal objective values within tolerance and reports measured runtimes without enforcing a method ordering.

## Original scheduling comparisons

The archived nominal records are `M1_constant`, `M2_native`, `M3_native`, and `M4_proposed`. M2 is the published multi-timescale model adaptation in `native_industrial_v14.py`, with the common configuration recorded under `m2_configuration_v28`. M3 is the carbon-oriented power-flow adaptation in `recent_methods_v12.py`. The exact selected dispatch arrays and their physical checks are in `results/robust_v28/<date>/` under the reference root. Model-native feasibility and original-process execution checks are recorded separately.
