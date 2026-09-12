# Data provenance

All paths below are relative to `data/reference/`.

| Data family | Location | Origin and interpretation |
|---|---|---|
| Source network and chronology | `data/RTS-GMLC-v0.2.3/` | Public benchmark v0.2.3, preserved original files and notice |
| Industrial parameters | `data/industrial_case_parameters_v15.json` | Declared heterogeneous simulation parameters |
| Three-day joint scenarios | `results/uncertainty_v26/` | Complete-day correlated renewable/load inputs and selection records |
| Production and recovery trajectories | `results/full_cycle_v13/`, `results/library/`, `results/pareto_v28/` | Nonlinear simulated execution and certified library inputs |
| Selected dispatch and common allowances | `results/robust_v28/` | Four-method results, scenario endpoints, cell certificates, shared decisions and audits |
| Core convergence | `revision_v32/algorithm_results/` | Saved outer/inner iterations and finite-model comparisons |
| Region certification | `revision_v37/experiments/region_dense/` | Seven fixed complete-trajectory charts with 2001-by-2001 grid classifications and depth-9 bounds |
| Holdout and stress evaluation | `revision_v37/experiments/oos_final/` | Twelve held-out donor trajectories and 200 generated stress trajectories per day; paired numerical retry records remain separate |
| Nonlinear AC recourse | `revision_v39/experiments/ac_complete_summary.json` | Final per-day source and array references for 720 operating points and 15 daily budgets |
| Algorithm scaling | `revision_v39/experiments/final_benchmark/` | All 36 final date/size/method records with status, bounds and timing provenance |
| Earlier evaluations | Other directories in `revision_v37/experiments/`, `revision_v38/experiments/`, `revision_v39/experiments/` | Retained calibration, solver attempts and alternative evaluations, not silently substituted for final results |

The original RTS file manifest includes a bibliography PDF that is intentionally not redistributed here. The packaged CSVs and accompanying source notices are covered by the original data-use notice. `data/REFERENCE_MANIFEST.csv` is the authoritative inventory of the files actually included in this repository.

## Stored array conventions

`dispatch_result_io.load` verifies the paired NPZ hash when `_arrays_sha256` is present in JSON. It combines numerical arrays with scalar metadata. Typical fields are:

| Field | Meaning | Unit / axes |
|---|---|---|
| `pg` | Dispatchable generation | MW, generator × period |
| `pr` | Utilized renewable and prescribed-source output | MW, source × period |
| `pl` | Industrial electrical load | MW, park × period |
| `rho` | Nodal carbon intensity in dispatch records | t/MWh, bus × period |
| `q` | Shared industrial carbon allowance adjustment | t, park |
| `allocation_t` | Integrated attributed industrial emissions | t, park |
| `budget` | Initial industrial carbon budget | t, park |
| `dt_h` | Scheduling interval | h |
| `bus`, `gen`, `branch` | Saved AC solution arrays | Period × component × standard PYPOWER columns |
| `alpha` | Convex chart coordinate between two grid endpoints | Dimensionless |
| `original_boundary` | Carbon-budget multiplier from original carbon equations | Dimensionless, chart coordinate |
| `certified_boundary` | Budget multiplier sufficient under the interval certificate | Dimensionless; infinity means no certificate at that coordinate/depth |

`data/ARRAY_INDEX.csv` records actual field shapes and dtypes. Legacy NPZ metadata stored as Python objects is not unpickled by the integrity checker. Numerical arrays are loaded with `allow_pickle=False`.

## Scope of validation

The 80 Floquet checks compare independently integrated propagation with the numerical implementation. They establish numerical consistency for the tested synthetic settings, not field calibration. The 720-point AC audit reconstructs balances and limits from the final corrected solutions; the original uncorrected AC mapping is archived separately. Corrective dispatch preserves industrial and renewable active-power plans and first-stage allowances, and includes the documented commitment of the existing `307_CT_1` unit for the low- and medium-renewable cases.

Generated stress trajectories multiply complete donor-day error vectors and preserve temporal/source correlation. They are distinguished from the held-out observed donor errors. Repository conclusions should use the same scope as the corresponding experiment.
