# Experiment map

Paths are relative to `data/reference/`; code modules are in `src/ecf_legacy/`.

| Experiment | Source and repeatable entry |
|---|---|
| Nominal operation | `python reproduce.py results`; main Figures 12–15 |
| Process propagation and power reconstruction | `python reproduce.py process`; supplemental S5 and main Figure 6 |
| Certified-section accuracy | `python reproduce.py region`; S10; full source `dense_region_v37.py` |
| Holdout and stress cases | `evidence_recourse_v37.py`; S11; `revision_v37/experiments/oos_final/` |
| AC original-constraint verification | `python reproduce.py ac`; S12; `audit_ac_v39.py` |
| Regional scalability | `python reproduce.py solve --regions 4 --scenarios 20 --method all`; S13 |
| Finite-model convergence | `nccg_solver_v32.py`; S8 and main Figure 16 |

The quick-start commands are tested entry points. Advanced legacy generation routines preserve experiment-specific caches and protocols; use a separate copy of the archived data tree via `ECF_DATA_ROOT` before running them. Rebuilding the entire archive is a distinct long-running workflow, not implied by rebuilding figures.
