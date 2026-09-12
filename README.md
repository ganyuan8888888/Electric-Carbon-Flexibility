# Electric–Carbon Flexibility

Research code, benchmark inputs, and archived numerical evidence for **Coupled electric–carbon flexibility region characterization and two-stage robust scheduling of industrial loads based on embedded process-recovery trajectories and carbon-flow perturbation envelopes**.

**Authors:** Yuan Gan, Zhen Cao, Ruby Feng, and Wei Li.  
**Contact:** ganyuan@hnu.edu.cn  
**Version:** 1.0.1.

**Repository:** https://github.com/ganyuan8888888/Electric-Carbon-Flexibility

The framework embeds complete electrothermal recovery trajectories into industrial demand response, certifies continuous electric–carbon sections using carbon-flow perturbation bounds, and coordinates shared carbon allowances with scenario-dependent recourse. Nested column-and-constraint generation (NCCG) reuses recovery modes while maintaining valid bounds for the finite certified model.

## Quick start

Use Python 3.12. Create a virtual environment from the repository root:

```bash
python -m venv .venv
```

Activate it with `.venv\Scripts\activate` on Windows, or `source .venv/bin/activate` on Linux/macOS, then run:

```bash
python -m pip install -r requirements.txt
python reproduce.py quick
python reproduce.py ac
python reproduce.py figures
python reproduce.py solve --regions 1 --scenarios 5 --method all
```

New files are written to `reproduced/`, which is excluded from version control. Archived observations in `data/reference/` are preserved. Use the wrappers above; some versioned legacy drivers were designed to create or cache experiment files in their input tree.

The quick route checks archived file hashes, recomputes the three-day operating metrics from dispatch arrays, repeats 80 independent Floquet propagation checks and nonlinear power reconstruction, and recomputes one nonlinear carbon section with interval bounds. The AC route reconstructs voltage-dependent power flows, nodal balances, limits, and daily carbon allocations for all 720 archived AC operating points. It verifies saved nonlinear solutions; it does not rerun AC optimization. The solve route performs fresh optimization and checks objective agreement when all methods terminate optimally.

## Reproduction tasks

| Command | Calculation | Main output |
|---|---|---|
| `python reproduce.py verify` | SHA-256 and numerical archive integrity | `reproduced/verification.json` |
| `python reproduce.py results` | Cost, emissions, curtailment and dense-grid classification counts | `reproduced/tables/` |
| `python reproduce.py process` | Independent propagation and nonlinear power reconstruction | `reproduced/process_validation.json` |
| `python reproduce.py region --depth 5` | Original carbon equations and interval envelopes on the first chart | `reproduced/region_validation.json` |
| `python reproduce.py ac` | Independent 720-point AC and carbon reconstruction | `reproduced/ac_validation.json` |
| `python reproduce.py solve --method all` | Fresh extensive-form MILP, NCCG without reuse, and proposed NCCG | `reproduced/solve_comparison.json` |
| `python reproduce.py figures` | Three data-derived figures using the archived plotting functions | `reproduced/figures/` |

Select a day with `--date 2020-04-07`, `2020-10-17`, or `2020-11-29`. A denser interval run uses `--depth 9`. To study the manuscript's four-region scaling instances:

```bash
python reproduce.py solve --date 2020-04-07 --regions 4 --scenarios 20 --method all --time-limit 600
```

Repeat for scenario counts 5, 10, 20, and 40 and the three dates. Run timing experiments sequentially on an otherwise idle machine. Archive the resulting logs with hardware and software information; fresh elapsed times are not expected to equal historical measurements. See [the benchmark protocol](docs/Benchmark_Protocol.md).

## Repository structure

```text
src/ecf_legacy/       Versioned scientific models, algorithms, and plotting functions
cases/               Study configurations and parameter-source guide
data/reference/      Public inputs, saved trajectories, dispatches, and experiment records
data/REFERENCE_MANIFEST.csv
                     Hash, size, and relative path of each archived file
data/ARRAY_INDEX.csv  NPZ field names, shapes, and dtypes
results/published/   Reconstructed numerical tables and package validation records
figures/reference/  Selected data-derived reference figures
docs/                Reproduction, figure mapping, provenance, and release instructions
tools/               Repository checks
reproduce.py         Supported command-line entry point
CITATION.cff         Machine-readable software citation
```

Version suffixes are retained in scientific modules and archived paths to preserve the relationship between a published result and its actual implementation. `docs/source_adaptations.json` records the original and packaged source hashes. Packaging changes relocate data roots and locate installed runtime libraries; they do not change model equations, solver settings, or numerical observations.

## Data and evidence

The network and source–load chronology are based on the public **RTS-GMLC v0.2.3** benchmark, commit `3ece0d3725c844056132393ee252b3083dd4eab4`. The study adds six modeled industrial parks and explicitly reconnects selected photovoltaic sources without duplicating renewable energy. Industrial trajectories are simulated; this repository does not present confidential plant measurements.

The nominal high-renewable-day operating totals reconstructed from the archived arrays yield reductions of **3.07% in operating cost, 1.92% in generation-side emissions, and 5.83% in renewable curtailment**, relative to the multi-timescale comparison. `results/published/nominal_operation.csv` contains all three days and all four methods, and `nominal_reductions.csv` gives the unrounded reductions.

Dense-grid section coverage refers to the specified complete-trajectory charts and budget interval. It is not the volume of the unrestricted nonlinear feasible region. Holdout and generated stress trajectories have separate provenance; the generated trajectories are not independent measured operating days. AC verification includes the documented corrective dispatch. See [data provenance](docs/Data_Provenance.md) for the exact scopes and selected record sets.

## Citation and reuse

Use `CITATION.cff` to cite this software version. The repository URL identifies this code and data collection. An archival DOI can be added after one is issued; no DOI or publication status is assumed.

The RTS-GMLC dataset retains its original attribution and redistribution notice in `data/reference/data/RTS-GMLC-v0.2.3/README.md`. See [third-party notices](THIRD_PARTY_NOTICES.md). A license for the authors' original code has not yet been selected; see `LICENSE_STATUS.md` before redistributing or relicensing it.
