# Code structure

Brief map of folders and main files. The installable package is **`divtree`** (see `pyproject.toml`); simulations live under `simulations/` and add paths as needed.

## Root

| Item | Purpose |
|------|---------|
| `pyproject.toml` | Package metadata and dependencies for `divtree` |
| `requirements.txt`, `requirements-exact.txt` | Pip installs (exact file for tighter reproducibility) |
| `README.md` | Project overview |
| `METHODS_EXPLANATION.txt` | Method notes |

## `src/` — libraries

| Path | Purpose |
|------|---------|
| `src/divtree/` | **Divergence Tree**: heterogeneous treatment effects on two outcomes (`tree.py`), Optuna tuning (`tune.py`), optional forest (`forest.py`), plotting (`viz.py`). |
| `src/twostepdivtree/` | **Two-step method**: EconML `CausalForestDML` for τ estimates, then a sklearn classifier on region labels (`tree.py`). |

Install from repo root: `pip install -e .`

## `simulations/binary_comparison/` — experiments

| Path | Purpose |
|------|---------|
| `binary_data_generator.py` | Legacy binary DGP. |
| `test_binary_generator.py` | Tests for the legacy generator. |
| `archive/README.md` | Notes on archived / legacy pieces. |
| `region_stratified_dgp/` | **Region-stratified DGP**: dispersion / rareness for region 2, combo and observation allocation (`generate_region_stratified_data.py`). |
| `comprehensive_simulation/` | **Legacy λ vs TwoStep study**: `config.py`, `simulation_base.py`, `lambda_comparison.py` (runner), `utils.py`, `metrics.py`, `analyze_lambda_comparison.py`, `run_meta_tree.py`. |
| `comprehensive_simulation_region_stratified/` | **Region-stratified study**: same roles with new DGP and outputs; `lambda_comparison.py` saves compressed tree artifacts; `analyze_region_stratified.py` and `analyze_region_stratified_dgp.py` for results and DGP diagnostics. |

The region-stratified runner imports `utils` and `metrics` from `comprehensive_simulation/` via `sys.path`.

## Generated output (usually not in git)

Simulation scripts write under each study’s `output/` (per-run data, aggregated tables, plots). Safe to delete on a fresh machine and regenerate.

## Typical entry points

- **Region-stratified simulation:** `simulations/binary_comparison/comprehensive_simulation_region_stratified/lambda_comparison.py`
- **Legacy simulation:** `simulations/binary_comparison/comprehensive_simulation/lambda_comparison.py`
