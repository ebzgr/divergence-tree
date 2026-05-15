# Code Structure

Brief map of the current research-project layout.

## Root

| Item | Purpose |
|------|---------|
| `README.md` | Main project documentation |
| `pyproject.toml` | Package metadata for `divtree` (source in `code/src`) |
| `.gitignore` | Ignore rules for Python + research artifacts |
| `config/` | Shared config (`requirements.txt`, `repo_paths.py`) |

## `code/`

| Path | Purpose |
|------|---------|
| `code/src/divtree/` | Core Divergence Tree (`tree.py`, `tune.py`, `forest.py`, `viz.py`) |
| `code/src/twostepdivtree/` | Two-step method (`tree.py`, `tune.py`) |
| `code/simulations/binary_features/` | Region-stratified binary-features simulation + DGP |
| `code/simulations/binary_features/dgp/` | Data generating process |

## `outputs/`

| Path | Purpose |
|------|---------|
| `outputs/simulations/binary_features/` | Simulation run outputs (gitignored) |

## Typical entry points

- Simulation: `code/simulations/binary_features/lambda_comparison.py`
- Analysis: `code/simulations/binary_features/analyze_region_stratified.py`
- DGP diagnostics: `code/simulations/binary_features/analyze_region_stratified_dgp.py`
