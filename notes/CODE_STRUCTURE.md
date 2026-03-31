# Code Structure

Brief map of the current research-project layout.

## Root

| Item | Purpose |
|------|---------|
| `README.md` | Main project documentation |
| `pyproject.toml` | Package metadata for `divtree` (package source in `code/src`) |
| `.gitignore` | Ignore rules for Python + research artifacts |
| `config/` | Shared config (`requirements.txt`, `repo_paths.py`) |

## `code/`

| Path | Purpose |
|------|---------|
| `code/src/divtree/` | Core Divergence Tree implementation (`tree.py`, `tune.py`, `forest.py`, `viz.py`) |
| `code/src/twostepdivtree/` | Two-step method implementation |
| `code/simulations/binary_comparison/` | Legacy and region-stratified simulation code |
| `code/export/` | Reserved for export utilities |

## `data/`

| Path | Purpose |
|------|---------|
| `data/binary_comparison/` | Legacy binary pickled datasets/artifacts |

## `outputs/`

| Path | Purpose |
|------|---------|
| `outputs/simulations/comprehensive_simulation/` | Legacy simulation run outputs |
| `outputs/simulations/comprehensive_simulation_region_stratified/` | Region-stratified run outputs |
| `outputs/figures/` | General figures outside per-simulation folders |
| `outputs/tables/` | General tables |
| `outputs/logs/` | Run logs |

## Writing and notes

| Path | Purpose |
|------|---------|
| `paper/sections`, `paper/figures`, `paper/tables`, `paper/style`, `paper/build` | Paper workspace |
| `refs/` | References/bibliography files |
| `notes/` | Project notes and migrated legacy markdown/text files |

## Typical entry points

- Region-stratified simulation: `code/simulations/binary_comparison/comprehensive_simulation_region_stratified/lambda_comparison.py`
- Legacy simulation: `code/simulations/binary_comparison/comprehensive_simulation/lambda_comparison.py`
