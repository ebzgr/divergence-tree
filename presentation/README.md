# Divergence Tree — HTML slide presentation

Static slides (**reveal.js**) + interactive charts (**Plotly.js**). Simulation data is loaded from `static/simulations.json` (generated from the aggregated pickle).

## Regenerate data from the project outputs

Run from the **repository root** (the folder that contains `outputs/` and `presentation/`):

```bash
# Default: finds outputs/simulations/.../all_simulations_results.pkl under the repo
python presentation/extract_data.py
```

Or pass the pickle explicitly (relative to repo root):

```bash
python presentation/extract_data.py --pkl outputs/simulations/binary_features/aggregated/lambda_twostep_comparison/all_simulations_results.pkl
```

On Linux, an **absolute** path must start with `/`. This is wrong (looks under `./data/...` in the current directory):

```text
data/ebrahim/Projects/.../all_simulations_results.pkl   # BAD
```

Use instead:

```text
/data/ebrahim/Projects/DivergenceTree/divergence-tree/outputs/simulations/...
```

Or set `export SIMULATIONS_PKL=/full/path/to/all_simulations_results.pkl`.

## Viewing the deck

Browsers often block `fetch()` to local files when opening `index.html` via `file://`. Use any static file server from the `presentation/` directory, for example:

```bash
cd presentation
python -m http.server 8765
```

Then open `http://127.0.0.1:8765/`.

## Contents

- `index.html` — slides and filter UI
- `css/theme.css` — light pastel theme
- `js/charts.js` — filters + Plotly charts (connected to `static/simulations.json`)
- `extract_data.py` — pickle → JSON export

No build step and no backend required.
