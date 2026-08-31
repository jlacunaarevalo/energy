# aspEnergy from aspAudioFeatures

Lasso and Ridge models that predict **aspEnergy** using only fields inside `aspAudioFeatures`. Spotify `audioFeatures`, experimental tags, popularity, and lyrics are ignored.

## Results on 100 ASP tracks (leave-one-out)

| Model | α | LOO R² | LOO RMSE | LOO MAE |
|---|---|---|---|---|
| Mean baseline | — | 0.000 | 0.147 | 0.116 |
| OLS | 0 | 0.942 | 0.035 | 0.027 |
| Ridge | 1.00 | 0.945 | 0.035 | 0.027 |
| **Lasso** | **5.9e-4** | **0.949** | **0.033** | **0.025** |

After z-scoring, the largest weights are **loudness (+)** and **relaxed (−)**. Raw `energy` is not a leak (Pearson r = 0.42 with aspEnergy). Full coefficients: `models/loo_report.json`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

## Train / predict

Point `--input` at an ASP tracks JSON array (each track has `aspAudioFeatures` with `aspEnergy`):

```bash
python -m asp_energy train --input path/to/aspTracks.json --output-dir models
python -m asp_energy predict --model lasso --input path/to/aspTracks.json --model-dir models
```

JSON arrays and JSONL are both accepted. Lasso drops near-zero coefficients (`SelectFromModel`) and refits on the rest.

## Closed-form formula (PySR)

Lasso-selected numeric features are passed to [PySR](https://github.com/MilesCranmer/PySR). Needs Julia (installed automatically by `pysr` on first run):

```bash
pip install -e '.[formula]'
python -m asp_formula fit --input /path/to/results_j.jsonl --output-dir models
python -m asp_formula serve --formula models/asp_energy_formula.json
```

`fit` writes `models/asp_energy_formula.json`. The Gradio UI shows the formula, accepts JSON/JSONL or manual coefficients, and reports deviation when `aspEnergy` is present.

That writes `models/lasso_pipeline.joblib` and `models/ridge_pipeline.joblib`.

```bash
pytest
```
