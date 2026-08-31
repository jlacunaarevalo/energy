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

That writes `models/lasso_pipeline.joblib` and `models/ridge_pipeline.joblib`.

```bash
pytest
```
