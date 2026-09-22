# aspEnergy from aspAudioFeatures

There is one energy score: **aspEnergy**. This repo keeps two ways to copy it from the other fields inside `aspAudioFeatures` (Spotify `audioFeatures`, experimental tags, popularity, and lyrics are ignored):

- **Lasso** — the sklearn model
- **Closed-form formula** — the single PySR equation distilled from Lasso, for reading and serving

## Test set

There is no committed test-track file in this repo. How numbers were (and now are) measured:

| Artifact | What was scored | Honest? |
|---|---|---|
| `models/loo_report.json` | Leave-one-out on **100** tracks (`aspTracks_selection_100.json`): each track is predicted by a Lasso trained on the other 99. Inner 5-fold CV picks α on those 99. | Cross-validation, **not** a separate test set |
| `models/asp_energy_formula.json` | The saved equation on the **62** tracks used to search it | **In-sample** |
| `python -m asp_energy train` / `asp_formula fit` now | Default **20%** of `--input` held out (seed 42). Or pass `--test-input other.json`. α / PySR never see the test tracks | Held-out test |

A mean baseline (always predict the **training** mean) is reported next to Lasso so the test R² is comparable. It is not a model you can serve.

## Results on 100 ASP tracks (leave-one-out)

| Model | α | LOO R² | LOO RMSE | LOO MAE |
|---|---|---|---|---|
| Mean baseline | — | 0.000 | 0.147 | 0.116 |
| **Lasso** | **5.9e-4** | **0.949** | **0.033** | **0.025** |

After z-scoring, the largest weights are **loudness (+)** and **relaxed (−)**. Raw `energy` is not a leak (Pearson r = 0.42 with aspEnergy). Full coefficients: `models/loo_report.json`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

## Train / predict (Lasso)

Point `--input` at an ASP tracks JSON array (each track has `aspAudioFeatures` with `aspEnergy`):

```bash
python -m asp_energy train --input path/to/aspTracks.json --output-dir models
python -m asp_energy predict --input path/to/aspTracks.json --model-dir models
```

JSON arrays and JSONL are both accepted. Lasso drops near-zero coefficients (`SelectFromModel`) and refits on the rest. That writes `models/lasso_pipeline.joblib`.

Hold out a different fraction, or a dedicated test file:

```bash
python -m asp_energy train --input path/to/train.json --test-frac 0.2
python -m asp_energy train --input path/to/train.json --test-input path/to/test.json
```

## Closed-form formula (PySR)

Lasso-selected numeric features are passed to [PySR](https://github.com/MilesCranmer/PySR). Needs Julia (installed automatically by `pysr` on first run):

```bash
pip install -e '.[formula]'
python -m asp_formula fit --input /path/to/tracks.json --output-dir models
python -m asp_formula serve --formula models/asp_energy_formula.json
```

`fit` writes `models/asp_energy_formula.json` (one equation). The Gradio UI shows that formula, accepts JSON/JSONL or manual coefficients, and reports deviation when `aspEnergy` is present. The same `--test-frac` / `--test-input` flags apply.

```bash
pytest
```
