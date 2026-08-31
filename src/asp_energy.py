"""Predict ``aspEnergy`` from ``aspAudioFeatures`` only, with Lasso and Ridge.

Spotify ``audioFeatures``, ``experimentalAudioFeatures``, popularity, lyrics,
and other track metadata are ignored. Categorical fields inside
``aspAudioFeatures`` (``key``, ``scale``, language code) are one-hot encoded;
the nested language-probability blob is reduced to the detected code plus its
confidence. Features are standardized before the penalty so L1/L2 are not
dominated by ``energy`` / ``durationMs`` scale. Lasso drops near-zero weights
via ``SelectFromModel`` and refits on the surviving columns.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import BaseCrossValidator, KFold, LeaveOneOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES: tuple[str, ...] = (
    "danceability",
    "acoustic",
    "aggressive",
    "happy",
    "party",
    "relaxed",
    "sad",
    "voice",
    "instrumental",
    "gender",
    "tempo",
    "loudness",
    "intensity",
    "onset_rate",
    "entropy",
    "energy",
    "dynamic_range",
    "larm",
    "leq",
    "durationMs",
    "initialSilence",
    "finalSilence",
    "language_probability",
)

CATEGORICAL_FEATURES: tuple[str, ...] = ("key", "scale", "language")
TARGET = "aspEnergy"
FEATURE_OBJECT = "aspAudioFeatures"

LASSO_ALPHAS = np.logspace(-4, 1, 40)
RIDGE_ALPHAS = np.logspace(-2, 4, 40)
LASSO_COEF_THRESHOLD = 1e-10
RANDOM_STATE = 42


def _inner_cv() -> KFold:
    return KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def extract_feature_row(asp: dict[str, Any]) -> dict[str, Any]:
    """Flatten one ``aspAudioFeatures`` object into a model row (no target)."""
    language = asp.get("language") or {}
    row: dict[str, Any] = {
        name: asp.get(name) for name in NUMERIC_FEATURES if name != "language_probability"
    }
    probability = language.get("probability")
    row["language_probability"] = 0.0 if probability is None else probability
    row["key"] = asp.get("key")
    row["scale"] = asp.get("scale")
    row["language"] = language.get("language") or "unknown"
    return row


def tracks_to_frame(
    tracks: list[dict[str, Any]],
    require_target: bool = True,
) -> tuple[pd.DataFrame, np.ndarray | None, list[str]]:
    rows: list[dict[str, Any]] = []
    targets: list[float] = []
    labels: list[str] = []
    skipped = 0
    for track in tracks:
        asp = track.get(FEATURE_OBJECT)
        if not isinstance(asp, dict):
            skipped += 1
            continue
        if require_target and asp.get(TARGET) is None:
            skipped += 1
            continue
        row = extract_feature_row(asp)
        if any(row[name] is None for name in NUMERIC_FEATURES):
            skipped += 1
            continue
        if any(row[name] is None for name in CATEGORICAL_FEATURES):
            skipped += 1
            continue
        rows.append(row)
        if require_target:
            targets.append(float(asp[TARGET]))
        artist = ""
        artists = track.get("artists") or []
        if artists and isinstance(artists[0], dict):
            artist = str(artists[0].get("artistName") or "")
        labels.append(f"{artist} — {track.get('name', track.get('_id', 'unknown'))}".strip(" —"))
    if skipped:
        print(f"skipped {skipped} tracks without complete {FEATURE_OBJECT}/{TARGET}")
    if not rows:
        raise ValueError(f"No usable {FEATURE_OBJECT} rows found")
    frame = pd.DataFrame(rows)
    y = np.asarray(targets, dtype=float) if require_target else None
    return frame, y, labels


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), list(NUMERIC_FEATURES)),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(CATEGORICAL_FEATURES),
            ),
        ],
        remainder="drop",
    )


def _lasso_cv() -> LassoCV:
    return LassoCV(
        alphas=LASSO_ALPHAS,
        cv=_inner_cv(),
        max_iter=50_000,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def make_lasso() -> Pipeline:
    """Lasso that drops near-zero coefficients, then refits Lasso on the rest."""
    return Pipeline(
        [
            ("prep", make_preprocessor()),
            (
                "select",
                SelectFromModel(_lasso_cv(), threshold=LASSO_COEF_THRESHOLD),
            ),
            ("model", _lasso_cv()),
        ]
    )


def make_ridge() -> Pipeline:
    return Pipeline(
        [
            ("prep", make_preprocessor()),
            ("model", RidgeCV(alphas=RIDGE_ALPHAS, cv=_inner_cv())),
        ]
    )


def make_ols() -> Pipeline:
    return Pipeline(
        [
            ("prep", make_preprocessor()),
            ("model", LinearRegression()),
        ]
    )


def _pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if np.std(y_pred) < 1e-12 or np.std(y_true) < 1e-12:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(rmse),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "pearson_r": _pearson(y_true, y_pred),
    }


def _loo_predict(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: np.ndarray,
    cv: BaseCrossValidator,
) -> np.ndarray:
    return cross_val_predict(estimator, X, y, cv=cv, n_jobs=-1)


def _round(value: float | None, digits: int = 6) -> float | None:
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return None
    return round(float(value), digits)


def _strip_encoder_prefix(name: str) -> str:
    return name.replace("num__", "").replace("cat__", "")


def _prep_names(pipeline: Pipeline) -> list[str]:
    return [_strip_encoder_prefix(n) for n in pipeline.named_steps["prep"].get_feature_names_out()]


def feature_names(pipeline: Pipeline) -> list[str]:
    names = list(pipeline.named_steps["prep"].get_feature_names_out())
    select = pipeline.named_steps.get("select")
    if select is not None:
        names = list(select.get_feature_names_out(names))
    return names


def lasso_selection(pipeline: Pipeline) -> dict[str, list[str]]:
    """Encoded feature names kept vs dropped by the Lasso selector."""
    if "select" not in pipeline.named_steps:
        kept = [_strip_encoder_prefix(n) for n in feature_names(pipeline)]
        return {"kept": kept, "dropped": []}
    all_names = _prep_names(pipeline)
    mask = np.asarray(pipeline.named_steps["select"].get_support(), dtype=bool)
    kept = [name for name, keep in zip(all_names, mask) if keep]
    dropped = [name for name, keep in zip(all_names, mask) if not keep]
    return {"kept": kept, "dropped": dropped}


def selected_numeric_features(pipeline: Pipeline) -> list[str]:
    """Original numeric columns whose Lasso coefficient survived the drop."""
    kept = set(lasso_selection(pipeline)["kept"])
    return [name for name in NUMERIC_FEATURES if name in kept]


def selector_abs_coef(pipeline: Pipeline, feature: str) -> float:
    if "select" not in pipeline.named_steps:
        return 0.0
    names = _prep_names(pipeline)
    coefs = np.abs(np.ravel(pipeline.named_steps["select"].estimator_.coef_))
    for name, coef in zip(names, coefs):
        if name == feature:
            return float(coef)
    return 0.0


def coefficient_table(pipeline: Pipeline) -> list[dict[str, Any]]:
    names = feature_names(pipeline)
    coefs = np.asarray(pipeline.named_steps["model"].coef_, dtype=float)
    intercept = float(pipeline.named_steps["model"].intercept_)
    rows = [
        {
            "feature": _strip_encoder_prefix(name),
            "coefficient": _round(float(coef), 6),
            "abs_coefficient": _round(abs(float(coef)), 6),
            "nonzero": bool(abs(float(coef)) > LASSO_COEF_THRESHOLD),
        }
        for name, coef in zip(names, coefs)
    ]
    rows.sort(key=lambda row: row["abs_coefficient"], reverse=True)
    rows.append(
        {
            "feature": "(intercept)",
            "coefficient": _round(intercept, 6),
            "abs_coefficient": _round(abs(intercept), 6),
            "nonzero": True,
        }
    )
    return rows


def univariate_correlations(X: pd.DataFrame, y: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in NUMERIC_FEATURES:
        series = X[name].to_numpy(dtype=float)
        corr = _pearson(y, series)
        rows.append(
            {
                "feature": name,
                "pearson_r": _round(corr, 4),
                "abs_r": _round(abs(corr) if not math.isnan(corr) else 0.0, 4),
            }
        )
    rows.sort(key=lambda row: row["abs_r"] or 0.0, reverse=True)
    return rows


def train_and_evaluate(
    tracks: list[dict[str, Any]],
    *,
    outer_cv: BaseCrossValidator | None = None,
) -> dict[str, Any]:
    X, y, labels = tracks_to_frame(tracks)
    assert y is not None
    cv = outer_cv if outer_cv is not None else LeaveOneOut()
    lasso = make_lasso()
    ridge = make_ridge()
    ols = make_ols()

    pred_mean = np.full_like(y, fill_value=float(y.mean()))
    pred_ols = _loo_predict(ols, X, y, cv)
    pred_lasso = _loo_predict(lasso, X, y, cv)
    pred_ridge = _loo_predict(ridge, X, y, cv)

    lasso.fit(X, y)
    ridge.fit(X, y)
    ols.fit(X, y)

    lasso_model = lasso.named_steps["model"]
    ridge_model = ridge.named_steps["model"]
    encoded_names = _prep_names(lasso)
    selection = lasso_selection(lasso)
    lasso_coefs = coefficient_table(lasso)
    n_nonzero = sum(1 for row in lasso_coefs if row["nonzero"] and row["feature"] != "(intercept)")
    selector_alpha = float(lasso.named_steps["select"].estimator_.alpha_)

    y_in_sample_lasso = lasso.predict(X)
    y_in_sample_ridge = ridge.predict(X)
    cv_name = type(cv).__name__

    report = {
        "n_tracks": int(len(y)),
        "n_raw_numeric": len(NUMERIC_FEATURES),
        "n_raw_categorical": len(CATEGORICAL_FEATURES),
        "n_encoded_features": len(encoded_names),
        "n_selected_features": len(selection["kept"]),
        "selected_features": selection["kept"],
        "dropped_features": selection["dropped"],
        "target": TARGET,
        "feature_object": FEATURE_OBJECT,
        "target_summary": {
            "min": _round(float(y.min()), 4),
            "max": _round(float(y.max()), 4),
            "mean": _round(float(y.mean()), 4),
            "std": _round(float(y.std(ddof=1)), 4),
        },
        "evaluation": f"{cv_name} (outer); 5-fold CV for alpha (inner)",
        "models": {
            "mean_baseline": {
                "note": "constant training-set mean; R² is 0 by construction",
                "loo": {k: _round(v, 4) for k, v in regression_metrics(y, pred_mean).items()},
            },
            "ols": {
                "loo": {k: _round(v, 4) for k, v in regression_metrics(y, pred_ols).items()},
                "in_sample": {k: _round(v, 4) for k, v in regression_metrics(y, ols.predict(X)).items()},
            },
            "lasso": {
                "alpha": _round(float(lasso_model.alpha_), 6),
                "selector_alpha": _round(selector_alpha, 6),
                "n_selected": len(selection["kept"]),
                "n_nonzero": int(n_nonzero),
                "loo": {k: _round(v, 4) for k, v in regression_metrics(y, pred_lasso).items()},
                "in_sample": {k: _round(v, 4) for k, v in regression_metrics(y, y_in_sample_lasso).items()},
            },
            "ridge": {
                "alpha": _round(float(ridge_model.alpha_), 6),
                "loo": {k: _round(v, 4) for k, v in regression_metrics(y, pred_ridge).items()},
                "in_sample": {k: _round(v, 4) for k, v in regression_metrics(y, y_in_sample_ridge).items()},
            },
        },
        "univariate_correlations": univariate_correlations(X, y),
        "lasso_coefficients": lasso_coefs,
        "ridge_coefficients": coefficient_table(ridge),
        "predictions": [
            {
                "track": label,
                "actual": _round(float(actual), 4),
                "lasso": _round(float(p_lasso), 4),
                "ridge": _round(float(p_ridge), 4),
                "ols": _round(float(p_ols), 4),
                "lasso_error": _round(float(p_lasso - actual), 4),
                "ridge_error": _round(float(p_ridge - actual), 4),
            }
            for label, actual, p_lasso, p_ridge, p_ols in zip(
                labels, y, pred_lasso, pred_ridge, pred_ols
            )
        ],
    }
    return {
        "report": report,
        "lasso": lasso,
        "ridge": ridge,
        "X": X,
        "y": y,
        "labels": labels,
    }


def slim_tracks(tracks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    slim: list[dict[str, Any]] = []
    for track in tracks:
        asp = track.get(FEATURE_OBJECT)
        if not isinstance(asp, dict):
            continue
        slim.append(
            {
                "_id": track.get("_id"),
                "name": track.get("name"),
                "artists": track.get("artists"),
                FEATURE_OBJECT: asp,
            }
        )
    return slim


def save_artifacts(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(result["lasso"], output_dir / "lasso_pipeline.joblib")
    joblib.dump(result["ridge"], output_dir / "ridge_pipeline.joblib")
    (output_dir / "metrics.json").write_text(json.dumps(result["report"], indent=2) + "\n")


def load_tracks(path: Path) -> list[dict[str, Any]]:
    raw = path.read_text()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        tracks: list[dict[str, Any]] = []
        for line_no, line in enumerate(raw.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"JSONL line {line_no} in {path} is not an object")
            tracks.append(obj)
        if not tracks:
            raise ValueError(f"No tracks found in {path}")
        return tracks
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return [payload]
    raise ValueError(f"Expected a JSON array, object, or JSONL of tracks in {path}")


def predict_tracks(tracks: list[dict[str, Any]], pipeline: Pipeline) -> list[dict[str, Any]]:
    has_target = any(
        isinstance(track.get(FEATURE_OBJECT), dict) and track[FEATURE_OBJECT].get(TARGET) is not None
        for track in tracks
    )
    X, y, labels = tracks_to_frame(tracks, require_target=has_target)
    preds = pipeline.predict(X)
    out = []
    for i, (label, pred) in enumerate(zip(labels, preds)):
        row: dict[str, Any] = {
            "track": label,
            "predicted": _round(float(pred), 4),
        }
        if y is not None:
            actual = float(y[i])
            row["actual"] = _round(actual, 4)
            row["error"] = _round(float(pred - actual), 4)
        out.append(row)
    return out


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train", help="Fit Lasso and Ridge and write artifacts.")
    train_p.add_argument("--input", required=True, type=Path, help="ASP tracks JSON array or JSONL.")
    train_p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models"),
        help="Directory for joblib pipelines and metrics.json.",
    )
    train_p.add_argument(
        "--slim-out",
        type=Path,
        default=None,
        help="Optional lyrics-free features-only copy of the tracks.",
    )
    train_p.add_argument(
        "--kfold",
        type=int,
        default=0,
        help="Use K-fold instead of leave-one-out for the outer evaluation (0 = LOO).",
    )

    pred_p = sub.add_parser("predict", help="Score tracks with a saved pipeline.")
    pred_p.add_argument("--input", required=True, type=Path)
    pred_p.add_argument("--model", choices=("lasso", "ridge"), default="lasso")
    pred_p.add_argument("--model-dir", type=Path, default=Path("models"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "train":
        tracks = load_tracks(args.input)
        outer = (
            KFold(n_splits=args.kfold, shuffle=True, random_state=RANDOM_STATE)
            if args.kfold and args.kfold > 1
            else None
        )
        result = train_and_evaluate(tracks, outer_cv=outer)
        save_artifacts(result, args.output_dir)
        if args.slim_out is not None:
            slim_path: Path = args.slim_out
            slim_path.parent.mkdir(parents=True, exist_ok=True)
            slim_path.write_text(json.dumps(slim_tracks(tracks), indent=2) + "\n")
        models = result["report"]["models"]
        print(
            json.dumps(
                {
                    "n_tracks": result["report"]["n_tracks"],
                    "n_encoded_features": result["report"]["n_encoded_features"],
                    "n_selected_features": result["report"]["n_selected_features"],
                    "dropped_features": result["report"]["dropped_features"],
                    "lasso": models["lasso"],
                    "ridge": models["ridge"],
                    "ols": models["ols"],
                    "mean_baseline": models["mean_baseline"],
                    "artifacts": str(args.output_dir),
                },
                indent=2,
            )
        )
        return 0

    model_file = args.model_dir / f"{args.model}_pipeline.joblib"
    pipeline = joblib.load(model_file)
    print(json.dumps(predict_tracks(load_tracks(args.input), pipeline), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
