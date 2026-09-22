"""Predict ``aspEnergy`` from ``aspAudioFeatures`` with Lasso.

Spotify ``audioFeatures``, ``experimentalAudioFeatures``, popularity, lyrics,
and other track metadata are ignored. Categorical fields inside
``aspAudioFeatures`` (``key``, ``scale``, language code) are one-hot encoded;
the nested language-probability blob is reduced to the detected code plus its
confidence. Features are standardized before the penalty so L1 is not
dominated by ``energy`` / ``durationMs`` scale. Lasso drops near-zero weights
via ``SelectFromModel`` and refits on the surviving columns.

Headline metrics are a held-out test split (default 20%). Lasso's penalty is
chosen with 5-fold CV on the training tracks only.
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
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import ElasticNetCV, LassoCV, LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

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
ELASTIC_L1_RATIOS = (0.2, 0.5, 0.8, 0.95)
LASSO_COEF_THRESHOLD = 1e-10
RANDOM_STATE = 42
DEFAULT_TEST_FRAC = 0.2
COMPACT_FEATURES: tuple[str, ...] = (
    "relaxed",
    "loudness",
    "intensity",
    "onset_rate",
    "dynamic_range",
    "happy",
    "acoustic",
)
PYSR_FROZEN = (
    "0.687504 - 0.436873*relaxed + 0.044354*onset_rate "
    "+ 0.063815*intensity - 0.019298*dynamic_range"
)


def _inner_cv() -> KFold:
    return KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def extract_feature_row(asp: dict[str, Any]) -> dict[str, Any]:
    """Flatten one ``aspAudioFeatures`` object into a model row (no target)."""
    language = asp.get("language") or {}
    if isinstance(language, str):
        language = {"language": language, "probability": 0.0}
    if not isinstance(language, dict):
        language = {}
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


def make_elastic_net() -> Pipeline:
    return Pipeline(
        [
            ("prep", make_preprocessor()),
            (
                "model",
                ElasticNetCV(
                    alphas=LASSO_ALPHAS,
                    l1_ratio=list(ELASTIC_L1_RATIOS),
                    cv=_inner_cv(),
                    max_iter=50_000,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def make_ols() -> Pipeline:
    return Pipeline(
        [
            ("prep", make_preprocessor()),
            ("model", LinearRegression()),
        ]
    )


def make_compact_ols() -> Pipeline:
    """Short linear formula on the features that keep a readable equation."""
    return Pipeline(
        [
            ("prep", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )


def make_compact_splines() -> Pipeline:
    """Additive spline formula on the compact feature set (still closed form)."""
    return Pipeline(
        [
            (
                "prep",
                SplineTransformer(n_knots=6, degree=3, include_bias=False),
            ),
            ("model", RidgeCV(alphas=RIDGE_ALPHAS, cv=_inner_cv())),
        ]
    )


def split_labeled(
    X: pd.DataFrame,
    y: np.ndarray,
    labels: list[str],
    *,
    test_frac: float = DEFAULT_TEST_FRAC,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, np.ndarray, list[str], pd.DataFrame, np.ndarray, list[str]]:
    """Hold out ``test_frac`` of labeled rows. Training and test never overlap."""
    if test_frac <= 0:
        y_arr = np.asarray(y, dtype=float)
        return X, y_arr, list(labels), X, y_arr, list(labels)
    if test_frac >= 1:
        raise ValueError("test_frac must be in [0, 1)")
    n = len(y)
    min_n = 8
    if n < min_n:
        raise ValueError(f"Need at least {min_n} labeled tracks to hold out a test set, got {n}")
    X_train, X_test, y_train, y_test, lab_train, lab_test = train_test_split(
        X,
        np.asarray(y, dtype=float),
        np.asarray(labels, dtype=object),
        test_size=test_frac,
        random_state=random_state,
    )
    return (
        X_train.reset_index(drop=True),
        np.asarray(y_train, dtype=float),
        lab_train.tolist(),
        X_test.reset_index(drop=True),
        np.asarray(y_test, dtype=float),
        lab_test.tolist(),
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
        "spearman_r": _spearman(y_true, y_pred),
    }


def _spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if np.std(y_pred) < 1e-12 or np.std(y_true) < 1e-12:
        return float("nan")
    true_rank = pd.Series(y_true).rank().to_numpy(dtype=float)
    pred_rank = pd.Series(y_pred).rank().to_numpy(dtype=float)
    return _pearson(true_rank, pred_rank)


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
    test_frac: float = DEFAULT_TEST_FRAC,
    test_tracks: list[dict[str, Any]] | None = None,
    random_state: int = RANDOM_STATE,
) -> dict[str, Any]:
    X, y, labels = tracks_to_frame(tracks)
    assert y is not None
    if test_tracks is not None:
        X_train, y_train, labels_train = X, y, labels
        X_test, y_test, labels_test = tracks_to_frame(test_tracks)
        assert y_test is not None
        evaluation = (
            f"held-out test file ({len(y_test)} tracks); "
            "5-fold CV for Lasso alpha on train only"
        )
    else:
        X_train, y_train, labels_train, X_test, y_test, labels_test = split_labeled(
            X, y, labels, test_frac=test_frac, random_state=random_state
        )
        if test_frac <= 0:
            evaluation = "in-sample (no held-out test set); 5-fold CV for Lasso alpha"
        else:
            evaluation = (
                f"held-out test ({test_frac:.0%} of {len(y)} labeled tracks, "
                f"seed {random_state}); 5-fold CV for Lasso alpha on train only"
            )

    lasso = make_lasso()
    lasso.fit(X_train, y_train)

    pred_test = lasso.predict(X_test)
    pred_train = lasso.predict(X_train)
    mean_train = float(y_train.mean())
    pred_mean = np.full_like(y_test, fill_value=mean_train)

    lasso_model = lasso.named_steps["model"]
    encoded_names = _prep_names(lasso)
    selection = lasso_selection(lasso)
    lasso_coefs = coefficient_table(lasso)
    n_nonzero = sum(1 for row in lasso_coefs if row["nonzero"] and row["feature"] != "(intercept)")
    selector_alpha = float(lasso.named_steps["select"].estimator_.alpha_)

    report = {
        "n_tracks": int(len(y_train) if test_tracks is None and test_frac <= 0 else len(y_train) + len(y_test)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "test_frac": None if test_tracks is not None or test_frac <= 0 else test_frac,
        "n_raw_numeric": len(NUMERIC_FEATURES),
        "n_raw_categorical": len(CATEGORICAL_FEATURES),
        "n_encoded_features": len(encoded_names),
        "n_selected_features": len(selection["kept"]),
        "selected_features": selection["kept"],
        "dropped_features": selection["dropped"],
        "target": TARGET,
        "feature_object": FEATURE_OBJECT,
        "target_summary": {
            "min": _round(float(y_train.min()), 4),
            "max": _round(float(y_train.max()), 4),
            "mean": _round(mean_train, 4),
            "std": _round(float(y_train.std(ddof=1)), 4),
        },
        "evaluation": evaluation,
        "models": {
            "mean_baseline": {
                "note": "predict the training-set mean on the test tracks",
                "test": {k: _round(v, 4) for k, v in regression_metrics(y_test, pred_mean).items()},
            },
            "lasso": {
                "alpha": _round(float(lasso_model.alpha_), 6),
                "selector_alpha": _round(selector_alpha, 6),
                "n_selected": len(selection["kept"]),
                "n_nonzero": int(n_nonzero),
                "test": {k: _round(v, 4) for k, v in regression_metrics(y_test, pred_test).items()},
                "train": {k: _round(v, 4) for k, v in regression_metrics(y_train, pred_train).items()},
            },
        },
        "univariate_correlations": univariate_correlations(X_train, y_train),
        "lasso_coefficients": lasso_coefs,
        "predictions": [
            {
                "track": label,
                "split": "test",
                "actual": _round(float(actual), 4),
                "lasso": _round(float(hat), 4),
                "lasso_error": _round(float(hat - actual), 4),
            }
            for label, actual, hat in zip(labels_test, y_test, pred_test)
        ],
    }
    return {
        "report": report,
        "lasso": lasso,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "labels_train": labels_train,
        "labels_test": labels_test,
    }


def _rounded_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | None]:
    return {k: _round(v, 4) for k, v in regression_metrics(y_true, y_pred).items()}


def predict_pysr_frozen(X: pd.DataFrame) -> np.ndarray:
    return (
        0.687504
        - 0.436873 * X["relaxed"].to_numpy(dtype=float)
        + 0.044354 * X["onset_rate"].to_numpy(dtype=float)
        + 0.063815 * X["intensity"].to_numpy(dtype=float)
        - 0.019298 * X["dynamic_range"].to_numpy(dtype=float)
    )


def compact_ols_formula(pipeline: Pipeline) -> str:
    """Write compact OLS in raw feature units (no z-scoring at inference)."""
    scaler: StandardScaler = pipeline.named_steps["prep"]
    model: LinearRegression = pipeline.named_steps["model"]
    means = scaler.mean_
    scales = scaler.scale_
    weights = np.asarray(model.coef_, dtype=float)
    intercept = float(model.intercept_)
    raw_intercept = intercept
    terms: list[str] = []
    for name, mean, scale, weight in zip(COMPACT_FEATURES, means, scales, weights):
        if abs(scale) < 1e-12:
            continue
        coef = weight / scale
        raw_intercept -= weight * mean / scale
        terms.append(f"{coef:+.6f}*{name}")
    body = " ".join(terms)
    return f"{raw_intercept:.6f} {body}".replace(" + -", " - ")


def compare_formula_models(
    tracks: list[dict[str, Any]],
    *,
    test_frac: float = DEFAULT_TEST_FRAC,
    random_state: int = RANDOM_STATE,
) -> dict[str, Any]:
    """Train closed-form models on an ASP hold-out split (not the 10 outside FLACs)."""
    X, y, labels = tracks_to_frame(tracks)
    assert y is not None
    X_train, y_train, _labels_train, X_test, y_test, _labels_test = split_labeled(
        X, y, labels, test_frac=test_frac, random_state=random_state
    )
    mean_train = float(y_train.mean())
    fitted: dict[str, Pipeline] = {}
    model_reports: dict[str, Any] = {
        "mean_baseline": {
            "formula": f"{mean_train:.6f}",
            "note": "constant training-set mean",
            "test": _rounded_metrics(y_test, np.full_like(y_test, mean_train)),
        }
    }

    compact_ols = make_compact_ols()
    compact_ols.fit(X_train[list(COMPACT_FEATURES)], y_train)
    fitted["compact_ols"] = compact_ols
    compact_pred = compact_ols.predict(X_test[list(COMPACT_FEATURES)])
    model_reports["compact_ols"] = {
        "formula": compact_ols_formula(compact_ols),
        "features": list(COMPACT_FEATURES),
        "note": "OLS on 7 numeric features including loudness; short linear equation",
        "test": _rounded_metrics(y_test, compact_pred),
        "coefficients": coefficient_table(compact_ols),
    }

    compact_splines = make_compact_splines()
    compact_splines.fit(X_train[list(COMPACT_FEATURES)], y_train)
    fitted["compact_splines"] = compact_splines
    spline_pred = compact_splines.predict(X_test[list(COMPACT_FEATURES)])
    model_reports["compact_splines"] = {
        "formula": "sum of cubic B-splines on compact features (6 knots) + Ridge",
        "features": list(COMPACT_FEATURES),
        "note": "still a closed-form additive function, not a tree or net",
        "test": _rounded_metrics(y_test, spline_pred),
    }

    ols = make_ols()
    ols.fit(X_train, y_train)
    fitted["ols"] = ols
    model_reports["ols"] = {
        "formula": "linear combination of all encoded aspAudioFeatures",
        "note": "ordinary least squares; dense linear formula",
        "test": _rounded_metrics(y_test, ols.predict(X_test)),
        "coefficients": coefficient_table(ols)[:16],
    }

    ridge = make_ridge()
    ridge.fit(X_train, y_train)
    fitted["ridge"] = ridge
    ridge_model = ridge.named_steps["model"]
    model_reports["ridge"] = {
        "formula": "linear combination of all encoded aspAudioFeatures (L2)",
        "alpha": _round(float(ridge_model.alpha_), 6),
        "test": _rounded_metrics(y_test, ridge.predict(X_test)),
        "coefficients": coefficient_table(ridge)[:16],
    }

    elastic = make_elastic_net()
    elastic.fit(X_train, y_train)
    fitted["elastic_net"] = elastic
    elastic_model = elastic.named_steps["model"]
    elastic_coefs = coefficient_table(elastic)
    n_nonzero = sum(1 for row in elastic_coefs if row["nonzero"] and row["feature"] != "(intercept)")
    model_reports["elastic_net"] = {
        "formula": "sparse linear combination (L1+L2) of encoded aspAudioFeatures",
        "alpha": _round(float(elastic_model.alpha_), 6),
        "l1_ratio": _round(float(elastic_model.l1_ratio_), 4),
        "n_nonzero": int(n_nonzero),
        "test": _rounded_metrics(y_test, elastic.predict(X_test)),
        "coefficients": elastic_coefs[:16],
    }

    lasso = make_lasso()
    lasso.fit(X_train, y_train)
    fitted["lasso"] = lasso
    lasso_model = lasso.named_steps["model"]
    lasso_coefs = coefficient_table(lasso)
    lasso_nonzero = sum(1 for row in lasso_coefs if row["nonzero"] and row["feature"] != "(intercept)")
    model_reports["lasso"] = {
        "formula": "sparse linear combination (L1) after dropping near-zero weights",
        "alpha": _round(float(lasso_model.alpha_), 6),
        "n_nonzero": int(lasso_nonzero),
        "test": _rounded_metrics(y_test, lasso.predict(X_test)),
        "coefficients": lasso_coefs[:16],
    }

    X_train_sel = lasso.named_steps["select"].transform(lasso.named_steps["prep"].transform(X_train))
    X_test_sel = lasso.named_steps["select"].transform(lasso.named_steps["prep"].transform(X_test))
    lasso_ols = LinearRegression()
    lasso_ols.fit(X_train_sel, y_train)
    model_reports["lasso_then_ols"] = {
        "formula": "OLS refit on the features Lasso kept (still linear)",
        "n_selected": int(X_train_sel.shape[1]),
        "test": _rounded_metrics(y_test, lasso_ols.predict(X_test_sel)),
    }

    model_reports["pysr_frozen"] = {
        "formula": PYSR_FROZEN,
        "note": "existing 4-term PySR equation; not refit on this split",
        "test": _rounded_metrics(y_test, predict_pysr_frozen(X_test)),
    }

    ranking = sorted(
        (
            {
                "model": name,
                "mae": payload["test"]["mae"],
                "r2": payload["test"]["r2"],
                "spearman_r": payload["test"]["spearman_r"],
            }
            for name, payload in model_reports.items()
        ),
        key=lambda row: (row["mae"] if row["mae"] is not None else 9e9, -(row["r2"] or 0)),
    )
    return {
        "evaluation": (
            f"held-out test ({test_frac:.0%} of {len(y)} labeled ASP tracks, "
            f"seed {random_state}); penalties tuned with 5-fold CV on train only. "
            "The 10 outside FLACs are not used."
        ),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "target": TARGET,
        "ranking_by_test_mae": ranking,
        "models": model_reports,
        "pipelines": fitted,
    }


def slim_tracks(tracks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only ids + aspAudioFeatures (rounded) so the file can be committed without LFS."""
    slim: list[dict[str, Any]] = []
    for track in tracks:
        asp = track.get(FEATURE_OBJECT)
        if not isinstance(asp, dict):
            continue
        compact_asp: dict[str, Any] = {}
        for key, value in asp.items():
            if isinstance(value, float):
                compact_asp[key] = round(value, 6)
            elif key == "language" and isinstance(value, dict):
                lang = {
                    "language": value.get("language") or value.get("code"),
                    "probability": value.get("probability") or value.get("language_probability"),
                }
                if isinstance(lang["probability"], float):
                    lang["probability"] = round(lang["probability"], 6)
                compact_asp[key] = lang
            else:
                compact_asp[key] = value
        slim.append({"_id": track.get("_id"), FEATURE_OBJECT: compact_asp})
    return slim


def save_artifacts(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(result["lasso"], output_dir / "lasso_pipeline.joblib")
    (output_dir / "metrics.json").write_text(json.dumps(result["report"], indent=2) + "\n")


def load_tracks(path: Path) -> list[dict[str, Any]]:
    raw = path.read_text(encoding="utf-8")
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

    train_p = sub.add_parser("train", help="Fit Lasso and write artifacts.")
    train_p.add_argument("--input", required=True, type=Path, help="ASP tracks JSON array or JSONL.")
    train_p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models"),
        help="Directory for the Lasso pipeline and metrics.json.",
    )
    train_p.add_argument(
        "--slim-out",
        type=Path,
        default=None,
        help="Optional lyrics-free features-only copy of the tracks.",
    )
    train_p.add_argument(
        "--test-frac",
        type=float,
        default=DEFAULT_TEST_FRAC,
        help="Fraction of --input held out as the test set (0 = score the training tracks).",
    )
    train_p.add_argument(
        "--test-input",
        type=Path,
        default=None,
        help="Optional separate test-set JSON/JSONL. Overrides --test-frac.",
    )

    pred_p = sub.add_parser("predict", help="Score tracks with the saved Lasso pipeline.")
    pred_p.add_argument("--input", required=True, type=Path)
    pred_p.add_argument("--model-dir", type=Path, default=Path("models"))

    cmp_p = sub.add_parser(
        "compare",
        help="Train closed-form models on an ASP train/test split and print test metrics.",
    )
    cmp_p.add_argument("--input", required=True, type=Path)
    cmp_p.add_argument(
        "--test-frac",
        type=float,
        default=DEFAULT_TEST_FRAC,
        help="Held-out fraction of --input (the 10 outside FLACs are not used).",
    )
    cmp_p.add_argument(
        "--output",
        type=Path,
        default=Path("models") / "formula_compare.json",
        help="Where to write the comparison report.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "train":
        tracks = load_tracks(args.input)
        test_tracks = load_tracks(args.test_input) if args.test_input is not None else None
        result = train_and_evaluate(
            tracks,
            test_frac=0.0 if test_tracks is not None else args.test_frac,
            test_tracks=test_tracks,
        )
        save_artifacts(result, args.output_dir)
        if args.slim_out is not None:
            slim_path: Path = args.slim_out
            slim_path.parent.mkdir(parents=True, exist_ok=True)
            slim_path.write_text(
                json.dumps(slim_tracks(tracks), separators=(",", ":"), ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        models = result["report"]["models"]
        print(
            json.dumps(
                {
                    "evaluation": result["report"]["evaluation"],
                    "n_train": result["report"]["n_train"],
                    "n_test": result["report"]["n_test"],
                    "n_encoded_features": result["report"]["n_encoded_features"],
                    "n_selected_features": result["report"]["n_selected_features"],
                    "dropped_features": result["report"]["dropped_features"],
                    "lasso": models["lasso"],
                    "mean_baseline": models["mean_baseline"],
                    "artifacts": str(args.output_dir),
                },
                indent=2,
            )
        )
        return 0

    if args.command == "compare":
        tracks = load_tracks(args.input)
        result = compare_formula_models(tracks, test_frac=args.test_frac)
        payload = {k: v for k, v in result.items() if k != "pipelines"}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "evaluation": payload["evaluation"],
                    "n_train": payload["n_train"],
                    "n_test": payload["n_test"],
                    "ranking_by_test_mae": payload["ranking_by_test_mae"],
                    "compact_ols_formula": payload["models"]["compact_ols"]["formula"],
                    "report": str(args.output),
                },
                indent=2,
            )
        )
        return 0

    pipeline = joblib.load(args.model_dir / "lasso_pipeline.joblib")
    print(json.dumps(predict_tracks(load_tracks(args.input), pipeline), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
