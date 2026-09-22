"""Discover a closed-form ``aspEnergy`` formula with PySR.

Lasso (from ``asp_energy``) drops near-zero coefficients; PySR then searches
an explicit expression on the surviving numeric features. Headline metrics
are a held-out test split (default 20%); selection and search use train only.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sympy as sp
from sklearn.pipeline import Pipeline

from asp_energy import (
    FEATURE_OBJECT,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET,
    DEFAULT_TEST_FRAC,
    extract_feature_row,
    lasso_selection,
    load_tracks,
    make_lasso,
    regression_metrics,
    selected_numeric_features,
    selector_abs_coef,
    split_labeled,
    tracks_to_frame,
)

DEFAULT_FORMULA_NAME = "asp_energy_formula.json"
MIN_STD = 1e-8


def coerce_tracks(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        if not payload:
            raise ValueError("JSON array is empty")
        if not all(isinstance(item, dict) for item in payload):
            raise ValueError("JSON array must contain track objects")
        return payload
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object, array, or JSONL of tracks")
    if FEATURE_OBJECT in payload:
        return [payload]
    return [
        {
            FEATURE_OBJECT: payload,
            "name": str(payload.get("name") or payload.get("_id") or "pasted"),
        }
    ]


def parse_tracks_text(text: str) -> list[dict[str, Any]]:
    stripped = text.strip()
    if not stripped:
        raise ValueError("No JSON provided")
    try:
        return coerce_tracks(json.loads(stripped))
    except json.JSONDecodeError:
        tracks: list[dict[str, Any]] = []
        for line_no, line in enumerate(stripped.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                tracks.extend(coerce_tracks(json.loads(line)))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}") from exc
        if not tracks:
            raise ValueError("No JSON objects found")
        return tracks


def fit_lasso_selector(X: pd.DataFrame, y: np.ndarray) -> Pipeline:
    pipeline = make_lasso()
    pipeline.fit(X, y)
    return pipeline


def pysr_feature_list(
    pipeline: Pipeline,
    X: pd.DataFrame,
    max_features: int = 8,
) -> list[str]:
    numeric = [
        name
        for name in selected_numeric_features(pipeline)
        if float(X[name].std(ddof=0)) > MIN_STD
    ]
    numeric.sort(key=lambda name: selector_abs_coef(pipeline, name), reverse=True)
    if max_features > 0:
        numeric = numeric[:max_features]
    if numeric:
        return numeric
    fallback = [
        name
        for name in NUMERIC_FEATURES
        if float(X[name].std(ddof=0)) > MIN_STD
    ]
    fallback.sort(key=lambda name: selector_abs_coef(pipeline, name), reverse=True)
    limit = max_features if max_features > 0 else 8
    return fallback[:limit]


def feature_stats(X: pd.DataFrame, features: list[str]) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for name in features:
        series = X[name].astype(float)
        stats[name] = {
            "mean": float(series.mean()),
            "std": float(series.std(ddof=0)),
            "min": float(series.min()),
            "max": float(series.max()),
        }
    return stats


def standardize(X: pd.DataFrame, stats: dict[str, dict[str, float]]) -> pd.DataFrame:
    out = pd.DataFrame(index=X.index)
    for name, st in stats.items():
        std = st["std"]
        out[name] = 0.0 if std < MIN_STD else (X[name].astype(float) - st["mean"]) / std
    return out


def round_sympy(expr: sp.Expr, digits: int = 6) -> sp.Expr:
    replacements = {}
    for number in expr.atoms(sp.Number):
        if number.is_Integer:
            continue
        replacements[number] = sp.Float(round(float(number), digits))
    return expr.xreplace(replacements)


def denormalize_sympy(expr: sp.Expr, stats: dict[str, dict[str, float]]) -> sp.Expr:
    substitutions = {}
    for name, st in stats.items():
        symbol = sp.Symbol(name)
        std = st["std"]
        substitutions[symbol] = 0 if std < MIN_STD else (symbol - st["mean"]) / std
    return sp.expand(expr.subs(substitutions))


def used_features(payload: dict[str, Any], equation_index: int | None = None) -> list[str]:
    equation = equation_at(payload, equation_index)
    expr = sp.sympify(equation["sympy"])
    free = {str(symbol) for symbol in expr.free_symbols}
    ordered = [name for name in payload["features"] if name in free]
    extras = sorted(free.difference(payload["features"]))
    return ordered + extras


def lambdify_formula(sympy_str: str, features: list[str]):
    expr = sp.sympify(sympy_str)
    if not features:
        const = float(expr)
        return lambda: const
    symbols = [sp.Symbol(name) for name in features]
    return sp.lambdify(symbols, expr, modules=["numpy"])


def evaluate_formula(payload: dict[str, Any], X: pd.DataFrame, equation_index: int | None = None) -> np.ndarray:
    equation = equation_at(payload, equation_index)
    features = used_features(payload, equation_index)
    missing = [name for name in features if name not in X.columns]
    if missing:
        raise ValueError(f"Missing features for formula: {missing}")
    fn = lambdify_formula(equation["sympy"], features)
    if not features:
        return np.full(len(X), float(fn()), dtype=float)
    values = [X[name].to_numpy(dtype=float) for name in features]
    pred = np.asarray(fn(*values), dtype=float)
    return np.broadcast_to(pred, (len(X),)).copy()


def equation_at(payload: dict[str, Any], equation_index: int | None) -> dict[str, Any]:
    equations = payload.get("equations") or []
    if not equations:
        return {
            "sympy": payload["sympy"],
            "latex": payload.get("latex") or payload["sympy"],
            "formula": payload.get("formula") or payload["sympy"],
        }
    if equation_index is None:
        equation_index = int(payload.get("chosen_index", len(equations) - 1))
    equation_index = max(0, min(int(equation_index), len(equations) - 1))
    return equations[equation_index]


def make_pysr_regressor(
    *,
    niterations: int,
    maxsize: int,
    timeout: int,
    output_dir: Path,
    feature_names: list[str] | None = None,
) -> Any:
    from pysr import PySRRegressor

    output_dir.mkdir(parents=True, exist_ok=True)
    guesses = [" + ".join(feature_names)] if feature_names else None
    return PySRRegressor(
        niterations=niterations,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["square"],
        maxsize=maxsize,
        maxdepth=8,
        populations=18,
        population_size=40,
        ncycles_per_iteration=250,
        model_selection="best",
        parsimony=1e-3,
        random_state=RANDOM_STATE,
        verbosity=1,
        progress=True,
        timeout_in_seconds=timeout,
        constraints={"/": (-1, 6), "square": 6},
        nested_constraints={"square": {"square": 0}},
        output_directory=str(output_dir / "pysr"),
        run_id="asp_energy",
        extra_sympy_mappings={"square": lambda x: x**2},
        guesses=guesses,
    )


def _equation_records(
    model: Any,
    X_raw: pd.DataFrame,
    y: np.ndarray,
    stats: dict[str, dict[str, float]],
    features: list[str],
) -> tuple[list[dict[str, Any]], int]:
    records: list[dict[str, Any]] = []
    equations = model.equations_
    selected_raw = str(round_sympy(denormalize_sympy(model.sympy(), stats)))
    chosen = 0
    for position, (index, row) in enumerate(equations.iterrows()):
        z_expr = model.sympy(index)
        raw_expr = round_sympy(denormalize_sympy(z_expr, stats))
        sympy_str = str(raw_expr)
        pred = evaluate_formula(
            {"features": features, "sympy": sympy_str, "equations": [{"sympy": sympy_str}]},
            X_raw,
            0,
        )
        metrics = regression_metrics(y, pred)
        records.append(
            {
                "index": int(index),
                "complexity": int(row.get("complexity", 0)),
                "loss": None if pd.isna(row.get("loss")) else float(row["loss"]),
                "sympy": sympy_str,
                "formula": sympy_str,
                "latex": str(sp.latex(raw_expr)),
                "r2": round(metrics["r2"], 4),
                "rmse": round(metrics["rmse"], 4),
                "mae": round(metrics["mae"], 4),
            }
        )
        if sympy_str == selected_raw:
            chosen = position
    if not records:
        raise ValueError("PySR returned no equations")
    winner = records[chosen]
    winner["index"] = 0
    return [winner], 0


def fit_formula(
    tracks: list[dict[str, Any]],
    *,
    max_features: int = 8,
    niterations: int = 60,
    maxsize: int = 20,
    timeout: int = 300,
    output_dir: Path = Path("models"),
    test_frac: float = DEFAULT_TEST_FRAC,
    test_tracks: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    X, y, labels = tracks_to_frame(tracks, require_target=True)
    assert y is not None
    if test_tracks is not None:
        X_train, y_train, labels_train = X, y, labels
        X_test, y_test, labels_test = tracks_to_frame(test_tracks, require_target=True)
        assert y_test is not None
        evaluation = (
            f"held-out test file ({len(y_test)} tracks); "
            "Lasso selection and PySR fit on train only"
        )
    else:
        X_train, y_train, labels_train, X_test, y_test, labels_test = split_labeled(
            X, y, labels, test_frac=test_frac
        )
        if test_frac <= 0:
            evaluation = "in-sample (no held-out test set); Lasso selection and PySR fit on all tracks"
        else:
            evaluation = (
                f"held-out test ({test_frac:.0%} of {len(y)} labeled tracks, "
                f"seed {RANDOM_STATE}); Lasso selection and PySR fit on train only"
            )
    lasso = fit_lasso_selector(X_train, y_train)
    selection = lasso_selection(lasso)
    features = pysr_feature_list(lasso, X_train, max_features=max_features)
    if not features:
        raise ValueError("Lasso dropped every numeric feature; cannot run PySR")
    stats = feature_stats(X_train, features)
    X_z = standardize(X_train, stats)
    model = make_pysr_regressor(
        niterations=niterations,
        maxsize=maxsize,
        timeout=timeout,
        output_dir=output_dir,
        feature_names=features,
    )
    model.fit(X_z[features], y_train)
    equations, chosen = _equation_records(model, X_train, y_train, stats, features)
    best = equations[chosen]
    skeleton = {"features": features, "sympy": best["sympy"], "equations": equations}
    pred_test = evaluate_formula(skeleton, X_test, chosen)
    pred_train = evaluate_formula(skeleton, X_train, chosen)
    metrics = {k: round(v, 4) for k, v in regression_metrics(y_test, pred_test).items()}
    train_metrics = {k: round(v, 4) for k, v in regression_metrics(y_train, pred_train).items()}
    n_all = int(len(y_train) if test_tracks is None and test_frac <= 0 else len(y_train) + len(y_test))
    payload = {
        "target": TARGET,
        "feature_object": FEATURE_OBJECT,
        "n_tracks": n_all,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "evaluation": evaluation,
        "features": features,
        "feature_stats": stats,
        "lasso_kept": selection["kept"],
        "lasso_dropped": selection["dropped"],
        "formula": best["formula"],
        "sympy": best["sympy"],
        "latex": best["latex"],
        "chosen_index": 0,
        "metrics": metrics,
        "train_metrics": train_metrics,
        "equations": equations,
        "predictions": [
            {
                "track": label,
                "split": "test",
                "actual": round(float(actual), 4),
                "predicted": round(float(hat), 4),
                "deviation": round(float(hat - actual), 4),
                "abs_deviation": round(abs(float(hat - actual)), 4),
            }
            for label, actual, hat in zip(labels_test, y_test, pred_test)
        ],
    }
    return payload


def save_formula(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def load_formula(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not payload.get("sympy") or not payload.get("features"):
        raise ValueError(f"Formula file {path} is missing sympy/features")
    return payload


def tracks_to_formula_frame(
    tracks: list[dict[str, Any]],
    features: list[str],
) -> tuple[pd.DataFrame, np.ndarray | None, list[str]]:
    rows: list[dict[str, Any]] = []
    targets: list[float] = []
    labels: list[str] = []
    has_any_target = False
    skipped = 0
    for track in tracks:
        asp = track.get(FEATURE_OBJECT)
        if not isinstance(asp, dict):
            if any(name in track for name in features):
                asp = track
            else:
                skipped += 1
                continue
        row = extract_feature_row(asp) if any(k in asp for k in NUMERIC_FEATURES) else dict(asp)
        if any(row.get(name) is None for name in features):
            skipped += 1
            continue
        rows.append({name: float(row[name]) for name in features})
        actual = asp.get(TARGET, track.get(TARGET))
        if actual is None:
            targets.append(math.nan)
        else:
            has_any_target = True
            targets.append(float(actual))
        artist = ""
        artists = track.get("artists") or []
        if artists and isinstance(artists[0], dict):
            artist = str(artists[0].get("artistName") or "")
        labels.append(f"{artist} — {track.get('name', track.get('_id', 'unknown'))}".strip(" —"))
    if skipped:
        print(f"skipped {skipped} tracks missing formula features")
    if not rows:
        raise ValueError("No tracks had the features required by the formula")
    y = np.asarray(targets, dtype=float) if has_any_target else None
    return pd.DataFrame(rows), y, labels


def score_tracks(
    tracks: list[dict[str, Any]],
    payload: dict[str, Any],
    equation_index: int | None = None,
) -> list[dict[str, Any]]:
    X, y, labels = tracks_to_formula_frame(tracks, used_features(payload, equation_index) or list(payload["features"]))
    pred = evaluate_formula(payload, X, equation_index)
    out: list[dict[str, Any]] = []
    for i, (label, hat) in enumerate(zip(labels, pred)):
        row: dict[str, Any] = {
            "track": label,
            "predicted": round(float(hat), 4),
        }
        if y is not None and not math.isnan(float(y[i])):
            actual = float(y[i])
            row["actual"] = round(actual, 4)
            row["deviation"] = round(float(hat) - actual, 4)
            row["abs_deviation"] = round(abs(float(hat) - actual), 4)
        out.append(row)
    return out


def score_manual(
    values: dict[str, float],
    payload: dict[str, Any],
    actual: float | None = None,
    equation_index: int | None = None,
) -> dict[str, Any]:
    used = used_features(payload, equation_index)
    missing_manual = [name for name in used if name not in values]
    if missing_manual:
        raise ValueError(f"Missing features for formula: {missing_manual}")
    X = pd.DataFrame([values])
    pred = float(evaluate_formula(payload, X, equation_index)[0])
    result: dict[str, Any] = {"predicted": round(pred, 4)}
    if actual is not None and not (isinstance(actual, float) and math.isnan(actual)):
        result["actual"] = round(float(actual), 4)
        result["deviation"] = round(pred - float(actual), 4)
        result["abs_deviation"] = round(abs(pred - float(actual)), 4)
    return result


def _formula_markdown(payload: dict[str, Any], equation_index: int | None = None) -> str:
    eq = equation_at(payload, equation_index)
    metrics = payload.get("metrics") or {}
    eq_metrics = {k: eq[k] for k in ("r2", "rmse", "mae") if k in eq}
    shown = eq_metrics or metrics
    kept = ", ".join(used_features(payload, equation_index) or payload.get("features") or [])
    dropped_n = len(payload.get("lasso_dropped") or [])
    lines = [
        f"**aspEnergy** = $${eq.get('latex') or eq.get('formula')}$$",
        "",
        f"`{eq.get('formula')}`",
        "",
        f"Features used: `{kept}`",
        f"Lasso dropped {dropped_n} encoded columns before PySR.",
        f"Fit on **{payload.get('n_train', payload.get('n_tracks', '?'))}** training tracks"
        + (
            f"; scored on **{payload['n_test']}** held-out test tracks."
            if payload.get("n_test") and payload.get("evaluation", "").startswith("held-out")
            else "."
        ),
    ]
    if payload.get("evaluation"):
        lines.append(payload["evaluation"])
    if shown:
        split_name = "Test" if "held-out" in str(payload.get("evaluation", "")) else "Fit"
        lines.append(
            f"{split_name}: "
            + ", ".join(f"{k.upper()}={shown[k]}" for k in ("r2", "rmse", "mae") if k in shown)
        )
    return "\n".join(lines)


def build_app(formula_path: Path):
    import gradio as gr

    payload = load_formula(formula_path)
    features: list[str] = used_features(payload) or list(payload["features"])
    stats = payload.get("feature_stats") or {}

    def from_file(file_obj):
        if file_obj is None:
            raise gr.Error("Upload a JSON or JSONL file first.")
        path = Path(file_obj)
        rows = score_tracks(load_tracks(path), payload)
        return pd.DataFrame(rows)

    def from_text(text: str):
        rows = score_tracks(parse_tracks_text(text), payload)
        return pd.DataFrame(rows)

    def from_manual(*args):
        *feature_values, actual = args
        values = {name: float(value) for name, value in zip(features, feature_values)}
        actual_value = None if actual is None or actual == "" else float(actual)
        result = score_manual(values, payload, actual_value)
        return (
            result.get("predicted"),
            result.get("actual"),
            result.get("deviation"),
            result.get("abs_deviation"),
        )

    with gr.Blocks(title="aspEnergy formula") as demo:
        gr.Markdown("# Closed-form aspEnergy")
        gr.Markdown(
            "The Lasso-selected PySR formula. "
            "Upload track JSON/JSONL, paste a track, or set coefficients manually."
        )
        gr.Markdown(_formula_markdown(payload))

        with gr.Tab("From JSON file"):
            file_in = gr.File(
                label="Tracks JSON or JSONL",
                file_types=[".json", ".jsonl", ".txt"],
                type="filepath",
            )
            file_btn = gr.Button("Score file", variant="primary")
            file_out = gr.Dataframe(label="Predictions")
            file_btn.click(fn=from_file, inputs=[file_in], outputs=file_out)

        with gr.Tab("Paste JSON"):
            text_in = gr.Textbox(
                label="Track, array, or aspAudioFeatures JSON",
                lines=14,
                placeholder='{"aspAudioFeatures": {"loudness": -15.0, "relaxed": 0.4, "aspEnergy": 0.55}}',
            )
            text_btn = gr.Button("Score pasted JSON", variant="primary")
            text_out = gr.Dataframe(label="Predictions")
            text_btn.click(fn=from_text, inputs=[text_in], outputs=text_out)

        with gr.Tab("Manual coefficients"):
            manual_inputs = []
            for name in features:
                st = stats.get(name) or {}
                manual_inputs.append(
                    gr.Number(
                        label=name,
                        value=round(float(st.get("mean", 0.0)), 6) if st else 0.0,
                        info=(
                            f"train range {st['min']:.4g} … {st['max']:.4g}"
                            if st
                            else None
                        ),
                    )
                )
            actual_in = gr.Number(label="aspEnergy (optional, for deviation)", value=None)
            manual_btn = gr.Button("Evaluate formula", variant="primary")
            pred_out = gr.Number(label="Predicted aspEnergy")
            actual_out = gr.Number(label="Actual aspEnergy")
            dev_out = gr.Number(label="Deviation (predicted − actual)")
            abs_out = gr.Number(label="Absolute deviation")
            manual_btn.click(
                fn=from_manual,
                inputs=[*manual_inputs, actual_in],
                outputs=[pred_out, actual_out, dev_out, abs_out],
            )

    return demo


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    fit_p = sub.add_parser("fit", help="Select features with Lasso, then search a PySR formula.")
    fit_p.add_argument("--input", required=True, type=Path, help="Tracks JSON array or JSONL.")
    fit_p.add_argument("--output-dir", type=Path, default=Path("models"))
    fit_p.add_argument("--max-features", type=int, default=8)
    fit_p.add_argument("--niterations", type=int, default=60)
    fit_p.add_argument("--maxsize", type=int, default=20)
    fit_p.add_argument("--timeout", type=int, default=300, help="PySR timeout in seconds.")
    fit_p.add_argument(
        "--test-frac",
        type=float,
        default=DEFAULT_TEST_FRAC,
        help="Fraction of --input held out as the test set (0 = score the training tracks).",
    )
    fit_p.add_argument(
        "--test-input",
        type=Path,
        default=None,
        help="Optional separate test-set JSON/JSONL. Overrides --test-frac.",
    )

    pred_p = sub.add_parser("predict", help="Evaluate a saved formula on tracks.")
    pred_p.add_argument("--input", required=True, type=Path)
    pred_p.add_argument(
        "--formula",
        type=Path,
        default=Path("models") / DEFAULT_FORMULA_NAME,
    )

    serve_p = sub.add_parser("serve", help="Open the Gradio formula interface.")
    serve_p.add_argument(
        "--formula",
        type=Path,
        default=Path("models") / DEFAULT_FORMULA_NAME,
    )
    serve_p.add_argument("--host", default="127.0.0.1")
    serve_p.add_argument("--port", type=int, default=7860)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "fit":
        tracks = load_tracks(args.input)
        test_tracks = load_tracks(args.test_input) if args.test_input is not None else None
        payload = fit_formula(
            tracks,
            max_features=args.max_features,
            niterations=args.niterations,
            maxsize=args.maxsize,
            timeout=args.timeout,
            output_dir=args.output_dir,
            test_frac=0.0 if test_tracks is not None else args.test_frac,
            test_tracks=test_tracks,
        )
        path = args.output_dir / DEFAULT_FORMULA_NAME
        save_formula(payload, path)
        print(
            json.dumps(
                {
                    "formula": payload["formula"],
                    "latex": payload["latex"],
                    "features": payload["features"],
                    "evaluation": payload.get("evaluation"),
                    "n_train": payload.get("n_train"),
                    "n_test": payload.get("n_test"),
                    "metrics": payload["metrics"],
                    "train_metrics": payload.get("train_metrics"),
                    "lasso_dropped": payload["lasso_dropped"],
                    "artifact": str(path),
                },
                indent=2,
            )
        )
        return 0

    if args.command == "predict":
        payload = load_formula(args.formula)
        print(json.dumps(score_tracks(load_tracks(args.input), payload), indent=2))
        return 0

    demo = build_app(args.formula)
    demo.launch(server_name=args.host, server_port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
