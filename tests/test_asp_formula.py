"""Formula helpers: denormalize, evaluate, JSON coercion (no Julia)."""

from __future__ import annotations

import json

import pandas as pd
import sympy as sp

from asp_energy import FEATURE_OBJECT, TARGET
from asp_formula import (
    coerce_tracks,
    denormalize_sympy,
    evaluate_formula,
    parse_tracks_text,
    score_manual,
    score_tracks,
)


def test_denormalize_linear_z_scores():
    x = sp.Symbol("loudness")
    expr = 0.5 + 0.2 * x
    stats = {"loudness": {"mean": 10.0, "std": 2.0}}
    raw = denormalize_sympy(expr, stats)
    # 0.5 + 0.2 * (loudness - 10) / 2 = 0.1 * loudness - 0.5
    loudness = sp.Symbol("loudness")
    expected = sp.expand(0.5 + 0.2 * (loudness - 10) / 2)
    assert sp.simplify(raw - expected) == 0


def test_evaluate_formula_and_deviation():
    payload = {
        "features": ["happy", "relaxed"],
        "sympy": "0.5 + 0.2*happy - 0.1*relaxed",
        "equations": [{"sympy": "0.5 + 0.2*happy - 0.1*relaxed"}],
    }
    X = pd.DataFrame({"happy": [1.0], "relaxed": [0.5]})
    pred = evaluate_formula(payload, X, 0)
    assert abs(float(pred[0]) - 0.65) < 1e-9
    result = score_manual({"happy": 1.0, "relaxed": 0.5}, payload, actual=0.70)
    assert result["predicted"] == 0.65
    assert result["deviation"] == -0.05
    assert result["abs_deviation"] == 0.05


def test_coerce_and_score_pasted_features():
    payload = {
        "features": ["happy", "relaxed"],
        "sympy": "0.4 + happy - relaxed",
        "equations": [{"sympy": "0.4 + happy - relaxed"}],
    }
    tracks = parse_tracks_text(
        json.dumps(
            {
                "happy": 0.8,
                "relaxed": 0.2,
                TARGET: 0.9,
                "name": "clip",
            }
        )
    )
    assert FEATURE_OBJECT in tracks[0]
    rows = score_tracks(tracks, payload)
    assert len(rows) == 1
    assert rows[0]["predicted"] == 1.0
    assert rows[0]["actual"] == 0.9
    assert rows[0]["deviation"] == 0.1


def test_parse_jsonl_text():
    text = "\n".join(
        [
            json.dumps({FEATURE_OBJECT: {"happy": 0.1, "relaxed": 0.2, TARGET: 0.3}, "name": "a"}),
            json.dumps({FEATURE_OBJECT: {"happy": 0.4, "relaxed": 0.1, TARGET: 0.5}, "name": "b"}),
        ]
    )
    tracks = parse_tracks_text(text)
    assert len(tracks) == 2
    assert coerce_tracks(tracks[0])[0]["name"] == "a"
