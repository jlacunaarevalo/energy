"""Synthetic aspAudioFeatures tracks with a known linear aspEnergy."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from asp_energy import (
    FEATURE_OBJECT,
    TARGET,
    compare_formula_models,
    extract_feature_row,
    load_tracks,
    main,
    make_lasso,
    predict_tracks,
    save_artifacts,
    tracks_to_frame,
    train_and_evaluate,
)

KEYS = ["C", "G", "D", "A", "F", "Bb"]
SCALES = ["major", "minor"]
LANGS = ["en", "es"]


def _asp_row(rng: np.random.Generator, i: int) -> dict:
    happy = float(rng.uniform(0.1, 0.95))
    relaxed = float(rng.uniform(0.05, 0.9))
    loudness = float(rng.uniform(-18.0, -4.0))
    acoustic = float(rng.uniform(0.02, 0.85))
    intensity = 0 if rng.random() > 0.3 else -1
    energy = (
        0.55
        + 0.22 * happy
        - 0.18 * relaxed
        + 0.012 * (loudness + 10)
        - 0.08 * acoustic
        + 0.04 * intensity
        + float(rng.normal(0, 0.02))
    )
    energy = float(np.clip(energy, 0.05, 0.95))
    return {
        "danceability": float(rng.uniform(0.2, 0.9)),
        "acoustic": acoustic,
        "aggressive": float(rng.uniform(0.01, 0.5)),
        "happy": happy,
        "party": float(rng.uniform(0.05, 0.8)),
        "relaxed": relaxed,
        "sad": float(rng.uniform(0.05, 0.7)),
        "voice": float(rng.uniform(0.3, 0.95)),
        "instrumental": float(rng.uniform(0.0, 0.4)),
        "gender": float(rng.uniform(0.3, 0.7)),
        "tempo": float(rng.uniform(70, 160)),
        "key": KEYS[i % len(KEYS)],
        "scale": SCALES[i % 2],
        "loudness": loudness,
        "intensity": intensity,
        "onset_rate": float(rng.uniform(1.5, 6.0)),
        "entropy": float(rng.uniform(18, 24)),
        "energy": float(rng.uniform(8e4, 9e5)),
        "dynamic_range": float(rng.uniform(3.0, 8.0)),
        "larm": float(rng.uniform(-12, -3)),
        "leq": float(rng.uniform(-20, -8)),
        "durationMs": int(rng.integers(150_000, 280_000)),
        "initialSilence": int(rng.integers(0, 800)),
        "finalSilence": int(rng.integers(200, 4000)),
        "language": {
            "language": LANGS[i % 2],
            "probability": float(rng.uniform(0.5, 0.99)),
        },
        TARGET: energy,
    }


def make_tracks(n: int = 40, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    tracks = []
    for i in range(n):
        tracks.append(
            {
                "_id": f"track-{i:03d}",
                "name": f"Song {i}",
                "artists": [{"artistId": "x", "artistName": f"Artist {i % 7}"}],
                FEATURE_OBJECT: _asp_row(rng, i),
                "lyrics": {"text": "should be ignored"},
                "popularity": 50,
            }
        )
    return tracks


def test_extract_uses_only_asp_audio_features():
    row = extract_feature_row(make_tracks(1)[0][FEATURE_OBJECT])
    assert "language_probability" in row
    assert row["language"] in LANGS
    assert TARGET not in row
    assert "popularity" not in row
    str_lang = extract_feature_row({**make_tracks(1)[0][FEATURE_OBJECT], "language": "en"})
    assert str_lang["language"] == "en"
    assert str_lang["language_probability"] == 0.0


def test_tracks_to_frame_skips_incomplete_and_ignores_lyrics():
    tracks = make_tracks(3)
    tracks.append({"_id": "bad", "name": "Missing", FEATURE_OBJECT: {"danceability": 0.5}})
    frame, y, labels = tracks_to_frame(tracks)
    assert len(frame) == 3
    assert y is not None and len(y) == 3
    assert all("Song" in label for label in labels)
    assert "lyrics" not in frame.columns


def test_lasso_beats_mean_on_held_out_test(tmp_path: Path):
    tracks = make_tracks(40)
    result = train_and_evaluate(tracks, test_frac=0.25)
    report = result["report"]
    models = report["models"]
    assert report["n_train"] == 30
    assert report["n_test"] == 10
    assert set(result["labels_train"]).isdisjoint(result["labels_test"])
    assert models["lasso"]["test"]["r2"] > 0.7
    assert models["lasso"]["test"]["r2"] > models["mean_baseline"]["test"]["r2"]
    assert "ridge" not in models
    assert "ols" not in models
    top = [row["feature"] for row in report["lasso_coefficients"][:8]]
    assert "happy" in top
    assert "relaxed" in top

    out = tmp_path / "models"
    save_artifacts(result, out)
    assert (out / "lasso_pipeline.joblib").exists()
    assert not (out / "ridge_pipeline.joblib").exists()
    scored = predict_tracks(tracks[:3], result["lasso"])
    assert len(scored) == 3
    assert scored[0]["predicted"] is not None


def test_separate_test_file_is_not_used_for_fitting():
    train = make_tracks(30, seed=0)
    test = make_tracks(12, seed=1)
    result = train_and_evaluate(train, test_tracks=test)
    assert result["report"]["n_train"] == 30
    assert result["report"]["n_test"] == 12
    assert "held-out test file" in result["report"]["evaluation"]
    assert len(result["report"]["predictions"]) == 12


def test_cli_train_and_predict(tmp_path: Path):
    tracks_path = tmp_path / "tracks.json"
    tracks_path.write_text(json.dumps(make_tracks(24)))
    model_dir = tmp_path / "models"
    code = main(
        [
            "train",
            "--input",
            str(tracks_path),
            "--output-dir",
            str(model_dir),
            "--test-frac",
            "0.25",
            "--slim-out",
            str(tmp_path / "slim.json"),
        ]
    )
    assert code == 0
    slim = json.loads((tmp_path / "slim.json").read_text())
    assert "lyrics" not in slim[0]
    assert FEATURE_OBJECT in slim[0]
    code = main(
        [
            "predict",
            "--input",
            str(tracks_path),
            "--model-dir",
            str(model_dir),
        ]
    )
    assert code == 0
    loaded = load_tracks(tracks_path)
    assert len(loaded) == 24


def test_lasso_pipeline_is_sklearn_pipeline():
    pipe = make_lasso()
    assert "prep" in pipe.named_steps
    assert "select" in pipe.named_steps
    assert "model" in pipe.named_steps


def test_lasso_drops_zero_coefficients():
    tracks = make_tracks(40)
    result = train_and_evaluate(tracks, test_frac=0.25)
    report = result["report"]
    kept = report["selected_features"]
    dropped = report["dropped_features"]
    encoded = report["n_encoded_features"]
    assert kept
    assert len(kept) + len(dropped) == encoded
    assert report["n_selected_features"] == len(kept)
    assert report["n_selected_features"] < encoded
    table_features = {
        row["feature"] for row in report["lasso_coefficients"] if row["feature"] != "(intercept)"
    }
    assert table_features.isdisjoint(dropped)
    assert any(name in table_features for name in ("happy", "relaxed"))


def test_load_tracks_jsonl(tmp_path: Path):
    tracks = make_tracks(3)
    path = tmp_path / "tracks.jsonl"
    path.write_text("\n".join(json.dumps(track) for track in tracks) + "\n")
    loaded = load_tracks(path)
    assert len(loaded) == 3
    assert loaded[0]["_id"] == "track-000"


def test_compare_formula_models_on_synthetic():
    tracks = make_tracks(48)
    result = compare_formula_models(tracks, test_frac=0.25)
    models = result["models"]
    assert result["n_train"] == 36
    assert result["n_test"] == 12
    for name in (
        "mean_baseline",
        "compact_ols",
        "compact_splines",
        "ols",
        "ridge",
        "elastic_net",
        "lasso",
        "lasso_then_ols",
        "pysr_frozen",
    ):
        assert models[name]["test"]["mae"] is not None
    assert models["lasso"]["test"]["r2"] > models["mean_baseline"]["test"]["r2"]
    assert models["elastic_net"]["test"]["r2"] > models["mean_baseline"]["test"]["r2"]
    assert "loudness" in models["compact_ols"]["formula"]
    assert result["ranking_by_test_mae"][0]["mae"] <= models["mean_baseline"]["test"]["mae"]

