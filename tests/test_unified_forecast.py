"""## Executive summary (read this first)

Run the public forecast entrypoint on synthetic inputs. Check conservative text
adjustments, explicit statistical mode, and offline fallback without network calls.
"""

import json

import numpy as np
import pandas as pd
import pytest

from baselines import reasoning_agent
from qfbench2_track_forecasting import cli


@pytest.fixture
def task(tmp_path, monkeypatch):
    unit = tmp_path / "input"
    panels = unit / "panels"
    panels.mkdir(parents=True)
    dates = pd.bdate_range("2020-01-01", periods=80)
    asof = str(dates[-1].date())
    pd.DataFrame({"date": dates, "asset": "SYN-A",
                  "value": 100 + np.cumsum(np.tile([1., -1.], 40))}).to_parquet(
        panels / "synthetic.parquet", index=False
    )
    (unit / "card.toml").write_text(
        '[task]\nid="synthetic-unified"\n[targets]\nasset_ids=["SYN-A"]\n'
        'horizons=[5,21]\ntarget_type="level"\n[scoring.params]\nn_draws_min=1200\n'
    )
    text = unit / "text"
    text.mkdir()
    effective = str((dates[-1] + pd.offsets.BDay(3)).date())
    (text / "evidence.txt").write_text(
        "Synthetic policy uncertainty, published " + asof + ". "
        + "Synthetic future increment takes effect on " + effective + "."
    )
    (text / "corpus_index.json").write_text(json.dumps({"documents": [{
        "doc_id": "synthetic-doc", "timestamp": asof,
        "file": "evidence.txt", "doc_type": "test"
    }]}))
    for name in ("MODEL_ENDPOINT", "MODEL_NAME", "MODEL_TOKEN", "QFBENCH_NETWORK"):
        monkeypatch.delenv(name, raising=False)
    return ["--panels", str(panels), "--text", str(text), "--asof", asof]


def read_output(out):
    return pd.read_parquet(out), json.loads((out.parent / "forecast_meta.json").read_text())


def test_forecast_defaults_to_conservative_house_adjustments(task, tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_ENDPOINT", "http://house.invalid")
    monkeypatch.setenv("MODEL_NAME", "synthetic-house")
    monkeypatch.setenv("MODEL_TOKEN", "synthetic-token")
    calls = []

    def reply(prompt):
        calls.append(prompt)
        return {"assets": {"SYN-A": {"drift_bp": 100, "vol_scale": 1.2,
                "because": "Synthetic uncertainty", "doc_ids": ["synthetic-doc"]}}}, "", ""

    monkeypatch.setattr(reasoning_agent, "call_model", reply)
    adjusted = tmp_path / "adjusted/forecast.parquet"
    base = tmp_path / "base/forecast.parquet"
    assert cli.main(task + ["--out", str(adjusted)]) == 0
    assert cli.main(task + ["--out", str(base), "--no-text"]) == 0
    assert len(calls) == 1
    assert "Conservative experiment" in calls[0]
    draws, meta = read_output(adjusted)
    original, base_meta = read_output(base)
    assert meta["text_policy"] == "conservative-v1"
    assert meta["reasoning_applied"] is True
    assert base_meta["reasoning_applied"] is False
    assert meta["n_draws"] == base_meta["n_draws"] == 1200
    for horizon in [5, 21]:
        before = original.loc[original.horizon == horizon, "value"].to_numpy()
        after = draws.loc[draws.horizon == horizon, "value"].to_numpy()
        np.testing.assert_allclose(after, before.mean() + 1.2 * (before - before.mean()) + 1)


@pytest.mark.parametrize("policy", ["conservative", "structured"])
@pytest.mark.parametrize("mode", ["missing-token", "network-none", "unavailable"])
def test_unified_forecast_labels_fallback(task, tmp_path, monkeypatch, mode, policy):
    monkeypatch.setenv("MODEL_ENDPOINT", "http://house.invalid")
    monkeypatch.setenv("MODEL_NAME", "synthetic-house")
    if mode != "missing-token":
        monkeypatch.setenv("MODEL_TOKEN", "synthetic-token")
    if mode == "network-none":
        monkeypatch.setenv("QFBENCH_NETWORK", "none")
    calls = []

    def reply(prompt):
        calls.append(prompt)
        return None, "synthetic upstream failure", ""

    monkeypatch.setattr(reasoning_agent, "call_model", reply)
    out = tmp_path / "fallback/forecast.parquet"
    base = tmp_path / "base/forecast.parquet"
    cli.main(task + ["--out", str(out), "--text-policy", policy])
    cli.main(task + ["--out", str(base), "--no-text"])
    draws, meta = read_output(out)
    original, _ = read_output(base)
    pd.testing.assert_frame_equal(draws, original)
    assert meta["reasoning_applied"] is False
    assert meta["reasoning_skipped_reason"]
    assert len(calls) == (1 if mode == "unavailable" else 0)


@pytest.mark.parametrize("decision", ["up", "conflict", "invalid", "empty"])
@pytest.mark.parametrize("text_scale", [1., .5])
def test_structured_forecast_uses_one_call_and_preserves_joint_draws(task, tmp_path, monkeypatch, decision, text_scale):
    from baselines.directional_text import evidence_spans
    from baselines.evidence_analysis import ANALYSIS_VERSION

    for name in ("MODEL_ENDPOINT", "MODEL_NAME", "MODEL_TOKEN"):
        monkeypatch.setenv(name, "synthetic")
    docs = reasoning_agent.read_corpus(tmp_path / "input/text", task[-1])[0]
    sid = next(iter(evidence_spans(docs)))
    driver = {"id": "d1", "change_basis": "explicit_future_path", "timing": "within_horizon",
              "announcement_date": task[-1],
              "effective_date": str((pd.Timestamp(task[-1]) + pd.offsets.BDay(3)).date()),
              "remaining_future_change": {
                  "description": "Synthetic future increment takes effect on "
                  + str((pd.Timestamp(task[-1]) + pd.offsets.BDay(3)).date()) + ".",
                  "date": str((pd.Timestamp(task[-1]) + pd.offsets.BDay(3)).date()),
                  "evidence": [{"span_id": sid}],
              },
              "effect": {"basis": "numeric", "entity": "SYN-A", "direction": "up"},
              "mechanism": "Synthetic future increment.", "evidence": [{"span_id": sid}]}
    drivers = [] if decision == "empty" else [driver]
    if decision == "conflict":
        drivers.append({**driver, "id": "d2", "effect": {**driver["effect"], "direction": "down"}})
    if decision == "invalid":
        driver["evidence"] = [{"span_id": "invented"}]
    raw = {"assets": {"SYN-A": {"drivers": drivers, "pricing": {
        "label": "unclear", "basis": "unavailable", "explanation": "No pricing evidence.", "evidence": []},
        "missing_evidence": "No supported future increment."}}}
    calls = []
    monkeypatch.setattr(reasoning_agent, "call_model", lambda prompt: (calls.append(prompt) or raw, "", ""))
    adjusted = tmp_path / "structured/forecast.parquet"
    base = tmp_path / "base/forecast.parquet"
    cli.main(task + ["--out", str(adjusted), "--text-policy", "structured", "--text-scale", str(text_scale)])
    cli.main(task + ["--out", str(base), "--text-policy", "structured", "--no-text"])
    draws, meta = read_output(adjusted)
    original, _ = read_output(base)
    assert len(calls) == 1 and ANALYSIS_VERSION in calls[0]
    assert meta["text_policy"] == ANALYSIS_VERSION
    assert meta["text_scale"] == text_scale
    assert meta["reasoning_applied"] is (decision == "up")
    assert meta["analysis_audit"]
    if decision == "up":
        shift = meta["text_adjustments"]["SYN-A"]["longest_horizon_shift"]
        assert shift > 0
        assert meta["text_adjustments"]["SYN-A"]["adjustment_scale"] == text_scale
        for horizon in (5, 21):
            before = original.loc[original.horizon == horizon, "value"].to_numpy()
            after = draws.loc[draws.horizon == horizon, "value"].to_numpy()
            np.testing.assert_allclose(after - before, shift * horizon / 21)
            assert after.std() == pytest.approx(before.std())
    else:
        pd.testing.assert_frame_equal(draws, original)
        assert meta["reasoning_skipped_reason"]


def test_zero_text_scale_skips_model_and_matches_no_text(task, tmp_path, monkeypatch):
    for name in ("MODEL_ENDPOINT", "MODEL_NAME", "MODEL_TOKEN"):
        monkeypatch.setenv(name, "synthetic")
    def forbidden(prompt):
        raise AssertionError("Zero text scale must not call the model")
    monkeypatch.setattr(reasoning_agent, "call_model", forbidden)
    zero = tmp_path / "zero/forecast.parquet"
    base = tmp_path / "base/forecast.parquet"
    cli.main(task + ["--out", str(zero), "--text-policy", "structured", "--text-scale", "0"])
    cli.main(task + ["--out", str(base), "--no-text"])
    pd.testing.assert_frame_equal(read_output(zero)[0], read_output(base)[0])


def test_structured_cache_reuses_analysis_across_draw_seeds(task, tmp_path, monkeypatch):
    for name in ("MODEL_ENDPOINT", "MODEL_NAME", "MODEL_TOKEN"):
        monkeypatch.setenv(name, "synthetic")
    raw = {"assets": {"SYN-A": {"drivers": [], "pricing": {
        "label": "unclear", "basis": "unavailable", "explanation": "No pricing evidence.", "evidence": []},
        "missing_evidence": "No supported increment."}}}
    calls = []
    monkeypatch.setattr(reasoning_agent, "call_model", lambda p: (calls.append(p) or raw, "", ""))
    args = task + ["--text-policy", "structured", "--text-cache", str(tmp_path / "cache")]
    for seed in (1, 7, 19):
        out = tmp_path / str(seed) / "forecast.parquet"
        cli.main(args + ["--out", str(out), "--seed", str(seed)])
        _, meta = read_output(out)
        assert meta["text_cache_hits"] == (0 if seed == 1 else 1)
        assert meta["analysis_audit"]["SYN-A"]["decision_reason"] == "no_supported_driver"
    assert len(calls) == 1
