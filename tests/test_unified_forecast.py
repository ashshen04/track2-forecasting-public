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
    (text / "evidence.txt").write_text("Synthetic policy uncertainty, published " + asof)
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


@pytest.mark.parametrize("mode", ["missing-token", "network-none", "unavailable"])
def test_unified_forecast_labels_fallback(task, tmp_path, monkeypatch, mode):
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
    cli.main(task + ["--out", str(out)])
    cli.main(task + ["--out", str(base), "--no-text"])
    draws, meta = read_output(out)
    original, _ = read_output(base)
    pd.testing.assert_frame_equal(draws, original)
    assert meta["reasoning_applied"] is False
    assert meta["reasoning_skipped_reason"]
    assert len(calls) == (1 if mode == "unavailable" else 0)
