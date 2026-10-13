"""## Executive summary (read this first)

Synthetic histories verify trend shrinkage and chronological model selection.
For example, half a daily trend moves day 10 by five daily mean changes.
These tests contain no public-card outcomes.
"""

import copy

import numpy as np
import pandas as pd
import pytest

from qfbench2_track_forecasting.cli import _draw
from qfbench2_track_forecasting.grid import GridSpec
from scripts.backtest import DRIFT_MODELS, drift_validation, forecast_samples, main


def history(target_type):
    dates = pd.bdate_range("2020-01-01", periods=160)
    steps = 0.002 + np.tile([-0.001, 0.001], 80)
    values = np.expm1(steps) if target_type == "log_return" else 100 + np.cumsum(steps)
    frame = pd.DataFrame({"date": dates, "asset": "SYN-A", "value": values})
    return {"synthetic": frame}, str(dates[-1].date())


@pytest.mark.parametrize("target_type", ["level", "log_return"])
def test_trend_only_translates_draws_and_respects_horizon_order(target_type):
    panels, asof = history(target_type)
    args = (panels, ["SYN-A"], [21, 5], asof, 1000, 1)
    zero, zero_meta = _draw(*args, target_type=target_type, drift_scale=0)
    half, half_meta = _draw(*args, target_type=target_type, drift_scale=0.5)
    full, full_meta = _draw(*args, target_type=target_type, drift_scale=1)
    shift = np.broadcast_to(np.array([21, 5]) * 0.002, (1000, 2))
    np.testing.assert_allclose(full[:, 0] - zero[:, 0], shift, atol=1e-12)
    np.testing.assert_allclose(half[:, 0] - zero[:, 0], shift / 2, atol=1e-12)
    assert zero_meta["daily_sd"] == half_meta["daily_sd"] == full_meta["daily_sd"]
    assert half_meta["daily_drift"]["SYN-A"] == pytest.approx(0.001)
    default, _ = _draw(*args, target_type=target_type)
    np.testing.assert_array_equal(default, full if target_type == "log_return" else zero)


def test_trend_ignores_future_and_changes_before_the_window():
    panels, asof = history("level")
    args = (["SYN-A"], [5], asof, 1000, 1)
    clean, _ = _draw(panels, *args, drift_scale=0.5)
    noisy = copy.deepcopy(panels)
    noisy["synthetic"].loc[:10, "value"] = 99999
    future = pd.DataFrame({"date": [pd.Timestamp("2030-01-01")],
                           "asset": ["SYN-A"], "value": [99999.]})
    noisy["synthetic"] = pd.concat([noisy["synthetic"], future])
    changed, _ = _draw(noisy, *args, drift_scale=0.5)
    np.testing.assert_array_equal(clean, changed)


@pytest.mark.parametrize("scale", [-1, 1.1, float("nan"), float("inf")])
def test_invalid_trend_strength_is_rejected(scale):
    panels, asof = history("level")
    with pytest.raises(ValueError, match="drift_scale"):
        _draw(panels, ["SYN-A"], [5], asof, 1000, 1, drift_scale=scale)


def test_monthly_trend_is_explicitly_unsupported():
    panels, asof = history("level")
    with pytest.raises(ValueError, match="daily-only"):
        _draw(panels, ["SYN-A"], [5], asof, 1000, 1,
              panel_steps=np.array([[1]]), drift_scale=0.5)


def folds():
    result = []
    for i in range(4):
        scores = {}
        for name, score in [("reference", 2), ("reference-drift-zero", 1),
                            ("reference-drift-half", 3), ("reference-drift-full", 4)]:
            scores[name] = {key: score for key in ("composite", "marginal", "joint", "tail")}
        result.append({"asof": f"2020-0{i + 1}-01", "model_scores": scores})
    return result


def test_selection_never_uses_validation_scores():
    data = folds()
    first = drift_validation(data, 2, 4)
    assert first["selected_model"] == "reference-drift-zero"
    assert first["validation"]["wins"] == 2
    for fold in data[2:]:
        fold["model_scores"]["reference-drift-zero"]["composite"] = 99
        fold["model_scores"]["reference-drift-full"]["composite"] = 0.01
    second = drift_validation(data, 2, 4)
    assert second["selected_model"] == first["selected_model"]
    assert second["validation"]["wins"] == 0
    assert second["selection_origins"] == ["2020-01-01", "2020-02-01"]
    assert second["validation_origins"] == ["2020-03-01", "2020-04-01"]


def test_short_run_does_not_claim_validation():
    result = drift_validation(folds()[:3], 2, 4)
    assert result["status"] == "insufficient_folds"
    assert "selected_model" not in result


def test_ties_prefer_existing_reference():
    data = folds()
    for fold in data:
        fold["model_scores"]["reference"] = dict(fold["model_scores"]["reference-drift-zero"])
    assert drift_validation(data, 2, 4)["selected_model"] == "reference"


@pytest.mark.parametrize("name, scale", DRIFT_MODELS.items())
def test_backtest_routes_trends_and_preserves_joint_noise(name, scale):
    panels, asof = history("level")
    first = panels["synthetic"]
    panels["synthetic"] = pd.concat([
        first, first.assign(asset="SYN-B", value=200 - 2 * (first.value - 100))])
    grid = GridSpec(("SYN-A", "SYN-B"), (21, 5))
    actual = forecast_samples(name, panels, pd.Timestamp(asof), grid, 1000, "level")
    direct, stats = _draw(panels, list(grid.assets), list(grid.horizons), asof, 1000, 1,
                          drift_scale=scale)
    np.testing.assert_array_equal(actual, direct.reshape(1000, -1))
    zero, _ = _draw(panels, list(grid.assets), list(grid.horizons), asof, 1000, 1,
                    drift_scale=0)
    shift = np.array([stats["daily_drift"][a] for a in grid.assets])[:, None] * grid.horizons
    np.testing.assert_allclose(direct - zero, np.broadcast_to(shift, direct.shape), atol=1e-12)


@pytest.mark.parametrize("selection", [0, 6, 7])
def test_invalid_split_is_refused_before_writing(tmp_path, selection):
    output = tmp_path / "run"
    with pytest.raises(SystemExit) as error:
        main(["--out", str(output), "--folds", "6", "--selection-folds", str(selection),
              "--models", "reference", *DRIFT_MODELS])
    assert error.value.code == 2
    assert not output.exists()


def test_incomplete_candidate_set_is_refused_before_writing(tmp_path):
    output = tmp_path / "run"
    with pytest.raises(SystemExit) as error:
        main(["--out", str(output), "--folds", "6", "--selection-folds", "3",
              "--models", "reference", "reference-drift-half"])
    assert error.value.code == 2
    assert not output.exists()
