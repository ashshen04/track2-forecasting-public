"""## Executive summary (read this first)

Synthetic changing-volatility histories check an optional daily EWMA estimate.
EWMA means exponentially weighted moving average. Only the forecast spread changes;
the centre, joint standardized paths, cutoff and default model stay unchanged.
"""

import numpy as np
import pandas as pd
import pytest

from qfbench2_track_forecasting.cli import _draw
from qfbench2_track_forecasting.grid import GridSpec
from scripts.backtest import forecast_samples


def panel(recent_scale):
    steps = np.tile([-0.001, 0.001], 60)
    steps[-40:] *= recent_scale
    dates = pd.bdate_range("2020-01-01", periods=121)
    values = 100 + np.r_[0, np.cumsum(steps)]
    first = pd.DataFrame({"date": dates, "asset": "SYN-A", "value": values})
    second = first.assign(asset="SYN-B", value=200 - 2 * (values - 100))
    return {"synthetic": pd.concat([first, second])}, str(dates[-1].date())


@pytest.mark.parametrize("recent_scale", [0.2, 5.0])
def test_ewma_changes_spread_but_preserves_standardized_paths(recent_scale):
    panels, asof = panel(recent_scale)
    args = (panels, ["SYN-A", "SYN-B"], [5, 21], asof, 1000, 1)
    base, base_stats = _draw(*args)
    weighted, weighted_stats = _draw(*args, volatility="ewma")
    assert base_stats["last"] == weighted_stats["last"]
    assert base_stats["daily_drift"] == weighted_stats["daily_drift"]
    for ai, asset in enumerate(args[1]):
        old_sd, new_sd = base_stats["daily_sd"][asset], weighted_stats["daily_sd"][asset]
        assert (new_sd > old_sd) == (recent_scale > 1)
        anchor = base_stats["last"][asset]
        np.testing.assert_allclose(
            (base[:, ai] - anchor) / old_sd,
            (weighted[:, ai] - anchor) / new_sd,
            atol=1e-10,
        )


def test_ewma_ignores_future_and_history_before_the_window():
    panels, asof = panel(5)
    args = (["SYN-A", "SYN-B"], [21], asof, 1000, 1)
    clean, _ = _draw(panels, *args, volatility="ewma")
    extra = pd.DataFrame({"date": pd.to_datetime(["2010-01-01", "2030-01-01"]),
                          "asset": "SYN-A", "value": [9999., 9999.]})
    noisy = {"synthetic": pd.concat([panels["synthetic"], extra])}
    changed, _ = _draw(noisy, *args, volatility="ewma")
    np.testing.assert_array_equal(clean, changed)


def test_backtest_routes_ewma_to_the_same_sampler():
    panels, asof = panel(5)
    grid = GridSpec(("SYN-A", "SYN-B"), (5, 21))
    actual = forecast_samples("reference-ewma", panels, pd.Timestamp(asof), grid, 1000, "level")
    direct, _ = _draw(panels, list(grid.assets), list(grid.horizons), asof, 1000, 1,
                      volatility="ewma")
    np.testing.assert_array_equal(actual, direct.reshape(1000, -1))


def test_unknown_volatility_mode_is_rejected():
    panels, asof = panel(5)
    with pytest.raises(ValueError, match="volatility"):
        _draw(panels, ["SYN-A"], [21], asof, 1000, 1, volatility="typo")
