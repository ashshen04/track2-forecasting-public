"""## Executive summary (read this first)

Estimate normalization from the published M0 Gaussian baseline procedure.
For example, independent synthetic outcomes estimate M0's expected CRPS error.
Monte Carlo means averages over simulated outcomes, rather than observed answers.
All metrics use the existing scorer and shared toolkit; no scoring math is copied.
These estimates are research proxies, never verified organizer scales or ranked scores.
Monthly callers must supply externally resolved observation-period step counts.
This module does not infer monthly target dates or publication lag.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from qfbench2_track_forecasting.cli import _series
from qfbench2_track_forecasting.normalization import NormalizationMode, RefScale
from qfbench2_track_forecasting.scoring import _score


def distribution(panels, grid, asof, target_type, *, panel_steps=None):
    """Published M0: trailing 300 rows, dated steps, gap removal and shared paths."""
    if target_type not in ('level', 'log_return'):
        raise ValueError('Unsupported M0 target')
    series = {a: _series(panels, a, asof).tail(300) for a in grid.assets}
    steps = {}
    for asset, values in series.items():
        dates = pd.DatetimeIndex(pd.to_datetime(values.index))
        gaps = dates.to_series().diff().dt.total_seconds().to_numpy() / 86400
        limit = max(10 * float(np.nanmedian(gaps)), 5.)
        # Published M0 consumes return rows directly, unlike the agent's log1p steps.
        increments = values.diff() if target_type == 'level' else values.copy()
        increments = increments.copy()
        increments.iloc[0] = np.nan
        increments.iloc[np.flatnonzero(gaps > limit)] = np.nan
        steps[asset] = increments
    aligned = pd.DataFrame(steps).dropna()
    if len(aligned) < 2:
        raise ValueError('Insufficient date-aligned M0 history')
    mu = aligned.mean().to_numpy()
    sigma = np.atleast_2d(np.cov(aligned.to_numpy(), rowvar=False, ddof=1))
    horizons = np.asarray(grid.horizons, dtype=float)
    if (horizons <= 0).any():
        raise ValueError('Invalid daily horizon')
    if panel_steps is None:
        counts = np.tile(horizons, len(grid.assets))
    else:
        monthly_counts = np.asarray(panel_steps, dtype=float)
        if (target_type != 'level' or monthly_counts.shape != (len(grid.assets), len(grid.horizons))
                or not np.isfinite(monthly_counts).all() or (monthly_counts <= 0).any()
                or (monthly_counts != np.floor(monthly_counts)).any()):
            raise ValueError('Invalid explicit monthly normalization steps')
        counts = monthly_counts.reshape(-1)
    indices = np.repeat(np.arange(len(grid.assets)), len(horizons))
    anchors = np.zeros(len(grid.assets)) if target_type == 'log_return' else np.array(
        [series[a].iloc[-1] for a in grid.assets])
    mean = anchors[indices] + counts * mu[indices]
    covariance = np.minimum.outer(counts, counts) * sigma[np.ix_(indices, indices)]
    covariance += np.eye(len(counts)) * 1.1e-9
    try:
        factor = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError:
        factor = np.linalg.cholesky(np.diag(np.diag(covariance)))
    return mean, factor


def estimate(panels, grid, asof, card, *, seed, outcomes=256, panel_steps=None):
    """Average existing component scores against independent baseline-drawn outcomes.

    The baseline has 500 draws, as specified for M0. Finite ensembles, simulated
    outcomes and the caller's step counts make this an approximation to exact scales.
    Error bars cover outcome simulation only, conditional on that forecast ensemble.
    Real holdout outcomes never enter this function.
    """
    if (card['targets'].get('target_frequency') == 'monthly' and panel_steps is None) or outcomes < 2:
        raise ValueError('Require explicit monthly steps and at least two simulated outcomes')
    mean, factor = distribution(panels, grid, asof, card['targets']['target_type'], panel_steps=panel_steps)
    rng = np.random.default_rng(seed)
    samples = mean + rng.standard_normal((500, len(mean))) @ factor.T
    synthetic = mean + rng.standard_normal((outcomes, len(mean))) @ factor.T
    context = {'expected_grid': grid, 'card': card, 'unit_handle': card['task']['id'],
               'grid_source': 'card', 'normalization_mode': NormalizationMode.RAW_UNRANKABLE,
               'ref_scale': None, '_samples': samples}
    scored = [_score({**context, 'realized': y}) for y in synthetic]
    components = ('marginal', 'joint', 'tail')
    averages = {k: float(np.mean([s[k] for s in scored])) for k in components}
    errors = {k: float(np.std([s[k] for s in scored], ddof=1) / np.sqrt(outcomes))
              for k in components}
    absent = grid.cell_count == 1 and averages['joint'] == 0
    scale = RefScale(marginal=averages['marginal'],
                     joint=None if absent else averages['joint'], tail=averages['tail'])
    return scale, {'method': 'm0_monthly_monte_carlo_proxy' if panel_steps is not None else 'm0_daily_monte_carlo_proxy', 'seed': seed,
                   'outcomes': outcomes, 'baseline_draws': 500,
                   'component_means': averages, 'conditional_standard_errors': errors}
