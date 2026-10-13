"""## Executive summary (read this first)

Experimental daily level forecasts move towards a fitted local equilibrium.
For example, a level above equilibrium moves partway back on each future day.
An AR(1) model predicts a level from the preceding business day's level.
Persistence is shrunk halfway towards a random walk to limit overfitting.
This candidate is for offline comparison. The submission default is unchanged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from qfbench2_track_forecasting.cli import _draw, _series


def draw_mean_reversion(panels, assets, horizons, asof, n_draws, seed, *,
                        target_type='level', centre_only=False):
    """Fit at most 120 adjacent business-day pairs using only pre-cutoff rows.

    Whole residual vectors estimate cross-asset innovation covariance. Daily
    paths preserve dependence across horizons. Nonstationary fits and constant
    series revert to zero-drift random walks. No event labels select parameters.
    """
    if target_type != 'level':
        raise ValueError('Mean reversion requires daily level targets')
    if n_draws < 200 or not horizons or any(h <= 0 or int(h) != h for h in horizons):
        raise ValueError('Require >=200 draws and positive integer horizons')
    histories = {a: _series(panels, a, asof) for a in assets}
    wide = pd.DataFrame(histories).dropna().tail(121)
    dates = pd.DatetimeIndex(pd.to_datetime(wide.index))
    adjacent = dates[1:] == dates[:-1] + pd.offsets.BDay(1)
    x = wide.to_numpy()[:-1][adjacent]
    y = wide.to_numpy()[1:][adjacent]
    if len(x) < 30:
        raise ValueError('Need 30 aligned adjacent business-day pairs')
    xm, ym = x.mean(axis=0), y.mean(axis=0)
    variance = ((x-xm)**2).sum(axis=0)
    raw_phi = np.divide(((x-xm)*(y-ym)).sum(axis=0), variance,
                        out=np.ones(len(assets)), where=variance > 0)
    fitted_phi = np.clip(raw_phi, 0., 1.)
    phi = .5 + .5*fitted_phi
    intercept = .5*(ym-fitted_phi*xm)
    intercept[phi == 1.] = 0.
    residual = y - intercept - phi*x
    residual -= residual.mean(axis=0)
    covariance = residual.T @ residual / (len(x)-1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    # Zero eigenvalues remain zero: no artificial shocks on constant series.
    root = eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0.)))
    anchor = np.array([histories[a].iloc[-1] for a in assets])
    centre = anchor.copy()
    centres = {}
    for day in range(1, max(horizons)+1):
        centre = intercept + phi*centre
        if day in horizons:
            centres[day] = centre.copy()
    if centre_only:
        draws, _ = _draw(panels, assets, horizons, asof, n_draws, seed)
        for hi, h in enumerate(horizons):
            draws[:, :, hi] += centres[h] - anchor
    else:
        rng = np.random.default_rng(seed)
        path = np.broadcast_to(anchor, (n_draws, len(assets))).copy()
        draws = np.empty((n_draws, len(assets), len(horizons)))
        for day in range(1, max(horizons)+1):
            path = intercept + phi*path + rng.standard_normal(path.shape) @ root.T
            for hi, h in enumerate(horizons):
                if day == h:
                    draws[:, :, hi] = path
    return draws, {
        'model': 'shrunk-local-ar1', 'centre_only': centre_only,
        'pair_rows': len(x), 'last_pair_date': str(dates[1:][adjacent][-1].date()),
        'phi': dict(zip(assets, phi.tolist())),
        'raw_phi': dict(zip(assets, raw_phi.tolist())),
        'horizon_centres': {a: {str(h): float(centres[h][ai]) for h in horizons}
                            for ai, a in enumerate(assets)},
    }
