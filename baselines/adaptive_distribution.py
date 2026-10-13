"""## Executive summary (read this first)

Forecast daily paths with changing conditional variance and optional heavy tails.
For example, recent large shocks increase near-term uncertainty, which then returns
towards the historical variance. A shared Student-t scale preserves joint paths.
Coefficients are fixed research assumptions, not fitted GARCH parameters. No scores
or future outcomes enter this model. Monthly forecasts use the existing sampler.
"""
from __future__ import annotations

import numpy as np

MODELS = ('gaussian', 'adaptive-gaussian', 'student-t', 'adaptive-t', 'historical-t')
ALPHA = .06
BETA = .92
DEGREES_OF_FREEDOM = 7
CORRELATION_SHRINKAGE = .10


def draw_paths(steps, last, drift, horizons, n_draws, rng, *, model):
    """Return joint draws and auditable conditional-variance parameters.

    A variance-mixture path draws one positive scale for each entire forecast.
    Sharing that scale across assets and horizons models persistent risk uncertainty.
    It is different from independently drawing a Student-t shock every day.
    The scale has mean squared value one, preserving modeled covariance.
    """
    if model not in MODELS[1:]:
        raise ValueError('Unsupported adaptive distribution')
    values = np.asarray(steps, dtype=float)
    if values.ndim != 2 or len(values) < 30 or not np.isfinite(values).all():
        raise ValueError('Require at least 30 finite aligned daily changes')
    if not horizons or any(h <= 0 or int(h) != h for h in horizons):
        raise ValueError('Daily horizons must be positive integers')
    residual = values - values.mean(axis=0)
    long_variance = values.var(axis=0, ddof=1)
    adaptive = model.startswith('adaptive-')
    heavy = model in ('student-t', 'adaptive-t', 'historical-t')
    degrees = DEGREES_OF_FREEDOM
    variance = long_variance.copy()
    if adaptive:
        for shock in residual:
            variance = (1-ALPHA-BETA)*long_variance + ALPHA*shock**2 + BETA*variance
    safe_sd = np.sqrt(long_variance)
    standardized = np.divide(residual, safe_sd, out=np.zeros_like(residual), where=safe_sd > 0)
    excess = None
    if model == 'historical-t':
        # Estimate a pooled shape from nonconstant, standardized historical assets.
        # Half shrinkage towards Gaussian reduces short-window moment noise.
        active = safe_sd > 0
        excess = max(float(np.median(np.mean(standardized[:, active]**4, axis=0)))-3., 0.) if active.any() else 0.
        if excess <= .25:
            heavy = False
        else:
            degrees = float(np.clip(4+12/excess, 5., 60.))
    corr = standardized.T @ standardized / (len(values)-1)
    np.fill_diagonal(corr, 1.)
    if adaptive:
        corr = (1-CORRELATION_SHRINKAGE)*corr + CORRELATION_SHRINKAGE*np.eye(values.shape[1])
    eigenvalues, vectors = np.linalg.eigh(corr)
    corr = (vectors*np.maximum(eigenvalues, 1e-10)) @ vectors.T
    diagonal = np.sqrt(np.diag(corr))
    corr = corr / np.outer(diagonal, diagonal)
    factor = np.linalg.cholesky(corr)
    # Unit-variance Student-t: E[(df-2)/chi_squared(df)] equals one.
    scale = np.sqrt((degrees-2)/rng.chisquare(degrees, n_draws))[:, None] if heavy else np.ones((n_draws, 1))
    path = np.zeros((n_draws, values.shape[1]))
    draws = np.empty((n_draws, values.shape[1], len(horizons)))
    cumulative = {}
    previous = np.zeros(values.shape[1])
    persistence = ALPHA+BETA
    for index in np.argsort(horizons):
        h = int(horizons[index])
        total = long_variance*h + (variance-long_variance)*(1-persistence**h)/(1-persistence)
        total = np.maximum(total, 0.)
        increment = np.maximum(total-previous, 0.)
        path += (rng.standard_normal(path.shape) @ factor.T)*np.sqrt(increment)*scale
        draws[:, :, index] = np.asarray(last) + np.asarray(drift)*h + path
        cumulative[str(h)] = total.tolist()
        previous = total
    if not np.isfinite(draws).all():
        raise ValueError('Non-finite adaptive forecast')
    return draws, {
        'model': model, 'alpha': ALPHA if adaptive else None,
        'beta': BETA if adaptive else None, 'coefficients_fitted': False,
        'degrees_of_freedom': degrees if heavy else None,
        'historical_excess_kurtosis': excess,
        'tail_shape_estimated': model == 'historical-t',
        'shared_scale_per_path': heavy,
        'correlation_shrinkage': CORRELATION_SHRINKAGE if adaptive else 0.,
        'long_run_variance': long_variance.tolist(),
        'next_step_variance': variance.tolist(),
        'cumulative_variance': cumulative,
    }
