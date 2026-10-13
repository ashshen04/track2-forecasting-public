"""## Executive summary (read this first)

Experimental monthly CPI trends move forecast centres without changing random shocks.
For example, a mean monthly increase of one moves a two-month centre by two.
Half trend moves it by one. Damped trend reduces each later monthly increment.
The fixed window is twelve monthly changes. The damping factor is 0.9.
Use explicit observation-period step counts, never the submission horizon integers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

POLICIES = ('none', 'recent', 'half', 'damped')
WINDOW = 12
DAMPING = .9


def validate_cpi_card(card, policy):
    """Limit formal opt-in to explicitly declared monthly CPI index targets."""
    if policy not in POLICIES:
        raise ValueError('Unknown monthly trend policy')
    if policy == 'none':
        return
    targets = card['targets']
    unit = targets.get('value_unit', '')
    if (targets.get('target_type') != 'level' or targets.get('target_frequency') != 'monthly'
            or not isinstance(unit, str) or not unit.startswith('cpi_index_')):
        raise ValueError('Monthly trend opt-in requires a declared monthly CPI index level target')


def resolve_cpi_policy(card, requested):
    """Apply the accepted CPI-only default; explicit none retains the old control."""
    if requested == 'auto':
        try:
            validate_cpi_card(card, 'damped')
        except ValueError:
            return 'none'
        return 'damped'
    validate_cpi_card(card, requested)
    return requested


def centre_shifts(hist, panel_steps, policy):
    """Fit only the supplied earlier history and use a consecutive monthly window."""
    if policy not in POLICIES:
        raise ValueError('Unknown monthly trend policy')
    counts = np.asarray(panel_steps)
    if (counts.ndim != 2 or counts.shape[0] != len(hist) or not np.isfinite(counts).all()
            or (counts <= 0).any() or (counts != np.floor(counts)).any()):
        raise ValueError('Provide positive integer monthly steps per asset and horizon')
    shifts = np.zeros(counts.shape, dtype=float)
    means = {}
    if policy != 'none':
        for ai, (asset, values) in enumerate(hist.items()):
            recent = values.tail(WINDOW + 1)
            periods = pd.PeriodIndex(recent.index, freq='M').asi8
            if len(recent) != WINDOW + 1 or not np.all(np.diff(periods) == 1):
                raise ValueError('Monthly trend requires thirteen consecutive monthly observations')
            if not np.isfinite(recent.to_numpy()).all():
                raise ValueError('Non-finite monthly trend history')
            mean = float(recent.diff().iloc[1:].mean())
            means[asset] = mean
            multiplier = .5 if policy == 'half' else 1.
            for hi, count in enumerate(counts[ai]):
                count = int(count)
                accumulated = float(np.sum(DAMPING ** np.arange(count))) if policy == 'damped' else count
                shifts[ai, hi] = multiplier * mean * accumulated
    return shifts, {'policy': policy, 'window_months': WINDOW,
                    'damping': DAMPING if policy == 'damped' else None,
                    'mean_monthly_changes': means}
