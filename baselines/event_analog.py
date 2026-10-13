"""## Executive summary (read this first)

Use completed past events to shift a statistical forecast's centre experimentally.
For example, past rate hikes can inform changes after a newly announced hike.
An analog is a historical event of the same declared type. Response means the
change from the next business day's close to a horizon's business-day endpoint.
Only the forecast centre changes. This module does not select or deploy a model.
"""
from __future__ import annotations

import re
import numpy as np
import pandas as pd


def classify_statement(text):
    """Recognize an announced hike, rather than an expectation of a future hike."""
    text = re.sub(r'\s+', ' ', text).lower()
    pattern = (r'committee decided(?: today)? to raise '
               r'(?:its target|the target range) for the federal funds rate\b')
    return 'fomc_hike' if re.search(pattern, text) else None


def analog_adjustment(wide, events, event_type, asof, assets, horizons, horizon_sd, *, shrink=True):
    """Estimate per-horizon mean changes from non-overlapping mature analogs.

    At least three peers are required. Shrinkage uses n/(n+10), fixed before
    evaluation, then caps the shift at half the baseline horizon standard
    deviation. Raw mode is an uncapped diagnostic. Matching never uses outcomes.
    Future rows are physically removed before constructing any peer response.
    No missing endpoint is filled. Horizons retain their authored order.
    """
    asof = pd.Timestamp(asof).normalize()
    sd = np.asarray(horizon_sd, dtype=float)
    if sd.shape != (len(assets), len(horizons)) or not np.isfinite(sd).all() or np.any(sd < 0):
        raise ValueError('Provide one finite nonnegative horizon standard deviation per cell')
    if not horizons or any(int(h)!=h or h<=0 for h in horizons):
        raise ValueError('Provide positive integer business-day horizons')
    history = wide.copy()
    history.index = pd.to_datetime(history.index)
    if history.index.has_duplicates:
        raise ValueError('Historical dates must be unique')
    history = history.loc[history.index <= asof, assets].sort_index()
    dates = sorted({pd.Timestamp(e['date']).normalize() for e in events
                    if e['type']==event_type and pd.Timestamp(e['date']).normalize()<asof}, reverse=True)
    shifts = np.zeros_like(sd)
    audit = {}
    for hi,h in enumerate(horizons):
        peers, responses = [], []
        next_origin = None
        for date in dates:
            origin = date + pd.offsets.BDay(1)
            end = origin + pd.offsets.BDay(h)
            if end > asof or (next_origin is not None and end > next_origin):
                continue
            if origin not in history.index or end not in history.index:
                continue
            delta = history.loc[end].to_numpy()-history.loc[origin].to_numpy()
            if not np.isfinite(delta).all():
                continue
            responses.append(delta)
            peers.append({'date':str(date.date()),'origin':str(origin.date()),'end':str(end.date())})
            next_origin = origin
        n = len(peers)
        weight = n/(n+10) if shrink else 1.
        if n >= 3:
            shift = np.mean(responses, axis=0)*weight
            shifts[:,hi] = np.clip(shift, -.5*sd[:,hi], .5*sd[:,hi]) if shrink else shift
        audit[str(h)] = {'count':n,'status':'adjusted' if n>=3 else 'insufficient_peers',
                         'weight':weight if n>=3 else 0.,'peers':peers}
    return shifts, audit
