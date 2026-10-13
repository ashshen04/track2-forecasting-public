"""## Executive summary (read this first)

Freeze real-time CPI research windows, select a fixed monthly trend on development,
then compare only that locked candidate with no trend on a later holdout.
Example: --plan --source-zip /tmp/cpi.zip --source-request /tmp/source.json --out /tmp/cpi-check.
Use --development and --holdout as separate subsequent commands.
ALFRED vintage dates govern training; first-release and fixed-snapshot targets stay distinct.
Every metric uses the existing scorer. Normalization is an input-only Monte Carlo proxy.
All observations, scores and experiment artifacts stay outside this public repository.
"""
from __future__ import annotations
import argparse
import copy
import io
import json
import sys
import tomllib
import zipfile
import zlib
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baselines.monthly_trend import POLICIES, centre_shifts
from qfbench2_track_forecasting.cli import _draw
from qfbench2_track_forecasting.grid import grid_from_card
from qfbench2_track_forecasting.horizons import monthly_horizon_steps
from qfbench2_track_forecasting.normalization import NormalizationMode
from qfbench2_track_forecasting.scoring import _score
from scripts.backtest import outside_repo, overlaps_excluded
from scripts.m0_normalization_proxy import estimate
from scripts.validate_return_calibration import prior_intervals, reserve
from scripts.validate_stability import atomic_json, digest
from scripts.validate_text_scale import aggregate

SEEDS = (1, 7, 19)
NORMALIZATION_SEEDS = (41, 43)
GRIDS = (('t2-F1-cpi-glidepath-2023', (8, 9), 'fixed_snapshot'),
         ('t2-F4-cpi-vintage-2022', (2,), 'first_release'))
SNAPSHOT = '2026-09-11'
BOUNDARY = '2020-01-01'


def read_vintages(path):
    with zipfile.ZipFile(path) as z:
        csvs = [f for f in z.infolist() if f.filename.endswith('.csv')]
        if len(csvs) != 1 or csvs[0].file_size > 10_000_000:
            raise ValueError('Require one bounded ALFRED CSV')
        frame = pd.read_csv(io.BytesIO(z.read(csvs[0])), dtype=str)
    required = ['period_start_date', 'CPIAUCSL', 'realtime_start_date', 'realtime_end_date']
    if list(frame.columns) != required:
        raise ValueError('Require CPIAUCSL real-time-period export')
    frame = frame.rename(columns={'period_start_date': 'date', 'CPIAUCSL': 'value',
                                  'realtime_start_date': 'start', 'realtime_end_date': 'end'})
    for key in ('date', 'start', 'end'):
        frame[key] = pd.to_datetime(frame[key], errors='raise')
    frame['value'] = pd.to_numeric(frame.value, errors='raise')
    if (frame.duplicated(['date', 'start']).any() or frame.date.isna().any()
            or frame.start.isna().any() or (frame.end.notna() & (frame.end < frame.start)).any()
            or (frame.date != frame.date.dt.to_period('M').dt.to_timestamp()).any()
            or np.isinf(frame.value).any() or (frame.value.dropna() <= 0).any()):
        raise ValueError('Invalid vintage dates, duplicate records or CPI levels')
    frame = frame.sort_values(['date', 'start']).reset_index(drop=True)
    for _, group in frame.groupby('date'):
        if len(group) > 1:
            ends = group.end.iloc[:-1]
            if ends.isna().any() or (ends.to_numpy() >= group.start.iloc[1:].to_numpy()).any():
                raise ValueError('Overlapping vintage validity intervals')
    return frame


def snapshot(frame, asof, *, lag_days=0):
    """Select only revisions valid at asof and observations allowed by the panel lag."""
    origin = pd.Timestamp(asof)
    rows = frame.loc[(frame.start <= origin) & (frame.end.isna() | (frame.end >= origin))
                     & (frame.date <= origin - pd.Timedelta(days=lag_days))].copy()
    if rows.date.duplicated().any():
        raise ValueError('Ambiguous vintage at origin')
    return rows[['date', 'value']].assign(asset='CPI_ALL').sort_values('date').reset_index(drop=True)


def target_row(frame, month, mode):
    if mode not in ('first_release', 'fixed_snapshot'):
        raise ValueError('Unknown CPI target vintage mode')
    selected = frame.loc[frame.date == pd.Period(month, freq='M').start_time].sort_values('start')
    if selected.empty:
        raise ValueError('Target month absent')
    first = selected.iloc[0]
    if not np.isfinite(first.value):
        raise ValueError('Initial release value missing')
    if mode == 'first_release':
        return float(first.value), str(first.start.date())
    current = snapshot(frame, SNAPSHOT)
    value = current.loc[current.date == first.date, 'value']
    if len(value) != 1 or not np.isfinite(value.iloc[0]):
        raise ValueError('Fixed snapshot value missing')
    return float(value.iloc[0]), str(first.start.date())


def research_card(item):
    card = copy.deepcopy(tomllib.loads((ROOT / 'units' / item['unit'] / 'card.toml').read_text()))
    card['provenance']['data_cutoff'] = item['asof']
    card['targets']['observation_periods'] = item['observation_periods']
    return card


def plan(out, source_zip, source_request, prior_paths):
    frame = read_vintages(source_zip)
    request = json.loads(source_request.read_text())
    if request['format'] != 'Observations by Real-Time Period' or request['vintage_dates'][-1] != SNAPSHOT:
        raise ValueError('Source request does not match the predeclared vintage snapshot')
    intervals, sources = prior_intervals(prior_paths)
    selected = []; rejected = Counter(); next_grid = 0
    for origin in pd.date_range('2010-01-31', '2026-08-31', freq='ME'):
        unit, steps, mode = GRIDS[next_grid]
        training = snapshot(frame, origin, lag_days=45)
        if len(training) < 121:
            rejected['less_than_120_training_changes'] += 1; continue
        last = pd.Period(training.date.iloc[-1], freq='M')
        periods = [str(last + s) for s in steps]
        try:
            _, info = centre_shifts({'CPI_ALL': training.set_index('date').value}, [steps], 'recent')
            releases = [target_row(frame, month, mode)[1] for month in periods]
            if any(pd.Timestamp(date) <= origin for date in releases):
                raise ValueError('Target already released at origin')
            end = max(pd.Timestamp(date) for date in releases)
            if end > pd.Timestamp(SNAPSHOT):
                raise ValueError('Target release after source snapshot')
            if overlaps_excluded(origin, end, ['CPI_ALL'], intervals):
                rejected['shared_asset_overlap'] += 1; continue
        except ValueError as exc:
            rejected[str(exc)] += 1; continue
        card_path = ROOT / 'units' / unit / 'card.toml'
        authored = tomllib.loads(card_path.read_text())
        split = 'development' if end < pd.Timestamp(BOUNDARY) else 'holdout' if origin >= pd.Timestamp(BOUNDARY) else 'purged'
        selected.append({'unit': unit, 'asof': str(origin.date()), 'assets': ['CPI_ALL'],
            'horizons': authored['targets']['horizons'], 'panel_steps': [list(steps)],
            'observation_periods': periods, 'target_dates': releases, 'target_mode': mode,
            'training_last_observation': str(training.date.iloc[-1].date()), 'split': split})
        reserve(intervals, ['CPI_ALL'], origin, end)
        next_grid = (next_grid + 1) % len(GRIDS)
    files = ['scripts/validate_monthly_cpi.py', 'baselines/monthly_trend.py',
             'scripts/m0_normalization_proxy.py', 'scripts/validate_text_scale.py',
             'qfbench2_track_forecasting/cli.py', 'qfbench2_track_forecasting/scoring.py',
             'qfbench2_track_forecasting/horizons.py']
    manifest = {'executive_summary': 'Frozen CPI research grids with real-time training; separate development and holdout.',
        'source_zip': str(source_zip.resolve()), 'source_request': str(source_request.resolve()),
        'source_hashes': {str(p.resolve()): digest(p) for p in [source_zip, source_request]},
        'prior_sources': sources, 'implementation_sha256': {p: digest(ROOT / p) for p in files},
        'card_sha256': {unit: digest(ROOT / 'units' / unit / 'card.toml') for unit, _, _ in GRIDS},
        'policies': POLICIES, 'window_months': 12, 'damping': .9, 'seeds': SEEDS,
        'normalization_seeds': NORMALIZATION_SEEDS, 'normalization_outcomes': 256,
        'n_draws': 1000, 'publication_lag_days': 45, 'fixed_snapshot': SNAPSHOT,
        'boundary': BOUNDARY, 'selection': 'Chronological month ends; alternate long and short authored grids; globally disjoint CPI forecast intervals.',
        'acceptance': {'minimum_development_intervals': 8, 'minimum_holdout_intervals': 6,
            'minimum_holdout_years': 3, 'improved_mean_at_every_draw_and_normalization_seed': True,
            'component_regression_is_diagnostic_not_automatic_veto': True},
        'common_version': version('qfbench2-common'), 'selected': selected,
        'counts': dict(Counter(r['split'] for r in selected)), 'rejected': dict(rejected)}
    if not selected:
        raise ValueError('No feasible CPI research intervals')
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(out / 'cohort.json', manifest)
    print(f"Frozen cohort: {manifest['counts']}; rejected: {dict(rejected)}")


def check_frozen(manifest):
    for path, value in manifest['source_hashes'].items():
        if digest(Path(path)) != value: raise ValueError('Frozen source changed')
    for path, value in manifest['implementation_sha256'].items():
        if digest(ROOT / path) != value: raise ValueError('Frozen implementation changed')
    for unit, value in manifest['card_sha256'].items():
        if digest(ROOT / 'units' / unit / 'card.toml') != value: raise ValueError('Frozen card changed')
    for path, info in manifest['prior_sources'].items():
        if digest(Path(path)) != info['sha256']: raise ValueError('Prior inspection ledger changed')
    if version('qfbench2-common') != manifest['common_version']:
        raise ValueError('Shared scoring toolkit changed')


def score_split(manifest, split, policies):
    frame = read_vintages(Path(manifest['source_zip'])); rows = []; normalizations = []
    for item in manifest['selected']:
        if item['split'] != split: continue
        card = research_card(item); grid = grid_from_card(card)
        panel = snapshot(frame, item['asof'], lag_days=manifest['publication_lag_days'])
        panel_steps = monthly_horizon_steps(grid.assets, grid.horizons,
            {'CPI_ALL': str(panel.date.iloc[-1].date())}, asof=item['asof'], card=card)
        if not np.array_equal(panel_steps, item['panel_steps']):
            raise ValueError('Frozen monthly mapping changed')
        # The first release and fixed-snapshot targets are extracted only in this scoring phase.
        truth = np.array([target_row(frame, month, item['target_mode'])[0] for month in item['observation_periods']])
        scales = {}
        for norm_seed in manifest['normalization_seeds']:
            seed = (zlib.crc32((item['unit'] + item['asof']).encode()) + norm_seed) & 0xFFFFFFFF
            scales[norm_seed], audit = estimate({'p': panel}, grid, item['asof'], card,
                seed=seed, outcomes=manifest['normalization_outcomes'], panel_steps=panel_steps)
            normalizations.append({**item, 'normalization_seed': norm_seed, 'audit': audit})
        for seed in manifest['seeds']:
            draws = {policy: _draw({'p': panel}, list(grid.assets), list(grid.horizons), item['asof'],
                manifest['n_draws'], seed, panel_steps=panel_steps, monthly_trend=policy)[0]
                for policy in policies}
            for norm_seed, scale in scales.items():
                context = {'expected_grid': grid, 'card': card, 'realized': truth,
                    'unit_handle': item['unit'], 'grid_source': 'card',
                    'normalization_mode': NormalizationMode.REF_SCALE, 'ref_scale': scale}
                for policy, samples in draws.items():
                    score = _score({**context, '_samples': samples.reshape(manifest['n_draws'], -1)})
                    rows.append({**item, 'policy': policy, 'seed': seed, 'normalization_seed': norm_seed,
                        'proxy_loss': score['score'], 'raw_components': {k: score[k] for k in ('marginal', 'joint', 'tail')},
                        'weights_effective': score['weights_effective']})
        print(f"[scored] {split} {item['unit']} {item['asof']}", flush=True)
    comparisons = []
    for seed in SEEDS:
        for norm_seed in NORMALIZATION_SEEDS:
            for policy in policies:
                value = aggregate([r for r in rows if r['seed'] == seed and r['normalization_seed'] == norm_seed and r['policy'] == policy])
                if value is not None: comparisons.append({'policy': policy, 'seed': seed, 'normalization_seed': norm_seed, **value})
    return {'executive_summary': 'Real-time CPI research scores; normalized proxies are not official ranked scores.',
        'split': split, 'model_api_calls': 0, 'intervals': sum(r['split'] == split for r in manifest['selected']),
        'policies': policies, 'comparisons': comparisons, 'results': rows, 'normalizations': normalizations}


def choose_development_policy(summary):
    means = {p: float(np.mean([r['mean_proxy_loss'] for r in summary['comparisons'] if r['policy'] == p])) for p in POLICIES}
    if not all(np.isfinite(v) for v in means.values()):
        raise ValueError('Incomplete development comparisons')
    return min(POLICIES, key=lambda p: (means[p], POLICIES.index(p))), means


def development(out, manifest):
    if (out / 'selection.json').exists(): raise ValueError('Selection already locked')
    summary = score_split(manifest, 'development', list(POLICIES))
    if summary['intervals'] < manifest['acceptance']['minimum_development_intervals']:
        raise ValueError('Insufficient development intervals')
    winner, means = choose_development_policy(summary)
    atomic_json(out / 'development.json', summary)
    selection = {'executive_summary': 'Candidate locked using development mean loss only; holdout not scored.',
        'selected_policy': winner, 'development_mean_proxy_losses': means,
        'cohort_sha256': digest(out / 'cohort.json'), 'development_sha256': digest(out / 'development.json')}
    atomic_json(out / 'selection.json', selection)
    print(f'Locked development candidate: {winner}; means: {means}')


def holdout(out, manifest):
    selection = json.loads((out / 'selection.json').read_text())
    if selection['cohort_sha256'] != digest(out / 'cohort.json') or selection['development_sha256'] != digest(out / 'development.json'):
        raise ValueError('Frozen selection changed')
    winner = selection['selected_policy']
    expected, means = choose_development_policy(json.loads((out / 'development.json').read_text()))
    if winner != expected or selection['development_mean_proxy_losses'] != means:
        raise ValueError('Selection no longer matches development-only decision')
    if (out / 'holdout.json').exists(): raise ValueError('Holdout already scored')
    policies = ['none'] if winner == 'none' else ['none', winner]
    summary = score_split(manifest, 'holdout', policies)
    deltas = []
    for seed in SEEDS:
        for norm_seed in NORMALIZATION_SEEDS:
            scores = {r['policy']: r['mean_proxy_loss'] for r in summary['comparisons'] if r['seed'] == seed and r['normalization_seed'] == norm_seed}
            if scores: deltas.append({'seed': seed, 'normalization_seed': norm_seed,
                'baseline_mean_proxy_loss': scores['none'], 'selected_mean_proxy_loss': scores[winner],
                'relative_change_percent': 100 * (scores[winner] / scores['none'] - 1)})
    years = len({r['asof'][:4] for r in manifest['selected'] if r['split'] == 'holdout'})
    rule = manifest['acceptance']
    accepted = bool(winner != 'none' and summary['intervals'] >= rule['minimum_holdout_intervals']
        and years >= rule['minimum_holdout_years'] and len(deltas) == 6
        and all(r['relative_change_percent'] < 0 for r in deltas))
    summary.update(selected_policy=winner, acceptance_passed=accepted)
    atomic_json(out / 'holdout.json', summary)
    lines = ['## Executive summary (read this first)', '',
        f"Development selected {winner}. Later real-time holdout acceptance: {accepted}. Scoring uses normalized research proxies, not organizer scales. No model calls or submission rebuild.", '',
        '| Seed | Normalization simulation | Baseline mean | Selected mean | Change |', '|---|---|---:|---:|---:|']
    for r in deltas: lines.append(f"| {r['seed']} | {r['normalization_seed']} | {r['baseline_mean_proxy_loss']:.6f} | {r['selected_mean_proxy_loss']:.6f} | {r['relative_change_percent']:+.2f}% |")
    lines += ['', f"Holdout intervals: {summary['intervals']}; origin years: {years}. Selection was locked before holdout scoring. Forecast intervals are globally disjoint by CPI asset; training histories overlap and macro regimes remain dependent.", '',
        'Training uses the vintage valid at origin and a 45-day panel lag. The first-release grid uses earliest archived observations. The current-vintage grid uses the fixed 2026-09-11 evaluation snapshot. Observation months and submission horizon keys stay separate. Research cards copy grid shapes, not official sealed targets.', '',
        'Normalization uses two independent simulations from published M0 inputs with explicit monthly step counts. Finite forecast draws and Monte Carlo uncertainty make these approximate scales. A research pass does not guarantee official leaderboard improvement.']
    (out / 'report.md').write_text('\n'.join(lines) + '\n')
    print(f'Holdout acceptance: {accepted}; deltas: {deltas}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source-zip', type=Path); parser.add_argument('--source-request', type=Path)
    parser.add_argument('--prior-summary', type=Path, action='append', default=[])
    action = parser.add_mutually_exclusive_group(required=True)
    for name in ('plan', 'development', 'holdout'): action.add_argument('--' + name, action='store_true')
    args = parser.parse_args(argv); out = outside_repo(args.out)
    if args.plan:
        if args.source_zip is None or args.source_request is None or not args.prior_summary:
            parser.error('Planning requires the official archive, request metadata and prior summaries')
        plan(out, args.source_zip, args.source_request, args.prior_summary)
    else:
        manifest = json.loads((out / 'cohort.json').read_text()); check_frozen(manifest)
        (development if args.development else holdout)(out, manifest)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
