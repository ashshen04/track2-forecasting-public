"""## Executive summary (read this first)
Run numerical historical holdouts across public units using the example baseline.
Example: python scripts/backtest.py --out /private/tmp/t2-backtest
Only earlier panel rows reach the model. Text is never used. Monthly tasks are
skipped because the scaffold lacks monthly mapping and release-vintage handling.
Scores are raw diagnostics, not official scores. Outputs must stay outside this repo.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tomllib
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baselines.base import ForecastRequest  # noqa: E402
from baselines.theta_arima import ThetaARIMABaseline  # noqa: E402
from qfbench2_track_forecasting.cli import _draw  # noqa: E402
from qfbench2_track_forecasting.grid import grid_from_card  # noqa: E402
from qfbench2_track_forecasting.normalization import NormalizationMode  # noqa: E402
from qfbench2_track_forecasting.scoring import _score  # noqa: E402
from qfbench2_track_forecasting.targets import log_return_steps  # noqa: E402


class SkipUnit(ValueError):
    """The available inputs cannot support this historical experiment."""


def forecast_samples(name, panels, origin, grid, n_draws, target_type):
    """Both implementations receive identical, physically truncated inputs."""
    history = {k: f.loc[f.date <= origin].copy() for k, f in panels.items()}
    assets, horizons = list(grid.assets), list(grid.horizons)
    asof = str(origin.date())
    if name == 'scaffold':
        request = ForecastRequest(history, asof, assets, horizons, n_draws, target_type)
        model = ThetaARIMABaseline()
        result = model.forecast(request)
        model.validate_output(result, request)
        samples = result.samples
    elif name in ('reference', 'reference-shuffled'):
        try:
            samples, _ = _draw(history, assets, horizons, asof, n_draws, 1,
                               target_type=target_type)
        except SystemExit as exc:
            raise ValueError(f'Reference forecaster refused: {exc}') from exc
        if name == 'reference-shuffled':
            # Preserve every marginal draw and each asset's cross-horizon path.
            # Only the pairing of different assets within scenarios is destroyed.
            rng = np.random.default_rng(9173)
            samples = samples.copy()
            for ai in range(1, len(assets)):
                samples[:, ai, :] = samples[rng.permutation(n_draws), ai, :]
    else:
        raise ValueError(f'Unknown model: {name}')
    if samples.shape != (n_draws, len(assets), len(horizons)) or not np.isfinite(samples).all():
        raise ValueError('Invalid forecast shape or non-finite draws')
    return samples.reshape(n_draws, -1)


def outside_repo(path: Path) -> Path:
    """Resolve symlinks before enforcing the public/private firewall."""
    path = path.expanduser().resolve()
    if path == ROOT or ROOT in path.parents:
        raise ValueError('Backtest outputs must be outside the public repository.')
    return path


def holdout(wide, origin, horizons, target_type):
    """Resolve exact weekday endpoints; never fill missing historical targets."""
    ends = [origin + pd.offsets.BDay(h) for h in horizons]
    if any(end not in wide.index for end in ends):
        raise SkipUnit('Missing exact business-day target date')
    columns = []
    for end in ends:
        if target_type == 'level':
            columns.append(wide.loc[end].to_numpy(dtype=float))
        else:
            dates = pd.bdate_range(origin + pd.offsets.BDay(1), end)
            future = wide.reindex(dates).to_numpy(dtype=float)
            if not np.isfinite(future).all():
                raise SkipUnit('Missing daily returns inside a forecast interval')
            columns.append(log_return_steps(future).sum(axis=0))
    values = np.stack(columns, axis=1)
    if not np.isfinite(values).all():
        raise SkipUnit('Missing or non-finite targets')
    return values.reshape(-1), [str(end.date()) for end in ends]


def run_unit(unit, args):
    card = tomllib.loads((unit / 'card.toml').read_text())
    targets = card['targets']
    grid = grid_from_card(card)
    target_type = targets['target_type']
    if targets.get('target_frequency') == 'monthly':
        raise SkipUnit('Monthly scaffold mapping and historical release vintages unsupported')
    if target_type not in ('level', 'log_return'):
        raise SkipUnit(f'Unsupported target type: {target_type}')
    cutoff = pd.Timestamp(card['provenance']['data_cutoff'])
    paths = sorted(unit.glob('*.parquet')) + sorted((unit / 'panels').glob('*.parquet'))
    if not paths:
        raise SkipUnit('No input panels')
    panels, hashes = {}, {}
    for path in paths:
        frame = pd.read_parquet(path)
        frame = frame.rename(columns={'asset_id': 'asset'})
        frame['date'] = pd.to_datetime(frame['date'])
        frame = frame.loc[frame['date'] <= cutoff].copy()
        panels[str(path.relative_to(unit))] = frame
        hashes[str(path.relative_to(unit))] = hashlib.sha256(path.read_bytes()).hexdigest()
    selected = []
    for asset in grid.assets:
        sources = [f.loc[f.asset == asset, ['date', 'value']] for f in panels.values()]
        sources = [f for f in sources if not f.empty]
        if len(sources) != 1:
            raise SkipUnit(f'Missing or ambiguous panel for {asset}')
        series = sources[0].set_index('date')['value'].sort_index()
        if series.index.has_duplicates:
            raise SkipUnit(f'Duplicate observations for {asset}')
        if len(series) < 2 or np.median(np.diff(series.index).astype('timedelta64[D]').astype(float)) > 3:
            raise SkipUnit(f'Target series is not daily: {asset}')
        selected.append(series.rename(asset))
    wide = pd.concat(selected, axis=1, sort=True).sort_index()
    complete = wide.dropna()
    if not np.isfinite(complete.to_numpy()).all():
        raise ValueError('Non-finite panel values')
    max_h = max(grid.horizons)
    candidates = complete.index[args.min_history - 1:]
    folds, rejected = [], Counter()
    next_end = None
    # Latest feasible, non-overlapping holdouts; each model still trains forwards.
    for origin in reversed(candidates):
        end = origin + pd.offsets.BDay(max_h)
        if end > cutoff or (next_end is not None and end > next_end):
            continue
        try:
            truth, dates = holdout(wide, origin, grid.horizons, target_type)
        except SkipUnit as exc:
            rejected[str(exc)] += 1
            continue
        model_scores = {}
        for name in args.models:
            samples = forecast_samples(name, panels, origin, grid, args.n_draws, target_type)
            model_scores[name] = _score({'realized': truth, '_samples': samples, 'expected_grid': grid,
                         'card': card, 'unit_handle': unit.name, 'grid_source': 'card',
                         'normalization_mode': NormalizationMode.RAW_UNRANKABLE,
                         'ref_scale': None})
        folds.append({'asof': str(origin.date()), 'target_dates': dates,
                      'history_dates': len(complete.loc[:origin]),
                      'scores': model_scores[args.models[0]], 'model_scores': model_scores})
        next_end = origin
        if len(folds) == args.folds:
            break
    if not folds:
        raise SkipUnit(f'No valid holdout with {args.min_history} history dates: {dict(rejected)}')
    manifest = (unit / 'manifest.json').read_text() if (unit / 'manifest.json').exists() else ''
    warnings = ['Fixed-snapshot retrospective experiment; historical publication vintages not verified.']
    if 'unverified' in manifest.lower() or 'synthetic' in manifest.lower():
        warnings.append('Manifest flags synthetic or unverified provenance; not evidence of market skill.')
    return {'status': 'completed', 'unit': unit.name, 'target_type': target_type,
            'assets': list(grid.assets), 'horizons': list(grid.horizons),
            'source_sha256': hashes, 'warnings': warnings, 'rejected_origins': dict(rejected),
            'mean_by_model': {name: {component: float(np.mean([
                f['model_scores'][name][component] for f in folds]))
                for component in ('marginal', 'joint', 'tail', 'composite')}
                for name in args.models},
            'folds': list(reversed(folds)), 'mean_raw_composite': float(np.mean(
                [f['scores']['composite'] for f in folds]))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--units', type=Path, default=ROOT / 'units')
    parser.add_argument('--pattern', default='*', help='Unit name glob, e.g. t2-F1-*')
    parser.add_argument('--out', type=Path, required=True, help='New directory outside this repo')
    parser.add_argument('--folds', type=int, default=3, help='Maximum holdouts per unit')
    parser.add_argument('--min-history', type=int, default=63, help='Minimum complete history dates')
    parser.add_argument('--n-draws', type=int, default=1000)
    parser.add_argument('--models', nargs='+', choices=['scaffold', 'reference', 'reference-shuffled'], default=['scaffold'])
    parser.add_argument('--include', nargs='+', help='Exact unit names; optional fixed experiment subset')
    args = parser.parse_args(argv)
    if args.folds < 1 or args.min_history < 31 or args.n_draws < 200:
        parser.error('Require folds >= 1, min-history >= 31, n-draws >= 200')
    try:
        out = outside_repo(args.out)
    except ValueError as exc:
        parser.error(str(exc))
    units = sorted(p for p in args.units.glob(args.pattern) if (p / 'card.toml').is_file())
    if args.include:
        missing = set(args.include) - {p.name for p in units}
        if missing:
            parser.error(f'Unknown or excluded units: {sorted(missing)}')
        units = [p for p in units if p.name in args.include]
    if not units:
        parser.error('No matching units')
    out.mkdir(parents=True, exist_ok=False)
    results = []
    for unit in units:
        try:
            result = run_unit(unit, args)
        except SkipUnit as exc:
            result = {'unit': unit.name, 'status': 'skipped', 'reason': str(exc)}
        except Exception as exc:
            result = {'unit': unit.name, 'status': 'failed', 'reason': f'{type(exc).__name__}: {exc}'}
        results.append(result)
        (out / f'{unit.name}.json').write_text(json.dumps(result, indent=2) + '\n')
        print(f"[{result['status']}] {unit.name}: " +
              (f"{len(result['folds'])} folds" if result['status'] == 'completed' else result['reason']), flush=True)
    counts = dict(Counter(r['status'] for r in results))
    summary = {'description': 'Raw historical diagnostics, not official scores or admissibility checks.',
               'created_utc': datetime.now(timezone.utc).isoformat(), 'common_version': version('qfbench2-common'),
               'models': args.models, 'seed': 1, 'text_used': False,
               'implementation_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in [Path(__file__), ROOT / 'baselines/base.py', ROOT / 'baselines/theta_arima.py',
                             ROOT / 'qfbench2_track_forecasting/cli.py', ROOT / 'qfbench2_track_forecasting/scoring.py']},
               'calendar': 'Monday-Friday; missing endpoints or daily return rows are rejected',
               'settings': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
               'counts': counts, 'units': results}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    lines = ['## Executive summary (read this first)', '',
             'Numerical retrospective holdouts. Raw scores are not comparable across different units.',
             'No overall score is calculated. Units share history; they are not independent experiments.',
             'Historical vintages are not reconstructed. Monthly tasks are skipped. No text is used.', '',
             str(counts), '', '| Unit | Status | Folds / reason |', '|---|---|---|']
    for r in results:
        detail = str(len(r['folds'])) if r['status'] == 'completed' else r['reason']
        lines.append(f"| {r['unit']} | {r['status']} | {detail.replace('|', '/')} |")
    if set(args.models) == {'scaffold', 'reference'}:
        lines += ['', '## Paired diagnostic comparison', '',
                  'Same origins and inputs. These models differ in drift and dependence; this is not a correlation ablation.',
                  'Negative reference-minus-scaffold means reference is better. Do not average raw deltas across units.', '',
                  '| Unit | Scaffold | Reference | Difference |', '|---|---:|---:|---:|']
        for r in results:
            if r['status'] != 'completed':
                continue
            a = r['mean_by_model']['scaffold']['composite']
            b = r['mean_by_model']['reference']['composite']
            lines.append(f"| {r['unit']} | {a:.6g} | {b:.6g} | {b-a:+.6g} |")
    if set(args.models) == {'reference', 'reference-shuffled'}:
        lines += ['', '## Asset dependence experiment', '',
                  'Reference draws versus the same draws with asset pairing shuffled (permutation seed 9173).',
                  'Each marginal and each within-asset horizon path are preserved exactly.',
                  'Negative reference-minus-shuffled favours original dependence. This is a fixed-seed diagnostic, not a significance test.', '',
                  '| Unit | Reference | Shuffled | Reference minus shuffled |', '|---|---:|---:|---:|']
        for r in results:
            if r['status'] != 'completed':
                continue
            a = r['mean_by_model']['reference']['composite']
            b = r['mean_by_model']['reference-shuffled']['composite']
            lines.append(f"| {r['unit']} | {a:.6g} | {b:.6g} | {a-b:+.6g} |")
    (out / 'report.md').write_text('\n'.join(lines) + '\n')
    print(f'Results: {out}\n{counts}')
    return 1 if counts.get('failed') or not counts.get('completed') else 0


if __name__ == '__main__':
    raise SystemExit(main())
