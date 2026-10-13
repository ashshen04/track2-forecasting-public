"""## Executive summary (read this first)

Replay previously inspected numerical forecasts to locate calibration and direction errors.
Example: python scripts/diagnose_errors.py --summaries /tmp/run/summary.json --out /tmp/diagnosis
All error records and reports stay outside the public repository. No model is selected here.
Calibration means agreement between forecast probabilities and observed frequencies.
A standardized residual is the error divided by the forecast's theoretical standard deviation.
These diagnostics are not competition scoring and cannot rank unlike cards by raw scores.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tomllib
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qfbench2_track_forecasting.cli import _draw, _series, _diff_without_gaps
from qfbench2_track_forecasting.grid import grid_from_card
from qfbench2_track_forecasting.normalization import NormalizationMode
from qfbench2_track_forecasting.scoring import _score
from qfbench2_track_forecasting.targets import log_return_steps
from scripts.backtest import COMPONENTS, calibration_validation, holdout, outside_repo


def cell_diagnostics(samples, truth, centre, anchor, sd, trend):
    """Describe each target cell; neutral forecasts abstain from direction prediction.

    PIT (probability integral transform) is the forecast probability below the observed
    target. A calibrated continuous forecast has uniform PIT values over many events.
    trend is a historical-mean direction probe, not an adopted forecasting policy.
    """
    samples = np.asarray(samples, dtype=float)
    vectors = [np.asarray(v, dtype=float) for v in (truth, centre, anchor, sd, trend)]
    if samples.ndim != 2 or any(v.shape != (samples.shape[1],) for v in vectors):
        raise ValueError('One finite value per target cell is required')
    if not np.isfinite(samples).all() or not all(np.isfinite(v).all() for v in vectors):
        raise ValueError('Non-finite diagnostic input')
    truth, centre, anchor, sd, trend = vectors
    if np.any(sd < 0):
        raise ValueError('Negative forecast standard deviation')
    residual = truth - centre
    z = np.divide(residual, sd, out=np.full(sd.shape, np.nan), where=sd > 0)
    delta, prediction = truth - anchor, centre - anchor
    attempt = (prediction != 0) & (delta != 0)
    trend_attempt = (trend != 0) & (delta != 0)
    result = pd.DataFrame({
        'truth': truth, 'centre': centre, 'anchor': anchor, 'sd': sd, 'z': z,
        'pit': ((samples < truth).mean(axis=0) + .5 * (samples == truth).mean(axis=0)),
        'direction_attempt': attempt,
        'direction_correct': attempt & (np.sign(prediction) == np.sign(delta)),
        'trend_attempt': trend_attempt,
        'trend_correct': trend_attempt & (np.sign(trend) == np.sign(delta)),
        'actual_up': delta > 0, 'actual_nonzero': delta != 0,
        'zero_sd_miss': (sd == 0) & (residual != 0),
    })
    for level in (50, 80, 90, 98):
        p = (1 - level / 100) / 2
        lo, hi = np.quantile(samples, [p, 1-p], axis=0)
        result[f'covered{level}'] = (truth >= lo) & (truth <= hi)
        result[f'lower{level}'] = truth < lo
        result[f'upper{level}'] = truth > hi
    return result


def aggregate_cells(frame):
    """Report frequencies and dimensionless residuals; never average raw scores."""
    z = frame['z'].dropna().to_numpy()
    attempts, trend_attempts = int(frame.direction_attempt.sum()), int(frame.trend_attempt.sum())
    absolute = np.abs(z)
    total = absolute.sum()
    return {
        'cells': len(frame), 'units': int(frame.unit.nunique()) if 'unit' in frame else None,
        **{f'coverage{level}': float(frame[f'covered{level}'].mean()) for level in (50,80,90,98)},
        'lower90': float(frame.lower90.mean()), 'upper90': float(frame.upper90.mean()),
        'median_z': float(np.median(z)) if len(z) else None,
        'median_abs_z': float(np.median(absolute)) if len(z) else None,
        'rms_z': float(np.sqrt(np.mean(z*z))) if len(z) else None,
        'abs_z_above3_fraction': float(np.mean(absolute > 3)) if len(z) else None,
        'abs_z_above3_error_share': float(absolute[absolute > 3].sum()/total) if total else None,
        'direction_attempts': attempts,
        'direction_accuracy': float(frame.direction_correct.sum()/attempts) if attempts else None,
        'trend_attempts': trend_attempts,
        'trend_accuracy': float(frame.trend_correct.sum()/trend_attempts) if trend_attempts else None,
        'zero_sd_cells': int((frame.sd == 0).sum()), 'zero_sd_misses': int(frame.zero_sd_miss.sum()),
        'pit_deciles': np.histogram(frame.pit, bins=np.linspace(0,1,11))[0].tolist(),
    }


def relative_components(base, changed):
    return {c: 100 * (changed[c]/base[c] - 1) if base[c] > 0 else None for c in COMPONENTS}


def component_evidence(summary, source):
    """Describe existing comparisons without inferring loss attribution from raw weights."""
    rows = []
    for unit in summary['units']:
        if unit['status'] != 'completed':
            continue
        for kind in ('drift_validation', 'width_validation'):
            validation = unit.get(kind, {})
            if validation.get('status') == 'completed':
                rows.append({'unit': unit['unit'], 'experiment': kind, 'source': str(source),
                             'design': 'older-fold selection; locked newer-fold evaluation',
                             'selected_model': validation['selected_model'],
                             'changes': validation['validation']['change_percent']})
        if set(summary['models']) == {'reference','reference-bootstrap'}:
            validation = calibration_validation(unit['folds'], 3, 6, ['reference-bootstrap'])
            if validation['status'] == 'completed':
                rows.append({'unit': unit['unit'], 'experiment': 'bootstrap_validation',
                             'source': str(source), 'design': 'older-fold selection; locked newer-fold evaluation',
                             'selected_model': validation['selected_model'],
                             'changes': validation['validation']['change_percent']})
        for model in ('reference-ewma', 'reference-shuffled'):
            if model in summary['models']:
                rows.append({'unit': unit['unit'], 'experiment': model, 'source': str(source),
                             'design': 'fixed paired development comparison; no locked validation',
                             'selected_model': model, 'changes': relative_components(
                                 unit['mean_by_model']['reference'], unit['mean_by_model'][model])})
    return rows


def replay(summary, unit_result, units):
    unit = units / unit_result['unit']
    if unit.parent.resolve() != units.resolve():
        raise ValueError('Invalid unit path in summary')
    card = tomllib.loads((unit / 'card.toml').read_text())
    grid = grid_from_card(card)
    if list(grid.assets) != unit_result['assets'] or list(grid.horizons) != unit_result['horizons']:
        raise ValueError('Card grid differs from recorded experiment')
    if card['targets']['target_type'] != unit_result['target_type']:
        raise ValueError('Card target type differs from recorded experiment')
    panels = {}
    cutoff = pd.Timestamp(card['provenance']['data_cutoff'])
    for name, recorded_hash in unit_result['source_sha256'].items():
        path = unit / name
        if unit.resolve() not in path.resolve().parents:
            raise ValueError('Panel path escapes unit')
        if hashlib.sha256(path.read_bytes()).hexdigest() != recorded_hash:
            raise ValueError('Panel hash differs from recorded experiment')
        frame = pd.read_parquet(path).rename(columns={'asset_id':'asset'})
        frame['date'] = pd.to_datetime(frame['date'])
        panels[name] = frame.loc[frame.date <= cutoff].copy()
    wide = pd.DataFrame({a: _series(panels, a, str(cutoff.date())) for a in grid.assets})
    wide.index = pd.to_datetime(wide.index)
    n_draws = summary['settings']['n_draws']
    if summary.get('seed') != 1 or n_draws < 200:
        raise ValueError('Unsupported replay seed or draw count')
    rows = []
    for fold in unit_result['folds']:
        origin = pd.Timestamp(fold['asof'])
        truth, dates = holdout(wide, origin, list(grid.horizons), unit_result['target_type'])
        if dates != fold['target_dates']:
            raise ValueError('Holdout dates differ from recorded experiment')
        history = {k: f.loc[f.date <= origin].copy() for k,f in panels.items()}
        draws, meta = _draw(history, list(grid.assets), list(grid.horizons), fold['asof'], n_draws, 1,
                            target_type=unit_result['target_type'])
        samples = draws.reshape(n_draws, -1)
        score = _score({'realized':truth, '_samples':samples, 'expected_grid':grid, 'card':card,
                        'unit_handle':unit.name, 'grid_source':'card',
                        'normalization_mode':NormalizationMode.RAW_UNRANKABLE, 'ref_scale':None})
        for c in COMPONENTS:
            if not np.isclose(score[c], fold['model_scores']['reference'][c], rtol=1e-9, atol=1e-12):
                raise ValueError(f'Reference replay differs for {unit.name} {fold["asof"]} {c}')
        _, trend_meta = _draw(history, list(grid.assets), list(grid.horizons), fold['asof'], 200, 1,
                              target_type=unit_result['target_type'], drift_scale=1.)
        shape = (len(grid.assets),len(grid.horizons))
        anchor = np.broadcast_to(np.array([meta['last'][a] for a in grid.assets])[:,None], shape)
        drift = np.array([meta['daily_drift'][a] for a in grid.assets])[:,None] * grid.horizons
        sd = np.array([meta['daily_sd'][a] for a in grid.assets])[:,None] * np.sqrt(grid.horizons)
        trend = np.array([trend_meta['daily_drift'][a] for a in grid.assets])[:,None] * grid.horizons
        cells = cell_diagnostics(samples, truth, (anchor+drift).ravel(), anchor.ravel(),sd.ravel(),trend.ravel())
        cells['unit'] = unit.name
        cells['asof'] = fold['asof']
        cells['family'] = card['metadata']['category']
        cells['panel'] = card['metadata'].get('asset_panel','unspecified')
        cells['target_type'] = unit_result['target_type']
        cells['asset'] = np.repeat(grid.assets,len(grid.horizons))
        cells['horizon'] = np.tile(grid.horizons,len(grid.assets))
        cells['target_date'] = np.tile(dates,len(grid.assets))
        cells['horizon_band'] = pd.cut(cells.horizon,[0,5,21,np.inf],labels=['1-5','6-21','22+']).astype(str)
        cells['provenance_flag'] = any('synthetic' in w.lower() or 'unverified provenance' in w.lower()
                                       for w in unit_result.get('warnings',[]))
        rows.append(cells)
    return rows


def grouped(frame, keys):
    return [{**dict(zip(keys, key if isinstance(key, tuple) else (key,))), **aggregate_cells(group)}
            for key, group in frame.groupby(keys,sort=True,observed=True)]


def audit_text_context(frame, units):
    """Count eligible dated documents without opening any document contents."""
    rows = []
    for unit_name, group in frame.groupby('unit', sort=True):
        text = units / unit_name / 'text'
        index = text / 'corpus_index.json'
        documents = json.loads(index.read_text()).get('documents', []) if index.exists() else []
        dates = []
        for doc in documents:
            if not doc.get('timestamp'):
                continue
            path = text / doc.get('file', '')
            if text.resolve() not in path.resolve().parents:
                raise ValueError('Corpus path escapes text directory')
            if path.is_file():
                dates.append(pd.Timestamp(str(doc['timestamp'])[:10]))
        for asof in sorted(group['asof'].unique()):
            origin = pd.Timestamp(asof)
            available = [date for date in dates if date <= origin]
            age = (origin - max(available)).days if available else None
            rows.append({'unit':unit_name, 'asof':asof, 'available_documents':len(available),
                         'recent90_documents':sum((origin-d).days <= 90 for d in available),
                         'youngest_document_age_days':age,
                         'index_sha256':hashlib.sha256(index.read_bytes()).hexdigest() if index.exists() else None})
    return rows


def basis_dates(panels, assets, asof, target_type):
    """Describe the age of the exact aligned daily steps used by the sampler."""
    history = {a:_series(panels,a,asof) for a in assets}
    changes = pd.DataFrame({a:pd.Series(log_return_steps(s),index=s.index)
                            if target_type == 'log_return' else _diff_without_gaps(s)
                            for a,s in history.items()}).dropna().tail(120)
    last = str(changes.index[-1])[:10] if len(changes) else None
    return {'aligned_step_rows':len(changes), 'last_aligned_step_date':last,
            'basis_age_calendar_days':(pd.Timestamp(asof)-pd.Timestamp(last)).days if last else None,
            'last_observation_dates':{a:str(s.index[-1])[:10] for a,s in history.items()}}


def authored_basis_audit(units):
    """Audit authored cutoffs without accessing any future target or answer."""
    rows = []
    for unit in sorted(units.iterdir()):
        if not (unit/'card.toml').is_file():
            continue
        card = tomllib.loads((unit/'card.toml').read_text())
        targets = card['targets']
        row = {'unit':unit.name, 'family':card['metadata']['category'],
               'asof':card['provenance']['data_cutoff'], 'target_type':targets['target_type']}
        if targets.get('target_frequency') == 'monthly':
            rows.append({**row,'status':'monthly_not_audited'})
            continue
        paths = sorted(unit.glob('*.parquet')) + sorted((unit/'panels').glob('*.parquet'))
        panels = {str(p.relative_to(unit)):pd.read_parquet(p) for p in paths}
        rows.append({**row,'status':'daily_audited',
                     'source_sha256':{str(p.relative_to(unit)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                     **basis_dates(panels,targets['asset_ids'],row['asof'],targets['target_type'])})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summaries',type=Path,nargs='+',required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--units',type=Path,default=ROOT/'units')
    args = parser.parse_args(argv)
    out = outside_repo(args.out)
    out.mkdir(parents=True,exist_ok=False)
    rows, evidence, inspected, statuses = [], [], set(), []
    hashes = {}
    for source in args.summaries:
        summary = json.loads(source.read_text())
        hashes[str(source.resolve())] = hashlib.sha256(source.read_bytes()).hexdigest()
        evidence += component_evidence(summary,source)
        for unit in summary['units']:
            statuses.append({'source':str(source),'unit':unit['unit'],'status':unit['status'],
                             'reason':unit.get('reason')})
            if unit['status'] != 'completed':
                continue
            copy = dict(unit)
            copy['folds'] = [f for f in unit['folds'] if (unit['unit'],f['asof']) not in inspected]
            if not copy['folds']:
                continue
            cells = replay(summary,copy,args.units)
            rows += cells
            inspected.update((unit['unit'], f['asof']) for f in copy['folds'])
            print(f'Replayed {unit["unit"]}: {len(cells)} previously inspected folds',flush=True)
    if not rows:
        raise ValueError('No reference folds could be replayed')
    frame = pd.concat(rows,ignore_index=True)
    event_key = ['asset','asof','target_type','horizon','target_date']
    events = frame.groupby(event_key,observed=True).truth.agg(['min','max'])
    if not np.allclose(events['min'],events['max'],rtol=1e-10,atol=1e-12):
        raise ValueError('Conflicting targets for repeated asset events')
    unique = frame.drop_duplicates(event_key,keep='first')
    context = audit_text_context(frame, args.units)
    authored_basis = authored_basis_audit(args.units)
    summary = {
        'description':'Development diagnosis only. No independent validation or official score.',
        'common_version':version('qfbench2-common'),
        'implementation_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in (Path(__file__), ROOT/'qfbench2_track_forecasting/cli.py')},
        'source_sha256':hashes, 'reference_replay':'Every used fold matches its recorded common-toolkit scores',
        'weighting':'Card cells and deterministic first-per-event cells; neither establishes independence',
        'card_weighted':aggregate_cells(frame), 'unique_event_weighted':aggregate_cells(unique),
        'unique_asset_events':len(unique), 'replayed_unit_folds':len(inspected),
        'by_family_target':grouped(frame,['family','target_type']),
        'by_panel_target':grouped(frame,['panel','target_type']),
        'by_horizon_target':grouped(frame,['horizon_band','target_type']),
        'by_unit':grouped(frame,['unit','family','target_type']),
        'by_provenance_flag':grouped(frame,['provenance_flag']),
        'text_context': {'unit_folds':len(context),
                         'with_eligible_text':sum(r['available_documents'] > 0 for r in context),
                         'with_text_within90days':sum(r['recent90_documents'] > 0 for r in context),
                         'rows':context},
        'authored_basis_audit':authored_basis,
        'status_records':statuses, 'component_evidence':evidence,
    }
    # Raw targets belong only in this explicitly external experiment directory.
    frame.to_csv(out/'cells.csv',index=False)
    (out/'diagnosis.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    lines=['## Executive summary (read this first)','',
           'Error diagnosis on already inspected historical periods. This is not a new validation.',
           'All reconstructed reference scores match the recorded shared-toolkit scores.',
           'No score is averaged across unlike units. Cells and cards share history.',
           'Monthly tasks are not covered by these backtests. No text or model API is used.','',
           f'{len(inspected)} unit-folds; {len(frame)} card cells; {len(unique)} unique asset events.',
           'Unique events keep the first forecast in input-summary order. Different grids may give different forecasts.',
           'They still overlap in time and are not independent.','',
           '## Forecast calibration by family and target','',
           '| Family | Target | Units | Cells | 80% coverage | 90% coverage | 98% coverage | Median signed error / SD | Median absolute error / SD | Trend direction accuracy |',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    def percent(v): return 'n/a' if v is None else f'{100*v:.1f}%'
    def number(v): return 'n/a' if v is None else f'{v:.3f}'
    for r in summary['by_family_target']:
        lines.append(f'| {r["family"]} | {r["target_type"]} | {r["units"]} | {r["cells"]} | '+
                     ' | '.join([percent(r['coverage80']),percent(r['coverage90']),percent(r['coverage98']),
                                  number(r['median_z']),number(r['median_abs_z']),percent(r['trend_accuracy'])])+' |')
    lines += ['', '## Data-source sensitivity', '',
              'Flags are unit-level synthetic or unverified-provenance warnings. Absence of a flag is not proof of a verified source.', '',
              '| Unit provenance warning | Units | Cells | 90% coverage | 98% coverage | RMS error / SD | Errors above 3 SD | Share of absolute standardized error from above 3 SD |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary['by_provenance_flag']:
        lines.append(f'| {r["provenance_flag"]} | {r["units"]} | {r["cells"]} | '+
                     ' | '.join([percent(r['coverage90']),percent(r['coverage98']),number(r['rms_z']),
                                  percent(r['abs_z_above3_fraction']),percent(r['abs_z_above3_error_share'])])+' |')
    lines += ['', 'RMS means root mean square. Standardized errors are diagnostics, not official loss attribution.',
              'Large ratios can reflect near-zero historical volatility, real jumps, or input defects. A warning does not establish which cause applies.',
              '', '## Historical text availability', '',
              f'{len(context)} replayed unit-folds: {summary["text_context"]["with_eligible_text"]} have any eligible document;',
              f'{summary["text_context"]["with_text_within90days"]} have at least one document within 90 days of the origin.',
              'Counts use existing files and indexed publication dates. No document content or model API is used.',
              'A numerical backtest on origins without eligible text cannot evaluate reasoning augmentation.']
    stale = [r for r in authored_basis if (r.get('basis_age_calendar_days') or 0) > 90]
    lines += ['', '## Authored forecast basis age', '',
              'This checks each authored cutoff, not earlier simulated origins. No future target is accessed.',
              f'{len(stale)} daily cards use aligned daily changes whose latest date is more than 90 calendar days old.',
              'A current anchor does not make an old uncertainty estimate current. The 90-day threshold is descriptive, not a calibrated policy.', '',
              '| Unit | Cutoff | Last aligned daily change | Age in calendar days |',
              '|---|---|---|---:|']
    for r in stale:
        lines.append(f'| {r["unit"]} | {r["asof"]} | {r["last_aligned_step_date"]} | {r["basis_age_calendar_days"]} |')
    lines += ['', 'SD means theoretical forecast standard deviation. Positive signed errors mean targets exceed the centre.',
              'Trend direction probes the historical mean; it is not a new evaluated policy.',
              'Neutral centres abstain. Tied observed changes are excluded from direction denominators.',
              'Coverage measures interval inclusion, not distance of misses or competition tail loss.','',
              '## Previously tested interventions','',
              '| Experiment | Units | Better / Worse / Unchanged | Median composite change | Median marginal change | Median joint change | Median tail change |',
              '|---|---:|---|---:|---:|---:|---:|']
    for experiment in sorted({e['experiment'] for e in evidence}):
        local=[e for e in evidence if e['experiment']==experiment]
        values=[e['changes']['composite'] for e in local if e['changes']['composite'] is not None]
        med=[]
        for c in ('composite','marginal','joint','tail'):
            v=[e['changes'][c] for e in local if e['changes'][c] is not None]
            med.append(f'{np.median(v):+.2f}%' if v else 'n/a')
        lines.append(f'| {experiment} | {len(local)} | {sum(v<0 for v in values)} / {sum(v>0 for v in values)} / {sum(v==0 for v in values)} | '+ ' | '.join(med)+' |')
    lines += ['', 'Negative component changes mean improvement within the same card.',
              'Zero baseline components have no percentage and are excluded from that component median.',
              'Drift, width and bootstrap rows use locked newer folds. EWMA and shuffled rows are small development comparisons.',
              'These experiments use different periods; do not rank them as if they shared a test set.',
              'Raw marginal, joint and tail weights cannot identify which component dominates the official normalized score.',
              'Historical release vintages and text availability are not reconstructed. Family labels describe the authored card,',
              'not necessarily a crisis present at each earlier simulated origin.',
              'Any model designed using this diagnosis needs separately untouched or prospective validation.']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'card_weighted':summary['card_weighted'],'unique_event_weighted':summary['unique_event_weighted']},indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
