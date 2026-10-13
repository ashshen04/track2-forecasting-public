"""## Executive summary (read this first)

Freeze and replay four structured-text amplitudes on saved version-3 replies.
Example: --plan --source-summary /tmp/v3/summary.json --availability /tmp/audit.json --out /tmp/scales
Then use --run --out /tmp/scales. No model requests are made.
Zero, quarter, half and original amplitudes share numerical shocks and uncertainty.
Existing score functions produce daily M0 Monte Carlo normalization proxies.
Previously inspected windows remain diagnostic; they cannot select a deployment.
Every research artifact stays outside the public repository.
"""
from __future__ import annotations

import argparse
import json
import sys
import tomllib
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baselines.evidence_analysis import ANALYSIS_VERSION
from qfbench2_track_forecasting.cli import _read_panels
from qfbench2_track_forecasting.grid import grid_from_card
from scripts.backtest import outside_repo
from scripts.build_text_cohort import temporal_partition
from scripts.m0_normalization_proxy import estimate
from scripts.validate_stability import atomic_json, digest
from scripts.validate_text import evaluate

AMPLITUDES = (0., .25, .5, 1.)
SEEDS = (1, 7, 19)
NORMALIZATION_SEEDS = (41, 43)


def aggregate(rows):
    """Equal-card mean after per-card research loss clipping, never a ranked result."""
    groups = defaultdict(list)
    for row in rows:
        if row['unit'].startswith('t2-EXAMPLE-'):
            continue
        groups[row['unit']].append(row)
    if not groups:
        return None
    # The scorer supplies every loss. This is research aggregation, not metric math.
    card_means = {unit: float(np.mean([np.clip(r['proxy_loss'], 0, 8) for r in items]))
                  for unit, items in groups.items()}
    return {'cards': len(groups), 'replay_rows': sum(len(items) for items in groups.values()),
            'mean_proxy_loss': float(np.mean(list(card_means.values()))),
            'card_mean_proxy_losses': card_means}


def plan(out, sources, availability):
    audit = json.loads(availability.read_text())
    windows = {}; jobs = []
    for batch, path in enumerate(sources):
        source = json.loads(path.read_text())
        if source['policy'] != 'structured' or source['thinking'] or source['model_max_tokens'] != 4000:
            raise ValueError('Require the fixed formal structured model configuration')
        for record in source['results']:
            identity = (record['unit'], record['asof'])
            windows[identity] = record
            jobs.append({'id': f'b{batch}-j{len(jobs):03d}', 'batch': batch,
                         'source': str(path.resolve()), 'record': record})
    partitioned, partition = temporal_partition(list(windows.values()), fraction=.5)
    split = {(r['unit'], r['asof']): r['split'] for r in partitioned}
    for job in jobs:
        r = job['record']; job['diagnostic_split'] = split.get((r['unit'], r['asof']), 'purged')
    files = ['scripts/validate_text_scale.py', 'scripts/m0_normalization_proxy.py',
             'scripts/validate_text.py', 'baselines/evidence_analysis.py',
             'baselines/directional_text.py', 'baselines/reasoning_agent.py',
             'qfbench2_track_forecasting/cli.py', 'qfbench2_track_forecasting/scoring.py']
    manifest = {'executive_summary': 'Inspected-window amplitude diagnostics. No model calls or automatic promotion.',
                'analysis_version': ANALYSIS_VERSION, 'amplitudes': AMPLITUDES,
                'seeds': SEEDS, 'normalization_seeds': NORMALIZATION_SEEDS,
                'normalization_outcomes': 256, 'n_draws': 1000,
                'objective': 'Equal-card mean clipped daily-M0 Monte Carlo normalized loss; research approximation.',
                'partition': partition, 'fresh_available_windows': len(audit['fresh']),
                'promotion_eligible': False,
                'half_drift_stage': 'Not started: no new independent acceptance cohort.',
                'source_hashes': {str(p.resolve()): digest(p) for p in [*sources, availability]},
                'implementation_sha256': {f: digest(ROOT / f) for f in files}, 'jobs': jobs}
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(out / 'manifest.json', manifest)
    return manifest


def fallback_reply(assets):
    return {'assets': {a: {'direction': 'neutral', 'strength': 'weak', 'priced_in': 'unclear',
            'mechanism': 'Statistical fallback after model failure.',
            'counterargument': 'No valid model reply.', 'evidence': []} for a in assets}}


def run(out, manifest):
    if manifest['analysis_version'] != ANALYSIS_VERSION:
        raise ValueError('Analysis version changed')
    for path, expected in manifest['source_hashes'].items():
        if digest(Path(path)) != expected:
            raise ValueError('Frozen source changed')
    for path, expected in manifest['implementation_sha256'].items():
        if digest(ROOT / path) != expected:
            raise ValueError('Frozen implementation changed')
    scale_cache = {}; rows = []; parity = 0
    for job in manifest['jobs']:
        item = job['record']; unit = ROOT / 'units' / item['unit']
        if unit.parent.resolve() != (ROOT / 'units').resolve():
            raise ValueError('Invalid unit path')
        card = tomllib.loads((unit / 'card.toml').read_text())
        grid = grid_from_card(card)
        failed = item['status'] != 'completed'
        identity = (item['unit'], item['asof'])
        if identity not in scale_cache:
            panels = _read_panels(unit / 'panels' if (unit / 'panels').is_dir() else unit)
            scale_cache[identity] = {}
            for norm_seed in manifest['normalization_seeds']:
                seed = (zlib.crc32('|'.join(identity).encode()) + norm_seed) & 0xFFFFFFFF
                scale_cache[identity][norm_seed] = estimate(panels, grid, item['asof'], card,
                    seed=seed, outcomes=manifest['normalization_outcomes'])
        def provider(prompt):
            if failed:
                return fallback_reply(grid.assets), '', ''
            if digest_text(prompt) != item['prompt_sha256']:
                raise ValueError('Saved reply does not match the current v3 prompt')
            return item['model_reply'], '', ''
        for seed in manifest['seeds']:
            for amplitude in manifest['amplitudes']:
                for norm_seed in manifest['normalization_seeds']:
                    scale, scale_audit = scale_cache[identity][norm_seed]
                    result = evaluate(item, ROOT / 'units', manifest['n_draws'], seed,
                        excerpt_policy='relevant', policy='directional' if failed else 'structured',
                        evidence_format='span', reply_provider=provider,
                        text_scale=amplitude, ref_scale=scale)
                    if result['status'] != 'completed':
                        raise ValueError('Replay must produce a scored forecast or statistical fallback')
                    if not failed and seed == 1 and amplitude == 1 and norm_seed == NORMALIZATION_SEEDS[0]:
                        for arm in ('reference', 'text'):
                            for component in ('marginal', 'joint', 'tail'):
                                if not np.isclose(result[arm][component], item[arm][component], rtol=1e-12, atol=1e-14):
                                    raise ValueError('Saved component score parity failed')
                        parity += 1
                    rows.append({'unit': item['unit'], 'asof': item['asof'], 'batch': job['batch'],
                        'job_id': job['id'], 'seed': seed, 'amplitude': amplitude,
                        'normalization_seed': norm_seed, 'diagnostic_split': job['diagnostic_split'],
                        'model_fallback': failed, 'changed_draws': result['changed_draws'],
                        'proxy_loss': result['text']['score'], 'baseline_proxy_loss': result['reference']['score'],
                        'raw_components': {k: result['text'][k] for k in ('marginal', 'joint', 'tail')},
                        'adjustments': result['adjustments']})
        print(f"[replayed] {job['id']} {item['unit']} {item['asof']} fallback={failed}", flush=True)
    comparisons = []
    for split in ('all', 'development', 'holdout', 'purged'):
        for seed in SEEDS:
            for norm_seed in NORMALIZATION_SEEDS:
                for amplitude in AMPLITUDES:
                    selected = [r for r in rows if r['seed'] == seed and r['normalization_seed'] == norm_seed
                        and r['amplitude'] == amplitude and (split == 'all' or r['diagnostic_split'] == split)]
                    value = aggregate(selected)
                    if value is not None:
                        base = aggregate([{**r, 'proxy_loss': r['baseline_proxy_loss']} for r in selected])
                        comparisons.append({'split': split, 'seed': seed, 'normalization_seed': norm_seed,
                            'amplitude': amplitude, **value,
                            'baseline_mean_proxy_loss': base['mean_proxy_loss'],
                            'paired_mean_loss_delta': value['mean_proxy_loss'] - base['mean_proxy_loss'],
                            'changed_rows': sum(r['changed_draws'] for r in selected)})
    summary = {'executive_summary': 'Input-only normalized amplitude diagnostics; no independent accuracy or ranked score claim.',
               'model_api_calls': 0, 'unique_windows': len(scale_cache), 'saved_reply_batches': len(set(j['batch'] for j in manifest['jobs'])),
               'saved_component_parity_matches': parity, 'scored_replay_rows': len(rows),
               'fallback_jobs': sum(j['record']['status'] != 'completed' for j in manifest['jobs']),
               'promotion_eligible': False, 'half_drift_stage': manifest['half_drift_stage'],
               'normalization': [{'unit': u, 'asof': a, 'estimates': {str(k): v[1] for k, v in scales.items()}}
                                  for (u, a), scales in scale_cache.items()],
               'comparisons': comparisons, 'results': rows}
    atomic_json(out / 'summary.json', summary)
    lines = ['## Executive summary (read this first)', '',
        'Compare fixed text amplitudes with the same numerical draws. Normalization uses daily M0 Monte Carlo proxies, not verified organizer scales. Previously inspected windows and reply repetitions are diagnostic, not independent validation. The exemplar is replayed for parity but excluded from aggregates. No model calls or model promotion.', '',
        '| Amplitude | Mean proxy loss range across draw and normalization seeds | Paired mean loss delta range |',
        '|---|---:|---:|']
    for amplitude in AMPLITUDES:
        rr = [r for r in comparisons if r['split'] == 'all' and r['amplitude'] == amplitude]
        loss = [r['mean_proxy_loss'] for r in rr]; delta = [r['paired_mean_loss_delta'] for r in rr]
        lines.append(f'| {amplitude:g} | {min(loss):.6f} to {max(loss):.6f} | {min(delta):+.6f} to {max(delta):+.6f} |')
    lines += ['', f"Unique windows: {len(scale_cache)}. Saved reply batches: {summary['saved_reply_batches']}. Component parity matches: {parity}. Model failures retained as statistical fallback: {summary['fallback_jobs']}.", '',
        'Daily M0 uses 300 trailing rows, direct return rows, a dated intersection and covariance shared across horizons. Each of two independent normalization simulations has 500 forecast draws and 256 synthetic outcomes. Simulation standard errors are saved and are conditional on the sampled forecast ensemble. Exact analytical organizer scales are not reproduced.', '',
        'The temporal diagnostic split was frozen before this replay. Every window had been inspected before the study. Zero fresh windows were found in the availability audit; no amplitude is selected for deployment and the half-drift stage remains pending independent validation.']
    (out / 'report.md').write_text('\n'.join(lines) + '\n')
    print(f'Results: {out}; rows: {len(rows)}; promotion eligible: False')


def digest_text(text):
    import hashlib
    return hashlib.sha256(text.encode()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source-summary', type=Path, action='append', default=[])
    parser.add_argument('--availability', type=Path)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--plan', action='store_true'); action.add_argument('--run', action='store_true')
    args = parser.parse_args(argv); out = outside_repo(args.out)
    if args.plan:
        if not args.source_summary or args.availability is None:
            parser.error('--plan requires saved v3 summaries and an availability audit')
        plan(out, args.source_summary, args.availability)
        print(f'Frozen: {out}')
    else:
        run(out, json.loads((out / 'manifest.json').read_text()))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
