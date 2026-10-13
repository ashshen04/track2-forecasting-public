"""## Executive summary (read this first)

Compare the current forecast with its conservative language-model text overlay.
Example: python scripts/validate_text.py --backtest /private/tmp/run/summary.json
The default audits dated-text availability without a model call. Add --run for
live calls using configured model access. Failed calls are never accuracy wins.
All results stay outside the public repository. Historical model memory remains
a limitation: dated prompts alone cannot establish freedom from contamination.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import math
import os
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

from baselines.reasoning_agent import (  # noqa: E402
    apply_adjustment, build_prompt, call_model, read_corpus,
)
from baselines.directional_text import directional_prompt, apply_directional  # noqa: E402
from qfbench2_track_forecasting.cli import _draw, _read_panels  # noqa: E402
from qfbench2_track_forecasting.grid import grid_from_card  # noqa: E402
from qfbench2_track_forecasting.normalization import NormalizationMode  # noqa: E402
from qfbench2_track_forecasting.scoring import _score  # noqa: E402
from scripts.backtest import COMPONENTS, holdout, outside_repo  # noqa: E402


def configuration_reason(local_api=False):
    """Report missing names only; never emit endpoint URLs or credentials."""
    if os.environ.get('QFBENCH_NETWORK', '').lower() == 'none':
        return 'QFBENCH_NETWORK=none; live model calls disabled'
    if local_api and 'MODEL_TOKEN' in os.environ:
        return 'Local experiment cannot use MODEL_TOKEN; unset it before using a vendor API'
    names = ['MODEL_ENDPOINT', 'MODEL_NAME']
    if not local_api:
        names.append('MODEL_TOKEN')
    missing = [name for name in names if not os.environ.get(name, '').strip()]
    return 'Missing model configuration: ' + ', '.join(missing) if missing else ''


def load_local_api_key(path):
    """Read only MODEL_API_KEY, without executing shell syntax or logging secrets.

    An existing environment setting takes precedence. This file is used only by
    local experiments, never by the House submission route.
    """
    if not path.is_file():
        return
    for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        name, separator, value = line.partition('=')
        if not separator or name.strip() != 'MODEL_API_KEY':
            raise ValueError(f'Invalid local key configuration at line {number}')
        value = value.strip()
        if value.startswith(('"', "'")):
            if len(value) < 2 or value[-1] != value[0]:
                raise ValueError(f'Invalid local key quoting at line {number}')
            value = value[1:-1]
        if value:
            os.environ.setdefault('MODEL_API_KEY', value)


def candidates(summary, units, max_document_age_days=90):
    """Pick one latest text-eligible origin per unit, without inspecting scores.

    Distribute the call budget across families in round-robin order. Full prior
    corpus history is retained, as in the deployed agent; freshness is diagnostic.
    """
    groups = {}
    rejected = Counter()
    for result in summary['units']:
        if result['status'] != 'completed':
            rejected['no historical folds'] += 1
            continue
        unit = units / result['unit']
        card = tomllib.loads((unit / 'card.toml').read_text())
        if card['targets'].get('target_frequency') == 'monthly':
            rejected['monthly release-vintage evaluation unsupported'] += 1
            continue
        chosen = None
        for fold in sorted(result['folds'], key=lambda f: f['asof'], reverse=True):
            docs, _, _ = read_corpus(unit / 'text', fold['asof'])
            age = ((pd.Timestamp(fold['asof']) - pd.Timestamp(docs[0]['timestamp'])).days
                   if docs else None)
            if docs and age <= max_document_age_days:
                family = card['metadata']['category']
                if family not in ('T2-F1', 'T2-F2', 'T2-F3', 'T2-F4'):
                    raise ValueError('Unknown card family')
                chosen = {'unit': result['unit'], 'asof': fold['asof'],
                          'family': family, 'documents': len(docs),
                          'newest_document_age_days': age,
                          'source_sha256': result['source_sha256']}
                break
        if chosen is None:
            rejected['no recent text predating a historical origin'] += 1
        else:
            groups.setdefault(chosen['family'], []).append(chosen)
    ordered = []
    for group in groups.values():
        group.sort(key=lambda item: (item['newest_document_age_days'], item['unit']))
    while any(groups.values()):
        for family in sorted(groups):
            if groups[family]:
                ordered.append(groups[family].pop(0))
    return ordered, dict(rejected)


def evaluate(item, units, n_draws, seed, excerpt_policy='head', policy='conservative', *,
             evidence_format='quote', reply_provider=None, documents_override=None,
             text_scale=1., drift_scale=None, ref_scale=None):
    """Recompute both scores from identical truncated panels and innovations."""
    unit = units / item['unit']
    if unit.parent.resolve() != units.resolve():
        raise ValueError('Invalid unit path')
    if item.get('card_sha256') and hashlib.sha256((unit / 'card.toml').read_bytes()).hexdigest() != item['card_sha256']:
        raise ValueError('Card changed since cohort construction')
    card = tomllib.loads((unit / 'card.toml').read_text())
    cutoff = pd.Timestamp(card['provenance']['data_cutoff'])
    origin = pd.Timestamp(item['asof'])
    if origin > cutoff:
        raise ValueError('Historical origin exceeds the card cutoff')
    for relative, digest in item['source_sha256'].items():
        if unit.resolve() not in (unit / relative).resolve().parents:
            raise ValueError('Panel path escapes the unit')
        if hashlib.sha256((unit / relative).read_bytes()).hexdigest() != digest:
            raise ValueError('Panel changed since the source backtest')
    for relative, digest in item.get('corpus_sha256', {}).items():
        if unit.resolve() not in (unit / relative).resolve().parents:
            raise ValueError('Corpus path escapes the unit')
        if hashlib.sha256((unit / relative).read_bytes()).hexdigest() != digest:
            raise ValueError('Corpus changed since cohort construction')
    panels = _read_panels(unit / 'panels' if (unit / 'panels').is_dir() else unit)
    panels = {k: f.rename(columns={'asset_id': 'asset'}).assign(date=pd.to_datetime(f.date))
              for k, f in panels.items()}
    panels = {k: f.loc[f.date <= cutoff].copy() for k, f in panels.items()}
    grid = grid_from_card(card)
    assets, horizons = list(grid.assets), list(grid.horizons)
    if ('assets' in item and item['assets'] != assets) or ('horizons' in item and item['horizons'] != horizons):
        raise ValueError('Recorded cohort grid differs from the card')
    series = {}
    for asset in assets:
        sources = [f.loc[f.asset == asset].set_index('date').value.sort_index()
                   for f in panels.values() if (f.asset == asset).any()]
        if len(sources) != 1 or sources[0].index.has_duplicates:
            raise ValueError('Missing, ambiguous or duplicated asset history')
        series[asset] = sources[0]
    truth, dates = holdout(pd.DataFrame(series), origin, horizons, card['targets']['target_type'])
    if 'target_dates' in item and dates != item['target_dates']:
        raise ValueError('Recorded target dates differ from the holdout')
    truncated = {k: f.loc[f.date <= origin].copy() for k, f in panels.items()}
    samples, stats = _draw(truncated, assets, horizons, item['asof'], n_draws, seed,
                           target_type=card['targets']['target_type'], drift_scale=drift_scale)
    docs, excluded, dropped = read_corpus(unit / 'text', item['asof'], excerpt_policy=excerpt_policy)
    if documents_override is not None:
        docs = documents_override
    if any(not d.get('timestamp') or pd.Timestamp(d['timestamp']) > origin for d in docs):
        raise ValueError('Document exceeds simulated origin')
    if not docs:
        raise ValueError('No dated text at this origin')
    sd_h = {a: stats['daily_sd'][a] * math.sqrt(max(horizons)) for a in assets}
    if policy in ('directional', 'structured'):
        context = {a: {'value_unit': card['targets'].get('value_unit', 'unknown')} for a in assets}
        if policy == 'structured':
            from baselines.evidence_analysis import analysis_prompt
            evidence_format = 'span'
            prompt = analysis_prompt(assets, horizons, item['asof'], card['targets']['target_type'], docs, context)
        else:
            _, trend_stats = _draw(truncated, assets, horizons, item['asof'], 200, seed,
                                   target_type=card['targets']['target_type'], drift_scale=1.)
            context = {a: {**context[a], 'baseline_daily_drift': stats['daily_drift'][a],
                           'historical_mean_daily_change': trend_stats['daily_drift'][a]} for a in assets}
            prompt = directional_prompt(assets, horizons, item['asof'], card['targets']['target_type'],
                                        stats['last'], sd_h, docs, context, evidence_format=evidence_format)
    elif policy == 'conservative':
        prompt = build_prompt(assets, horizons, item['asof'], card['targets']['target_type'],
                              stats['last'], sd_h, docs, conservative=True)
    else:
        raise ValueError('Unknown text policy')
    parsed, reason, _ = (reply_provider or call_model)(prompt)
    result = {**item, 'policy': policy, 'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
              'target_dates': dates, 'excerpt_policy': excerpt_policy, 'documents_excluded': excluded,
              'evidence_format':evidence_format, 'model_reply':parsed,
              'documents_dropped_by_budget': dropped}
    if parsed is None:
        return {**result, 'status': 'model_failed', 'reason': reason}
    if policy in ('directional', 'structured'):
        adjustment_reply = parsed
        if policy == 'structured':
            from baselines.evidence_analysis import resolve_analysis
            adjustment_reply, result['analysis_audit'] = resolve_analysis(
                parsed, assets, horizons, item['asof'], card['targets']['target_type'], docs, context,
            )
        adjusted, adjustments, matched = apply_directional(samples, assets, horizons, sd_h, adjustment_reply, docs,
                                                          evidence_format=evidence_format, adjustment_scale=text_scale)
    else:
        adjusted, adjustments, matched = apply_adjustment(
            samples, assets, stats['last'], sd_h, parsed,
            target_type=card['targets']['target_type'], conservative=True,
            valid_doc_ids={str(d['doc_id']) for d in docs})
    if not matched:
        return {**result, 'status': 'invalid_adjustment',
                'reason': 'No asset passed evidence and numeric checks',
                'adjustments': adjustments, 'model_reply': parsed}
    ctx = {'realized': truth, 'expected_grid': grid, 'card': card,
           'unit_handle': unit.name, 'grid_source': 'card',
           'normalization_mode': NormalizationMode.REF_SCALE if ref_scale else NormalizationMode.RAW_UNRANKABLE,
           'ref_scale': ref_scale}
    base = _score({**ctx, '_samples': samples.reshape(n_draws, -1)})
    text = _score({**ctx, '_samples': adjusted.reshape(n_draws, -1)})
    anchors = np.array([stats['last'][a] for a in assets])[:, None]
    centres = anchors + np.array([stats['daily_drift'][a] for a in assets])[:, None] * horizons
    shifts = np.array([adjustments[a].get('longest_horizon_shift' if policy in ('directional', 'structured') else 'shift', 0.)
                       for a in assets])[:, None]
    text_centres = centres + shifts * (np.asarray(horizons) / max(horizons) if policy in ('directional', 'structured') else 1.)
    if policy == 'structured':
        text_centres = centres + np.array([
            [adjustments[a].get('horizon_shifts', {}).get(str(h), 0.) for h in horizons] for a in assets
        ])
    actual = truth.reshape(len(assets), len(horizons)) - anchors
    direction = {}
    for label, centre in [('reference', centres), ('text', text_centres)]:
        predicted = centre - anchors
        attempt = (predicted != 0) & (actual != 0)
        direction[label] = {'attempts': int(attempt.sum()), 'cells': int(actual.size),
                            'correct': int((attempt & (np.sign(predicted) == np.sign(actual))).sum()),
                            'accuracy': float((np.sign(predicted[attempt]) == np.sign(actual[attempt])).mean())
                                        if attempt.any() else None}
    return {**result, 'status': 'completed', 'accepted_assets': matched,
            'assets_requested': len(assets), 'adjustments': adjustments,
            'changed_draws': not np.array_equal(samples, adjusted),
            'direction_diagnostics': direction,
            'reference': base, 'text': text,
            'change_percent': {c: 100 * (text[c] / base[c] - 1) if base[c] > 0 else None
                               for c in COMPONENTS}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument('--backtest', type=Path, help='Existing summary.json; scores are not reused')
    sources.add_argument('--cohort', type=Path, help='Frozen text-aligned cohort.json')
    parser.add_argument('--split', choices=['development', 'holdout'], default='development')
    parser.add_argument('--policy', choices=['conservative', 'directional', 'structured', 'both'], default='conservative')
    parser.add_argument('--evidence-format', choices=['quote','span'], default='quote')
    parser.add_argument('--thinking', action='store_true', help='Enable model reasoning for this experiment')
    parser.add_argument('--model-max-tokens', type=int, default=3000, help='Completion budget; House cap still applies')
    parser.add_argument('--cache-dir', type=Path, help='Optional exact-input structured reply cache outside the repo')
    parser.add_argument('--units', type=Path, default=ROOT / 'units')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--max-calls', type=int, default=12, help='Maximum actual model calls, including both policies')
    parser.add_argument('--include', nargs='+', help='Exact eligible unit names for diagnostics')
    parser.add_argument('--excerpt-policy', choices=['head', 'relevant'], default='head',
                        help='Experimental policy windows use the same per-document character budget')
    parser.add_argument('--max-document-age-days', type=int, default=90,
                        help='Require at least one document within this many days of the origin')
    parser.add_argument('--run', action='store_true', help='Make live model calls; otherwise audit only')
    parser.add_argument('--local-api', action='store_true', help='Allow an experimental local model interface')
    parser.add_argument('--nvidia', action='store_true', help=(
        'Use the public NVIDIA Nemotron 3 Super endpoint for a local experiment'))
    parser.add_argument('--prompt-api-key', action='store_true', help=(
        'Read a local MODEL_API_KEY from a hidden terminal prompt; do not save it'))
    args = parser.parse_args(argv)
    if args.max_calls < 1:
        parser.error('max-calls must be positive')
    if args.cache_dir is not None and args.policy != 'structured':
        parser.error('--cache-dir requires --policy structured')
    if not 1 <= args.model_max_tokens <= 8192:
        parser.error('Require model-max-tokens from 1 to 8192')
    os.environ['MODEL_THINKING'] = 'on' if args.thinking else 'off'
    os.environ['MODEL_MAX_TOKENS'] = str(args.model_max_tokens)
    if args.max_document_age_days < 0:
        parser.error('max-document-age-days must be nonnegative')
    if args.nvidia:
        args.local_api = True
        os.environ['MODEL_ENDPOINT'] = 'https://integrate.api.nvidia.com/v1'
        os.environ['MODEL_NAME'] = 'nvidia/nemotron-3-super-120b-a12b'
    if args.local_api:
        try:
            load_local_api_key(ROOT / '.env.local')
        except ValueError as exc:
            parser.error(str(exc))
    if args.prompt_api_key and not (args.run and args.local_api):
        parser.error('prompt-api-key requires --run and --local-api or --nvidia')
    try:
        out = outside_repo(args.out)
    except ValueError as exc:
        parser.error(str(exc))
    source_path = args.cohort or args.backtest
    source = json.loads(source_path.read_text())
    draws = source['settings']['n_draws']
    if not 200 <= draws <= 20000:
        parser.error('Source backtest must use 200 to 20000 draws')
    if args.cohort:
        eligible = [r for r in source['items'] if r['split'] == args.split]
        rejected = {'other temporal split': len(source['items']) - len(eligible)}
    else:
        eligible, rejected = candidates(source, args.units, args.max_document_age_days)
    if args.include:
        missing = set(args.include) - {item['unit'] for item in eligible}
        if missing:
            parser.error(f'Unknown or text-ineligible units: {sorted(missing)}')
        eligible = [item for item in eligible if item['unit'] in args.include]
    policies_per_window = 2 if args.policy == 'both' else 1
    if args.max_calls < policies_per_window:
        parser.error('Call budget must cover all requested policies for at least one window')
    selected = eligible[:args.max_calls // policies_per_window]
    if args.prompt_api_key:
        key = getpass.getpass('Local model API key (hidden; not saved): ').strip()
        if not key:
            parser.error('An API key is required')
        os.environ['MODEL_API_KEY'] = key
    blocker = configuration_reason(args.local_api)
    if args.nvidia and not os.environ.get('MODEL_API_KEY', '').strip():
        blocker = blocker or 'NVIDIA API key missing; set MODEL_API_KEY in .env.local or use --prompt-api-key'
    out.mkdir(parents=True, exist_ok=False)
    results = []
    actual_calls = 0
    def live_model(prompt):
        nonlocal actual_calls
        actual_calls += 1
        return call_model(prompt)
    provider = live_model
    reply_cache = None
    if args.cache_dir is not None:
        from baselines.evidence_analysis import PromptReplyCache
        reply_cache = PromptReplyCache(outside_repo(args.cache_dir), live_model)
        provider = reply_cache
    if args.run and not blocker:
        for item in selected:
            for policy in (['conservative', 'directional'] if args.policy == 'both' else [args.policy]):
                try:
                    result = evaluate(item, args.units, draws, source['seed'], args.excerpt_policy, policy,
                                      evidence_format=args.evidence_format, reply_provider=provider)
                except Exception as exc:
                    # Exception messages can include credentials; only disclose the type.
                    result = {**item, 'policy': policy, 'status': 'failed', 'reason': type(exc).__name__}
                results.append(result)
                name = f"{item['unit']}-{item['asof']}-{policy}.json"
                (out / name).write_text(json.dumps(result, indent=2) + '\n')
                print(f"[{result['status']}] {policy}: {item['unit']} at {item['asof']}", flush=True)
    complete = [r for r in results if r['status'] == 'completed']
    summary = {'created_utc': datetime.now(timezone.utc).isoformat(),
               'status': 'blocked' if args.run and blocker else 'completed' if complete else 'audit_only',
               'model_configuration_issue': blocker,
               'access_mode': 'local_experiment' if args.local_api else 'house',
               'model_name': os.environ.get('MODEL_NAME', ''),
               'snapshot': 'not verified against the House snapshot',
               'excerpt_policy': args.excerpt_policy,
               'policy': args.policy, 'split': args.split if args.cohort else 'prior_development',
               'thinking': args.thinking, 'model_max_tokens': args.model_max_tokens,
               'partition': source.get('partition'),
               'common_version': version('qfbench2-common'), 'n_draws': draws, 'seed': source['seed'],
               'max_document_age_days': args.max_document_age_days,
               'candidate_rule': source.get('selection_rule', 'one latest eligible fold per unit; freshest text first within family; round robin families'),
               'max_actual_calls': args.max_calls,
               'cache_hits': reply_cache.hits if reply_cache else 0,
               'actual_calls': actual_calls,
               'eligible_units': len({r['unit'] for r in eligible}), 'eligible_windows': len(eligible),
               'rejected_units': rejected,
               'selected': selected, 'counts': dict(Counter(r['status'] for r in results)),
               'results': results, 'recommendation': 'Accuracy benefit not established; retain the statistical control.',
               'source_backtest_sha256': hashlib.sha256(source_path.read_bytes()).hexdigest(),
               'implementation_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in [Path(__file__), ROOT / 'baselines/reasoning_agent.py',
                              ROOT / 'baselines/directional_text.py',
                              ROOT / 'baselines/evidence_analysis.py',
                              ROOT / 'qfbench2_track_forecasting/cli.py']}}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    lines = ['## Executive summary (read this first)', '',
             'Paired historical text diagnostics; raw scores are not official scores.',
             'Shared histories and model knowledge of later events limit historical evidence.',
             'A successful call and citation check do not establish forecast improvement.', '',
             f"Eligible windows: {len(eligible)}. Selected windows: {len(selected)}. Maximum actual calls: {args.max_calls}.",
             f"Actual evaluated replies: {len(complete)}. Configuration: {blocker or 'present; access not proven by configuration alone'}.",
             'Audit mode makes no calls. Failures and rejected replies are excluded from accuracy comparisons.', '',
             '| Unit | Origin | Policy | Status | Composite change | Changed draws |',
             '|---|---|---|---|---:|---|']
    for result in results:
        change = result.get('change_percent', {}).get('composite')
        formatted = f'{change:+.2f}%' if change is not None else 'n/a'
        lines.append(f"| {result['unit']} | {result['asof']} | {result['policy']} | {result['status']} | {formatted} | {result.get('changed_draws', 'n/a')} |")
    lines += ['', summary['recommendation']]
    (out / 'report.md').write_text('\n'.join(lines) + '\n')
    print(f'Results: {out}; evaluated replies: {len(complete)}')
    expected = len(selected) * (2 if args.policy == 'both' else 1)
    return 2 if args.run and (blocker or len(complete) != expected or not complete) else 0


if __name__ == '__main__':
    raise SystemExit(main())
