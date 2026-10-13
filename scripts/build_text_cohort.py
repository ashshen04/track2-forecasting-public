"""## Executive summary (read this first)

Build historical validation windows around frozen, dated text rather than numeric scores.
Example: python scripts/build_text_cohort.py --out /private/tmp/text-cohort
Training panels stop at each origin. Authored horizons never change. Targets and
scores are not stored here. Every generated file stays outside the public repository.
Development windows end before a global date split. Holdout windows start after it.
Historical model memory and fixed data vintages still limit independence claims.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
import tomllib
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from baselines.reasoning_agent import read_corpus
from qfbench2_track_forecasting.cli import _draw
from qfbench2_track_forecasting.grid import grid_from_card
from scripts.backtest import holdout, outside_repo, overlaps_excluded, SkipUnit


def temporal_partition(items, fraction=.7):
    """Purge target intervals crossing a global origin-date boundary."""
    dates = sorted({item['asof'] for item in items})
    if len(dates) < 2:
        return [], {'status':'insufficient_dates'}
    boundary = dates[min(len(dates)-1,max(1,int(len(dates)*fraction)))]
    kept, purged = [], 0
    for item in items:
        end = max(item['target_dates'])
        if end < boundary:
            kept.append({**item,'split':'development'})
        elif item['asof'] >= boundary:
            kept.append({**item,'split':'holdout'})
        else:
            purged += 1
    return kept, {'status':'completed','boundary':boundary,'purged':purged,
                  'rule':'development final target < boundary; holdout origin >= boundary'}


def round_robin(items):
    groups = {}
    for item in sorted(items,key=lambda r:(r['newest_document_age_days'],r['unit'],r['asof'])):
        groups.setdefault(item['family'],[]).append(item)
    result = []
    while any(groups.values()):
        for family in sorted(groups):
            if groups[family]: result.append(groups[family].pop(0))
    return result


def policy_documents(docs):
    """A score-free lexical eligibility filter, not proof of asset relevance."""
    direct = {'fomc_statement', 'fomc_minutes', 'central_bank_decision',
              'monetary_policy_statement', 'macro_release', 'beige_book'}
    terms = ('monetary policy', 'interest rate', 'federal funds', 'policy rate',
             'inflation', 'yield curve', 'asset purchase', 'exchange rate')
    return [d for d in docs if d.get('doc_type') in direct or
            sum(term in d['text'].lower() for term in terms) >= 2]


def build_unit(unit, max_folds=3, max_age=90, require_policy_text=False, excluded_intervals=None):
    card = tomllib.loads((unit/'card.toml').read_text())
    targets = card['targets']
    if targets.get('target_frequency') == 'monthly':
        raise SkipUnit('Monthly historical release vintages unsupported')
    grid = grid_from_card(card)
    cutoff = pd.Timestamp(card['provenance']['data_cutoff'])
    paths = sorted(unit.glob('*.parquet')) + sorted((unit/'panels').glob('*.parquet'))
    panels, hashes = {}, {}
    for p in paths:
        frame = pd.read_parquet(p).rename(columns={'asset_id':'asset'})
        frame['date'] = pd.to_datetime(frame.date)
        panels[str(p.relative_to(unit))] = frame.loc[frame.date <= cutoff].copy()
        hashes[str(p.relative_to(unit))] = hashlib.sha256(p.read_bytes()).hexdigest()
    series = {}
    for asset in grid.assets:
        sources = [f.loc[f.asset==asset].set_index('date').value.sort_index()
                   for f in panels.values() if (f.asset==asset).any()]
        if len(sources)!=1 or sources[0].index.has_duplicates:
            raise SkipUnit('Missing or ambiguous target history')
        series[asset] = sources[0]
    wide = pd.DataFrame(series).sort_index()
    # The prompt reader's document cap must not hide earlier indexed availability.
    index = unit/'text/corpus_index.json'
    indexed = json.loads(index.read_text()).get('documents',[]) if index.exists() else []
    dates = [pd.Timestamp(str(d['timestamp'])[:10]) for d in indexed
             if d.get('timestamp') and (unit/'text'/d.get('file','')).is_file()]
    if not dates: raise SkipUnit('No dated corpus')
    available = wide.dropna().index
    candidates = available[62:]
    candidates = candidates[candidates >= min(dates)]
    items, rejected, next_end = [], Counter(), None
    for origin in reversed(candidates):
        end = origin+pd.offsets.BDay(max(grid.horizons))
        if end>cutoff or (next_end is not None and end>next_end): continue
        if excluded_intervals and overlaps_excluded(origin, end, grid.assets, excluded_intervals):
            rejected['previously_inspected_shared_asset'] += 1
            continue
        eligible,_,_ = read_corpus(unit/'text',str(origin.date()),excerpt_policy='relevant')
        if not eligible or (origin-pd.Timestamp(eligible[0]['timestamp'])).days>max_age:
            rejected['no recent eligible text'] += 1; continue
        policy = policy_documents(eligible)
        recent_policy = [d for d in policy if (origin-pd.Timestamp(d['timestamp'])).days <= max_age]
        if require_policy_text and not recent_policy:
            rejected['no recent policy or macro text'] += 1; continue
        try:
            _, target_dates = holdout(wide,origin,list(grid.horizons),targets['target_type'])
            history = {k:f.loc[f.date <= origin].copy() for k,f in panels.items()}
            _draw(history,list(grid.assets),list(grid.horizons),str(origin.date()),200,1,
                  target_type=targets['target_type'])
        except (SkipUnit, ValueError, SystemExit) as exc:
            rejected[type(exc).__name__] += 1; continue
        manifest_path = unit/'manifest.json'
        manifest = manifest_path.read_text().lower() if manifest_path.exists() else ''
        items.append({'unit':unit.name,'asof':str(origin.date()),'target_dates':target_dates,
            'card_sha256':hashlib.sha256((unit/'card.toml').read_bytes()).hexdigest(),
            'family':card['metadata']['category'],'assets':list(grid.assets),'horizons':list(grid.horizons),
            'target_type':targets['target_type'],'documents':len(eligible),
            'policy_documents':len(policy),'recent_policy_documents':len(recent_policy),
            'newest_document_age_days':(origin-pd.Timestamp(eligible[0]['timestamp'])).days,
            'source_sha256':hashes,
            'corpus_sha256':{str(p.relative_to(unit)):hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in [index,*[unit/'text'/d['file'] for d in indexed]] if p.is_file()},
            'provenance_warning':('unverified' in manifest or 'synthetic' in manifest)})
        next_end=origin
        if len(items)==max_folds: break
    if not items: raise SkipUnit(f'No eligible exact holdout: {dict(rejected)}')
    return items


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--units',type=Path,default=ROOT/'units')
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--folds-per-unit',type=int,default=3)
    parser.add_argument('--max-document-age-days',type=int,default=90)
    parser.add_argument('--require-policy-text',action='store_true',
                        help='Require a recent statement, macro release or lexically policy-related speech')
    args=parser.parse_args(argv)
    if args.folds_per_unit<1 or args.max_document_age_days<0: parser.error('Invalid fold count or text age')
    out=outside_repo(args.out); out.mkdir(parents=True,exist_ok=False)
    items,statuses=[],[]
    for unit in sorted(args.units.iterdir()):
        if not (unit/'card.toml').exists(): continue
        try:
            local=build_unit(unit,args.folds_per_unit,args.max_document_age_days,args.require_policy_text)
            items+=local; statuses.append({'unit':unit.name,'status':'completed','folds':len(local)})
        except SkipUnit as exc:
            statuses.append({'unit':unit.name,'status':'skipped','reason':str(exc)})
    partitioned, partition=temporal_partition(items)
    ordered=[]
    for split in ('development','holdout'):
        ordered += round_robin([r for r in partitioned if r['split']==split])
    result={'description':'Text-aligned development cohort; existing historical outcomes, not independent model-memory validation.',
        'seed':1,'settings':{'n_draws':1000,'max_document_age_days':args.max_document_age_days,
                           'require_policy_text':args.require_policy_text},
        'selection_rule':'latest feasible non-overlapping windows with eligible text; optional recent policy filter; no scores inspected',
        'partition':partition,'items':ordered,'unit_status':statuses,
        'counts':dict(Counter(r['split'] for r in ordered)),
        'implementation_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/'cohort.json').write_text(json.dumps(result,indent=2)+'\n')
    (out/'report.md').write_text('## Executive summary (read this first)\n\n'+
        'Text-aligned windows preserve authored horizons and dated evidence. Scores are not used for selection.\n\n'+
        f'{len(items)} eligible windows before temporal purging. Partition: {partition}.\n\n'+
        f'Split counts: {result["counts"]}. Unit statuses: {dict(Counter(r["status"] for r in statuses))}.\n\n'+
        'Missing targets and monthly tasks remain excluded. Historical LLM memory, shared assets and final data vintages limit evidence.\n')
    print(result['counts'],dict(Counter(r['status'] for r in statuses)))
    return 0 if ordered else 1


if __name__=='__main__': raise SystemExit(main())
