"""## Executive summary (read this first)

Compare Gaussian forecasts with historical event-mean centre adjustments.
Example: python scripts/validate_event_analog.py --out /private/tmp/event-test
Historical event responses and scoring reports stay outside the public repository.
This is an offline, fixed-snapshot development experiment, not official validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tomllib
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baselines.event_analog import analog_adjustment, classify_statement
from qfbench2_track_forecasting.cli import _draw, _read_panels, _series
from qfbench2_track_forecasting.grid import grid_from_card
from qfbench2_track_forecasting.normalization import NormalizationMode
from qfbench2_track_forecasting.scoring import _score
from scripts.backtest import SkipUnit, holdout, outside_repo


def event_catalog(units, cutoff):
    """Deduplicate dated statements by date; classification never reads price responses."""
    by_date = {}
    for index in sorted(units.glob('*/text/corpus_index.json')):
        for doc in json.loads(index.read_text()).get('documents', []):
            if doc.get('doc_type')!='fomc_statement' or not doc.get('timestamp'):
                continue
            date = str(pd.Timestamp(doc['timestamp']).date())
            if date > cutoff:
                continue
            path = (index.parent/doc['file']).resolve()
            if index.parent.resolve() not in path.parents:
                raise ValueError('Corpus file escapes text directory')
            if not path.is_file():
                continue
            kind = classify_statement(path.read_text())
            if kind:
                record = {'date':date,'type':kind,'source':str(path.relative_to(ROOT)),
                          'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
                by_date.setdefault(date, record)
    return [by_date[d] for d in sorted(by_date)]


def evaluate(wide, panels, events, event, card, grid, seed, n_draws):
    """Score only after predictions are built from physically truncated histories."""
    origin = pd.Timestamp(event['date']) + pd.offsets.BDay(1)
    assets, horizons = list(grid.assets), list(grid.horizons)
    history = {k:f.loc[pd.to_datetime(f.date)<=origin].copy() for k,f in panels.items()}
    draws, meta = _draw(history, assets, horizons, str(origin.date()), n_draws, seed)
    sd = np.array([meta['daily_sd'][a] for a in assets])[:,None] * np.sqrt(horizons)
    adjustments = {}
    samples = {'reference':draws}
    for name, shrink in (('event-mean',False), ('event-mean-shrunk',True)):
        shift, audit = analog_adjustment(wide.loc[:origin], events, event['type'], origin,
                                        assets, horizons, sd, shrink=shrink)
        samples[name] = draws + shift
        adjustments[name] = {'shift':shift.tolist(),'audit':audit}
    trend, _ = _draw(history, assets, horizons, str(origin.date()), n_draws, seed, drift_scale=.5)
    samples['historical-half-trend'] = trend
    truth, dates = holdout(wide, origin, horizons, 'level')
    scores = {name:_score({'realized':truth,'_samples':sample.reshape(n_draws,-1),
                'expected_grid':grid,'card':card,'unit_handle':card['task']['id'],
                'grid_source':'card','normalization_mode':NormalizationMode.RAW_UNRANKABLE,
                'ref_scale':None}) for name,sample in samples.items()}
    return {'status':'completed','event':event,'asof':str(origin.date()),'target_dates':dates,
            'seed':seed,'adjustments':adjustments,'scores':scores,
            'change_percent':{name:{c:100*(s[c]/scores['reference'][c]-1)
                if scores['reference'][c]>0 else None for c in ('composite','marginal','joint','tail')}
                for name,s in scores.items() if name!='reference'}}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--unit',default='t2-F3-sep-dots-curve-2023')
    parser.add_argument('--evaluation-start',default='2015-01-01')
    parser.add_argument('--n-draws',type=int,default=1000)
    args=parser.parse_args(argv)
    out=outside_repo(args.out)
    if args.n_draws<200: parser.error('Require >=200 draws')
    unit=ROOT/'units'/args.unit
    if unit.parent.resolve()!=(ROOT/'units').resolve(): parser.error('Invalid unit')
    card=tomllib.loads((unit/'card.toml').read_text())
    grid=grid_from_card(card)
    if card['targets']['target_type']!='level' or card['targets'].get('target_frequency')=='monthly':
        parser.error('Require daily level targets')
    cutoff=card['provenance']['data_cutoff']
    panels=_read_panels(unit)
    for name,f in panels.items():
        f['date']=pd.to_datetime(f.date)
        panels[name]=f.loc[f.date<=pd.Timestamp(cutoff)].copy()
    wide=pd.DataFrame({a:_series(panels,a,cutoff) for a in grid.assets})
    wide.index=pd.to_datetime(wide.index)
    events=event_catalog(ROOT/'units',cutoff)
    candidates=[e for e in events if e['date']>=args.evaluation_start]
    out.mkdir(parents=True,exist_ok=False)
    # Select old-to-new non-overlapping evaluation intervals without examining scores.
    records=[]; last_end=None
    for event in candidates:
        origin=pd.Timestamp(event['date'])+pd.offsets.BDay(1)
        end=origin+pd.offsets.BDay(max(grid.horizons))
        if end>pd.Timestamp(cutoff) or (last_end is not None and origin<last_end):
            records.append({'status':'skipped','event':event,'reason':'incomplete_or_overlapping_evaluation_interval'})
            continue
        try:
            batch=[evaluate(wide,panels,events,event,card,grid,seed,args.n_draws) for seed in (1,7,19)]
        except (SkipUnit,ValueError,SystemExit) as exc:
            records.append({'status':'skipped','event':event,'reason':str(exc)})
            continue
        records.extend(batch)
        last_end=end
        print(f'[completed] {event["date"]}: three seeds',flush=True)
    comparisons=[]
    for seed in (1,7,19):
        good=[r for r in records if r['status']=='completed' and r['seed']==seed]
        for name in ('event-mean','event-mean-shrunk','historical-half-trend'):
            changes=[r['change_percent'][name] for r in good]
            eligible=[r for r in good if name not in r['adjustments'] or
                      any(a['count']>=3 for a in r['adjustments'][name]['audit'].values())]
            comparisons.append({'seed':seed,'model':name,'events':len(good),'actionable_events':len(eligible),
                'better':sum(c['composite']<0 for c in changes),
                'worse':sum(c['composite']>0 for c in changes),
                'unchanged':sum(c['composite']==0 for c in changes),
                'median_change':{c:float(np.median([r[c] for r in changes if r[c] is not None]))
                    if any(r[c] is not None for r in changes) else None
                    for c in ('composite','marginal','joint','tail')}})
    summary={'executive_summary':'Sparse same-action analog development experiment; no independent accuracy claim.',
        'unit':args.unit,'assets':list(grid.assets),'horizons':list(grid.horizons),
        'common_version':version('qfbench2-common'),'text_used':'dated event classification only',
        'model_api_calls':0,'n_draws':args.n_draws,'seeds':[1,7,19],
        'matching':'same announced FOMC hike action only; no surprise, magnitude or macro matching',
        'origin':'next business day close; excludes announcement-day reaction',
        'shrunk_rule':'n/(n+10), minimum 3 nonoverlapping completed peers, capped at 0.5 baseline SD',
        'catalog':events,'evaluation_start':args.evaluation_start,
        'counts':dict(Counter(r['status'] for r in records)),'comparisons':comparisons,'results':records,
        'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in unit.glob('*.parquet')},
        'implementation_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__),ROOT/'baselines/event_analog.py',ROOT/'qfbench2_track_forecasting/cli.py')},
        'limitations':['Corpus is an incomplete retrospective statement sample, not a full event calendar.',
            'Panel vintages and intraday availability are not reconstructed.',
            'Historical periods may have been inspected in previous experiments.',
            'Historical average changes are associations, not causal event effects.']}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['## Executive summary (read this first)','',summary['executive_summary'],'',
        'Same authored assets and horizon keys; Gaussian baseline shocks are unchanged by event shifts.',
        'Evaluation intervals do not overlap. Analog response intervals do not overlap within each horizon.',
        'The catalog is incomplete and dates already inspected elsewhere can be reused. This is development evidence.',
        'Negative percentages mean lower raw scoring error within this one grid; these are not official normalized scores.','',
        '| Seed | Model | Events | Actionable | Better / Worse / Unchanged | Median composite change |',
        '|---|---|---:|---:|---|---:|']
    for r in comparisons:
        value=r['median_change']['composite']
        display=f'{value:+.2f}%' if value is not None else 'n/a'
        lines.append(f'| {r["seed"]} | {r["model"]} | {r["events"]} | {r["actionable_events"]} | {r["better"]} / {r["worse"]} / {r["unchanged"]} | {display} |')
    lines+=['','Minimum peer requirement is horizon-specific. Zero changes can reflect abstention.',
        'No current-case response enters matching or fitting. Completed peer responses end no later than the forecast origin.',
        'Matching only hike direction cannot distinguish surprise, hike size or inflation regimes.',
        'Formal forecast and existing submission package remain unchanged.']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'counts':summary['counts'],'comparisons':comparisons},indent=2))
    return 0 if any(r['status']=='completed' for r in records) else 1


if __name__=='__main__':
    raise SystemExit(main())
