"""## Executive summary (read this first)

Freeze uninspected return-target intervals, then compare four fixed Gaussian policies.
Example: run with --plan --out /private/tmp/return-check --prior-summary prior.json,
then run with --run --out /private/tmp/return-check. All artifacts remain outside
the public repository. Scoring formulas are imported from the shared toolkit.
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
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from qfbench2_track_forecasting.cli import _draw, _series
from qfbench2_track_forecasting.grid import grid_from_card
from qfbench2_track_forecasting.scoring import _score
from qfbench2_track_forecasting.normalization import NormalizationMode
from scripts.backtest import outside_repo, holdout, overlaps_excluded, SkipUnit

POLICIES={'reference':(None,1.),'half-drift':(.5,1.),'wide':(None,1.2),'combined':(.5,1.2)}
SEEDS=(1,7,19)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reserve(intervals,assets,start,end):
    if end<=start: raise ValueError('Interval must end after origin')
    for asset in assets: intervals.setdefault(asset,[]).append((start,end))


def prior_intervals(paths):
    """Read numerical, text and event-response records; count shared assets globally."""
    intervals={}; sources={}
    for path in paths:
        d=json.loads(path.read_text()); count=0; inferred={}
        for unit in d.get('units',[]):
            for fold in unit.get('folds',[]):
                reserve(intervals,unit['assets'],pd.Timestamp(fold['asof']),
                        max(pd.Timestamp(t) for t in fold['target_dates'])); count+=1
        for r in d.get('results',[]):
            if r.get('status')!='completed' or not r.get('target_dates'): continue
            assets=r.get('assets',d.get('assets'))
            if not assets:
                # Older experiment records omitted their grid. Conservatively exclude
                # the authored grid plus adjustment keys, recording the inference.
                unit=ROOT/'units'/r['unit']
                if unit.parent.resolve()!=(ROOT/'units').resolve(): raise ValueError('Invalid prior unit')
                card_path=unit/'card.toml'
                authored=tomllib.loads(card_path.read_text())['targets']['asset_ids']
                assets=sorted(set(authored)|set(r.get('adjustments',{})))
                inferred[r['unit']]={'assets':assets,'card_sha256':digest(card_path)}
            reserve(intervals,assets,pd.Timestamp(r['asof']),max(pd.Timestamp(t) for t in r['target_dates']))
            count+=1
            for adjustment in r.get('adjustments',{}).values():
                for audit in adjustment.get('audit',{}).values():
                    for peer in audit.get('peers',[]):
                        reserve(intervals,assets,pd.Timestamp(peer['origin']),pd.Timestamp(peer['end']))
                        count+=1
        sources[str(path.resolve())]={'sha256':digest(path),'records':count,'legacy_asset_inference':inferred}
    if not intervals: raise ValueError('Prior summaries contain no historical intervals')
    # Compact repeated replay intervals before scanning many possible origins.
    for asset,ranges in intervals.items():
        merged=[]
        for start,end in sorted(ranges):
            if merged and start<=merged[-1][1]: merged[-1]=(merged[-1][0],max(end,merged[-1][1]))
            else: merged.append((start,end))
        intervals[asset]=merged
    return intervals,sources


def read_unit(unit):
    card=tomllib.loads((unit/'card.toml').read_text()); grid=grid_from_card(card)
    panels={}; hashes={'card.toml':digest(unit/'card.toml')}
    for p in sorted(unit.glob('*.parquet'))+sorted((unit/'panels').glob('*.parquet')):
        frame=pd.read_parquet(p).rename(columns={'asset_id':'asset'})
        frame['date']=pd.to_datetime(frame.date)
        panels[str(p.relative_to(unit))]=frame.loc[frame.date<=pd.Timestamp(card['provenance']['data_cutoff'])].copy()
        hashes[str(p.relative_to(unit))]=digest(p)
    wide=pd.DataFrame({a:_series(panels,a,card['provenance']['data_cutoff']) for a in grid.assets})
    wide.index=pd.to_datetime(wide.index)
    if wide.index.has_duplicates: raise ValueError('Duplicate target dates')
    return card,grid,panels,wide.sort_index(),hashes


def four_forecasts(panels,grid,origin,n_draws,seed):
    """Change only centre and width, sharing each seed's correlated Gaussian shocks."""
    history={k:f.loc[f.date<=origin].copy() for k,f in panels.items()}
    return {name:_draw(history,list(grid.assets),list(grid.horizons),str(origin.date()),n_draws,seed,
                       target_type='log_return',drift_scale=drift,width_scale=width)[0]
            for name,(drift,width) in POLICIES.items()}


def plan(out,paths,max_folds=12):
    intervals,sources=prior_intervals(paths)
    units=[]
    for p in (ROOT/'units').glob('*/card.toml'):
        card=tomllib.loads(p.read_text())
        if card['targets']['target_type']=='log_return' and card['targets'].get('target_frequency')!='monthly':
            grid=grid_from_card(card); units.append((-len(grid.assets),-max(grid.horizons),p.parent.name,p.parent))
    selected=[]; statuses=[]
    for _,_,name,unit in sorted(units):
        card,grid,panels,wide,hashes=read_unit(unit)
        rejected=Counter(); chosen=[]
        for origin in reversed(wide.dropna().index[120:]):
            end=origin+pd.offsets.BDay(max(grid.horizons))
            if end>pd.Timestamp(card['provenance']['data_cutoff']): continue
            if overlaps_excluded(origin,end,grid.assets,intervals):
                rejected['previously_inspected_or_reserved_shared_asset']+=1; continue
            try:
                _,dates=holdout(wide,origin,list(grid.horizons),'log_return')
                # Check fitting feasibility without accessing or recording any score.
                history={k:f.loc[f.date<=origin] for k,f in panels.items()}
                _draw(history,list(grid.assets),list(grid.horizons),str(origin.date()),200,1,target_type='log_return')
            except (SkipUnit,ValueError,SystemExit) as exc:
                rejected[str(exc)]+=1; continue
            chosen.append({'unit':name,'assets':list(grid.assets),'horizons':list(grid.horizons),
                           'asof':str(origin.date()),'target_dates':dates,'source_sha256':hashes})
            reserve(intervals,grid.assets,origin,end)
            if len(chosen)>=max_folds: break
        selected.extend(chosen)
        statuses.append({'unit':name,'selected':len(chosen),'rejected':dict(rejected)})
        print(f'[planned] {name}: {len(chosen)} uninspected intervals',flush=True)
    cohort={'executive_summary':'Frozen score-free cohort; shared-asset forecast intervals do not overlap.',
        'policies':POLICIES,'seeds':list(SEEDS),'n_draws':1000,'max_folds_per_unit':max_folds,
        'selection':'multi-asset grids first, longer maximum horizon first, lexical unit; latest feasible origins',
        'prior_sources':sources,'selected':sorted(selected,key=lambda r:(r['asof'],r['unit'])),
        'unit_statuses':statuses,'acceptance':{
            'minimum_intervals':12,'minimum_origin_years':3,
            'rule':'At each seed: median paired composite change < 0, more wins than losses, median joint and tail changes <= 0.',
            'promotion':'No automatic deployment; retrospective vintages and regime coverage require review.'}}
    out.mkdir(parents=True,exist_ok=False)
    (out/'cohort.json').write_text(json.dumps(cohort,indent=2)+'\n')
    print(json.dumps({'selected_intervals':len(selected),'source_summaries':len(sources)}))


def run(out):
    path=out/'cohort.json'; cohort=json.loads(path.read_text())
    if cohort['policies']!={k:list(v) for k,v in POLICIES.items()} or cohort['seeds']!=list(SEEDS):
        raise ValueError('Frozen configuration differs from the fixed experiment')
    if (out/'summary.json').exists(): raise ValueError('Run already recorded; use a new cohort')
    records=[]; loaded={}
    for item in cohort['selected']:
        unit=ROOT/'units'/item['unit']
        if unit.parent.resolve()!=(ROOT/'units').resolve(): raise ValueError('Invalid unit path')
        if item['unit'] not in loaded: loaded[item['unit']]=read_unit(unit)
        card,grid,panels,wide,hashes=loaded[item['unit']]
        if hashes!=item['source_sha256']: raise ValueError('Source data changed since cohort freeze')
        origin=pd.Timestamp(item['asof'])
        for seed in SEEDS:
            draws=four_forecasts(panels,grid,origin,cohort['n_draws'],seed)
            truth,dates=holdout(wide,origin,list(grid.horizons),'log_return')
            if dates!=item['target_dates']: raise ValueError('Target mapping changed')
            scores={name:_score({'realized':truth,'_samples':sample.reshape(cohort['n_draws'],-1),
                       'expected_grid':grid,'card':card,'unit_handle':item['unit'],'grid_source':'card',
                       'normalization_mode':NormalizationMode.RAW_UNRANKABLE,'ref_scale':None})
                    for name,sample in draws.items()}
            records.append({**item,'status':'completed','seed':seed,'scores':scores,
                'change_percent':{name:{c:100*(s[c]/scores['reference'][c]-1) if scores['reference'][c]>0 else None
                    for c in ('composite','marginal','joint','tail')} for name,s in scores.items() if name!='reference'}})
        print(f'[scored] {item["unit"]} at {item["asof"]}',flush=True)
    comparisons=[]
    for seed in SEEDS:
        for name in ('half-drift','wide','combined'):
            group=[r['change_percent'][name] for r in records if r['seed']==seed]
            medians={c:float(np.median([r[c] for r in group if r[c] is not None]))
                     if any(r[c] is not None for r in group) else None for c in ('composite','marginal','joint','tail')}
            comparisons.append({'seed':seed,'model':name,'intervals':len(group),
                'better':sum(r['composite']<0 for r in group),'worse':sum(r['composite']>0 for r in group),
                'median_change':medians})
    years=len({r['asof'][:4] for r in cohort['selected']})
    enough=len(cohort['selected'])>=12 and years>=3
    combined=[r for r in comparisons if r['model']=='combined']
    passed=enough and all(r['better']>r['worse'] and r['median_change']['composite'] is not None
        and r['median_change']['composite']<0 and all(r['median_change'][c] is not None
        and r['median_change'][c]<=1e-10 for c in ('joint','tail')) for r in combined)
    summary={'executive_summary':'Fixed four-policy experiment on frozen uninspected intervals; no deployment.',
        'cohort_sha256':digest(path),'common_version':version('qfbench2-common'),'model_api_calls':0,
        'intervals':len(cohort['selected']),'origin_years':years,'scored_runs':len(records),
        'acceptance_passed':passed,'enough_intervals_and_years':enough,
        'implementation_sha256':{str(p.relative_to(ROOT)):digest(p) for p in
            (Path(__file__),ROOT/'qfbench2_track_forecasting/cli.py',ROOT/'qfbench2_track_forecasting/scoring.py')},
        'comparisons':comparisons,'results':records}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['## Executive summary (read this first)','',summary['executive_summary'],'',
        f'{summary["intervals"]} intervals across {years} origin years; {len(records)} three-seed runs.',
        f'Predeclared combined-policy acceptance passed: {passed}. Sample sufficiency passed: {enough}.',
        'All prior supplied target intervals and new shared-asset overlaps are excluded.',
        'Training history can overlap; fixed snapshots do not reconstruct publication vintages.',
        'Percentages are paired per-interval raw-error changes, not official normalized scores.',
        'Repeated seeds are sampling checks, not additional independent intervals.','',
        '| Seed | Policy | Better / Worse | Median composite | Median joint | Median tail |',
        '|---|---|---|---:|---:|---:|']
    for r in comparisons:
        cells=[f'{r["median_change"][c]:+.2f}%' if r['median_change'][c] is not None else 'n/a' for c in ('composite','joint','tail')]
        lines.append(f'| {r["seed"]} | {r["model"]} | {r["better"]}/{r["worse"]} | '+' | '.join(cells)+' |')
    lines+=['','No policy is promoted by this script. If insufficient new intervals remain, additional eligible data is needed.',
        'Results on older remaining periods do not establish performance in a newer market regime.']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('results','implementation_sha256')},indent=2))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--plan',action='store_true'); mode.add_argument('--run',action='store_true')
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--prior-summary',type=Path,nargs='+',default=[])
    args=parser.parse_args(argv); out=outside_repo(args.out)
    if args.plan:
        if not args.prior_summary: parser.error('Supply all prior historical summary files')
        plan(out,args.prior_summary)
    else: run(out)
    return 0


if __name__=='__main__': raise SystemExit(main())
