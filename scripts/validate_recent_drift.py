"""## Executive summary (read this first)

Validate fixed return-drift policies on externally supplied recent French factor data.
First freeze a cohort with --plan and provider ZIPs, then score with --run.
All downloads, normalized observations and scored artifacts stay outside this repo.
The provider's current snapshot is not a reconstruction of historical vintages.
"""
from __future__ import annotations
import argparse
import csv
import io
import json
import re
import sys
import tomllib
import zipfile
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from qfbench2_track_forecasting.cli import _draw
from qfbench2_track_forecasting.grid import grid_from_card
from qfbench2_track_forecasting.scoring import _score
from qfbench2_track_forecasting.normalization import NormalizationMode
from scripts.backtest import outside_repo, overlaps_excluded, holdout, SkipUnit
from scripts.validate_return_calibration import digest, prior_intervals, reserve

POLICIES={'reference':None,'half':.5,'zero':0.}
SEEDS=(1,7,19)
GRIDS=('t2-F3-readout-calendar-factors-2020','t2-F4-factor-joint-credit-2007','t2-F4-covid-mkt-2020')
PROVIDER='https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/'


def parse_provider_csv(text,mapping):
    """Parse daily percent returns; retain missing sentinels as NaN without filling."""
    rows=list(csv.reader(io.StringIO(text))); header=None; data=[]
    for row in rows:
        row=[v.strip() for v in row]
        if set(mapping)<=set(row) and row[0]=='': header=row; continue
        if not row or not re.fullmatch(r'\d{8}',row[0]): continue
        if header is None: raise ValueError('Daily table header missing')
        if len(row)!=len(header): raise ValueError('Malformed daily row')
        values=[]
        for source in mapping:
            v=float(row[header.index(source)])
            values.append(np.nan if v in (-99.99,-999.) else v/100.)
        data.append([pd.to_datetime(row[0],format='%Y%m%d'),*values])
    if not data: raise ValueError('No daily data rows')
    frame=pd.DataFrame(data,columns=['date',*mapping.values()]).set_index('date').sort_index()
    if frame.index.has_duplicates: raise ValueError('Duplicate provider dates')
    values=frame.to_numpy()
    if np.isinf(values).any() or np.any(values[np.isfinite(values)]<=-1):
        raise ValueError('Invalid decimal simple returns')
    return frame


def read_zip(path,mapping):
    with zipfile.ZipFile(path) as z:
        files=[p for p in z.infolist() if p.filename.lower().endswith('.csv')]
        if len(files)!=1 or files[0].file_size>10_000_000: raise ValueError('Expected one bounded CSV archive member')
        return parse_provider_csv(z.read(files[0]).decode('utf-8-sig'),mapping)


def drift_forecasts(panel,grid,origin,seed):
    history={'factors':panel.loc[panel.date<=origin].copy()}
    return {name:_draw(history,list(grid.assets),list(grid.horizons),str(origin.date()),1000,seed,
                       target_type='log_return',drift_scale=scale)[0] for name,scale in POLICIES.items()}


def plan(args,out):
    paths=[outside_repo(p) for p in (args.three_zip,args.momentum_zip)]
    if not args.prior_summary: raise ValueError('Supply all previous experiment summaries')
    ledger,sources=prior_intervals(args.prior_summary)
    three=read_zip(paths[0],{'Mkt-RF':'MKT','SMB':'SMB','HML':'HML'})
    momentum=read_zip(paths[1],{'Mom':'MOM'})
    wide=three.join(momentum,how='outer').sort_index()
    end=min(pd.Timestamp(args.end),three.index.max(),momentum.index.max())
    wide=wide.loc[:end]
    panel=wide.reset_index().melt(id_vars='date',var_name='asset',value_name='value')
    selected=[]; statuses=[]; cards={}
    for name in GRIDS:
        path=ROOT/'units'/name/'card.toml'; card=tomllib.loads(path.read_text()); grid=grid_from_card(card)
        cards[name]={'sha256':digest(path),'assets':list(grid.assets),'horizons':list(grid.horizons)}
        local=wide.loc[:,list(grid.assets)]; rejected=Counter(); count=0
        origins=local.dropna().index
        for origin in reversed(origins[(origins>=pd.Timestamp(args.start))]):
            target_end=origin+pd.offsets.BDay(max(grid.horizons))
            if target_end>end: continue
            if overlaps_excluded(origin,target_end,grid.assets,ledger):
                rejected['prior_or_new_shared_asset_overlap']+=1; continue
            try:
                _,dates=holdout(local,origin,list(grid.horizons),'log_return')
                history={'factors':panel.loc[panel.date<=origin]}
                _draw(history,list(grid.assets),list(grid.horizons),str(origin.date()),200,1,target_type='log_return')
            except (SkipUnit,ValueError,SystemExit) as exc:
                rejected[str(exc)]+=1; continue
            reserve(ledger,grid.assets,origin,target_end)
            selected.append({'unit':name,'assets':list(grid.assets),'horizons':list(grid.horizons),
                             'asof':str(origin.date()),'target_dates':dates})
            count+=1
        statuses.append({'grid':name,'selected':count,'rejected':dict(rejected)})
        print(f'[planned] {name}: {count}',flush=True)
    out.mkdir(parents=True,exist_ok=False)
    panel.to_parquet(out/'input_panel.parquet',index=False)
    cohort={'executive_summary':'Frozen recent external-data research grids; not official card evaluations.',
        'start':args.start,'requested_end':args.end,'data_end':str(end.date()),'policies':POLICIES,
        'seeds':list(SEEDS),'n_draws':1000,'selected':sorted(selected,key=lambda r:(r['asof'],r['unit'])),
        'grid_statuses':statuses,'cards':cards,'prior_sources':sources,'panel_sha256':digest(out/'input_panel.parquet'),
        'sources':[{'path':str(p),'sha256':digest(p),'url':PROVIDER+'ftp/'+file}
                   for p,file in zip(paths,('F-F_Research_Data_Factors_daily_CSV.zip','F-F_Momentum_Factor_daily_CSV.zip'))],
        'return_mapping':'MKT=Mkt-RF excess return; SMB/HML/MOM factor returns, percent divided by 100.',
        'acceptance':'Half policy: at every seed median composite<0, wins>losses, median joint/tail<=0; >=20 intervals and >=2 origin years. No automatic deployment.',
        'limits':['Current revised provider snapshot, including its 2025 CIZ transition; historical availability not reconstructed.',
            'Monday-Friday horizons; holidays are not filled. Skipped long horizons cannot establish long-horizon accuracy.',
            'Disjoint asset sets can share market dates; repeated seeds and correlated regimes are not independent evidence.']}
    (out/'cohort.json').write_text(json.dumps(cohort,indent=2)+'\n')
    print(json.dumps({'intervals':len(selected),'data_end':cohort['data_end']}))


def run(out):
    cohort=json.loads((out/'cohort.json').read_text())
    if cohort['policies']!=POLICIES or cohort['seeds']!=list(SEEDS): raise ValueError('Frozen policy mismatch')
    if (out/'summary.json').exists(): raise ValueError('Already scored; create a new cohort')
    if digest(out/'input_panel.parquet')!=cohort['panel_sha256']: raise ValueError('Frozen data changed')
    panel=pd.read_parquet(out/'input_panel.parquet'); wide=panel.pivot(index='date',columns='asset',values='value')
    records=[]
    for item in cohort['selected']:
        path=ROOT/'units'/item['unit']/'card.toml'
        if digest(path)!=cohort['cards'][item['unit']]['sha256']: raise ValueError('Grid card changed')
        card=tomllib.loads(path.read_text()); grid=grid_from_card(card); origin=pd.Timestamp(item['asof'])
        for seed in SEEDS:
            forecasts=drift_forecasts(panel,grid,origin,seed)
            truth,dates=holdout(wide.loc[:,list(grid.assets)],origin,list(grid.horizons),'log_return')
            if dates!=item['target_dates']: raise ValueError('Target date mapping changed')
            scores={name:_score({'realized':truth,'_samples':draws.reshape(1000,-1),'expected_grid':grid,
                'card':card,'unit_handle':'external-research-grid','grid_source':'card',
                'normalization_mode':NormalizationMode.RAW_UNRANKABLE,'ref_scale':None}) for name,draws in forecasts.items()}
            records.append({**item,'status':'completed','seed':seed,'scores':scores,
                'change_percent':{name:{c:100*(s[c]/scores['reference'][c]-1) if scores['reference'][c]>0 else None
                    for c in ('composite','marginal','joint','tail')} for name,s in scores.items() if name!='reference'}})
    comparisons=[]
    for group in ('all','multi_asset','single_asset'):
        for seed in SEEDS:
            rows=[r for r in records if r['seed']==seed and
                  (group=='all' or group=='multi_asset' and len(r['assets'])>1 or group=='single_asset' and len(r['assets'])==1)]
            for name in ('half','zero'):
                changes=[r['change_percent'][name] for r in rows]
                comparisons.append({'group':group,'seed':seed,'policy':name,'intervals':len(rows),
                    'better':sum(r['composite']<0 for r in changes),'worse':sum(r['composite']>0 for r in changes),
                    'component_intervals':{c:sum(r[c] is not None for r in changes) for c in ('composite','joint','tail')},
                    'median_change':{c:float(np.median([r[c] for r in changes if r[c] is not None]))
                        if any(r[c] is not None for r in changes) else None for c in ('composite','marginal','joint','tail')}})
    half=[r for r in comparisons if r['group']=='all' and r['policy']=='half']
    years=len({r['asof'][:4] for r in cohort['selected']})
    enough=len(cohort['selected'])>=20 and years>=2
    passed=enough and all(r['better']>r['worse'] and r['median_change']['composite'] is not None and
        r['median_change']['composite']<0 and all(r['median_change'][c] is not None and r['median_change'][c]<=1e-10
        for c in ('joint','tail')) for r in half)
    summary={'executive_summary':'Recent fixed-snapshot research validation; no automatic model promotion.',
        'cohort_sha256':digest(out/'cohort.json'),'intervals':len(cohort['selected']),'origin_years':years,
        'scored_runs':len(records),'model_api_calls':0,'enough_data':enough,'half_acceptance_passed':passed,
        'implementation_sha256':{str(p.relative_to(ROOT)):digest(p) for p in
            (Path(__file__),ROOT/'qfbench2_track_forecasting/cli.py',ROOT/'qfbench2_track_forecasting/scoring.py')},
        'comparisons':comparisons,'results':records}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['## Executive summary (read this first)','',summary['executive_summary'],'',
        f'{summary["intervals"]} intervals and {summary["scored_runs"]} three-seed runs; half acceptance: {passed}.',
        f'Recent data ends {cohort["data_end"]}. These are research grids on external data, not official task scores.',
        'Provider percent returns are converted to decimal simple returns. MKT denotes excess return.',
        'Missing holiday observations are refused. See cohort.json for zero-coverage grids.',
        'Negative paired percentage changes mean lower raw error. Zero-baseline components are omitted.','',
        '| Group | Seed | Policy | N | Better/Worse | Composite | Joint | Tail |',
        '|---|---|---|---:|---|---:|---:|---:|']
    for r in comparisons:
        cells=[f'{r["median_change"][c]:+.2f}%' if r['median_change'][c] is not None else 'n/a' for c in ('composite','joint','tail')]
        lines.append(f'| {r["group"]} | {r["seed"]} | {r["policy"]} | {r["intervals"]} | {r["better"]}/{r["worse"]} | '+' | '.join(cells)+' |')
    lines+=['','Source: '+PROVIDER+'data_library.html','',*cohort['limits']]
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('results','implementation_sha256')},indent=2))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__); mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--plan',action='store_true'); mode.add_argument('--run',action='store_true')
    parser.add_argument('--out',type=Path,required=True); parser.add_argument('--three-zip',type=Path)
    parser.add_argument('--momentum-zip',type=Path); parser.add_argument('--prior-summary',type=Path,nargs='+',default=[])
    parser.add_argument('--start',default='2025-01-02'); parser.add_argument('--end',default='2026-10-11')
    args=parser.parse_args(argv); out=outside_repo(args.out)
    if args.plan:
        if args.three_zip is None or args.momentum_zip is None: parser.error('Supply both official daily CSV ZIPs')
        plan(args,out)
    else: run(out)
    return 0


if __name__=='__main__': raise SystemExit(main())
