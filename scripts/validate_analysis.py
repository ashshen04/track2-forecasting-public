"""## Executive summary (read this first)

Freeze and smoke-test structured analysis on 12 previously selected diagnostic windows.
Example: python scripts/validate_analysis.py --source-manifest /tmp/run/manifest.json --out /tmp/analysis
Add --resume --nvidia --run for live calls. Replies save before validation and score
offline at three seeds through the existing evaluator and shared toolkit. Optional
exact-input caching saves calls, never counts as independent repetition. All outputs
stay outside this public repository. These selected cases cannot establish accuracy.
"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
import sys
import tomllib
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from baselines.evidence_analysis import ANALYSIS_VERSION, PromptReplyCache, analysis_prompt, resolve_analysis
from baselines.reasoning_agent import call_model, read_corpus
from scripts.backtest import outside_repo
from scripts.validate_stability import CallBudget, atomic_json, check_frozen, digest, fully_valid, key
from scripts.validate_text import configuration_reason, evaluate, load_local_api_key


def freeze(source_path,limit,cache_directory):
    source=json.loads(source_path.read_text());check_frozen(source)
    selected=[]
    for case in source['cases']:
        job=next(j for j in source['jobs'] if j['group']=='first_pass' and key(j['item'])==(case['unit'],case['asof']))
        selected.append({**job,'id':f'analysis-{len(selected):03d}','source_job_id':job['id'],
                         'group':'structured_diagnostic','variant':'structured','selection_reason':case['reason']})
    if len(selected)!=12 or limit<12: raise ValueError('Budget must cover the fixed 12 diagnostic windows')
    files=[Path(__file__),ROOT/'baselines/evidence_analysis.py',ROOT/'scripts/validate_stability.py']
    return {**source,'description':'Structured analysis smoke test on previously selected, inspected diagnostic cases; no independent accuracy validation.',
        'analysis_version':ANALYSIS_VERSION,'source_manifest':str(source_path.resolve()),
        'source_files':{**source['source_files'],str(source_path.resolve()):digest(source_path)},
        'implementation_sha256':{**source['implementation_sha256'],**{str(p.relative_to(ROOT)):digest(p) for p in files}},
        'call_limit':limit,'cache_directory':str(cache_directory) if cache_directory else None,'jobs':selected}


def execute(job,out,budget,manifest):
    path=out/(job['id']+'.json')
    if path.exists(): return
    unit=ROOT/'units'/job['item']['unit'];item=job['item']
    card=tomllib.loads((unit/'card.toml').read_text());assets=item['assets'];horizons=item['horizons']
    target=card['targets']['target_type'];context={a:{'value_unit':card['targets'].get('value_unit','unknown')} for a in assets}
    docs=read_corpus(unit/'text',item['asof'],excerpt_policy='relevant')[0]
    prompt=analysis_prompt(assets,horizons,item['asof'],target,docs,context)
    prompt_hash=hashlib.sha256(prompt.encode()).hexdigest();capture=out/(job['id']+'-reply.json')
    cache_hits=0
    def live(text):
        if not budget.claim(job['id']):
            return None,'Connectivity circuit open; request not attempted' if budget.network_stopped else 'Claimed request has no saved reply; not retried',''
        parsed,reason,trace=call_model(text)
        atomic_json(capture,{'prompt_sha256':prompt_hash,'parsed':parsed,'reason':reason})
        budget.record_reply(reason)
        return parsed,reason,trace
    try:
        if capture.exists():
            saved=json.loads(capture.read_text())
            if saved['prompt_sha256']!=prompt_hash: raise ValueError('Cached prompt mismatch')
            raw,reason=saved['parsed'],saved['reason']
        else:
            provider=PromptReplyCache(Path(manifest['cache_directory']),live) if manifest['cache_directory'] else live
            raw,reason,_=provider(prompt)
            cache_hits=provider.hits if isinstance(provider,PromptReplyCache) else 0
            if raw is not None and not capture.exists():
                atomic_json(capture,{'prompt_sha256':prompt_hash,'parsed':raw,'reason':reason,'cache_hit':True})
        if raw is None:
            result={**item,'status':'not_attempted_network' if 'request not attempted' in reason else 'model_failed','reason':reason}
        else:
            converted,audit=resolve_analysis(raw,assets,horizons,item['asof'],target,docs,context)
            scores=[]
            for seed in (1,7,19):
                scores.append(evaluate(item,ROOT/'units',manifest['settings']['n_draws'],seed,'relevant','directional',
                    evidence_format='span',reply_provider=lambda _: (converted,'','')))
                if scores[-1]['status']!='completed': break
            result=scores[0]
            result={**result,'evaluation_prompt_sha256':result['prompt_sha256'],'prompt_sha256':prompt_hash,
                    'model_reply':raw,'directional_reply':converted,'analysis_audit':audit,
                    'seed_changes':{str(seed):r['change_percent'] for seed,r in zip((1,7,19),scores) if r['status']=='completed'}}
        result={**result,'job_id':job['id'],'source_job_id':job['source_job_id'],'policy':'structured_analysis',
                'cache_hits':cache_hits,'selection_reason':job['selection_reason']}
    except Exception as exc:
        result={**item,'job_id':job['id'],'status':'failed','reason':type(exc).__name__}
    atomic_json(path,result)
    print(f'[{result["status"]}] {job["id"]} {item["unit"]}',flush=True)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-manifest',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--max-calls',type=int,default=12);p.add_argument('--cache-dir',type=Path)
    p.add_argument('--resume',action='store_true');p.add_argument('--nvidia',action='store_true');p.add_argument('--run',action='store_true')
    args=p.parse_args(argv)
    if not 12<=args.max_calls<=100: p.error('The fixed design requires a budget from 12 to 100')
    out=outside_repo(args.out);cache=outside_repo(args.cache_dir) if args.cache_dir else None
    if args.resume:
        manifest=json.loads((out/'manifest.json').read_text())
        if manifest['call_limit']!=args.max_calls or manifest['cache_directory']!=(str(cache) if cache else None):
            p.error('Resume must preserve budget and cache settings')
    else:
        manifest=freeze(args.source_manifest,args.max_calls,cache);out.mkdir(parents=True,exist_ok=False)
        atomic_json(out/'manifest.json',manifest)
    check_frozen(manifest)
    if args.run:
        if not args.nvidia: p.error('The frozen local model requires --nvidia')
        load_local_api_key(ROOT/'.env.local');os.environ['MODEL_ENDPOINT']='https://integrate.api.nvidia.com/v1'
        os.environ['MODEL_NAME']=manifest['settings']['model'];os.environ['MODEL_THINKING']='on';os.environ['MODEL_MAX_TOKENS']='6000'
        blocker=configuration_reason(local_api=True)
        if blocker or not os.environ.get('MODEL_API_KEY'): p.error(blocker or 'Missing local API key')
        with (out/'run.lock').open('a') as lock:
            try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError: p.error('Another run already holds the process lock')
            budget=CallBudget(out,manifest['call_limit'])
            for job in manifest['jobs']: execute(job,out,budget,manifest)
    results=[json.loads((out/(j['id']+'.json')).read_text()) for j in manifest['jobs'] if (out/(j['id']+'.json')).exists()]
    summary={'description':manifest['description'],'analysis_version':ANALYSIS_VERSION,'planned_jobs':12,
             'finished_jobs':len(results),'claimed_calls':len(json.loads((out/'call_claims.json').read_text())) if (out/'call_claims.json').exists() else 0,
             'cache_enabled':bool(cache),'cache_hits':sum(r.get('cache_hits',0) for r in results),'fully_valid':sum(fully_valid(r) for r in results),
             'status_counts':dict(Counter(r['status'] for r in results)),
             'recommendation':'Keep the default. This is a selected-case functional smoke test, not independent accuracy or repeatability evidence.',
             'results':results}
    atomic_json(out/'summary.json',summary)
    print(f'Results: {out}; fully valid: {summary["fully_valid"]}/12; claimed: {summary["claimed_calls"]}')
    return 0


if __name__=='__main__': raise SystemExit(main())
