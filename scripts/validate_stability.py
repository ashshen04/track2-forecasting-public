"""## Executive summary (read this first)

Freeze and run attribution, repetition and negative-control diagnostics on inspected history.
Example: python scripts/validate_stability.py --cohort /tmp/cohort.json --prior-assessment /tmp/assessment.json --out /tmp/stability
The default freezes a manifest without API calls. --run makes at most 98 planned calls.
Claims survive interruptions; a claimed request is never silently retried. Replies are
saved before validation, allowing offline scoring at three seeds. All outputs stay
outside the public repository. These diagnostics cannot establish independent accuracy.
Three consecutive connection errors halt new requests until a new experiment is created.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import sys
import threading
import fcntl
import tomllib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from baselines.reasoning_agent import read_corpus, call_model
from baselines.directional_text import evidence_spans
from scripts.build_text_cohort import policy_documents
from scripts.backtest import outside_repo
from scripts.validate_text import evaluate, load_local_api_key, configuration_reason


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path, value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


class CallBudget:
    """Persistent pessimistic accounting, including interrupted or failed requests."""
    def __init__(self, directory, limit):
        self.path=directory/'call_claims.json'
        self.limit=limit
        self.lock=threading.Lock()
        self.claims=json.loads(self.path.read_text()) if self.path.exists() else []
        self.circuit_path=directory/'network_circuit.json'
        self.connection_failures=json.loads(self.circuit_path.read_text())['consecutive_errors'] if self.circuit_path.exists() else 0
        if len(self.claims)>limit:
            raise ValueError('Existing claims exceed the call budget')

    @property
    def used(self):
        return len(self.claims)

    @property
    def network_stopped(self):
        return self.connection_failures>=3

    def record_reply(self, reason):
        """Persist a latched stop for connection errors; timeouts alone do not trip it.

        A model taking too long or returning bad JSON is not proof of a broken
        connection. A concurrent in-flight response cannot clear an open circuit.
        """
        with self.lock:
            if self.network_stopped: return
            transport_error=reason in ('model request failed (URLError)', 'model request failed (HTTPError)')
            self.connection_failures=self.connection_failures+1 if transport_error else 0
            atomic_json(self.circuit_path,{'consecutive_errors':self.connection_failures,
                                          'stopped':self.network_stopped})

    def claim(self, job_id):
        with self.lock:
            if self.network_stopped or job_id in self.claims or self.used>=self.limit:
                return False
            self.claims.append(job_id)
            atomic_json(self.path,self.claims)
            return True


def repeat_agreement(signs):
    """Invalid observations never form an agreeing pair, including two failures."""
    pairs=[(a,b) for i,a in enumerate(signs) for b in signs[i+1:]]
    return sum(a is not None and b is not None and a==b for a,b in pairs)/len(pairs) if pairs else 0.


def key(item):
    return item['unit'],item['asof']


def diagnostic_cases(prior):
    """Outcome-informed cases diagnose stability only, not population accuracy."""
    recent=prior['experiments'][-1]['results']
    groups=[('previous_improvement',[r for r in recent if r.get('changed_draws') and
                r.get('change_percent',{}).get('composite',0)<0][:3]),
            ('previous_worse',[r for e in prior['experiments'] for r in e['results'] if
                r.get('policy')=='directional' and r.get('changed_draws') and
                r.get('change_percent',{}).get('composite',0)>0][:1]),
            ('previous_rejection',[r for r in recent if r['status']!='completed'][:4]),
            ('previous_neutral',[r for r in recent if r['status']=='completed' and not r.get('changed_draws')][:4])]
    cases=[]; seen=set()
    for reason,rows in groups:
        for row in rows:
            if key(row) not in seen:
                cases.append({'item':row,'diagnostic_reason':reason}); seen.add(key(row))
    if len(cases)!=12:
        raise ValueError('The frozen diagnostic design requires exactly 12 distinct windows')
    return cases


def controls_for(case, units):
    """Choose one existing administrative document, preferring the target unit.

    A lexical non-policy classification is a diagnostic proxy for irrelevance.
    Source dates are enforced; missing controls are recorded rather than invented.
    """
    origin=case['asof']; candidates=[]
    ordered=[units/case['unit'],*[u for u in sorted(units.iterdir()) if u.name!=case['unit']]]
    for unit in ordered:
        if not (unit/'text/corpus_index.json').exists(): continue
        docs,_,_=read_corpus(unit/'text',origin,excerpt_policy='relevant')
        indexed=json.loads((unit/'text/corpus_index.json').read_text()).get('documents',[])
        paths={str(d.get('doc_id')):unit/'text'/d.get('file','') for d in indexed}
        for doc in docs:
            if policy_documents([doc]): continue
            source=paths.get(str(doc['doc_id']))
            if source is None or not source.is_file() or (unit/'text').resolve() not in source.resolve().parents:
                continue
            candidates.append((0 if unit.name==case['unit'] else 1,
                               -pd.Timestamp(doc['timestamp']).value, str(source),doc,source))
    if not candidates: return None
    _,_,_,doc,path=min(candidates,key=lambda r:r[:3])
    return {'docs':[doc], 'source_files':{str(path.relative_to(ROOT)):digest(path),
             str(path.parent/'corpus_index.json'):digest(path.parent/'corpus_index.json')},
            'selection':'same-unit preferred; newest eligible lexical non-policy text; no scores used'}


def freeze_manifest(cohort_path, prior_path, units, limit):
    cohort=json.loads(cohort_path.read_text()); prior=json.loads(prior_path.read_text())
    items=cohort['items']
    if len(items)!=50 or len({key(r) for r in items})!=50:
        raise ValueError('Expected the existing 50-window policy cohort')
    cases=diagnostic_cases(prior)
    all_items={key(r):r for r in items}
    jobs=[]
    def append(item,variant,replicate=0,group='first_pass',control=None):
        clean={k:item[k] for k in ('unit','asof','family','assets','horizons','target_type',
               'target_dates','source_sha256','corpus_sha256','provenance_warning') if k in item}
        clean['card_sha256']=digest(units/item['unit']/'card.toml')
        clean['exposure']='previously_inspected_history'
        jobs.append({'id':f'job-{len(jobs):03d}','item':clean,'variant':variant,'replicate':replicate,
                     'group':group,'control':control})
    for item in items: append(item,'span')
    for case in cases:
        item=all_items.get(key(case['item']),case['item'])
        if 'assets' not in item:
            card=tomllib.loads((units/item['unit']/'card.toml').read_text())
            item={**item,'assets':card['targets']['asset_ids'],'horizons':card['targets']['horizons'],
                  'target_type':card['targets']['target_type']}
        for replicate in (1,2): append(item,'span',replicate,'targeted_repeat')
        append(item,'quote',0,'quote_comparison')
        control=controls_for(item,units)
        append(item,'span',0,'negative_control',control)
    missing=[c for c in cases if key(c['item']) not in all_items]
    if missing:
        raise ValueError('All 12 diagnostic windows must belong to the frozen 50-window cohort')
    if len(jobs)>98: raise ValueError('Diagnostic design exceeds 98 planned requests')
    files=[ROOT/'scripts/validate_stability.py',ROOT/'scripts/validate_text.py',
           ROOT/'baselines/directional_text.py',ROOT/'baselines/reasoning_agent.py',
           ROOT/'qfbench2_track_forecasting/cli.py',ROOT/'qfbench2_track_forecasting/scoring.py']
    return {'version':1,'description':'Reliability diagnosis on previously inspected history; no independent holdout.',
        'source_files':{str(cohort_path.resolve()):digest(cohort_path),str(prior_path.resolve()):digest(prior_path)},
        'implementation_sha256':{str(p.relative_to(ROOT)):digest(p) for p in files},
        'common_version':version('qfbench2-common'),'call_limit':limit,
        'settings':{'model':'nvidia/nemotron-3-super-120b-a12b','thinking':True,'max_tokens':6000,
                    'n_draws':1000,'score_seeds':[1,7,19],'excerpt_policy':'relevant','workers':2},
        'gates':{'first_pass_fully_valid_rate':.9,'repeat_pair_agreement':.8,
                 'minimum_controls':8,'maximum_control_shift_rate':.1},
        'cases':[{'unit':c['item']['unit'],'asof':c['item']['asof'],'reason':c['diagnostic_reason']} for c in cases],
        'jobs':jobs}


def check_frozen(manifest):
    for name, expected in manifest['source_files'].items():
        if digest(Path(name))!=expected: raise ValueError('Frozen source manifest changed')
    for name,expected in manifest['implementation_sha256'].items():
        if digest(ROOT/name)!=expected: raise ValueError('Implementation changed; create a new experiment')
    if version('qfbench2-common')!=manifest['common_version']:
        raise ValueError('Shared toolkit changed')
    for job in manifest['jobs']:
        item=job['item']; unit=ROOT/'units'/item['unit']
        if unit.parent.resolve()!=(ROOT/'units').resolve(): raise ValueError('Invalid unit path')
        if digest(unit/'card.toml')!=item['card_sha256']: raise ValueError('Card changed')
        for name,expected in {**item['source_sha256'],**item.get('corpus_sha256',{})}.items():
            path=unit/name
            if unit.resolve() not in path.resolve().parents or digest(path)!=expected:
                raise ValueError('Frozen panel or corpus changed')
        if job['control']:
            for name,expected in job['control']['source_files'].items():
                path=Path(name) if Path(name).is_absolute() else ROOT/name
                if (ROOT/'units').resolve() not in path.resolve().parents or digest(path)!=expected:
                    raise ValueError('Negative-control source changed')
            if any(pd.Timestamp(d['timestamp'])>pd.Timestamp(item['asof']) for d in job['control']['docs']):
                raise ValueError('Future negative-control document')


def fully_valid(result):
    return result.get('status')=='completed' and result.get('accepted_assets')==result.get('assets_requested')


def signs(result,assets):
    output=[]
    for asset in assets:
        entry=result.get('adjustments',{}).get(asset,{})
        if entry.get('status') not in ('abstained','priced_in_abstention','direction_applied'):
            output.append(None)
        else: output.append(int(np.sign(entry.get('longest_horizon_shift',0.))))
    return output


def report(manifest,out):
    results={j['id']:json.loads((out/(j['id']+'.json')).read_text()) for j in manifest['jobs']
             if (out/(j['id']+'.json')).exists()}
    first=[results[j['id']] for j in manifest['jobs'] if j['group']=='first_pass' and j['id'] in results]
    repeat_scores=[]; repeat_details=[]
    for case in manifest['cases']:
        jobs=[j for j in manifest['jobs'] if key(j['item'])==(case['unit'],case['asof']) and
              j['variant']=='span' and j['group'] in ('first_pass','targeted_repeat','targeted_repeat_baseline')]
        jobs=sorted(jobs,key=lambda j:j['replicate'])
        assets=jobs[0]['item']['assets']
        for i,asset in enumerate(assets):
            values=[signs(results.get(j['id'],{}),assets)[i] for j in jobs]
            # Missing replicate is an explicit failed observation.
            values+= [None]*(3-len(values))
            agreement=repeat_agreement(values)
            repeat_scores.append(agreement)
            repeat_details.append({'unit':case['unit'],'asof':case['asof'],'asset':asset,
                                   'reason':case['reason'],'signs':values,'pair_agreement':agreement})
    controls=[results[j['id']] for j in manifest['jobs'] if j['group']=='negative_control' and
              j['id'] in results and results[j['id']].get('status')!='control_unavailable']
    control_valid=[r for r in controls if fully_valid(r)]
    shifted=sum(r.get('changed_draws',False) for r in control_valid)
    rate=shifted/len(control_valid) if control_valid else None
    valid=sum(fully_valid(r) for r in first)
    first_rate=valid/50
    agreement=float(np.mean(repeat_scores)) if repeat_scores else 0.
    gates={'first_pass':first_rate>=.9,'repetition':agreement>=.8,
           'negative_control':len(control_valid)>=8 and rate is not None and rate<=.1}
    changes=[r['change_percent']['composite'] for r in first if fully_valid(r)]
    by_unit=[]
    for unit in sorted({j['item']['unit'] for j in manifest['jobs'] if j['group']=='first_pass'}):
        local=[r for r in first if r['unit']==unit and fully_valid(r)]
        v=[r['change_percent']['composite'] for r in local]
        by_unit.append({'unit':unit,'valid_windows':len(local),'median_relative_change':float(np.median(v)) if v else None})
    quote_jobs=[j for j in manifest['jobs'] if j['group']=='quote_comparison']
    quote_results=[results.get(j['id'],{}) for j in quote_jobs]
    matched_span=[next((r for r in first if key(r)==key(j['item'])),{}) for j in quote_jobs]
    comparisons={'windows':len(quote_jobs),'span_fully_valid':sum(fully_valid(r) for r in matched_span),
                 'quote_fully_valid':sum(fully_valid(r) for r in quote_results)}
    mc=[r for r in first if fully_valid(r) and r.get('changed_draws')]
    summary={'description':manifest['description'],'planned_jobs':len(manifest['jobs']),
        'finished_jobs':len(results),'claimed_calls':len(json.loads((out/'call_claims.json').read_text())) if (out/'call_claims.json').exists() else 0,
        'first_pass':{'windows':len(first),'fully_valid':valid,'fully_valid_rate':first_rate,
                      'status_counts':dict(Counter(r['status'] for r in first)),
                      'improved':sum(v<0 for v in changes),'worse':sum(v>0 for v in changes),
                      'unchanged':sum(v==0 for v in changes),'median_relative_change':float(np.median(changes)) if changes else None},
        'repetition':{'mean_pair_agreement':agreement,'assets':repeat_details},
        'negative_controls':{'available_results':len(controls),'fully_valid':len(control_valid),
                             'shifted':shifted,'shift_rate':rate},
        'gates':gates,'by_unit':by_unit,
        'citation_comparison':comparisons,
        'monte_carlo':{'changed_fully_valid_windows':len(mc),
                       'improved_all_three_seeds':sum(r.get('all_seed_composite_improvement',False) for r in mc),
                       'seed_sign_disagreements':sum(len(set(r.get('monte_carlo_change_signs',[])))>1 for r in mc)},
        'recommendation':'Keep the default. Even passing reliability gates would not establish independent forecasting benefit.',
        'results':list(results.values())}
    atomic_json(out/'summary.json',summary)
    lines=['## Executive summary (read this first)','',manifest['description'],
        'No independent accuracy or cross-unit raw score is calculated. All failures remain visible.','',
        f'Finished jobs: {len(results)}/{len(manifest["jobs"])}. Claimed calls: {summary["claimed_calls"]}.','',
        f'First pass fully valid: {valid}/50 ({first_rate:.1%}); required 90%.',
        f'Repeat pair agreement: {agreement:.1%}; required 80%. Failed pairs disagree.',
        f'Valid negative controls: {len(control_valid)}; changed: {shifted}; rate: {rate}. Required >=8 controls and <=10% shifts.',
        f'Gates: {gates}.','',
        f'Paired citation validity: {comparisons}. Monte Carlo checks: {summary["monte_carlo"]}.','',
        f'First-pass valid-window composite changes: {summary["first_pass"]}.','',
        '## Results','', '| Job | Unit | Origin | Variant / group | Status | Fully valid | Composite change |',
        '|---|---|---|---|---|---|---:|']
    for job in manifest['jobs']:
        r=results.get(job['id'],{})
        delta=r.get('change_percent',{}).get('composite')
        formatted=f'{delta:+.3f}%' if delta is not None else '—'
        lines.append(f'| {job["id"]} | {job["item"]["unit"]} | {job["item"]["asof"]} | {job["variant"]} / {job["group"]} | {r.get("status","pending")} | {fully_valid(r)} | {formatted} |')
    lines+=['',summary['recommendation'],'',
            'Repeat and comparison cases were selected from earlier outcomes for diagnosis only.',
            'Negative controls use a lexical non-policy filter; this does not certify semantic irrelevance.',
            'Per-seed score changes and Monte Carlo sign consistency are recorded in each result.']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    return summary


def execute_job(job,out,budget,manifest):
    path=out/(job['id']+'.json'); captured=out/(job['id']+'-reply.json')
    if path.exists(): return
    if job['group']=='negative_control' and job['control'] is None:
        atomic_json(path,{**job['item'],'job_id':job['id'],'status':'control_unavailable'})
        return
    def provider(prompt):
        prompt_hash=hashlib.sha256(prompt.encode()).hexdigest()
        if captured.exists():
            raw=json.loads(captured.read_text())
            if raw['prompt_sha256']!=prompt_hash: raise ValueError('Cached prompt mismatch')
            return raw['parsed'],raw['reason'],''
        if not budget.claim(job['id']):
            reason='Connectivity circuit open; request not attempted' if budget.network_stopped else 'Claimed request has no saved reply; not retried'
            return None,reason,''
        parsed,reason,trace=call_model(prompt)
        atomic_json(captured,{'prompt_sha256':prompt_hash,'parsed':parsed,'reason':reason,
                              'reasoning_sha256':hashlib.sha256(trace.encode()).hexdigest()})
        budget.record_reply(reason)
        return parsed,reason,trace
    docs=job['control']['docs'] if job['control'] else None
    try:
        result=evaluate(job['item'],ROOT/'units',1000,1,'relevant','directional',
            evidence_format=job['variant'],reply_provider=provider,documents_override=docs)
        if result.get('reason')=='Connectivity circuit open; request not attempted':
            result['status']='not_attempted_network'
        seed_scores={}
        if result.get('status')=='completed':
            parsed=result['model_reply']
            for seed in (1,7,19):
                replay=result if seed==1 else evaluate(job['item'],ROOT/'units',1000,seed,'relevant','directional',
                    evidence_format=job['variant'],reply_provider=lambda _: (parsed,'',''),documents_override=docs)
                seed_scores[str(seed)]=replay['change_percent']
            values=[v['composite'] for v in seed_scores.values()]
            result['seed_changes']=seed_scores
            result['monte_carlo_change_signs']=[int(np.sign(v)) for v in values]
            result['all_seed_composite_improvement']=all(v<0 for v in values)
        # Save exact input evidence IDs separately, with source timestamp and excerpt offsets.
        source_docs=docs if docs is not None else read_corpus(ROOT/'units'/job['item']['unit']/'text',
                                       job['item']['asof'],excerpt_policy='relevant')[0]
        atomic_json(out/(job['id']+'-evidence.json'),evidence_spans(source_docs))
        result={**result,'job_id':job['id'],'variant':job['variant'],'replicate':job['replicate'],
                'group':job['group']}
    except Exception as exc:
        result={**job['item'],'job_id':job['id'],'status':'failed','reason':type(exc).__name__}
    atomic_json(path,result)
    print(f'[{result["status"]}] {job["id"]} {job["group"]} {job["item"]["unit"]}',flush=True)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cohort',type=Path,required=True)
    p.add_argument('--prior-assessment',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--max-calls',type=int,default=100)
    p.add_argument('--run',action='store_true')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--nvidia',action='store_true')
    args=p.parse_args(argv)
    if not 1<=args.max_calls<=100: p.error('Call limit must be between 1 and 100')
    out=outside_repo(args.out)
    manifest_path=out/'manifest.json'
    if args.resume:
        manifest=json.loads(manifest_path.read_text())
        if manifest['call_limit']!=args.max_calls: p.error('Resume must preserve the frozen call limit')
    else:
        out.mkdir(parents=True,exist_ok=False)
        manifest=freeze_manifest(args.cohort,args.prior_assessment,ROOT/'units',args.max_calls)
        atomic_json(manifest_path,manifest)
    check_frozen(manifest)
    if args.run:
        if not args.nvidia: p.error('This frozen local experiment requires --nvidia')
        load_local_api_key(ROOT/'.env.local')
        os.environ['MODEL_ENDPOINT']='https://integrate.api.nvidia.com/v1'
        os.environ['MODEL_NAME']=manifest['settings']['model']
        os.environ['MODEL_THINKING']='on'; os.environ['MODEL_MAX_TOKENS']='6000'
        reason=configuration_reason(local_api=True)
        if reason or not os.environ.get('MODEL_API_KEY'): p.error(reason or 'Missing local API key')
        budget=CallBudget(out,args.max_calls)
        with (out/'run.lock').open('a') as lock_file:
            try:
                fcntl.flock(lock_file,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                p.error('Another process is already running this experiment')
            # Reload claims only after obtaining the cross-process lock.
            budget=CallBudget(out,args.max_calls)
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures=[pool.submit(execute_job,job,out,budget,manifest) for job in manifest['jobs']]
                for future in as_completed(futures): future.result()
    summary=report(manifest,out)
    print(f'Results: {out}; gates: {summary["gates"]}')
    return 0


if __name__=='__main__': raise SystemExit(main())
