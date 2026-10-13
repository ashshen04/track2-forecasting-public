"""## Executive summary (read this first)

Synthetic evidence tests span attribution, repeat consistency and persistent call budgets.
No public-card outcomes appear here. Stability checks do not prove predictive accuracy.
"""
import numpy as np
import pytest
from baselines.directional_text import evidence_spans, apply_directional
from scripts.validate_stability import repeat_agreement, CallBudget


def spec(evidence):
    return {'assets':{'SYN-A':{'direction':'up','strength':'weak','priced_in':'unclear',
        'mechanism':'Synthetic forward guidance changes the yield expectation.',
        'counterargument':'The information may already be priced.', 'evidence':evidence}}}


def test_span_ids_are_stable_and_refill_actual_source():
    docs=[{'doc_id':'a','timestamp':'2020-01-01','text':'Interest rates will remain elevated. '*20}]
    spans=evidence_spans(docs)
    assert spans==evidence_spans(docs)
    sid=next(iter(spans))
    samples=np.zeros((200,1,2))
    out,ledger,matched=apply_directional(samples,['SYN-A'],[5,20],{'SYN-A':2.},
        spec([{'span_id':sid}]),docs,evidence_format='span')
    assert matched==1
    assert ledger['SYN-A']['evidence'][0]['quote']==spans[sid]['quote']
    np.testing.assert_allclose(out[0,0],[.05,.2])


@pytest.mark.parametrize('evidence',[{'span_id':'missing'}, {'span_id':'real','doc_id':'wrong'}])
def test_unknown_or_cross_document_span_is_refused(evidence):
    docs=[{'doc_id':'a','timestamp':'2020-01-01','text':'Interest rates will remain elevated.'}]
    if evidence['span_id']=='real': evidence={**evidence,'span_id':next(iter(evidence_spans(docs)))}
    out,ledger,matched=apply_directional(np.zeros((200,1,1)),['SYN-A'],[20],{'SYN-A':2.},
        spec([evidence]),docs,evidence_format='span')
    assert matched==0
    assert ledger['SYN-A']['status']=='invalid_evidence'
    assert not out.any()


def test_failed_repeats_are_disagreements_not_stable_abstentions():
    assert repeat_agreement([None,None,None])==0.
    assert repeat_agreement([0,0,0])==1.
    assert repeat_agreement([1,1,-1])==pytest.approx(1/3)


def test_call_budget_survives_restart_and_never_retries_claimed_calls(tmp_path):
    budget=CallBudget(tmp_path,2)
    assert budget.claim('job-a')
    assert not budget.claim('job-a')
    resumed=CallBudget(tmp_path,2)
    assert resumed.used==1
    assert resumed.claim('job-b')
    assert not resumed.claim('job-c')
    assert resumed.used==2


def test_connection_failures_stop_new_calls_and_persist_across_restart(tmp_path):
    budget=CallBudget(tmp_path,100)
    for i in range(3):
        assert budget.claim(f'job-{i}')
        budget.record_reply('model request failed (URLError)')
    assert budget.network_stopped
    assert not budget.claim('job-next')
    # A previously in-flight success cannot unlatch the stop.
    budget.record_reply('')
    resumed=CallBudget(tmp_path,100)
    assert resumed.network_stopped
    assert not resumed.claim('job-next')
    assert resumed.used==3


def test_success_breaks_the_failure_streak_and_model_timeout_does_not_trip_it(tmp_path):
    budget=CallBudget(tmp_path,100)
    budget.record_reply('model request failed (HTTPError)')
    budget.record_reply('')
    budget.record_reply('model request failed (URLError)')
    assert not budget.network_stopped
    budget.record_reply('model request failed (TimeoutError)')
    assert budget.connection_failures==0


def test_cached_job_replays_three_seeds_without_another_call(tmp_path,monkeypatch):
    from scripts import validate_stability as runner
    calls=[]
    docs=[{'doc_id':'a','timestamp':'2020-01-01','text':'Synthetic monetary policy evidence.'}]
    job={'id':'job-test','item':{'unit':'synthetic','asof':'2020-01-01','assets':['SYN-A']},
         'variant':'span','replicate':0,'group':'negative_control','control':{'docs':docs}}
    parsed=spec([{'span_id':next(iter(evidence_spans(docs)))}])
    monkeypatch.setattr(runner,'call_model',lambda prompt:(calls.append(prompt) or parsed,'',''))
    def fake_evaluate(item,units,n_draws,seed,*args,reply_provider,**kwargs):
        reply_provider('fixed synthetic prompt')
        return {**item,'status':'completed','accepted_assets':1,'assets_requested':1,
                'model_reply':parsed,'change_percent':{'composite':-.5},'changed_draws':True}
    monkeypatch.setattr(runner,'evaluate',fake_evaluate)
    budget=CallBudget(tmp_path,1)
    runner.execute_job(job,tmp_path,budget,{})
    assert len(calls)==1
    result=__import__('json').loads((tmp_path/'job-test.json').read_text())
    assert list(result['seed_changes'])==['1','7','19']
    # Simulate a crash after the reply was saved but before the result committed.
    (tmp_path/'job-test.json').unlink()
    runner.execute_job(job,tmp_path,CallBudget(tmp_path,1),{})
    assert len(calls)==1
