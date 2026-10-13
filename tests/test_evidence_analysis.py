"""## Executive summary (read this first)

Synthetic drivers test quote conversion, conflicting evidence, date boundaries
and exact-input caching. They do not establish real-world predictive benefit.
"""
import copy

import numpy as np
import pytest

from baselines.directional_text import apply_directional, evidence_spans
from baselines.evidence_analysis import PromptReplyCache, resolve_analysis


DOCS=[{'doc_id':'synthetic','timestamp':'2020-01-01',
       'text':'Synthetic policy was announced on January 1, 2020. '
              'Synthetic future policy changes take effect on January 6, 2020. '
              'Additional purchases will occur on January 8, 2020.'}]
SID=next(iter(evidence_spans(DOCS)))


def parsed(entity='JPY',direction='up'):
    driver={'id':'d1','change_basis':'explicit_future_path','timing':'within_horizon',
            'announcement_date':'2020-01-01','effective_date':'2020-01-06',
            'remaining_future_change':{'description':'Synthetic future policy changes take effect on January 6, 2020.',
                                       'date':'2020-01-06','evidence':[{'span_id':SID}]},
            'effect':{'basis':'currency_strength','entity':entity,'direction':direction},
            'mechanism':'Synthetic future policy transmission.', 'evidence':[{'span_id':SID}]}
    return {'assets':{'JPY':{'drivers':[driver],'pricing':{'label':'unclear',
        'basis':'unavailable','explanation':'No supplied pricing evidence.','evidence':[]},
        'missing_evidence':'none'}}}


def resolve(reply,docs=DOCS):
    return resolve_analysis(reply,['JPY'],[20],'2020-01-02','level',docs,
                            {'JPY':{'value_unit':'jpy_per_usd'}})


@pytest.mark.parametrize('entity,strength,numeric',[
    ('JPY','up','down'),('JPY','down','up'),('USD','up','up'),('USD','down','down')])
def test_currency_strength_is_converted_before_existing_bounded_shift(entity,strength,numeric):
    reply,audit=resolve(parsed(entity,strength))
    assert reply['assets']['JPY']['direction']==numeric
    samples=np.zeros((200,1,1))
    out,ledger,matched=apply_directional(samples,['JPY'],[20],{'JPY':2.},reply,DOCS,evidence_format='span')
    assert matched==1
    np.testing.assert_allclose(out,.2 if numeric=='up' else -.2)


def test_conflicting_cited_drivers_cannot_select_a_direction_by_list_order():
    r=parsed();d=copy.deepcopy(r['assets']['JPY']['drivers'][0]);d['id']='d2';d['effect']['direction']='down'
    r['assets']['JPY']['drivers'].append(d)
    first,a=resolve(r)
    r['assets']['JPY']['drivers'].reverse()
    second,b=resolve(r)
    assert first['assets']['JPY']['direction']==second['assets']['JPY']['direction']=='neutral'
    assert a['JPY']['decision_reason']=='balanced_evidence'


@pytest.mark.parametrize('basis,timing',[
    ('maintained_stance_only','within_horizon'),('conditional_scenario_only','within_horizon'),
    ('past_change','within_horizon'),('scheduled_action','unknown'),('scheduled_action','outside_horizon')])
def test_unsupported_increment_or_timing_does_not_trigger_a_shift(basis,timing):
    r=parsed();d=r['assets']['JPY']['drivers'][0];d['change_basis']=basis;d['timing']=timing
    reply,a=resolve(r)
    assert reply['assets']['JPY']['direction']=='neutral'
    assert a['JPY']['discarded_drivers']


def test_timestamp_only_pricing_cannot_silence_a_supported_driver():
    r=parsed();r['assets']['JPY']['pricing'].update(label='likely',basis='timestamp_only')
    reply,a=resolve(r)
    assert reply['assets']['JPY']['priced_in']=='unclear'
    assert reply['assets']['JPY']['direction']=='down'
    assert a['JPY']['pricing_downgraded']


def test_fabricated_sources_wrong_asset_keys_and_future_documents_are_refused():
    r=parsed();r['assets']['JPY']['drivers'][0]['evidence']=[{'span_id':'fabricated'}]
    reply,a=resolve(r)
    assert not reply['assets'] and a['JPY']['status']=='invalid_driver'
    r=parsed();r['assets']['UST_10Y']=r['assets'].pop('JPY')
    assert resolve(r)[1]['schema']=='asset_keys_mismatch'
    with pytest.raises(ValueError,match='Future or undated'):
        resolve(parsed(),[{**DOCS[0],'timestamp':'2021-01-01'}])


@pytest.mark.parametrize('field', ['announcement_date', 'effective_date', 'remaining_future_change'])
def test_legacy_or_incomplete_timeline_cannot_move_a_forecast(field):
    r = parsed()
    del r['assets']['JPY']['drivers'][0][field]
    reply, audit = resolve(r)
    assert reply['assets']['JPY']['direction'] == 'neutral'
    assert audit['JPY']['discarded_drivers'][0]['reason'] == 'missing_event_timing'


def test_completed_event_is_refused_even_when_model_labels_it_scheduled_and_within_horizon():
    r = parsed()
    d = r['assets']['JPY']['drivers'][0]
    d.update(change_basis='scheduled_action', effective_date='2020-01-01')
    d['remaining_future_change'].update(date='2020-01-01', description='Synthetic policy was announced on January 1, 2020.')
    reply, audit = resolve(r)
    assert reply['assets']['JPY']['direction'] == 'neutral'
    assert audit['JPY']['discarded_drivers'][0]['reason'] == 'no_remaining_future_change'


def test_past_effective_event_needs_a_separately_cited_future_increment():
    r = parsed()
    d = r['assets']['JPY']['drivers'][0]
    d.update(change_basis='scheduled_action', effective_date='2020-01-01')
    d['remaining_future_change'].update(date='2020-01-08', description='Additional purchases will occur on January 8, 2020.')
    reply, audit = resolve(r)
    assert reply['assets']['JPY']['direction'] == 'down'
    assert audit['JPY']['event_timing'][0]['past_effective_with_future_increment'] is True
    d['remaining_future_change']['description'] = 'Unspecified further weakening is expected.'
    reply, audit = resolve(r)
    assert reply['assets']['JPY']['direction'] == 'neutral'
    assert audit['JPY']['discarded_drivers'][0]['reason'] == 'uncited_future_increment'


@pytest.mark.parametrize('value', ['2020-02-30', '2020-1-6', '2020-01-06T00:00:00', True])
def test_invalid_effective_dates_do_not_pass_declared_timing(value):
    r = parsed()
    r['assets']['JPY']['drivers'][0]['effective_date'] = value
    reply, audit = resolve(r)
    assert reply['assets']['JPY']['direction'] == 'neutral'
    assert audit['JPY']['discarded_drivers'][0]['reason'] == 'invalid_event_date'


def test_date_absent_from_cited_spans_cannot_be_invented_from_horizon():
    r = parsed()
    d = r['assets']['JPY']['drivers'][0]
    d['effective_date'] = d['remaining_future_change']['date'] = '2020-01-09'
    reply, audit = resolve(r)
    assert reply['assets']['JPY']['direction'] == 'neutral'
    assert audit['JPY']['discarded_drivers'][0]['reason'] == 'uncited_effective_date'


def test_announcement_date_is_bound_to_a_cited_source_release():
    r = parsed()
    r['assets']['JPY']['drivers'][0]['announcement_date'] = '2020-01-02'
    reply, audit = resolve(r)
    assert reply['assets']['JPY']['direction'] == 'neutral'
    assert audit['JPY']['discarded_drivers'][0]['reason'] == 'announcement_not_bound_to_source'


def test_future_increment_cannot_precede_event_effective_date():
    r = parsed()
    d = r['assets']['JPY']['drivers'][0]
    d['effective_date'] = '2020-01-08'
    reply, audit = resolve(r)
    assert reply['assets']['JPY']['direction'] == 'neutral'
    assert audit['JPY']['discarded_drivers'][0]['reason'] == 'increment_before_effective_date'


def test_event_after_short_horizon_does_not_shift_that_horizon():
    r = parsed()
    reply, audit = resolve_analysis(r, ['JPY'], [1, 20], '2020-01-02', 'level', DOCS,
                                   {'JPY': {'value_unit': 'jpy_per_usd'}})
    assert reply['assets']['JPY']['eligible_horizons'] == [20]
    samples = np.zeros((200, 1, 2))
    out, _, _ = apply_directional(samples, ['JPY'], [1, 20], {'JPY': 2.}, reply, DOCS, evidence_format='span')
    np.testing.assert_array_equal(out[:, :, 0], samples[:, :, 0])
    np.testing.assert_allclose(out[:, :, 1], -.2)


def test_actual_date_overrides_within_horizon_claim_and_end_boundary_is_inclusive():
    r = parsed()
    # Friday January 3 is the one-business-day boundary; Monday January 6 is outside.
    reply, audit = resolve_analysis(r, ['JPY'], [1], '2020-01-02', 'level', DOCS,
                                   {'JPY': {'value_unit': 'jpy_per_usd'}})
    assert reply['assets']['JPY']['direction'] == 'neutral'
    assert audit['JPY']['discarded_drivers'][0]['reason'] == 'event_outside_horizon'
    reply, _ = resolve_analysis(r, ['JPY'], [2], '2020-01-02', 'level', DOCS,
                               {'JPY': {'value_unit': 'jpy_per_usd'}})
    assert reply['assets']['JPY']['eligible_horizons'] == [2]


def test_cache_reuses_exact_inputs_but_invalidates_model_and_prompt_changes(tmp_path,monkeypatch):
    calls=[]
    cache=PromptReplyCache(tmp_path,lambda p:(calls.append(p) or parsed(),'',''))
    monkeypatch.setenv('MODEL_NAME','synthetic-a')
    cache('prompt-a');cache('prompt-a')
    assert cache.calls==1 and cache.hits==1
    monkeypatch.setenv('MODEL_NAME','synthetic-b');cache('prompt-a');cache('prompt-b')
    assert cache.calls==3 and len(calls)==3


def test_failed_model_reply_is_not_cached(tmp_path):
    cache=PromptReplyCache(tmp_path,lambda p:(None,'model request failed (URLError)',''))
    cache('prompt');cache('prompt')
    assert cache.calls==2 and cache.hits==0
    assert not list(tmp_path.glob('*.json'))


@pytest.mark.parametrize('target_type', ['log_return', 'cumulative_log_return'])
@pytest.mark.parametrize('direction', ['up', 'down'])
def test_factor_return_driver_uses_card_target_semantics(target_type, direction):
    r = parsed()
    spec = r['assets'].pop('JPY')
    r['assets']['MOM'] = spec
    spec['drivers'][0]['effect'] = {
        'basis': 'return', 'entity': 'MOM', 'direction': direction,
    }
    context = {'MOM': {'value_unit': 'cumulative_log_return'}}
    reply, audit = resolve_analysis(
        r, ['MOM'], [5, 20], '2020-01-02', target_type, DOCS, context,
    )
    assert reply['assets']['MOM']['direction'] == direction
    assert not audit['MOM']['discarded_drivers']
    samples = np.zeros((200, 1, 2))
    out, _, matched = apply_directional(
        samples, ['MOM'], [5, 20], {'MOM': .02}, reply, DOCS,
        evidence_format='span',
    )
    sign = 1 if direction == 'up' else -1
    np.testing.assert_allclose(out, np.broadcast_to(sign * np.array([.0005, .002]), out.shape))
    assert matched == 1
    # A yield or generic numeric effect must still be refused for a return target.
    for basis in ('yield', 'numeric'):
        spec['drivers'][0]['effect']['basis'] = basis
        invalid, audit = resolve_analysis(
            r, ['MOM'], [5, 20], '2020-01-02', target_type, DOCS, context,
        )
        assert invalid['assets']['MOM']['direction'] == 'neutral'
        assert audit['MOM']['discarded_drivers'][0]['reason'] == 'unknown_or_incompatible_numeric_effect'


def test_runner_saves_raw_analysis_and_scores_three_seeds_with_one_model_call(tmp_path,monkeypatch):
    import json
    from scripts import validate_analysis as runner
    from scripts.validate_stability import CallBudget
    unit=tmp_path/'units'/'synthetic';unit.mkdir(parents=True)
    (unit/'card.toml').write_text('# Executive summary: synthetic FX metadata only.\n[targets]\ntarget_type="level"\nvalue_unit="jpy_per_usd"\n')
    out=tmp_path/'out';out.mkdir()
    job={'id':'analysis-test','source_job_id':'source-test','selection_reason':'synthetic',
         'item':{'unit':'synthetic','assets':['JPY'],'horizons':[20],'asof':'2020-01-02'}}
    calls=[];seeds=[]
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    monkeypatch.setattr(runner,'read_corpus',lambda *a,**k:(DOCS,0,0))
    monkeypatch.setattr(runner,'call_model',lambda p:(calls.append(p) or parsed(),'',''))
    def evaluate(item,units,draws,seed,*args,reply_provider,**kwargs):
        assert (out/'analysis-test-reply.json').exists()
        converted,_,_=reply_provider('statistical evaluation prompt')
        assert converted['assets']['JPY']['direction']=='down'
        seeds.append(seed)
        return {**item,'status':'completed','accepted_assets':1,'assets_requested':1,
                'prompt_sha256':'evaluation-hash','change_percent':{'composite':-.5}}
    monkeypatch.setattr(runner,'evaluate',evaluate)
    manifest={'cache_directory':None,'settings':{'n_draws':1000}}
    runner.execute(job,out,CallBudget(out,12),manifest)
    result=json.loads((out/'analysis-test.json').read_text())
    assert len(calls)==1 and seeds==[1,7,19]
    assert result['model_reply']==parsed()
    assert list(result['seed_changes'])==['1','7','19']
    # A saved reply survives loss of the final result without another model call.
    (out/'analysis-test.json').unlink()
    runner.execute(job,out,CallBudget(out,12),manifest)
    assert len(calls)==1
