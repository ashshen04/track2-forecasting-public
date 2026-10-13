"""## Executive summary (read this first)
Synthetic evidence checks bounded direction shifts and abstention without fake citations.
The experiment changes centres only. It does not establish accuracy improvement.
"""
import numpy as np
import pytest
from baselines.directional_text import apply_directional, directional_prompt
from scripts.build_text_cohort import temporal_partition


def response(direction='up', quote='Policy rates will remain elevated'):
    return {'assets':{'SYN-A':{'direction':direction,'strength':'moderate','priced_in':'unlikely',
        'mechanism':'Higher expected policy rates shift the yield level upward.',
        'counterargument':'The forward curve may already reflect this guidance.',
        'evidence':[{'doc_id':'d1','quote':quote}]}}}


def test_direction_shift_accumulates_with_horizon_and_preserves_joint_shocks():
    samples=np.random.default_rng(7).normal(size=(1000,1,2))
    docs=[{'doc_id':'d1','text':'Policy rates will remain elevated for a prolonged period.'}]
    result,ledger,matched=apply_directional(samples,['SYN-A'],[5,20],{'SYN-A':2.},response(),docs)
    np.testing.assert_allclose(result-samples,np.broadcast_to([.125,.5],samples.shape))
    assert matched==1
    assert ledger['SYN-A']['status']=='direction_applied'


def test_fabricated_quote_is_refused_even_with_valid_document_id():
    samples=np.zeros((200,1,1))
    result,ledger,matched=apply_directional(samples,['SYN-A'],[20],{'SYN-A':2.},response(),
        [{'doc_id':'d1','text':'Administrative arrangements only.'}])
    np.testing.assert_array_equal(result,samples)
    assert matched==0
    assert ledger['SYN-A']['status']=='invalid_evidence'


def test_explicit_neutral_is_valid_abstention_without_a_fake_reference():
    reply=response('neutral'); reply['assets']['SYN-A']['evidence']=[]
    samples=np.zeros((200,1,1))
    result,ledger,matched=apply_directional(samples,['SYN-A'],[20],{'SYN-A':2.},reply,[])
    assert matched==1
    np.testing.assert_array_equal(result,samples)
    assert ledger['SYN-A']['status']=='abstained'


def test_already_priced_information_cannot_move_the_centre():
    reply=response();reply['assets']['SYN-A']['priced_in']='likely'
    samples=np.zeros((200,1,1))
    result,ledger,matched=apply_directional(samples,['SYN-A'],[20],{'SYN-A':2.},reply,
        [{'doc_id':'d1','text':'Policy rates will remain elevated'}])
    np.testing.assert_array_equal(result,samples)
    assert matched==1
    assert ledger['SYN-A']['status']=='priced_in_abstention'


def test_global_time_split_purges_crossing_intervals_across_units():
    items=[{'unit':str(i),'asof':f'2020-0{i+1}-01','target_dates':[end]}
           for i,end in enumerate(['2020-01-10','2020-04-10','2020-03-10','2020-04-10'])]
    kept,info=temporal_partition(items,.5)
    assert info['boundary']=='2020-03-01'
    assert info['purged']==1
    assert all(max(r['target_dates'])<info['boundary'] for r in kept if r['split']=='development')
    assert all(r['asof']>=info['boundary'] for r in kept if r['split']=='holdout')


def test_policy_filter_rejects_generic_admin_text_without_using_outcomes():
    from scripts.build_text_cohort import policy_documents
    docs=[{'doc_id':'admin','doc_type':'cb_speech','text':'Accounting standards and supervisory procedures.'},
          {'doc_id':'policy','doc_type':'cb_speech','text':'Monetary policy will address persistent inflation.'},
          {'doc_id':'statement','doc_type':'fomc_statement','text':'A dated decision statement.'}]
    assert [r['doc_id'] for r in policy_documents(docs)]==['policy','statement']


def test_long_genuine_quote_is_verified_before_audit_truncation():
    text='Monetary policy and inflation expectations remain uncertain. '*15
    reply=response(quote=text)
    samples=np.zeros((200,1,1))
    _,ledger,matched=apply_directional(samples,['SYN-A'],[20],{'SYN-A':2.},reply,
                                      [{'doc_id':'d1','text':text}])
    assert matched==1
    assert ledger['SYN-A']['evidence'][0]['quote_truncated']
    assert len(ledger['SYN-A']['evidence'][0]['quote'])==500
    reply['assets']['SYN-A']['evidence'][0]['quote'] += ' FABRICATED_SUFFIX'
    _,ledger,matched=apply_directional(samples,['SYN-A'],[20],{'SYN-A':2.},reply,
                                      [{'doc_id':'d1','text':text}])
    assert matched==0
    assert ledger['SYN-A']['status']=='invalid_evidence'


@pytest.mark.parametrize('unit,up',[
    ('chf_per_usd','USD strengthens against CHF'),
    ('USD-per-CHF','CHF strengthens against USD'),
])
def test_numeric_fx_direction_uses_units_not_the_asset_ticker(unit,up):
    from baselines.directional_text import asset_direction_contract
    contract=asset_direction_contract(['CHF'],'level',{'CHF':{'value_unit':unit}})
    assert contract['CHF']['up_means']==up


def test_mixed_fx_unit_groups_preserve_opposite_quote_conventions():
    from baselines.directional_text import asset_direction_contract
    unit='H.10 native quote (JPY,CHF: ccy-per-USD; AUD,NZD: USD-per-ccy)'
    assets=['JPY','CHF','AUD','NZD']
    c=asset_direction_contract(assets,'level',{a:{'value_unit':unit} for a in assets})
    assert c['JPY']['up_means']=='USD strengthens against JPY'
    assert c['CHF']['up_means']=='USD strengthens against CHF'
    assert c['AUD']['up_means']=='AUD strengthens against USD'
    assert c['NZD']['up_means']=='NZD strengthens against USD'


def test_missing_or_conflicting_fx_units_do_not_guess_a_convention():
    from baselines.directional_text import asset_direction_contract
    for unit in ('unknown','H.10 native quotes','CHF-per-USD; USD-per-CHF'):
        c=asset_direction_contract(['CHF'],'level',{'CHF':{'value_unit':unit}})
        assert c['CHF']['quote_convention']=='not established by supplied unit metadata'


@pytest.mark.parametrize('target_type', ['log_return', 'cumulative_log_return'])
def test_return_direction_does_not_treat_a_factor_as_a_yield(target_type):
    from baselines.directional_text import asset_direction_contract
    c=asset_direction_contract(['MOM'],target_type,{'MOM':{'value_unit':'cumulative_log_return'}})
    assert c['MOM']['up_means']=='more positive cumulative log return'
    assert 'quote_convention' not in c['MOM']
