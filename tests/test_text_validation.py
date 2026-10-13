"""## Executive summary (read this first)

Synthetic replies check paired text evaluation, dated inputs and failure labels.
For example, a failed model call yields no accuracy comparison. These tests
establish mechanics only, never the predictive value of a real language model.
"""

import hashlib
import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from scripts import validate_text as subject
from baselines import reasoning_agent as agent


@pytest.fixture
def experiment(tmp_path):
    unit = tmp_path / 'synthetic-unit'
    (unit / 'text').mkdir(parents=True)
    dates = pd.bdate_range('2020-01-01', periods=80)
    origin = str(dates[60].date())
    (unit / 'card.toml').write_text(f'''# Executive summary: synthetic daily test card.
[task]
id = "synthetic-unit"
[metadata]
category = "T2-F1"
[provenance]
data_cutoff = "{dates[-1].date()}"
[targets]
asset_ids = ["SYN-A"]
horizons = [5]
target_type = "level"
target_frequency = "daily"
''')
    panel = pd.DataFrame({'date': dates, 'asset': 'SYN-A',
                          'value': 100 + np.cumsum(np.tile([-0.01, 0.02], 40))})
    panel.to_parquet(unit / 'panel.parquet', index=False)
    effective = str((pd.Timestamp(origin) + pd.offsets.BDay(3)).date())
    (unit / 'text/old.txt').write_text(
        'Executive summary: supplied synthetic policy evidence. '
        + 'Synthetic future increment takes effect on ' + effective + '.'
    )
    (unit / 'text/future.txt').write_text('FUTURE_MARKER_NOT_ALLOWED')
    (unit / 'text/corpus_index.json').write_text(json.dumps({'documents': [
        {'doc_id': 'doc-old', 'timestamp': origin, 'file': 'old.txt'},
        {'doc_id': 'doc-future', 'timestamp': '2030-01-01', 'file': 'future.txt'}]}))
    item = {'unit': unit.name, 'asof': origin, 'family': 'T2-F1',
            'source_sha256': {'panel.parquet': hashlib.sha256((unit / 'panel.parquet').read_bytes()).hexdigest()}}
    return tmp_path, item


def reply(shift=1):
    return {'assets': {'SYN-A': {'drift_bp': shift, 'vol_scale': 1,
                                'because': 'Synthetic policy evidence in doc-old.',
                                'doc_ids': ['doc-old']}}}


def test_success_uses_truncated_history_and_same_draws(experiment):
    units, item = experiment
    original_draw = subject._draw
    original_apply = subject.apply_adjustment
    scored = []

    def draw(panels, *args, **kwargs):
        assert all(f.date.max() <= pd.Timestamp(item['asof']) for f in panels.values())
        return original_draw(panels, *args, **kwargs)

    def model(prompt):
        assert 'FUTURE_MARKER_NOT_ALLOWED' not in prompt
        assert 'doc-old' in prompt
        return reply(), '', ''

    def score(ctx):
        scored.append(ctx['_samples'].copy())
        return {c: 1.0 for c in subject.COMPONENTS}

    with patch.object(subject, '_draw', side_effect=draw), \
            patch.object(subject, 'call_model', side_effect=model), \
            patch.object(subject, '_score', side_effect=score), \
            patch.object(subject, 'apply_adjustment', wraps=original_apply):
        result = subject.evaluate(item, units, 1000, 1)
    assert result['status'] == 'completed'
    assert result['changed_draws']
    assert result['documents_excluded'] == 1
    assert len(scored) == 2
    # Text only translates these paired draws; no fresh random sampling occurred.
    difference = scored[1] - scored[0]
    np.testing.assert_allclose(difference, difference[0, 0], atol=1e-12)


@pytest.mark.parametrize('parsed,status', [(None, 'model_failed'), ({}, 'invalid_adjustment')])
def test_failed_calls_never_become_accuracy_results(experiment, parsed, status):
    units, item = experiment
    with patch.object(subject, 'call_model', return_value=(parsed, 'synthetic failure', '')), \
            patch.object(subject, '_score') as scorer:
        result = subject.evaluate(item, units, 1000, 1)
    assert result['status'] == status
    assert 'change_percent' not in result
    scorer.assert_not_called()


def test_valid_zero_adjustment_is_success_but_not_a_changed_forecast(experiment):
    units, item = experiment
    with patch.object(subject, 'call_model', return_value=(reply(0), '', '')), \
            patch.object(subject, '_score', return_value={c: 1 for c in subject.COMPONENTS}):
        result = subject.evaluate(item, units, 1000, 1)
    assert result['status'] == 'completed'
    assert not result['changed_draws']


def test_changed_panels_fail_before_a_model_call(experiment):
    units, item = experiment
    item['source_sha256']['panel.parquet'] = 'wrong-hash'
    with patch.object(subject, 'call_model') as model, pytest.raises(ValueError, match='Panel changed'):
        subject.evaluate(item, units, 1000, 1)
    model.assert_not_called()


def test_candidate_selection_uses_dates_and_not_scores(experiment):
    units, item = experiment
    data = {'units': [{'unit': item['unit'], 'status': 'completed',
                       'source_sha256': item['source_sha256'],
                       'folds': [{'asof': '2010-01-01'}, {'asof': item['asof']}]}]}
    chosen, _ = subject.candidates(data, units)
    assert chosen[0]['asof'] == item['asof']
    assert chosen[0]['newest_document_age_days'] == 0
    data['units'][0]['folds'].append({'asof': '2021-01-01'})
    chosen_again, _ = subject.candidates(data, units)
    assert chosen_again == chosen


def test_missing_configuration_is_an_explicit_blocker(monkeypatch):
    for name in ('MODEL_ENDPOINT', 'MODEL_NAME', 'MODEL_TOKEN', 'QFBENCH_NETWORK'):
        monkeypatch.delenv(name, raising=False)
    reason = subject.configuration_reason()
    assert 'MODEL_ENDPOINT' in reason and 'MODEL_TOKEN' in reason


def test_local_experiment_cannot_silently_use_house_credentials(monkeypatch):
    monkeypatch.delenv('QFBENCH_NETWORK', raising=False)
    monkeypatch.setenv('MODEL_TOKEN', '')
    assert 'cannot use MODEL_TOKEN' in subject.configuration_reason(local_api=True)


def test_local_key_file_reads_quotes_without_shell_expansion(tmp_path, monkeypatch):
    monkeypatch.delenv('MODEL_API_KEY', raising=False)
    path = tmp_path / '.env.local'
    path.write_text('# Executive summary: synthetic test key.\nMODEL_API_KEY="synthetic-$(do-not-execute)=value"\n')
    subject.load_local_api_key(path)
    assert subject.os.environ['MODEL_API_KEY'] == 'synthetic-$(do-not-execute)=value'


def test_environment_key_takes_precedence(tmp_path, monkeypatch):
    monkeypatch.setenv('MODEL_API_KEY', 'synthetic-existing')
    path = tmp_path / '.env.local'
    path.write_text('MODEL_API_KEY=synthetic-file\n')
    subject.load_local_api_key(path)
    assert subject.os.environ['MODEL_API_KEY'] == 'synthetic-existing'


@pytest.mark.parametrize('line', ['MODEL_TOKEN=synthetic-secret', 'MODEL_API_KEY="synthetic-secret'])
def test_invalid_local_key_file_does_not_echo_secret(tmp_path, line):
    path = tmp_path / '.env.local'
    path.write_text(line)
    with pytest.raises(ValueError) as error:
        subject.load_local_api_key(path)
    assert 'synthetic-secret' not in str(error.value)


def test_relevant_excerpts_recover_late_policy_without_increasing_budget():
    header = 'Executive summary: synthetic central bank speech.\n'
    filler = 'Administrative information and introductory remarks. ' * 300
    policy = 'POLICY_MARKER: Accommodative monetary policy and asset purchases will support liquidity. '
    text = header + filler + policy + filler
    excerpt = agent.document_excerpt(text, 'relevant')
    assert excerpt.startswith(header)
    assert 'POLICY_MARKER' in excerpt
    assert len(excerpt) <= agent._MAX_DOC_CHARS
    assert 'POLICY_MARKER' not in agent.document_excerpt(text, 'head')
    assert '[Excerpt from character ' in excerpt


def test_excerpt_preserves_short_documents_and_falls_back_without_keywords():
    short = 'Executive summary: inflation expectations increased.'
    assert agent.document_excerpt(short, 'relevant') == short
    long = 'Generic administrative remarks. ' * 400
    assert agent.document_excerpt(long, 'relevant') == long[:agent._MAX_DOC_CHARS]


def test_relevant_corpus_still_excludes_future_text(experiment):
    units, item = experiment
    docs, excluded, _ = agent.read_corpus(units / item['unit'] / 'text', item['asof'],
                                         excerpt_policy='relevant')
    assert excluded == 1
    assert all('FUTURE_MARKER_NOT_ALLOWED' not in d['text'] for d in docs)


def test_directional_evaluation_checks_quotes_and_changes_only_centres(experiment):
    units,item=experiment
    parsed={'assets':{'SYN-A':{'direction':'up','strength':'moderate','priced_in':'unlikely',
        'mechanism':'Synthetic policy evidence increases the level relative to the baseline.',
        'counterargument':'The statement could already be reflected in prices.',
        'evidence':[{'doc_id':'doc-old','quote':'supplied synthetic policy evidence.'}]}}}
    captured=[]
    def model(prompt):
        assert 'FUTURE_MARKER_NOT_ALLOWED' not in prompt
        assert 'historical_mean_daily_change' in prompt
        return parsed,'',''
    def score(ctx):
        captured.append(ctx['_samples'].copy())
        return {c:1. for c in subject.COMPONENTS}
    with patch.object(subject,'call_model',side_effect=model),patch.object(subject,'_score',side_effect=score):
        result=subject.evaluate(item,units,1000,1,'relevant','directional')
    assert result['changed_draws']
    assert result['direction_diagnostics']['text']['attempts']==1
    assert np.ptp(captured[1]-captured[0]) < 1e-12


def test_structured_historical_evaluation_matches_formal_forecast(experiment, monkeypatch):
    from baselines.directional_text import evidence_spans
    from qfbench2_track_forecasting import cli

    units, item = experiment
    unit = units / item['unit']
    card_path = unit / 'card.toml'
    card_path.write_text(card_path.read_text().replace('horizons = [5]', 'horizons = [1, 5]'))
    docs = agent.read_corpus(unit / 'text', item['asof'], excerpt_policy='relevant')[0]
    sid = next(iter(evidence_spans(docs)))
    raw = {'assets': {'SYN-A': {'drivers': [{
        'id': 'd1', 'change_basis': 'explicit_future_path', 'timing': 'within_horizon',
        'announcement_date': item['asof'],
        'effective_date': str((pd.Timestamp(item['asof']) + pd.offsets.BDay(3)).date()),
        'remaining_future_change': {
            'description': 'Synthetic future increment takes effect on '
            + str((pd.Timestamp(item['asof']) + pd.offsets.BDay(3)).date()) + '.',
            'date': str((pd.Timestamp(item['asof']) + pd.offsets.BDay(3)).date()),
            'evidence': [{'span_id': sid}],
        },
        'effect': {'basis': 'numeric', 'entity': 'SYN-A', 'direction': 'up'},
        'mechanism': 'Synthetic future increment.', 'evidence': [{'span_id': sid}]}],
        'pricing': {'label': 'unclear', 'basis': 'unavailable',
                    'explanation': 'No pricing evidence.', 'evidence': []}, 'missing_evidence': 'none'}}}
    prompts = []
    def model(prompt):
        assert 'FUTURE_MARKER_NOT_ALLOWED' not in prompt
        prompts.append(prompt)
        return raw, '', ''
    captured = []
    def score(ctx):
        captured.append(ctx['_samples'].copy())
        return {c: 1. for c in subject.COMPONENTS}
    with patch.object(subject, 'call_model', side_effect=model), patch.object(subject, '_score', side_effect=score):
        result = subject.evaluate(item, units, 1000, 1, 'relevant', 'structured')
    for name in ('MODEL_ENDPOINT', 'MODEL_NAME', 'MODEL_TOKEN'):
        monkeypatch.setenv(name, 'synthetic')
    monkeypatch.setattr(agent, 'call_model', model)
    out = units / 'formal/forecast.parquet'
    cli.main(['--panels', str(unit), '--text', str(unit / 'text'), '--asof', item['asof'],
              '--out', str(out), '--text-policy', 'structured', '--n-draws', '1000', '--seed', '1'])
    np.testing.assert_array_equal(pd.read_parquet(out).value.to_numpy(), captured[1].ravel())
    assert prompts[0] == prompts[1]
    assert result['changed_draws'] and result['analysis_audit']
    assert result['evidence_format'] == 'span'
    np.testing.assert_array_equal(captured[1][:, 0], captured[0][:, 0])
    assert np.all(captured[1][:, 1] > captured[0][:, 1])
    assert result['direction_diagnostics']['text']['attempts'] == 1


def test_cohort_builder_selects_exact_text_aligned_holdouts(experiment):
    from scripts.build_text_cohort import build_unit
    units,item=experiment
    windows=build_unit(units/item['unit'],max_folds=2,max_age=90)
    assert windows
    assert all(r['documents']>=1 and r['newest_document_age_days']<=90 for r in windows)
    assert all(r['asof']>=item['asof'] for r in windows)
    assert all(max(r['target_dates'])<='2020-04-21' for r in windows)
    assert all(r['horizons']==[5] for r in windows)


def test_cohort_excludes_inspected_intervals_before_greedy_selection(experiment):
    from scripts.build_text_cohort import build_unit
    units, item = experiment
    latest = build_unit(units / item['unit'], max_folds=1)[0]
    prior = {'SYN-A': [(pd.Timestamp(latest['asof']),
                       max(pd.Timestamp(t) for t in latest['target_dates']))]}
    fresh = build_unit(units / item['unit'], max_folds=1, excluded_intervals=prior)[0]
    assert max(pd.Timestamp(t) for t in fresh['target_dates']) <= pd.Timestamp(latest['asof'])
    assert fresh['asof'] != latest['asof']


def test_frozen_cohort_text_change_fails_before_model_call(experiment):
    units,item=experiment
    item['corpus_sha256']={'text/old.txt':'incorrect-hash'}
    with patch.object(subject,'call_model') as model,pytest.raises(ValueError,match='Corpus changed'):
        subject.evaluate(item,units,1000,1,'relevant','directional')
    model.assert_not_called()
