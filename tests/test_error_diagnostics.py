"""## Executive summary (read this first)
Synthetic forecasts verify calibration diagnostics and honest direction denominators.
These metrics describe forecast errors; they do not replace competition scoring.
"""
import numpy as np
import pandas as pd
import pytest
from scripts.diagnose_errors import cell_diagnostics, aggregate_cells, relative_components


def test_neutral_direction_is_abstention_not_a_correct_prediction():
    samples = np.tile(np.arange(-10., 11.)[:, None], (1, 2))
    frame = cell_diagnostics(samples, np.array([1., -1.]), np.zeros(2),
                             np.zeros(2), np.ones(2), np.array([1., 1.]))
    assert not frame['direction_attempt'].any()
    assert frame['trend_attempt'].all()
    assert frame['trend_correct'].tolist() == [True, False]
    result = aggregate_cells(frame)
    assert result['direction_accuracy'] is None
    assert result['trend_accuracy'] == .5


def test_outside_interval_and_tied_pit_are_recorded():
    samples = np.tile(np.arange(-10., 11.)[:, None], (1, 2))
    frame = cell_diagnostics(samples, np.array([20., 0.]), np.zeros(2),
                             np.zeros(2), np.array([2., 2.]), np.zeros(2))
    assert frame['covered90'].tolist() == [False, True]
    assert frame['upper90'].tolist() == [True, False]
    assert frame['pit'].tolist() == [1., .5]
    assert frame['z'].tolist() == [10., 0.]


def test_zero_variance_is_reported_not_divided_by_zero():
    frame = cell_diagnostics(np.zeros((200, 2)), np.array([0., 1.]), np.zeros(2),
                             np.zeros(2), np.zeros(2), np.zeros(2))
    result = aggregate_cells(frame)
    assert result['zero_sd_cells'] == 2
    assert result['zero_sd_misses'] == 1
    assert result['median_abs_z'] is None
    assert result['coverage90'] == .5


def test_component_deltas_do_not_invent_percentages_for_zero_baselines():
    base = {c: v for c, v in [('marginal', 2), ('joint', 0), ('tail', 1), ('composite', 2)]}
    changed = {c: v for c, v in [('marginal', 1), ('joint', 1), ('tail', 2), ('composite', 1)]}
    result = relative_components(base, changed)
    assert result == {'marginal': -50., 'joint': None, 'tail': 100., 'composite': -50.}


def test_invalid_shapes_refuse_silent_broadcast():
    with pytest.raises(ValueError):
        cell_diagnostics(np.zeros((200, 2)), np.zeros(1), np.zeros(2),
                          np.zeros(2), np.ones(2), np.zeros(2))


def test_text_audit_counts_only_existing_documents_before_origin(tmp_path):
    import json
    from scripts.diagnose_errors import audit_text_context
    text = tmp_path / 'synthetic-card' / 'text'
    text.mkdir(parents=True)
    for name in ('old.txt','fresh.txt','future.txt'):
        (text/name).write_text('Synthetic document.')
    (text/'corpus_index.json').write_text(json.dumps({'documents':[
        {'file':'old.txt','timestamp':'2019-01-01'},
        {'file':'fresh.txt','timestamp':'2020-01-01'},
        {'file':'future.txt','timestamp':'2020-03-01'},
        {'file':'missing.txt','timestamp':'2020-01-02'}]}))
    frame = pd.DataFrame({'unit':['synthetic-card','synthetic-card'],
                           'asof':['2018-01-01','2020-02-01']})
    rows = audit_text_context(frame,tmp_path)
    assert rows[0]['available_documents'] == 0
    assert rows[0]['youngest_document_age_days'] is None
    assert rows[1]['available_documents'] == 2
    assert rows[1]['recent90_documents'] == 1
    assert rows[1]['youngest_document_age_days'] == 31


def test_diagnostics_refuse_to_write_results_inside_public_repo(tmp_path):
    from scripts.diagnose_errors import main, ROOT
    with pytest.raises(ValueError, match='outside'):
        main(['--summaries',str(tmp_path/'absent.json'),'--out',str(ROOT/'diagnostic-output')])
    assert not (ROOT/'diagnostic-output').exists()


def test_current_anchor_does_not_hide_stale_uncertainty_basis():
    from scripts.diagnose_errors import basis_dates
    dates = pd.bdate_range('2000-01-03',periods=150)
    history = pd.DataFrame({'date':dates,'asset':'SYN-A','value':100+np.arange(150)*.01})
    anchor = pd.DataFrame({'date':[pd.Timestamp('2020-01-01')],'asset':['SYN-A'],'value':[200.]})
    result = basis_dates({'synthetic':pd.concat([history,anchor])},['SYN-A'],'2020-01-01','level')
    assert result['last_observation_dates']['SYN-A'] == '2020-01-01'
    assert result['last_aligned_step_date'] == str(dates[-1].date())
    assert result['aligned_step_rows'] == 120
    assert result['basis_age_calendar_days'] > 7000
