"""## Executive summary (read this first)

Synthetic data verify combined drift/width calibration and exclusion of inspected periods.
For example, an interval inspected under another card remains excluded for a shared asset.
"""
import json
import numpy as np
import pandas as pd
import pytest
from scripts.validate_return_calibration import prior_intervals, four_forecasts, reserve
from qfbench2_track_forecasting.cli import _draw
from qfbench2_track_forecasting.grid import GridSpec
from scripts.backtest import overlaps_excluded


def test_exclusions_cover_numeric_text_and_event_peer_formats(tmp_path):
    p=tmp_path/'summary.json'
    p.write_text(json.dumps({'assets':['A'], 'units':[{'assets':['A'],
        'folds':[{'asof':'2020-01-01','target_dates':['2020-01-10']}]}],
        'results':[{'status':'completed','asof':'2020-02-01','target_dates':['2020-02-10'],
        'adjustments':{'x':{'audit':{'5':{'peers':[{'origin':'2019-01-01','end':'2019-01-10'}]}}}}}]}))
    intervals,_=prior_intervals([p])
    for start,end in [('2020-01-02','2020-01-03'),('2020-02-02','2020-02-03'),('2019-01-02','2019-01-03')]:
        assert overlaps_excluded(pd.Timestamp(start),pd.Timestamp(end),['A'],intervals)
    assert not overlaps_excluded(pd.Timestamp('2020-01-02'),pd.Timestamp('2020-01-03'),['B'],intervals)


def test_new_cohort_reservation_is_global_across_cards():
    intervals={}
    reserve(intervals,['A','B'],pd.Timestamp('2020-01-01'),pd.Timestamp('2020-01-10'))
    assert overlaps_excluded(pd.Timestamp('2020-01-05'),pd.Timestamp('2020-01-15'),['B','C'],intervals)
    assert not overlaps_excluded(pd.Timestamp('2020-01-10'),pd.Timestamp('2020-01-15'),['A'],intervals)


def test_legacy_asset_inference_is_recorded(tmp_path,monkeypatch):
    import scripts.validate_return_calibration as calibration
    unit=tmp_path/'units'/'SYN'; unit.mkdir(parents=True)
    (unit/'card.toml').write_text('[targets]\nasset_ids=["A"]\n')
    monkeypatch.setattr(calibration,'ROOT',tmp_path)
    p=tmp_path/'summary.json'
    p.write_text(json.dumps({'results':[{'unit':'SYN','status':'completed','asof':'2020-01-01',
        'target_dates':['2020-01-10'],'adjustments':{'B':{}}}]}))
    intervals,sources=prior_intervals([p])
    assert set(intervals)=={'A','B'}
    assert sources[str(p.resolve())]['legacy_asset_inference']['SYN']['assets']==['A','B']


def test_empty_prior_ledger_is_refused(tmp_path):
    p=tmp_path/'summary.json'; p.write_text('{"results":[]}')
    with pytest.raises(ValueError,match='no historical intervals'): prior_intervals([p])


def test_changed_source_is_rejected_before_scoring(tmp_path,monkeypatch):
    import scripts.validate_return_calibration as calibration
    monkeypatch.setattr(calibration,'ROOT',tmp_path)
    (tmp_path/'cohort.json').write_text(json.dumps({'policies':calibration.POLICIES,
        'seeds':list(calibration.SEEDS),'selected':[{'unit':'SYN','source_sha256':{'p':'old'}}]}))
    monkeypatch.setattr(calibration,'read_unit',lambda _: ({},None,{},None,{'p':'new'}))
    with pytest.raises(ValueError,match='changed since cohort freeze'): calibration.run(tmp_path)


def test_combination_uses_same_joint_noise_and_ignores_future_rows():
    dates=pd.bdate_range('2020-01-01',periods=180)
    rng=np.random.default_rng(4)
    shocks=rng.normal(.001,.002,(180,2))
    panel=pd.concat([pd.DataFrame({'date':dates,'asset':a,'value':np.expm1(shocks[:,i])})
                     for i,a in enumerate(['A','B'])])
    asof=dates[160]
    grid=GridSpec(('A','B'),(21,5))
    forecasts=four_forecasts({'p':panel},grid,asof,1000,7)
    history={'p':panel.loc[panel.date<=asof]}
    base,meta=_draw(history,list(grid.assets),list(grid.horizons),str(asof.date()),1000,7,target_type='log_return')
    centre=np.array([meta['daily_drift'][a] for a in grid.assets])[:,None]*grid.horizons
    np.testing.assert_allclose(forecasts['combined'],.5*centre+1.2*(base-centre),atol=1e-15)
    altered=panel.copy(); altered.loc[altered.date>asof,'value']=999.
    second=four_forecasts({'p':altered},grid,asof,1000,7)
    for name in forecasts: np.testing.assert_array_equal(forecasts[name],second[name])
