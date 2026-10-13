"""## Executive summary (read this first)

Synthetic tests verify heavy-tail covariance, conditional-risk response, shared
paths and cutoff safety. Integration checks both formal forecast entry points.
No benchmark answers or real future outcomes are embedded in these tests.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from baselines.adaptive_distribution import draw_paths
from qfbench2_track_forecasting.cli import _draw


def history():
    rng = np.random.default_rng(42)
    x = rng.standard_normal((120, 2))
    x[:, 1] = .7*x[:, 0] + np.sqrt(1-.7**2)*x[:, 1]
    dates = pd.bdate_range('2020-01-01', periods=121)
    frame = pd.DataFrame([{'date':d,'asset':a,'value':float(v)} for i,a in enumerate(('SYN-A','SYN-B')) for d,v in zip(dates, 100+np.r_[0,np.cumsum(x[:,i])])])
    return x, {'synthetic':frame}, str(dates[-1].date())


def test_student_t_has_unit_variance_and_more_extreme_tails():
    x, _, _ = history()
    draws, meta = draw_paths(x,[0,0],[0,0],[1,5],20000,np.random.default_rng(3),model='student-t')
    covariance=np.cov(draws[:,:,0],rowvar=False)
    np.testing.assert_allclose(covariance,np.cov(x,rowvar=False),rtol=.10,atol=.04)
    standardized=draws[:,0,0]/x[:,0].std(ddof=1)
    assert np.quantile(np.abs(standardized),.999)>3.6
    assert np.corrcoef(draws[:,0,0],draws[:,0,1])[0,1]==pytest.approx(np.sqrt(1/5),abs=.04)
    assert meta['shared_scale_per_path'] is True


def test_recent_shocks_raise_near_risk_then_variance_reverts():
    x, _, _ = history()
    x[-10:] *= 5
    _, meta = draw_paths(x,[0,0],[0,0],[1,250],200,np.random.default_rng(3),model='adaptive-t')
    long=np.array(meta['long_run_variance'])
    near=np.array(meta['cumulative_variance']['1'])
    far=np.array(meta['cumulative_variance']['250'])/250
    assert (near>long).all()
    assert (np.abs(far-long)<np.abs(near-long)).all()
    assert meta['coefficients_fitted'] is False


def test_historical_tail_shape_uses_only_visible_distribution():
    # Symmetric bounded changes carry no evidence of excess tail mass.
    bounded=np.tile(np.array([[-1.,-2.],[1.,2.]]),(60,1))
    _, meta=draw_paths(bounded,[0,0],[0,0],[1],200,np.random.default_rng(7),model='historical-t')
    assert meta['degrees_of_freedom'] is None
    assert not meta['shared_scale_per_path']
    heavy=bounded.copy(); heavy[-2:,0]=[-20.,20.]; heavy[-2:,1]=[-40.,40.]
    _, fitted=draw_paths(heavy,[0,0],[0,0],[1],200,np.random.default_rng(7),model='historical-t')
    assert 5 <= fitted['degrees_of_freedom'] <= 60
    assert fitted['historical_excess_kurtosis']>0
    assert fitted['tail_shape_estimated']


def test_shared_earlier_horizon_and_future_rows_do_not_change_draws():
    _, panels, asof = history()
    draws,_=_draw(panels,['SYN-A','SYN-B'],[5,1],asof,1000,7,statistical_model='adaptive-t')
    first,_=_draw(panels,['SYN-A','SYN-B'],[1],asof,1000,7,statistical_model='adaptive-t')
    np.testing.assert_array_equal(draws[:,:,1],first[:,:,0])
    panels['synthetic']=pd.concat([panels['synthetic'],pd.DataFrame({'date':[pd.Timestamp('2030-01-01')],'asset':['SYN-A'],'value':[1e9]})])
    again,_=_draw(panels,['SYN-A','SYN-B'],[5,1],asof,1000,7,statistical_model='adaptive-t')
    np.testing.assert_array_equal(draws,again)


def test_gaussian_control_unchanged_and_constant_asset_stays_constant():
    _,panels,asof=history()
    args=(panels,['SYN-A','SYN-B'],[1,5],asof,200,7)
    a,_=_draw(*args)
    b,_=_draw(*args,statistical_model='gaussian')
    np.testing.assert_array_equal(a,b)
    panels['synthetic'].loc[panels['synthetic'].asset=='SYN-B','value']=100.
    constant,_=_draw(*args,statistical_model='adaptive-t')
    np.testing.assert_array_equal(constant[:,1,:],100.)


@pytest.mark.parametrize('kwargs',[{'statistical_model':'unknown'}, {'statistical_model':'adaptive-t','innovation':'bootstrap'}, {'statistical_model':'adaptive-t','drift_scale':.5}])
def test_invalid_combinations_fail(kwargs):
    _,panels,asof=history()
    with pytest.raises(ValueError):
        _draw(panels,['SYN-A','SYN-B'],[1],asof,200,7,**kwargs)


@pytest.mark.parametrize('text_policy',['no-text','conservative','structured'])
def test_formal_entrypoint_reports_distribution(tmp_path,monkeypatch,text_policy):
    from qfbench2_track_forecasting.cli import main
    root=Path(__file__).resolve().parents[1]
    unit=root/'units/t2-EXAMPLE-ust-curve-1m'
    monkeypatch.setenv('QFBENCH_NETWORK','none')
    extra=['--no-text'] if text_policy=='no-text' else ['--text-policy',text_policy]
    assert main(['--panels',str(unit/'panels'),'--text',str(unit/'text'),'--asof','2024-06-28','--out',str(tmp_path/'forecast.parquet'),'--statistical-model','adaptive-t',*extra])==0
    meta=json.loads((tmp_path/'forecast_meta.json').read_text())
    assert meta['statistical_model']=='adaptive-t'
    assert meta['distribution_parameters']['degrees_of_freedom']==7
    assert not meta['reasoning_applied']


def test_monthly_cpi_retains_existing_distribution(tmp_path):
    from qfbench2_track_forecasting.cli import main
    import tomllib
    root=Path(__file__).resolve().parents[1]
    unit=root/'units/t2-F1-cpi-glidepath-2023'
    card=tomllib.loads((unit/'card.toml').read_text())
    base=['--panels',str(unit/'panels'),'--text',str(unit/'text'),'--asof',str(card['provenance']['data_cutoff']),'--no-text']
    a=tmp_path/'a/forecast.parquet'; b=tmp_path/'b/forecast.parquet'
    main([*base,'--out',str(a)])
    main([*base,'--out',str(b),'--statistical-model','adaptive-t'])
    pd.testing.assert_frame_equal(pd.read_parquet(a),pd.read_parquet(b))
