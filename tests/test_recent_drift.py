"""## Executive summary (read this first)

Synthetic provider CSVs verify percent-to-decimal conversion and missing-value handling.
No market outcomes are included in these test fixtures.
"""
import numpy as np
import pandas as pd
import pytest
from scripts.validate_recent_drift import parse_provider_csv, drift_forecasts
from qfbench2_track_forecasting.grid import GridSpec
from qfbench2_track_forecasting.cli import _draw


def test_csv_units_headers_and_sentinels():
    text='Provider metadata\n,Mkt-RF,SMB,HML,RF\n20300102,1.00,-2.00,-99.99,0.01\n20300103,0.50,0.20,0.10,0.01\n\nCopyright\n'
    frame=parse_provider_csv(text,{'Mkt-RF':'MKT','SMB':'SMB','HML':'HML'})
    assert frame.loc['2030-01-02','MKT']==pytest.approx(.01)
    assert frame.loc['2030-01-02','SMB']==pytest.approx(-.02)
    assert np.isnan(frame.loc['2030-01-02','HML'])
    assert 'RF' not in frame


def test_duplicate_daily_dates_are_refused():
    with pytest.raises(ValueError,match='Duplicate'):
        parse_provider_csv(',Mom\n20300102,1\n20300102,2\n',{'Mom':'MOM'})


def test_recent_candidates_preserve_joint_innovations():
    dates=pd.bdate_range('2030-01-01',periods=180)
    rng=np.random.default_rng(4)
    frame=pd.concat([pd.DataFrame({'date':dates,'asset':a,'value':np.expm1(rng.normal(.001,.002,180))})
                     for a in ('A','B')])
    origin=dates[150]; grid=GridSpec(('A','B'),(7,21))
    draws=drift_forecasts(frame,grid,origin,7)
    _,meta=_draw({'factors':frame.loc[frame.date<=origin]},list(grid.assets),list(grid.horizons),
                 str(origin.date()),1000,7,target_type='log_return')
    centre=np.array([meta['daily_drift'][a] for a in grid.assets])[:,None]*grid.horizons
    np.testing.assert_allclose(draws['half']-draws['zero'],.5*np.broadcast_to(centre,draws['half'].shape),atol=1e-15)
    altered=frame.copy(); altered.loc[altered.date>origin,'value']=1000
    other=drift_forecasts(altered,grid,origin,7)
    for name in draws: np.testing.assert_array_equal(draws[name],other[name])
