"""## Executive summary (read this first)
Check holdout calendar semantics, return compounding, and the output firewall.
All values below are synthetic test fixtures, not answers to public cards.
"""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.backtest import ROOT, SkipUnit, holdout, outside_repo, forecast_samples
from qfbench2_track_forecasting.grid import GridSpec


class BacktestTests(unittest.TestCase):
    def setUp(self):
        self.dates = pd.bdate_range('2030-01-04', periods=5)
        self.frame = pd.DataFrame({'A': [0.0, 0.1, -0.1, 0.2, 0.0]}, index=self.dates)

    def test_weekend_is_not_a_step(self):
        values, dates = holdout(self.frame, self.dates[0], [1], 'level')
        self.assertEqual(dates, ['2030-01-07'])
        np.testing.assert_allclose(values, [0.1])

    def test_returns_are_compounded_and_origin_is_excluded(self):
        values, _ = holdout(self.frame, self.dates[0], [2], 'log_return')
        np.testing.assert_allclose(values, [np.log(1.1 * 0.9)])

    def test_missing_interior_return_is_not_filled(self):
        with self.assertRaises(SkipUnit):
            holdout(self.frame.drop(self.dates[1]), self.dates[0], [2], 'log_return')

    def test_flattening_preserves_asset_then_horizon_order(self):
        frame = self.frame.assign(B=self.frame.A + 10)
        values, _ = holdout(frame, self.dates[0], [2, 1], 'level')
        np.testing.assert_allclose(values, [-0.1, 0.1, 9.9, 10.1])

    def test_public_repository_output_is_refused(self):
        with self.assertRaises(ValueError):
            outside_repo(ROOT / 'out' / 'backtest')

    def test_reference_receives_no_future_rows(self):
        panel = self.frame.reset_index(names='date').rename(columns={'A': 'value'})
        panel['asset'] = 'A'
        grid = GridSpec(('A',), (1,))
        def fake_draw(panels, assets, horizons, asof, draws, seed, **kwargs):
            self.assertEqual(len(panels['p']), 2)
            self.assertLessEqual(panels['p'].date.max(), self.dates[1])
            self.assertEqual(asof, str(self.dates[1].date()))
            return np.zeros((draws, 1, 1)), {}
        with patch('scripts.backtest._draw', side_effect=fake_draw):
            result = forecast_samples('reference', {'p': panel}, self.dates[1], grid, 200, 'level')
        self.assertEqual(result.shape, (200, 1))

    def test_reference_system_exit_becomes_reportable_error(self):
        with patch('scripts.backtest._draw', side_effect=SystemExit('short history')):
            with self.assertRaisesRegex(ValueError, 'short history'):
                forecast_samples('reference', {}, self.dates[0], GridSpec(('A',), (1,)), 200, 'level')

    def test_shuffled_reference_preserves_marginals_and_asset_paths(self):
        draws = np.arange(200.0)
        samples = np.stack([draws, draws + 5, draws * 2, draws * 2 + 7], axis=1).reshape(200, 2, 2)
        grid = GridSpec(('A', 'B'), (1, 2))
        with patch('scripts.backtest._draw', return_value=(samples, {})):
            shuffled = forecast_samples('reference-shuffled', {}, self.dates[0], grid, 200, 'level')
        np.testing.assert_array_equal(np.sort(shuffled, axis=0), np.sort(samples.reshape(200, 4), axis=0))
        np.testing.assert_array_equal(shuffled[:, 1] - shuffled[:, 0], np.full(200, 5))
        np.testing.assert_array_equal(shuffled[:, 3] - shuffled[:, 2], np.full(200, 7))
        self.assertLess(abs(np.corrcoef(shuffled[:, 0], shuffled[:, 2])[0, 1]), 0.2)


if __name__ == '__main__':
    unittest.main()
