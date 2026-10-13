"""## Executive summary (read this first)

Check that text attenuation changes centres while preserving joint uncertainty.
Zero attenuation must reproduce no-text forecasts and cannot make model calls.
Synthetic checks establish mechanics, not accuracy.
"""
import numpy as np
import pytest
from baselines.directional_text import apply_directional


def inputs():
    samples = np.random.default_rng(8).normal(size=(200, 2, 2))
    docs = [{'doc_id': 'd', 'text': 'Synthetic policy evidence sufficiently long.'}]
    spec = {'direction': 'up', 'strength': 'moderate', 'priced_in': 'unlikely',
            'mechanism': 'Synthetic policy mechanism.', 'counterargument': 'Synthetic uncertainty.',
            'evidence': [{'doc_id': 'd', 'quote': docs[0]['text']}], 'eligible_horizons': [10]}
    return samples, docs, {'assets': {'A': spec, 'B': {**spec, 'direction': 'neutral'}}}


def test_attenuation_preserves_joint_shocks_and_horizon_mask():
    samples, docs, reply = inputs()
    full, _, _ = apply_directional(samples, ['A', 'B'], [5, 10], {'A': 2, 'B': 1}, reply, docs)
    half, ledger, _ = apply_directional(samples, ['A', 'B'], [5, 10], {'A': 2, 'B': 1}, reply, docs,
                                      adjustment_scale=.5)
    assert np.allclose(half - samples, .5 * (full - samples))
    assert np.array_equal(half[:, :, 0], samples[:, :, 0])
    assert np.array_equal(half[:, 1, :], samples[:, 1, :])
    assert np.allclose(np.cov(half.reshape(200, -1), rowvar=False),
                       np.cov(samples.reshape(200, -1), rowvar=False))
    assert ledger['A']['adjustment_scale'] == .5


def test_zero_is_exact_baseline():
    samples, docs, reply = inputs()
    zero, _, _ = apply_directional(samples, ['A', 'B'], [5, 10], {'A': 2, 'B': 1}, reply, docs,
                                  adjustment_scale=0)
    assert np.array_equal(zero, samples)


@pytest.mark.parametrize('scale', [-.1, 1.1, float('nan'), float('inf')])
def test_invalid_attenuation_is_refused(scale):
    samples, docs, reply = inputs()
    with pytest.raises(ValueError):
        apply_directional(samples, ['A', 'B'], [5, 10], {'A': 2, 'B': 1}, reply, docs,
                          adjustment_scale=scale)
