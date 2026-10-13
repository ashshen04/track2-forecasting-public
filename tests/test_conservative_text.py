"""## Executive summary (read this first)

Use synthetic model replies to check bounded, document-backed text adjustments.
These tests validate mechanics, not forecast accuracy or citation entailment.
"""

import numpy as np
import pytest

from baselines import reasoning_agent as agent


@pytest.mark.parametrize("vol,applied_vol", [(0.1, 1.0), (1.2, 1.2), (10, 1.5)])
def test_conservative_adjustments_bound_mean_and_spread(vol, applied_vol):
    samples = np.random.default_rng(4).normal(100, 2, (1000, 1, 1))
    reply = {"assets": {"A": {"drift_bp": 1e6, "vol_scale": vol,
                               "because": "Synthetic uncertainty evidence.",
                               "doc_ids": ["doc-1"]}}}
    adjusted, ledger, matched = agent.apply_adjustment(
        samples, ["A"], {"A": 100}, {"A": 2}, reply,
        conservative=True, valid_doc_ids={"doc-1"},
    )
    assert matched == 1
    assert adjusted.mean() - samples.mean() == pytest.approx(1)
    assert adjusted.std() / samples.std() == pytest.approx(applied_vol)
    assert ledger["A"]["doc_ids"] == ["doc-1"]


@pytest.mark.parametrize("doc_ids,because", [([], "Reason"), (["invented"], "Reason"),
                                         (["doc-1"], "")])
def test_conservative_adjustments_drop_unsupported_replies(doc_ids, because):
    samples = np.random.default_rng(4).normal(100, 2, (1000, 1, 1))
    reply = {"assets": {"A": {"drift_bp": 100, "vol_scale": 1.5,
                               "because": because, "doc_ids": doc_ids}}}
    adjusted, ledger, matched = agent.apply_adjustment(
        samples, ["A"], {"A": 100}, {"A": 2}, reply,
        conservative=True, valid_doc_ids={"doc-1"},
    )
    np.testing.assert_array_equal(adjusted, samples)
    assert matched == 0
    assert "evidence" in ledger["A"]["note"]


def test_zero_adjustment_preserves_every_draw_exactly():
    # Subtracting and adding the centre can round tiny return samples, even when
    # the model requests no change. A neutral reply must preserve the control.
    samples = np.array([1e-12, 0.002, 0.003, 0.001]).reshape(4, 1, 1)
    reply = {"assets": {"A": {"drift_bp": 0, "vol_scale": 1,
                               "because": "No directional evidence in doc-1.",
                               "doc_ids": ["doc-1"]}}}
    adjusted, _, matched = agent.apply_adjustment(
        samples, ["A"], {"A": 0}, {"A": 0.01}, reply,
        target_type="log_return", conservative=True, valid_doc_ids={"doc-1"},
    )
    assert matched == 1
    np.testing.assert_array_equal(adjusted, samples)
