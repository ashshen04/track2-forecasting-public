"""## Executive summary (read this first)

Research aggregation must give cards equal weight and retain adverse runs.
For example, many windows from one card cannot outweigh a second card.
The published exemplar is a smoke check, not part of an accuracy aggregate.
"""
import pytest
from scripts.validate_text_scale import aggregate, fallback_reply


def test_equal_card_weighting_clipping_and_exemplar_exclusion():
    rows = [{'unit': 'one', 'proxy_loss': 1.}] * 10
    rows += [{'unit': 'two', 'proxy_loss': 20.}, {'unit': 't2-EXAMPLE-smoke', 'proxy_loss': 0.}]
    result = aggregate(rows)
    assert result['cards'] == 2
    assert result['replay_rows'] == 11
    assert result['mean_proxy_loss'] == pytest.approx(4.5)
    assert aggregate([{'unit': 't2-EXAMPLE-smoke', 'proxy_loss': 0.}]) is None


def test_failed_reply_fallback_is_neutral_for_every_asset():
    reply = fallback_reply(['A', 'B'])
    assert set(reply['assets']) == {'A', 'B'}
    assert all(s['direction'] == 'neutral' and not s['evidence'] for s in reply['assets'].values())
