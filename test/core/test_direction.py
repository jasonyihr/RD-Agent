"""Tests for direction management system (Phase 1: direction labeling)."""

import math
from math import inf

import pytest

from rdagent.core.direction import Direction, DirectionState, DirectionTracker


# ---------------------------------------------------------------------------
# Direction & DirectionState basics
# ---------------------------------------------------------------------------


class TestDirection:
    def test_creation(self):
        d = Direction(name="momentum", description="Momentum-based factors")
        assert d.name == "momentum"
        assert d.description == "Momentum-based factors"
        assert d.category == "factor"

    def test_creation_with_category(self):
        d = Direction(name="lstm", description="LSTM models", category="model")
        assert d.category == "model"


class TestDirectionState:
    def test_initial_state(self):
        d = Direction(name="momentum", description="test")
        ds = DirectionState(direction=d)
        assert ds.attempt_count == 0
        assert ds.best_score == -inf
        assert ds.last_score == 0.0
        assert ds.consecutive_no_improvement == 0
        assert not ds.is_saturated

    def test_update_improves_best(self):
        d = Direction(name="momentum", description="test")
        ds = DirectionState(direction=d)
        ds.update(score=0.5, loop_idx=1)
        assert ds.attempt_count == 1
        assert ds.best_score == 0.5
        assert ds.last_score == 0.5
        assert ds.consecutive_no_improvement == 0
        assert not ds.is_saturated

    def test_update_no_improvement(self):
        d = Direction(name="momentum", description="test")
        ds = DirectionState(direction=d)
        ds.update(score=0.5, loop_idx=1)
        ds.update(score=0.3, loop_idx=2)  # worse
        assert ds.attempt_count == 2
        assert ds.best_score == 0.5
        assert ds.last_score == 0.3
        assert ds.consecutive_no_improvement == 1
        assert not ds.is_saturated

    def test_saturated_after_three_no_improvement(self):
        d = Direction(name="momentum", description="test")
        ds = DirectionState(direction=d)
        ds.update(score=0.5, loop_idx=1)
        ds.update(score=0.3, loop_idx=2)
        ds.update(score=0.4, loop_idx=3)
        ds.update(score=0.2, loop_idx=4)
        assert ds.consecutive_no_improvement == 3
        assert ds.is_saturated

    def test_saturated_resets_on_improvement(self):
        d = Direction(name="momentum", description="test")
        ds = DirectionState(direction=d)
        ds.update(score=0.5, loop_idx=1)
        ds.update(score=0.3, loop_idx=2)
        ds.update(score=0.4, loop_idx=3)
        ds.update(score=0.6, loop_idx=4)  # new best!
        assert ds.consecutive_no_improvement == 0
        assert ds.best_score == 0.6
        assert not ds.is_saturated

    def test_ucb_score_unexplored(self):
        d = Direction(name="momentum", description="test")
        ds = DirectionState(direction=d)
        # unexplored direction should have infinite UCB score
        assert ds.ucb_score(total_attempts=10) == inf

    def test_ucb_score_explored(self):
        d = Direction(name="momentum", description="test")
        ds = DirectionState(direction=d)
        ds.update(score=0.5, loop_idx=1)
        ucb = ds.ucb_score(total_attempts=10)
        expected = 0.5 + 1.41 * math.sqrt(math.log(10) / 1)
        assert abs(ucb - expected) < 1e-6

    def test_last_attempt_loop(self):
        d = Direction(name="momentum", description="test")
        ds = DirectionState(direction=d)
        ds.update(score=0.5, loop_idx=5)
        assert ds.last_attempt_loop == 5


# ---------------------------------------------------------------------------
# DirectionTracker
# ---------------------------------------------------------------------------


class TestDirectionTracker:
    def test_get_or_create_new(self):
        tracker = DirectionTracker()
        d = tracker.get_or_create(name="momentum", description="Momentum factors")
        assert d.name == "momentum"
        assert "momentum" in tracker.directions

    def test_get_or_create_existing(self):
        tracker = DirectionTracker()
        d1 = tracker.get_or_create(name="momentum", description="Momentum factors")
        d2 = tracker.get_or_create(name="momentum")
        assert d1 is d2  # same object

    def test_update_direction(self):
        tracker = DirectionTracker()
        d = tracker.get_or_create(name="momentum", description="test")
        tracker.update(d, score=0.5, loop_idx=1)
        assert tracker.directions["momentum"].best_score == 0.5
        assert tracker.directions["momentum"].attempt_count == 1

    def test_update_nonexistent_direction_is_noop(self):
        tracker = DirectionTracker()
        d = Direction(name="nonexistent", description="test")
        tracker.update(d, score=0.5, loop_idx=1)
        assert "nonexistent" not in tracker.directions

    def test_ucb_ranking(self):
        tracker = DirectionTracker()
        d1 = tracker.get_or_create(name="momentum", description="test")
        d2 = tracker.get_or_create(name="volatility", description="test")
        tracker.update(d1, score=0.5, loop_idx=1)
        tracker.update(d2, score=0.3, loop_idx=2)
        # volatility has lower best_score but UCB bonus for being less explored
        ranking = tracker.get_ucb_ranking()
        assert len(ranking) == 2
        # Both have 1 attempt, so ranking is by best_score
        assert ranking[0][0].name == "momentum"

    def test_ucb_ranking_underexplored_first(self):
        tracker = DirectionTracker()
        d1 = tracker.get_or_create(name="momentum", description="test")
        d2 = tracker.get_or_create(name="volatility", description="test")
        # momentum tried many times
        for i in range(5):
            tracker.update(d1, score=0.5, loop_idx=i)
        # volatility never tried
        ranking = tracker.get_ucb_ranking()
        assert ranking[0][0].name == "volatility"  # unexplored first

    def test_get_saturated_directions(self):
        tracker = DirectionTracker()
        d1 = tracker.get_or_create(name="momentum", description="test")
        d2 = tracker.get_or_create(name="volatility", description="test")
        # saturate momentum
        tracker.update(d1, score=0.5, loop_idx=1)
        tracker.update(d1, score=0.4, loop_idx=2)
        tracker.update(d1, score=0.3, loop_idx=3)
        tracker.update(d1, score=0.2, loop_idx=4)
        saturated = tracker.get_saturated_directions()
        assert len(saturated) == 1
        assert saturated[0].name == "momentum"

    def test_get_underexplored_directions(self):
        tracker = DirectionTracker()
        d1 = tracker.get_or_create(name="momentum", description="test")
        d2 = tracker.get_or_create(name="volatility", description="test")
        tracker.update(d1, score=0.5, loop_idx=1)
        tracker.update(d1, score=0.6, loop_idx=2)
        tracker.update(d1, score=0.7, loop_idx=3)
        # volatility has 0 attempts
        underexplored = tracker.get_underexplored_directions(threshold=2)
        assert len(underexplored) == 1
        assert underexplored[0].name == "volatility"

    def test_get_summary(self):
        tracker = DirectionTracker()
        d1 = tracker.get_or_create(name="momentum", description="Momentum factors")
        d2 = tracker.get_or_create(name="volatility", description="Volatility factors")
        tracker.update(d1, score=0.5, loop_idx=1)
        tracker.update(d1, score=0.6, loop_idx=2)
        tracker.update(d2, score=0.3, loop_idx=3)
        summary = tracker.get_summary()
        assert "momentum" in summary
        assert "volatility" in summary
        assert "2" in summary  # momentum attempt count
        assert "1" in summary  # volatility attempt count
