"""Tests for DirectionManager (Phase 2: direction management engine)."""

from unittest.mock import MagicMock

import pytest

from rdagent.core.direction import Direction, DirectionTracker
from rdagent.core.direction_mgr import DirectionManager
from rdagent.core.proposal import ExperimentFeedback, Hypothesis, Trace
from rdagent.core.scenario import Scenario


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_trace(hist_entries):
    """Create a mock Trace with the given hist entries.

    Each entry is (mock_experiment, feedback).
    The mock_experiment has a .hypothesis attribute.
    """
    trace = MagicMock(spec=Trace)
    trace.hist = []
    for exp, fb in hist_entries:
        trace.hist.append((exp, fb))
    return trace


def _make_mock_experiment(hypothesis_text: str):
    """Create a mock experiment with a hypothesis."""
    exp = MagicMock()
    hyp = Hypothesis(
        hypothesis=hypothesis_text,
        reason="test",
        concise_reason="test",
        concise_observation="test",
        concise_justification="test",
        concise_knowledge="test",
    )
    exp.hypothesis = hyp
    return exp


def _make_feedback(decision: bool = True):
    return ExperimentFeedback(reason="test", decision=decision)


# ---------------------------------------------------------------------------
# Test: Initialization
# ---------------------------------------------------------------------------


class TestDirectionManagerInit:
    def test_default_state(self):
        mgr = DirectionManager()
        assert isinstance(mgr.tracker, DirectionTracker)
        assert mgr._direction_map == {}
        assert mgr._processed_up_to == 0

    def test_custom_tracker(self):
        tracker = DirectionTracker()
        mgr = DirectionManager(tracker=tracker)
        assert mgr.tracker is tracker


# ---------------------------------------------------------------------------
# Test: update_from_trace
# ---------------------------------------------------------------------------


class TestUpdateFromTrace:
    def test_empty_trace(self):
        mgr = DirectionManager(
            classifier=lambda text, existing: ("test_dir", "test description"),
            score_fn=lambda exp, fb: 0.5,
        )
        trace = _make_mock_trace([])
        mgr.update_from_trace(trace)
        assert mgr._processed_up_to == 0
        assert len(mgr.tracker.directions) == 0

    def test_processes_new_entries(self):
        call_log = []

        def classifier(text, existing):
            call_log.append(text)
            if "momentum" in text:
                return ("momentum", "Momentum-based factors")
            return ("other", "Other factors")

        mgr = DirectionManager(
            classifier=classifier,
            score_fn=lambda exp, fb: 0.5,
        )

        exp1 = _make_mock_experiment("A momentum-based factor using price trends")
        fb1 = _make_feedback(True)
        trace = _make_mock_trace([(exp1, fb1)])

        mgr.update_from_trace(trace)
        assert mgr._processed_up_to == 1
        assert "momentum" in mgr.tracker.directions
        assert mgr.tracker.directions["momentum"].attempt_count == 1
        assert mgr.tracker.directions["momentum"].best_score == 0.5

    def test_incremental_processing(self):
        mgr = DirectionManager(
            classifier=lambda text, existing: ("dir_a", "Direction A"),
            score_fn=lambda exp, fb: 0.5,
        )

        exp1 = _make_mock_experiment("hypothesis 1")
        exp2 = _make_mock_experiment("hypothesis 2")
        trace = _make_mock_trace([(exp1, _make_feedback(True))])

        mgr.update_from_trace(trace)
        assert mgr._processed_up_to == 1

        # Add another entry
        trace.hist.append((exp2, _make_feedback(True)))
        mgr.update_from_trace(trace)
        assert mgr._processed_up_to == 2
        assert mgr.tracker.directions["dir_a"].attempt_count == 2

    def test_does_not_reprocess_entries(self):
        classify_count = [0]

        def classifier(text, existing):
            classify_count[0] += 1
            return ("dir_a", "Direction A")

        mgr = DirectionManager(
            classifier=classifier,
            score_fn=lambda exp, fb: 0.5,
        )

        exp1 = _make_mock_experiment("hypothesis 1")
        trace = _make_mock_trace([(exp1, _make_feedback(True))])

        mgr.update_from_trace(trace)
        assert classify_count[0] == 1

        # Update again without new entries — classifier should NOT be called again
        mgr.update_from_trace(trace)
        assert classify_count[0] == 1


# ---------------------------------------------------------------------------
# Test: Classifier integration
# ---------------------------------------------------------------------------


class TestClassifier:
    def test_classifier_receives_existing_directions(self):
        received_existing = []

        def classifier(text, existing):
            received_existing.append(list(existing))
            if "momentum" in text:
                return ("momentum", "Momentum")
            return ("volatility", "Volatility")

        mgr = DirectionManager(
            classifier=classifier,
            score_fn=lambda exp, fb: 0.5,
        )

        exp1 = _make_mock_experiment("momentum hypothesis")
        exp2 = _make_mock_experiment("volatility hypothesis")
        trace = _make_mock_trace([(exp1, _make_feedback(True)), (exp2, _make_feedback(True))])

        mgr.update_from_trace(trace)
        # First call: no existing directions
        assert len(received_existing[0]) == 0
        # Second call: momentum already exists
        assert "momentum" in received_existing[1]

    def test_classifier_reuses_existing_direction(self):
        """Same direction name → same Direction object in tracker."""

        def classifier(text, existing):
            return ("momentum", "Momentum")

        mgr = DirectionManager(
            classifier=classifier,
            score_fn=lambda exp, fb: 0.5,
        )

        exp1 = _make_mock_experiment("momentum hypothesis 1")
        exp2 = _make_mock_experiment("momentum hypothesis 2")
        trace = _make_mock_trace([(exp1, _make_feedback(True)), (exp2, _make_feedback(True))])

        mgr.update_from_trace(trace)
        assert len(mgr.tracker.directions) == 1
        assert "momentum" in mgr.tracker.directions
        assert mgr.tracker.directions["momentum"].attempt_count == 2


# ---------------------------------------------------------------------------
# Test: Score function
# ---------------------------------------------------------------------------


class TestScoreFn:
    def test_uses_injected_score_fn(self):
        score_log = []

        def score_fn(exp, fb):
            score_log.append((exp, fb))
            return 0.8

        mgr = DirectionManager(
            classifier=lambda text, existing: ("dir_a", "Direction A"),
            score_fn=score_fn,
        )

        exp1 = _make_mock_experiment("hypothesis 1")
        fb1 = _make_feedback(True)
        trace = _make_mock_trace([(exp1, fb1)])

        mgr.update_from_trace(trace)
        assert len(score_log) == 1
        assert mgr.tracker.directions["dir_a"].best_score == 0.8

    def test_default_score_uses_decision_true(self):
        mgr = DirectionManager(
            classifier=lambda text, existing: ("dir_a", "Direction A"),
        )

        exp1 = _make_mock_experiment("hypothesis 1")
        trace = _make_mock_trace([(exp1, _make_feedback(True))])

        mgr.update_from_trace(trace)
        assert mgr.tracker.directions["dir_a"].best_score == 1.0

    def test_default_score_uses_decision_false(self):
        mgr = DirectionManager(
            classifier=lambda text, existing: ("dir_a", "Direction A"),
        )

        exp1 = _make_mock_experiment("hypothesis 1")
        trace = _make_mock_trace([(exp1, _make_feedback(False))])

        mgr.update_from_trace(trace)
        assert mgr.tracker.directions["dir_a"].best_score == 0.0


# ---------------------------------------------------------------------------
# Test: get_guidance
# ---------------------------------------------------------------------------


class TestGetGuidance:
    def test_empty_returns_empty_string(self):
        mgr = DirectionManager()
        assert mgr.get_guidance() == ""

    def test_returns_direction_summary(self):
        mgr = DirectionManager(
            classifier=lambda text, existing: (
                "momentum" if "momentum" in text else "volatility",
                "Description",
            ),
            score_fn=lambda exp, fb: 0.5,
        )

        exp1 = _make_mock_experiment("momentum hypothesis")
        exp2 = _make_mock_experiment("volatility hypothesis")
        trace = _make_mock_trace([(exp1, _make_feedback(True)), (exp2, _make_feedback(True))])

        mgr.update_from_trace(trace)
        guidance = mgr.get_guidance()

        assert "momentum" in guidance
        assert "volatility" in guidance

    def test_warns_about_saturated_direction(self):
        mgr = DirectionManager(
            classifier=lambda text, existing: ("momentum", "Momentum"),
            score_fn=lambda exp, fb: 0.3,  # low score, won't improve
        )

        # First entry with higher score
        exp0 = _make_mock_experiment("momentum init")
        trace_entries = [
            (exp0, _make_feedback(True)),
        ]
        # Override score for first entry
        scores = [0.5, 0.3, 0.3, 0.3]
        mgr = DirectionManager(
            classifier=lambda text, existing: ("momentum", "Momentum"),
            score_fn=lambda exp, fb: scores.pop(0),
        )

        trace = _make_mock_trace([])
        for i in range(4):
            trace.hist.append((_make_mock_experiment(f"momentum {i}"), _make_feedback(True)))

        mgr.update_from_trace(trace)
        assert mgr.tracker.directions["momentum"].is_saturated

        guidance = mgr.get_guidance()
        assert "saturat" in guidance.lower() or "saturat" in guidance

    def test_suggests_underexplored_direction(self):
        mgr = DirectionManager(
            classifier=lambda text, existing: (
                "momentum" if "momentum" in text else "volatility",
                "Description",
            ),
            score_fn=lambda exp, fb: 0.5,
        )

        # Only explore momentum, not volatility
        trace = _make_mock_trace([
            (_make_mock_experiment("momentum 1"), _make_feedback(True)),
            (_make_mock_experiment("momentum 2"), _make_feedback(True)),
        ])

        mgr.update_from_trace(trace)
        underexplored = mgr.tracker.get_underexplored_directions(threshold=2)
        assert any(d.name == "volatility" for d in underexplored)
