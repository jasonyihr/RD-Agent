"""Tests for DirectedQlibQuantHypothesisGen (Phase 2: qlib direction extension)."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rdagent.core.proposal import ExperimentFeedback, Hypothesis, Scenario, Trace
from rdagent.scenarios.qlib.proposal.directed_quant_proposal import (
    DirectedQlibQuantHypothesisGen,
    qlib_direction_classifier,
    qlib_direction_score_fn,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_hypothesis(text: str, action: str = "factor") -> Hypothesis:
    """Create a mock QlibQuantHypothesis-like object."""
    hyp = Hypothesis(
        hypothesis=text,
        reason="test",
        concise_reason="test",
        concise_observation="test",
        concise_justification="test",
        concise_knowledge="test",
    )
    hyp.action = action
    return hyp


def _make_mock_experiment(hypothesis_text: str, action: str = "factor", result=None):
    """Create a mock experiment."""
    exp = MagicMock()
    exp.hypothesis = _make_hypothesis(hypothesis_text, action)
    exp.result = result or {
        "IC": 0.05,
        "ICIR": 0.5,
        "Rank IC": 0.04,
        "Rank ICIR": 0.4,
        "1day.excess_return_with_cost.annualized_return ": 0.1,
        "1day.excess_return_with_cost.information_ratio": 0.8,
        "1day.excess_return_with_cost.max_drawdown": -0.05,
    }
    return exp


def _make_mock_trace(entries):
    """Create a Trace with given (experiment, feedback) entries."""
    trace = Trace(scen=MagicMock(spec=Scenario))
    trace.hist = list(entries)
    return trace


def _make_feedback(decision: bool = True):
    return ExperimentFeedback(reason="test", decision=decision)


# ---------------------------------------------------------------------------
# Test: qlib_direction_classifier
# ---------------------------------------------------------------------------


class TestQlibDirectionClassifier:
    def test_momentum_keywords(self):
        name, desc = qlib_direction_classifier("A momentum-based factor using price trends", [])
        assert name == "momentum"
        assert "momentum" in desc.lower()

    def test_volatility_keywords(self):
        name, desc = qlib_direction_classifier("Volatility clustering factor with GARCH model", [])
        assert name == "volatility"

    def test_mean_reversion_keywords(self):
        name, desc = qlib_direction_classifier("Mean reversion strategy using z-score", [])
        assert name == "mean_reversion"

    def test_ml_model_keywords(self):
        name, desc = qlib_direction_classifier("LSTM neural network for prediction", [])
        assert name == "ml_model"

    def test_value_keywords(self):
        name, desc = qlib_direction_classifier("Value investing based on PE ratio", [])
        assert name == "value"

    def test_volume_keywords(self):
        name, desc = qlib_direction_classifier("Volume weighted average price factor", [])
        assert name == "volume"

    def test_unknown_returns_other(self):
        name, desc = qlib_direction_classifier("A completely novel approach with butterflies", [])
        assert name == "other"

    def test_case_insensitive(self):
        name, desc = qlib_direction_classifier("MOMENTUM factor using RSI", [])
        assert name == "momentum"

    def test_first_match_wins(self):
        """If text matches multiple categories, the first match wins."""
        name, desc = qlib_direction_classifier("Momentum and volatility combined", [])
        assert name == "momentum"


# ---------------------------------------------------------------------------
# Test: qlib_direction_score_fn
# ---------------------------------------------------------------------------


class TestQlibDirectionScoreFn:
    def test_positive_metrics(self):
        exp = _make_mock_experiment("test", result={
            "IC": 0.1,
            "ICIR": 1.0,
            "Rank IC": 0.08,
            "Rank ICIR": 0.8,
            "1day.excess_return_with_cost.annualized_return ": 0.2,
            "1day.excess_return_with_cost.information_ratio": 1.5,
            "1day.excess_return_with_cost.max_drawdown": -0.05,
        })
        fb = ExperimentFeedback(reason="test", decision=True)
        score = qlib_direction_score_fn(exp, fb)
        assert score > 0.0

    def test_zero_metrics(self):
        exp = _make_mock_experiment("test", result={
            "IC": 0.0,
            "ICIR": 0.0,
            "Rank IC": 0.0,
            "Rank ICIR": 0.0,
            "1day.excess_return_with_cost.annualized_return ": 0.0,
            "1day.excess_return_with_cost.information_ratio": 0.0,
            "1day.excess_return_with_cost.max_drawdown": 0.0,
        })
        fb = ExperimentFeedback(reason="test", decision=False)
        score = qlib_direction_score_fn(exp, fb)
        assert score == pytest.approx(0.0)

    def test_negative_metrics(self):
        exp = _make_mock_experiment("test", result={
            "IC": -0.05,
            "ICIR": -0.5,
            "Rank IC": -0.04,
            "Rank ICIR": -0.4,
            "1day.excess_return_with_cost.annualized_return ": -0.1,
            "1day.excess_return_with_cost.information_ratio": -0.8,
            "1day.excess_return_with_cost.max_drawdown": -0.2,
        })
        fb = ExperimentFeedback(reason="test", decision=False)
        score = qlib_direction_score_fn(exp, fb)
        assert score < 0.0

    def test_fallback_on_none_result(self):
        """If result is None, extract_metrics catches error and returns zero Metrics."""
        exp = MagicMock()
        exp.result = None  # extract_metrics catches and returns Metrics()
        fb = ExperimentFeedback(reason="test", decision=True)
        score = qlib_direction_score_fn(exp, fb)
        # All-zero Metrics → weighted dot product = 0.0
        assert score == pytest.approx(0.0)

    def test_fallback_decision_false(self):
        exp = MagicMock()
        exp.result = None
        fb = ExperimentFeedback(reason="test", decision=False)
        score = qlib_direction_score_fn(exp, fb)
        assert score == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Test: DirectedQlibQuantHypothesisGen
# ---------------------------------------------------------------------------


class TestDirectedQlibQuantHypothesisGen:
    def test_inherits_from_qlib_quant(self):
        from rdagent.scenarios.qlib.proposal.quant_proposal import QlibQuantHypothesisGen

        gen = DirectedQlibQuantHypothesisGen(scen=MagicMock(spec=Scenario))
        assert isinstance(gen, QlibQuantHypothesisGen)

    def test_has_direction_manager(self):
        gen = DirectedQlibQuantHypothesisGen(scen=MagicMock(spec=Scenario))
        assert hasattr(gen, "direction_mgr")
        from rdagent.core.direction_mgr import DirectionManager

        assert isinstance(gen.direction_mgr, DirectionManager)

    @patch(
        "rdagent.scenarios.qlib.proposal.quant_proposal.QlibQuantHypothesisGen.prepare_context",
    )
    def test_guidance_injected_into_rag(self, mock_super_prepare):
        """Direction guidance should be appended to the RAG field."""
        mock_super_prepare.return_value = (
            {
                "hypothesis_and_feedback": "mock hist",
                "last_hypothesis_and_feedback": "mock last",
                "SOTA_hypothesis_and_feedback": None,
                "RAG": "Base RAG content.",
                "hypothesis_output_format": "format",
                "hypothesis_specification": "spec",
            },
            True,
        )

        gen = DirectedQlibQuantHypothesisGen(scen=MagicMock(spec=Scenario))

        # Create a trace with momentum entries to generate direction info
        entries = [
            (_make_mock_experiment("A momentum-based factor", "factor"), _make_feedback(True)),
            (_make_mock_experiment("Another momentum factor", "factor"), _make_feedback(True)),
            (_make_mock_experiment("A volatility factor", "factor"), _make_feedback(True)),
        ]
        trace = _make_mock_trace(entries)

        context_dict, json_flag = gen.prepare_context(trace)

        # RAG should contain direction guidance
        assert "Base RAG content." in context_dict["RAG"]
        assert "Direction" in context_dict["RAG"] or "direction" in context_dict["RAG"].lower()
        assert "momentum" in context_dict["RAG"]

    @patch(
        "rdagent.scenarios.qlib.proposal.quant_proposal.QlibQuantHypothesisGen.prepare_context",
    )
    def test_empty_trace_no_guidance(self, mock_super_prepare):
        """Empty trace should not add direction guidance."""
        mock_super_prepare.return_value = (
            {
                "hypothesis_and_feedback": "",
                "last_hypothesis_and_feedback": None,
                "SOTA_hypothesis_and_feedback": None,
                "RAG": "Base RAG content.",
                "hypothesis_output_format": "format",
                "hypothesis_specification": "spec",
            },
            True,
        )

        gen = DirectedQlibQuantHypothesisGen(scen=MagicMock(spec=Scenario))
        trace = _make_mock_trace([])

        context_dict, json_flag = gen.prepare_context(trace)

        # No direction info to inject, RAG should be unchanged
        assert context_dict["RAG"] == "Base RAG content."

    @patch(
        "rdagent.scenarios.qlib.proposal.quant_proposal.QlibQuantHypothesisGen.prepare_context",
    )
    def test_guidance_warns_about_saturated(self, mock_super_prepare):
        """Guidance should warn when a direction is saturated."""
        mock_super_prepare.return_value = (
            {
                "hypothesis_and_feedback": "mock",
                "last_hypothesis_and_feedback": "mock",
                "SOTA_hypothesis_and_feedback": None,
                "RAG": "Base RAG.",
                "hypothesis_output_format": "format",
                "hypothesis_specification": "spec",
            },
            True,
        )

        gen = DirectedQlibQuantHypothesisGen(scen=MagicMock(spec=Scenario))

        # Create entries that saturate momentum (4 entries, only first improves)
        # Use different scores: first is high, rest are low
        entries = [
            (_make_mock_experiment("momentum factor 1"), _make_feedback(True)),
            (_make_mock_experiment("momentum factor 2"), _make_feedback(True)),
            (_make_mock_experiment("momentum factor 3"), _make_feedback(True)),
            (_make_mock_experiment("momentum factor 4"), _make_feedback(True)),
        ]
        trace = _make_mock_trace(entries)

        # Manually set scores to simulate saturation
        # First: high score, rest: low scores → consecutive_no_improvement = 3
        scores = [0.8, 0.2, 0.3, 0.1]
        gen.direction_mgr = DirectedQlibQuantHypothesisGen(
            scen=MagicMock(spec=Scenario),
        ).direction_mgr
        # We need to use a custom score_fn that gives declining scores
        from rdagent.core.direction_mgr import DirectionManager

        gen.direction_mgr = DirectionManager(
            classifier=qlib_direction_classifier,
            score_fn=lambda exp, fb: scores.pop(0),
        )

        context_dict, _ = gen.prepare_context(trace)
        assert "saturat" in context_dict["RAG"].lower()
