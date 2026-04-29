"""Integration tests for DirectedQlibQuantHypothesisGen.

Tests the full chain: config → import_class → instantiation → multi-round usage.
"""

from unittest.mock import MagicMock, patch

import pytest

from rdagent.core.proposal import ExperimentFeedback, Hypothesis, Scenario, Trace
from rdagent.core.utils import import_class
from rdagent.scenarios.qlib.proposal.directed_quant_proposal import (
    DirectedQlibQuantHypothesisGen,
    qlib_direction_classifier,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DIRECTED_CLASS_PATH = (
    "rdagent.scenarios.qlib.proposal.directed_quant_proposal.DirectedQlibQuantHypothesisGen"
)


def _make_hypothesis(text: str, action: str = "factor") -> Hypothesis:
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


def _make_mock_experiment(text: str, action: str = "factor"):
    exp = MagicMock()
    exp.hypothesis = _make_hypothesis(text, action)
    exp.result = {
        "IC": 0.05,
        "ICIR": 0.5,
        "Rank IC": 0.04,
        "Rank ICIR": 0.4,
        "1day.excess_return_with_cost.annualized_return ": 0.1,
        "1day.excess_return_with_cost.information_ratio": 0.8,
        "1day.excess_return_with_cost.max_drawdown": -0.05,
    }
    return exp


def _make_feedback(decision: bool = True):
    return ExperimentFeedback(reason="test", decision=decision)


def _make_mock_trace(entries):
    trace = Trace(scen=MagicMock(spec=Scenario))
    trace.hist = list(entries)
    return trace


# ---------------------------------------------------------------------------
# Test: import_class chain
# ---------------------------------------------------------------------------


class TestImportClassChain:
    def test_import_class_loads_directed_gen(self):
        """import_class should resolve the class from its full path."""
        cls = import_class(_DIRECTED_CLASS_PATH)
        assert cls is DirectedQlibQuantHypothesisGen

    def test_instantiate_via_import_class(self):
        """import_class(path)(scen) pattern should work (matches QuantRDLoop)."""
        cls = import_class(_DIRECTED_CLASS_PATH)
        scen = MagicMock(spec=Scenario)
        gen = cls(scen)
        assert isinstance(gen, DirectedQlibQuantHypothesisGen)
        assert hasattr(gen, "direction_mgr")


# ---------------------------------------------------------------------------
# Test: Multi-round direction accumulation
# ---------------------------------------------------------------------------


class TestMultiRoundAccumulation:
    @patch(
        "rdagent.scenarios.qlib.proposal.quant_proposal.QlibQuantHypothesisGen.prepare_context",
    )
    def test_directions_accumulate_across_rounds(self, mock_super_prepare):
        """Direction state should accumulate across multiple prepare_context calls."""
        call_count = [0]
        base_rags = [
            "First round RAG.",
            "Second round RAG.",
            "Third round RAG.",
        ]

        def side_effect(trace):
            idx = call_count[0]
            call_count[0] += 1
            return (
                {
                    "hypothesis_and_feedback": "mock",
                    "last_hypothesis_and_feedback": "mock",
                    "SOTA_hypothesis_and_feedback": None,
                    "RAG": base_rags[idx] if idx < len(base_rags) else base_rags[-1],
                    "hypothesis_output_format": "format",
                    "hypothesis_specification": "spec",
                },
                True,
            )

        mock_super_prepare.side_effect = side_effect

        gen = DirectedQlibQuantHypothesisGen(scen=MagicMock(spec=Scenario))

        # Round 1: momentum
        trace = _make_mock_trace([
            (_make_mock_experiment("A momentum-based factor"), _make_feedback(True)),
        ])
        ctx1, _ = gen.prepare_context(trace)
        assert "momentum" in ctx1["RAG"]
        assert len(gen.direction_mgr.tracker.directions) == 1

        # Round 2: add volatility
        trace.hist.append((_make_mock_experiment("Volatility clustering factor"), _make_feedback(True)))
        ctx2, _ = gen.prepare_context(trace)
        assert "volatility" in ctx2["RAG"]
        assert len(gen.direction_mgr.tracker.directions) == 2

        # Round 3: more momentum → momentum should show 2 attempts
        trace.hist.append((_make_mock_experiment("Another momentum factor"), _make_feedback(True)))
        ctx3, _ = gen.prepare_context(trace)
        assert gen.direction_mgr.tracker.directions["momentum"].attempt_count == 2

    @patch(
        "rdagent.scenarios.qlib.proposal.quant_proposal.QlibQuantHypothesisGen.prepare_context",
    )
    def test_saturated_direction_noted_in_subsequent_round(self, mock_super_prepare):
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

        scores = [0.8, 0.2, 0.3, 0.1]
        from rdagent.core.direction_mgr import DirectionManager

        gen = DirectedQlibQuantHypothesisGen(scen=MagicMock(spec=Scenario))
        gen.direction_mgr = DirectionManager(
            classifier=qlib_direction_classifier,
            score_fn=lambda exp, fb: scores.pop(0),
        )

        # Feed 4 momentum entries to saturate the direction
        trace = _make_mock_trace([
            (_make_mock_experiment("momentum 1"), _make_feedback(True)),
            (_make_mock_experiment("momentum 2"), _make_feedback(True)),
            (_make_mock_experiment("momentum 3"), _make_feedback(True)),
            (_make_mock_experiment("momentum 4"), _make_feedback(True)),
        ])

        ctx, _ = gen.prepare_context(trace)
        assert gen.direction_mgr.tracker.directions["momentum"].is_saturated
        assert "saturat" in ctx["RAG"].lower()


# ---------------------------------------------------------------------------
# Test: Template rendering
# ---------------------------------------------------------------------------


class TestTemplateRendering:
    @patch(
        "rdagent.scenarios.qlib.proposal.quant_proposal.QlibQuantHypothesisGen.prepare_context",
    )
    def test_guidance_uses_template_format(self, mock_super_prepare):
        """Guidance should be rendered through prompts_direction.yaml template."""
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
        trace = _make_mock_trace([
            (_make_mock_experiment("A momentum-based factor"), _make_feedback(True)),
        ])

        ctx, _ = gen.prepare_context(trace)

        # Template should produce structured output with these markers
        assert "[Direction Exploration Status]" in ctx["RAG"]
        assert "Explored directions (ranked by potential):" in ctx["RAG"]
