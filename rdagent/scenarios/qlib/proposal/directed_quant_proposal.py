"""
Directed hypothesis generation for Qlib quant scenario.

Extends QlibQuantHypothesisGen with structured direction control.
Uses DirectionManager to track exploration directions and inject
guidance text into the LLM prompt via the RAG context field.

Zero source-code modification — pure inheritance extension.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from rdagent.core.direction_mgr import DirectionManager
from rdagent.core.proposal import Scenario, Trace
from rdagent.scenarios.qlib.proposal.bandit import (
    EnvController,
    extract_metrics_from_experiment,
)
from rdagent.scenarios.qlib.proposal.quant_proposal import QlibQuantHypothesisGen
from rdagent.utils.agent.tpl import T


# ---------------------------------------------------------------------------
# Qlib-specific classifier (keyword-based, no LLM call)
# ---------------------------------------------------------------------------

_KEYWORD_DIRECTIONS: list[tuple[str, list[str], str]] = [
    (
        "momentum",
        ["momentum", "trend", "moving average", "macd", "rsi", "price rate"],
        "Momentum and trend-following strategies",
    ),
    (
        "volatility",
        ["volatility", "garch", "atr", "bollinger", "realized variance"],
        "Volatility-based strategies",
    ),
    (
        "mean_reversion",
        ["mean reversion", "cointegration", "z-score", "pair trading", "distance"],
        "Mean reversion strategies",
    ),
    (
        "ml_model",
        [
            "machine learning",
            "neural",
            "lstm",
            "gru",
            "transformer",
            "xgboost",
            "random forest",
            "attention",
        ],
        "Machine learning model-based strategies",
    ),
    (
        "value",
        ["value", "fundamental", "earnings", "pe ratio", "pb ratio", "dividend"],
        "Value and fundamental strategies",
    ),
    (
        "volume",
        ["volume", "vwap", "obv", "money flow", "turnover"],
        "Volume-based strategies",
    ),
]


def qlib_direction_classifier(
    hypothesis_text: str,
    existing_names: list[str],
) -> tuple[str, str]:
    """Classify a hypothesis into a quant direction using keyword matching.

    Uses word-boundary matching to avoid false positives (e.g. "rsi"
    matching inside "reversion").

    Parameters
    ----------
    hypothesis_text:
        The hypothesis text to classify.
    existing_names:
        Names of existing directions (unused by keyword classifier but
        kept for interface compatibility with DirectionManager).

    Returns
    -------
    tuple[str, str]
        (direction_name, description)
    """
    text_lower = hypothesis_text.lower()
    for name, keywords, description in _KEYWORD_DIRECTIONS:
        for kw in keywords:
            if re.search(r"\b" + re.escape(kw) + r"\b", text_lower):
                return (name, description)
    return ("other", "Unclassified direction")


# ---------------------------------------------------------------------------
# Qlib-specific score function (based on backtest metrics)
# ---------------------------------------------------------------------------

# Use the same weights as EnvController for consistency
_DEFAULT_WEIGHTS = np.array([0.1, 0.1, 0.05, 0.05, 0.25, 0.15, 0.1, 0.2])


def qlib_direction_score_fn(experiment: Any, feedback: Any) -> float:
    """Compute a direction score from backtest metrics.

    Uses the same weighted metric vector as ``EnvController.reward()``.
    Falls back to ``feedback.decision`` if metrics extraction fails.

    Parameters
    ----------
    experiment:
        The experiment object (must have a ``.result`` attribute).
    feedback:
        The feedback object (must have a ``.decision`` attribute for fallback).

    Returns
    -------
    float
        A scalar score reflecting experiment quality.
    """
    try:
        metrics = extract_metrics_from_experiment(experiment)
        return float(np.dot(_DEFAULT_WEIGHTS, metrics.as_vector()))
    except Exception:
        return 1.0 if getattr(feedback, "decision", False) else 0.0


# ---------------------------------------------------------------------------
# DirectedQlibQuantHypothesisGen
# ---------------------------------------------------------------------------


class DirectedQlibQuantHypothesisGen(QlibQuantHypothesisGen):
    """Qlib quant hypothesis generator with direction-aware guidance injection.

    Wraps ``QlibQuantHypothesisGen.prepare_context`` and appends direction
    exploration guidance to the ``RAG`` field of the context dictionary.
    The guidance summarizes which directions have been explored, which are
    saturated, and which are underexplored — steering the LLM toward more
    structured exploration.
    """

    def __init__(
        self,
        scen: Scenario,
        direction_mgr: DirectionManager | None = None,
    ) -> None:
        super().__init__(scen)
        self.direction_mgr = direction_mgr or DirectionManager(
            classifier=qlib_direction_classifier,
            score_fn=qlib_direction_score_fn,
        )

    def prepare_context(self, trace: Trace) -> tuple[dict, bool]:
        context_dict, json_flag = super().prepare_context(trace)

        # Update direction tracking from the full trace history
        self.direction_mgr.update_from_trace(trace)

        # Render guidance using template
        ctx = self.direction_mgr.get_guidance_context()
        if ctx:
            guidance = T("scenarios.qlib.prompts_direction:direction_guidance").r(**ctx)
            if guidance:
                context_dict["RAG"] = context_dict.get("RAG", "") + "\n\n" + guidance

        return context_dict, json_flag
