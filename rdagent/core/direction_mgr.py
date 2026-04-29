"""
Direction management engine for structured exploration in R&D loops.

Phase 2: DirectionManager observes trace history, classifies hypotheses into
directions, and generates guidance text for prompt injection.

This module is scenario-agnostic. Scenario-specific classifiers and score
functions are injected via constructor parameters.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from rdagent.core.direction import Direction, DirectionTracker
from rdagent.core.proposal import Trace


class DirectionManager:
    """Manages direction classification, scoring, and guidance generation.

    Observes ``trace.hist`` incrementally, classifies each hypothesis into a
    direction, updates direction state with scores, and produces guidance text
    that can be injected into LLM prompts.

    Parameters
    ----------
    tracker:
        An existing ``DirectionTracker``, or ``None`` to create a fresh one.
    classifier:
        A callable ``(hypothesis_text: str, existing_direction_names: list[str])
        -> tuple[str, str]`` that returns ``(direction_name, description)``.
        If ``None``, a default classifier using the LLM is used.
    score_fn:
        A callable ``(experiment, feedback) -> float`` that computes a score.
        If ``None``, uses ``1.0`` for positive feedback and ``0.0`` otherwise.
    """

    def __init__(
        self,
        tracker: DirectionTracker | None = None,
        classifier: Callable[[str, list[str]], tuple[str, str]] | None = None,
        score_fn: Callable[[Any, Any], float] | None = None,
    ) -> None:
        self.tracker: DirectionTracker = tracker or DirectionTracker()
        self._direction_map: dict[int, Direction] = {}
        self._processed_up_to: int = 0
        self._classifier = classifier
        self._score_fn = score_fn

    def _classify_direction(self, hypothesis_text: str) -> Direction:
        """Classify a hypothesis into a direction using the injected classifier."""
        existing_names = list(self.tracker.directions.keys())

        if self._classifier is not None:
            name, description = self._classifier(hypothesis_text, existing_names)
        else:
            name, description = self._default_classifier(hypothesis_text, existing_names)

        return self.tracker.get_or_create(name, description=description)

    def _default_classifier(
        self,
        hypothesis_text: str,
        existing_names: list[str],
    ) -> tuple[str, str]:
        """LLM-based default classifier. Calls the API to classify hypothesis text."""
        from rdagent.oai.llm_utils import APIBackend

        import json

        existing_str = ", ".join(existing_names) if existing_names else "None"

        system_prompt = (
            "You are a research direction classifier. "
            "Given a hypothesis from a quantitative finance R&D process, "
            "classify it into a concise exploration direction name.\n"
            "Existing directions: [" + existing_str + "]\n"
            "If the hypothesis fits an existing direction, reuse that name. "
            "Otherwise create a new short name (e.g., momentum, volatility, "
            "mean_reversion, ml_model, etc.).\n"
            'Return JSON: {"direction_name": "...", "description": "..."}'
        )
        user_prompt = f"Hypothesis: {hypothesis_text}"

        resp = APIBackend().build_messages_and_create_chat_completion(
            user_prompt,
            system_prompt,
            json_mode=True,
        )
        result = json.loads(resp)
        return result.get("direction_name", "unclassified"), result.get("description", "")

    def _compute_score(self, experiment: Any, feedback: Any) -> float:
        """Compute a score for a trace entry."""
        if self._score_fn is not None:
            return self._score_fn(experiment, feedback)
        return 1.0 if getattr(feedback, "decision", False) else 0.0

    def update_from_trace(self, trace: Trace) -> None:
        """Process new entries in ``trace.hist`` since the last call."""
        for i in range(self._processed_up_to, len(trace.hist)):
            exp, feedback = trace.hist[i]
            hypothesis = exp.hypothesis

            if i not in self._direction_map:
                direction = self._classify_direction(hypothesis.hypothesis)
                self._direction_map[i] = direction

            score = self._compute_score(exp, feedback)
            self.tracker.update(self._direction_map[i], score, i)

        self._processed_up_to = len(trace.hist)

    def get_guidance(self) -> str:
        """Generate guidance text for prompt injection.

        Returns a human-readable summary of direction exploration status
        with recommendations, or an empty string if no directions exist.
        """
        ctx = self.get_guidance_context()
        if not ctx:
            return ""

        lines = ["[Direction Exploration Status]"]
        lines.append("Explored directions (ranked by potential):")
        for d in ctx["directions"]:
            lines.append(
                f"  - {d['name']} (attempts={d['attempts']}, "
                f"best={d['best_score']:.4f}, UCB={d['ucb']:.4f}) [{d['status']}]"
            )

        if ctx["recommendations"]:
            lines.append("Recommendations:")
            for r in ctx["recommendations"]:
                lines.append(f"  - {r}")

        return "\n".join(lines)

    def get_guidance_context(self) -> dict:
        """Return structured direction data for template rendering.

        Returns
        -------
        dict
            A dict with ``directions`` (list of direction info dicts) and
            ``recommendations`` (list of recommendation strings), or an
            empty dict if no directions have been explored.
        """
        if not self.tracker.directions:
            return {}

        ranking = self.tracker.get_ucb_ranking()
        saturated = self.tracker.get_saturated_directions()
        underexplored = self.tracker.get_underexplored_directions()

        directions = []
        for d, ucb in ranking:
            ds = self.tracker.directions[d.name]
            directions.append({
                "name": d.name,
                "description": d.description,
                "attempts": ds.attempt_count,
                "best_score": ds.best_score,
                "ucb": ucb,
                "status": "SATURATED" if ds.is_saturated else "ACTIVE",
            })

        recommendations = []
        if saturated:
            names = [d.name for d in saturated]
            recommendations.append(
                f"Warning: {', '.join(names)} appear(s) saturated "
                f"(consecutive no-improvement). Consider switching direction."
            )
        if underexplored:
            names = [d.name for d in underexplored]
            recommendations.append(
                f"Priority: {', '.join(names)} are underexplored. "
                f"Consider trying these directions."
            )

        return {"directions": directions, "recommendations": recommendations}
