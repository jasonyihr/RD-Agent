"""
Direction management for structured exploration in R&D loops.

This module provides data structures for tracking exploration directions
in the hypothesis generation process, enabling structured direction control
instead of random exploration.

Phase 1: Direction labeling and state tracking.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from math import inf


@dataclass
class Direction:
    """A exploration direction for hypothesis generation.

    Represents a coherent theme or approach in the R&D process,
    such as "momentum factors" or "LSTM-based models".
    """

    name: str  # Direction identifier, e.g. "momentum", "volatility_cluster"
    description: str  # Description of the direction
    category: str = "factor"  # Category: factor / model / feature_eng


@dataclass
class DirectionState:
    """Exploration state for a single direction.

    Tracks how deeply a direction has been explored, its performance,
    and whether it has reached saturation.
    """

    direction: Direction
    attempt_count: int = 0
    best_score: float = -inf
    last_score: float = 0.0
    improvement_trend: float = 0.0
    last_attempt_loop: int = 0
    consecutive_no_improvement: int = 0

    SATURATION_THRESHOLD: int = 3  # consecutive no-improvement to be saturated

    def update(self, score: float, loop_idx: int) -> None:
        """Update direction state with a new experiment result."""
        self.attempt_count += 1
        self.last_score = score
        if score > self.best_score:
            self.best_score = score
            self.consecutive_no_improvement = 0
        else:
            self.consecutive_no_improvement += 1
        self.last_attempt_loop = loop_idx

    @property
    def is_saturated(self) -> bool:
        """Whether the direction has saturated (consecutive no-improvement)."""
        return self.consecutive_no_improvement >= self.SATURATION_THRESHOLD

    def ucb_score(self, total_attempts: int, c: float = 1.41) -> float:
        """UCB score = best_score + c * sqrt(ln(total) / attempts).

        Uses Upper Confidence Bound to balance exploration and exploitation.
        Unexplored directions get infinite score (always try them first).
        """
        if self.attempt_count == 0:
            return inf
        return self.best_score + c * math.sqrt(math.log(total_attempts) / self.attempt_count)


class DirectionTracker:
    """Tracks exploration state across all directions.

    Maintains a dictionary of Direction -> DirectionState mappings
    and provides query methods for direction selection strategies.
    """

    def __init__(self) -> None:
        self.directions: dict[str, DirectionState] = {}

    def get_or_create(self, name: str, description: str = "", category: str = "factor") -> Direction:
        """Get an existing direction or create a new one."""
        if name not in self.directions:
            self.directions[name] = DirectionState(
                direction=Direction(name=name, description=description, category=category)
            )
        return self.directions[name].direction

    def update(self, direction: Direction, score: float, loop_idx: int) -> None:
        """Update the state of a specific direction."""
        if direction.name in self.directions:
            self.directions[direction.name].update(score, loop_idx)

    def get_summary(self) -> str:
        """Generate a human-readable summary of all directions for LLM prompts."""
        if not self.directions:
            return "No directions explored yet."
        lines = ["Direction Exploration Summary:"]
        for ds in self.directions.values():
            status = "SATURATED" if ds.is_saturated else "ACTIVE"
            lines.append(
                f"  [{status}] {ds.direction.name} "
                f"(attempts={ds.attempt_count}, best={ds.best_score:.4f}, "
                f"last={ds.last_score:.4f}, no_improve={ds.consecutive_no_improvement})"
            )
        return "\n".join(lines)

    def get_ucb_ranking(self) -> list[tuple[Direction, float]]:
        """Rank directions by UCB score (highest first).

        Returns list of (Direction, ucb_score) tuples.
        """
        if not self.directions:
            return []
        total = sum(ds.attempt_count for ds in self.directions.values())
        if total == 0:
            # All unexplored - return in arbitrary order with inf score
            return [(ds.direction, inf) for ds in self.directions.values()]
        ranked = [(ds.direction, ds.ucb_score(total)) for ds in self.directions.values()]
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

    def get_saturated_directions(self) -> list[Direction]:
        """Return all directions that have reached saturation."""
        return [ds.direction for ds in self.directions.values() if ds.is_saturated]

    def get_underexplored_directions(self, threshold: int = 2) -> list[Direction]:
        """Return directions with fewer than threshold attempts."""
        return [ds.direction for ds in self.directions.values() if ds.attempt_count < threshold]
