#!/usr/bin/env python3
"""Pure anytime discovery admission and deterministic segment routing.

This module owns scheduling only.  It cannot admit candidates, validate
localization, or authorize motion/grasping.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Iterable


START_DISCOVERY_SEGMENT = "START_DISCOVERY_SEGMENT"
START_CANDIDATE_ATTEMPT = "START_CANDIDATE_ATTEMPT"
STOP_NO_FEASIBLE_WORK = "STOP_NO_FEASIBLE_WORK"


@dataclass(frozen=True)
class DiscoverySegment:
    segment_id: str
    coverage_key: Any
    estimated_cost_s: float
    can_observe_missing_pending_kind: bool = False
    historical_context_complete_yield: int = 0
    current_run_candidate_yield: int = 0
    travel_cost_s: float = 0.0
    route_index: int = 0

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        key = value["coverage_key"]
        if hasattr(key, "__dataclass_fields__"):
            value["coverage_key"] = asdict(key)
        return value


@dataclass(frozen=True)
class AnytimeDiscoveryDecision:
    action: str
    remaining_hard_s: float
    completion_reserve_s: float
    safety_margin_s: float
    available_discovery_budget_s: float
    selected_segment: DiscoverySegment | None = None
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        if self.selected_segment is not None:
            value["selected_segment"] = self.selected_segment.as_dict()
        return value


def rank_discovery_segments(
        segments: Iterable[DiscoverySegment]) -> list[DiscoverySegment]:
    """Apply the evidence-only discovery value order from SCORE_R10."""
    return sorted(segments, key=lambda segment: (
        0 if segment.can_observe_missing_pending_kind else 1,
        -max(0, int(segment.historical_context_complete_yield)),
        -max(0, int(segment.current_run_candidate_yield)),
        max(0.0, float(segment.travel_cost_s)),
        int(segment.route_index),
        str(segment.segment_id),
    ))


class AnytimeDiscoveryPolicy:
    """Reserve one plausible completion before admitting one scan segment."""

    def decide(
            self, *, elapsed_s: float, hard_deadline_s: float,
            pending_orders: Iterable[Any], current_run_coverage: Any,
            discovery_cursor: Any, current_candidates: Iterable[Any],
            candidate_completion_estimate_s: float,
            next_scan_segments: Iterable[DiscoverySegment],
            safety_margin_s: float,
            candidate_attempt_feasible: bool = False,
            candidate_attempt_available: bool = False,
            ) -> AnytimeDiscoveryDecision:
        del current_run_coverage, discovery_cursor  # audit-only inputs
        pending = tuple(pending_orders)
        candidates = tuple(current_candidates)
        remaining = round(max(
            0.0, float(hard_deadline_s) - float(elapsed_s)), 6)
        reserve = max(0.0, float(candidate_completion_estimate_s))
        margin = max(0.0, float(safety_margin_s))
        available = round(remaining - reserve - margin, 6)

        if not pending:
            return AnytimeDiscoveryDecision(
                STOP_NO_FEASIBLE_WORK, remaining, reserve, margin, available,
                reason="no_pending_orders")
        if candidate_attempt_available and candidate_attempt_feasible:
            return AnytimeDiscoveryDecision(
                START_CANDIDATE_ATTEMPT, remaining, reserve, margin, available,
                reason="best_candidate_deadline_feasible")

        ranked = rank_discovery_segments(next_scan_segments)
        feasible = [segment for segment in ranked
                    if math.isfinite(float(segment.estimated_cost_s))
                    and float(segment.estimated_cost_s) <= available]
        if feasible:
            return AnytimeDiscoveryDecision(
                START_DISCOVERY_SEGMENT, remaining, reserve, margin, available,
                selected_segment=feasible[0],
                reason=("candidate_unavailable_segment_fits_reserve"
                        if not candidates else
                        "no_feasible_candidate_segment_fits_reserve"))
        return AnytimeDiscoveryDecision(
            STOP_NO_FEASIBLE_WORK, remaining, reserve, margin, available,
            reason=("no_uncovered_segment" if not ranked
                    else "next_segment_would_consume_completion_reserve"))


def stable_segment_id(run_prefix: str, station_id: int, pose_name: str,
                      shelf_band: str) -> str:
    return (f"{run_prefix}:station-{int(station_id)}:"
            f"{str(pose_name)}:{str(shelf_band)}")
