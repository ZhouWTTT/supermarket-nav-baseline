from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples" / "supermarket_sorting"))

from anytime_discovery_policy import (  # noqa: E402
    AnytimeDiscoveryPolicy,
    DiscoverySegment,
    START_CANDIDATE_ATTEMPT,
    START_DISCOVERY_SEGMENT,
    STOP_NO_FEASIBLE_WORK,
    rank_discovery_segments,
)


def decide(segment_cost, **updates):
    values = dict(
        elapsed_s=235.073, hard_deadline_s=600.0,
        pending_orders=("a",), current_run_coverage={}, discovery_cursor=3,
        current_candidates=(), candidate_completion_estimate_s=233.0,
        next_scan_segments=(DiscoverySegment("s", "key", segment_cost),),
        safety_margin_s=6.0, candidate_attempt_available=False,
        candidate_attempt_feasible=False,
    )
    values.update(updates)
    return AnytimeDiscoveryPolicy().decide(**values)


def test_full_discovery_is_irrelevant_when_one_segment_fits():
    result = decide(12.0)
    assert result.action == START_DISCOVERY_SEGMENT
    assert result.available_discovery_budget_s == 125.927


def test_segment_blocked_when_it_would_consume_completion_reserve():
    assert decide(126.0).action == STOP_NO_FEASIBLE_WORK


def test_candidate_pauses_discovery_and_is_reassessed_first():
    result = decide(
        12.0, current_candidates=({"candidate_id": "c"},),
        candidate_attempt_available=True, candidate_attempt_feasible=True)
    assert result.action == START_CANDIDATE_ATTEMPT
    assert result.selected_segment is None


def test_candidate_negative_slack_does_not_bypass_reservation():
    result = decide(
        12.0, elapsed_s=590.0,
        current_candidates=({"candidate_id": "c"},),
        candidate_attempt_available=True, candidate_attempt_feasible=False)
    assert result.action == STOP_NO_FEASIBLE_WORK


def test_discovery_reserve_is_separate_from_candidate_attempt_estimate():
    result = decide(12.0)
    assert result.completion_reserve_s == 233.0
    assert result.safety_margin_s == 6.0
    assert result.available_discovery_budget_s == round(
        result.remaining_hard_s - 233.0 - 6.0, 6)


def test_value_order_and_tie_break_are_stable():
    segments = [
        DiscoverySegment("z", "z", 10, route_index=0),
        DiscoverySegment("b", "b", 10, True, 0, 0, 2, 2),
        DiscoverySegment("a", "a", 10, True, 0, 0, 2, 1),
        DiscoverySegment("yield", "y", 10, True, 1, 0, 9, 4),
    ]
    assert [item.segment_id for item in rank_discovery_segments(segments)] == [
        "yield", "a", "b", "z"]
