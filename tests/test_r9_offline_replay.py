from pathlib import Path
import sys

MODULE = Path(__file__).resolve().parents[1] / "examples" / "supermarket_sorting"
sys.path.insert(0, str(MODULE))

from anytime_discovery_policy import (  # noqa: E402
    AnytimeDiscoveryPolicy,
    DiscoverySegment,
    START_DISCOVERY_SEGMENT,
)


R9_EVENTS = (
    {"event": "match_start", "monotonic_s": 100.0},
    *(
        {
            "event": "candidate_attempt_end",
            "monotonic_s": 110.0 + index,
            "first_failure_stage": "target_kind_detection",
            "first_failure_reason": "no_target_kind",
            "spread_reject": False,
            "sample_reject": False,
            "validated_marker_id": None,
        }
        for index in range(5)
    ),
    {"event": "match_end", "monotonic_s": 335.073},
)


def events():
    return [dict(item) for item in R9_EVENTS]


def test_r9_real_chain_admits_a_bounded_segment_at_stop():
    chain = events()
    start = next(item for item in chain if item["event"] == "match_start")
    stop = next(item for item in reversed(chain) if item["event"] == "match_end")
    elapsed = stop["monotonic_s"] - start["monotonic_s"]
    decision = AnytimeDiscoveryPolicy().decide(
        elapsed_s=elapsed, hard_deadline_s=600.0,
        pending_orders=("sanmingzhi", "heweidao", "shupian", "zhijin", "maidong"),
        current_run_coverage={}, discovery_cursor=1, current_candidates=(),
        candidate_completion_estimate_s=233.0,
        next_scan_segments=(DiscoverySegment("r9-next", "r9-next", 2.0),),
        safety_margin_s=6.0)
    assert round(elapsed, 3) == 235.073
    assert round(decision.remaining_hard_s, 3) == 364.927
    assert round(decision.available_discovery_budget_s, 3) == 125.927
    assert decision.action == START_DISCOVERY_SEGMENT


def test_r9_failure_funnel_is_target_reacquisition_not_spread_or_samples():
    attempts = [item for item in events()
                if item["event"] == "candidate_attempt_end"]
    assert len(attempts) == 5
    assert {item["first_failure_stage"] for item in attempts} == {
        "target_kind_detection"}
    assert {item["first_failure_reason"] for item in attempts} == {
        "no_target_kind"}
    assert sum(item["spread_reject"] for item in attempts) == 0
    assert sum(item["sample_reject"] for item in attempts) == 0
    assert sum(item["validated_marker_id"] is not None for item in attempts) == 0
