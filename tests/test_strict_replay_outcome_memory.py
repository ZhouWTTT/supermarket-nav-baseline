from __future__ import annotations

import copy
import pathlib
import sys

import pytest


MODULE_DIR = pathlib.Path(__file__).parents[1] / "examples" / "supermarket_sorting"
sys.path.insert(0, str(MODULE_DIR))

from strict_replay_outcome_memory import (  # noqa: E402
    BACKUP_AVAILABLE,
    EXHAUSTED_UNTIL_MATERIAL_CHANGE,
    LOCALIZED,
    PREGRASP_FAILED,
    REACTIVATED,
    RUNTIME_INVALID,
    SPREAD_RECOVERY_AVAILABLE,
    UNTRIED,
    NO_ASSOCIATION,
    SPREAD_REJECT,
    StrictReplayOutcomeMemory,
    make_equivalence_key,
    strict_state_priority,
)
from score_first_order_policy import OrderOption, select_order  # noqa: E402


def candidate(**updates):
    value = {
        "candidate_id": "candidate-41",
        "kind": "maidong",
        "marker_id": 41,
        "source_yolo_stamp_ns": 100,
        "source_aruco_stamp_ns": 100,
        "confidence": 0.9727,
        "confirmations": 3,
        "observed_base_pose": [1.7999, 2.4101, 1.5692],
        "observed_head_pose": ["overview_mid", 0.1161, -0.0007, -0.4425],
        "head_pose_hint": ["overview_mid", 0.1161, -0.0007, -0.4425],
        "observed_pose_name": "overview_mid",
        "observed_scan_station": {"index": 0},
        "context_quality": "CONTEXT_COMPLETE",
        "target_bbox_summary": [483.0, 141.0, 615.0, 325.0],
        "marker_pixel_summary": [513.5, 352.0],
        "association_summary": {
            "bbox_center": [549.0, 233.0],
            "marker_pixel": [513.5, 352.0],
            "bottom_center_distance_px": 44.601,
        },
    }
    value.update(updates)
    return value


def key(value=None, *, run="run-a", marker=41):
    value = candidate() if value is None else value
    return make_equivalence_key(
        run_prefix=run, kind=value.get("kind", "maidong"),
        marker_id=marker, candidate=value)


def test_first_spread_reject_allows_exactly_one_recovery():
    memory = StrictReplayOutcomeMemory("run-a")
    decision = memory.record_spread_reject(key(), attempt_id="a1")
    assert decision.allowed is True
    assert decision.state == SPREAD_RECOVERY_AVAILABLE
    assert decision.recovery_slot == "spread_recovery_1_of_1"


def test_stage_d_first_spread_then_recovery_localizes():
    memory = StrictReplayOutcomeMemory("run-a")
    material = key()
    memory.record_spread_reject(material, attempt_id="a1")
    decision = memory.record_localized(material, attempt_id="a2")
    assert decision.state == LOCALIZED
    assert decision.allowed is True


def test_second_equivalent_spread_reject_exhausts():
    memory = StrictReplayOutcomeMemory("run-a")
    material = key()
    memory.record_spread_reject(material, attempt_id="a1")
    decision = memory.record_spread_reject(material, attempt_id="a2")
    assert decision.state == EXHAUSTED_UNTIL_MATERIAL_CHANGE
    assert decision.allowed is False


def test_primary_no_association_allows_untried_backup():
    memory = StrictReplayOutcomeMemory("run-a")
    material = key()
    decision = memory.record_no_association(
        material, attempt_id="a1", pose_ids=["0:0:overview_mid"])
    assert decision.state == BACKUP_AVAILABLE
    assert decision.allowed is True


def test_primary_and_defined_backup_no_association_exhausts():
    memory = StrictReplayOutcomeMemory("run-a")
    material = key()
    decision = memory.record_no_association(
        material, attempt_id="a1",
        pose_ids=["0:0:overview_mid", "0:1:overview_high"])
    assert decision.state == EXHAUSTED_UNTIL_MATERIAL_CHANGE
    assert decision.allowed is False


@pytest.mark.parametrize("field,new_value", [
    ("source_yolo_stamp_ns", 900),
    ("source_aruco_stamp_ns", 901),
    ("confirmations", 67),
    ("confidence", 0.9782),
])
def test_churn_fields_do_not_change_material(field, new_value):
    first = candidate()
    changed = copy.deepcopy(first)
    changed[field] = new_value
    assert key(first) == key(changed)


def test_attempt_number_is_not_part_of_material_key():
    first = candidate(attempt_number=1)
    second = candidate(attempt_number=99)
    assert key(first) == key(second)


def test_new_geometry_is_material_change_and_reactivates():
    memory = StrictReplayOutcomeMemory("run-a")
    old = key()
    memory.record_no_association(
        old, attempt_id="a1", pose_ids=["overview_mid", "overview_high"])
    changed = candidate(target_bbox_summary=[480.0, 141.0, 615.0, 325.0])
    decision = memory.decision(key(changed))
    assert decision.state == REACTIVATED
    assert decision.allowed is True
    assert "geometry_changed" in decision.reactivation_reason


def test_complete_context_upgrade_reactivates():
    memory = StrictReplayOutcomeMemory("run-a")
    old = key(candidate(context_quality="DERIVED"))
    memory.record_no_association(
        old, attempt_id="a1", pose_ids=["overview_mid", "overview_high"])
    decision = memory.decision(key(candidate(context_quality="CONTEXT_COMPLETE")))
    assert decision.state == REACTIVATED
    assert "context_quality_changed" in decision.reactivation_reason


def test_new_marker_has_independent_untried_state():
    memory = StrictReplayOutcomeMemory("run-a")
    old = key()
    memory.record_no_association(
        old, attempt_id="a1", pose_ids=["overview_mid", "overview_high"])
    new_candidate = candidate(marker_id=42, candidate_id="candidate-42")
    assert memory.decision(key(new_candidate, marker=42)).state == UNTRIED


def test_memory_does_not_cross_run_prefix():
    memory = StrictReplayOutcomeMemory("run-a")
    with pytest.raises(ValueError):
        memory.decision(key(run="run-b"))


def test_sensor_invalid_does_not_penalize_candidate():
    memory = StrictReplayOutcomeMemory("run-a")
    decision = memory.record_runtime_invalid(key(), attempt_id="a1")
    assert decision.state == RUNTIME_INVALID
    assert decision.allowed is True


def test_pregrasp_timeout_does_not_strictly_exhaust_candidate():
    memory = StrictReplayOutcomeMemory("run-a")
    material = key()
    memory.record_localized(material, attempt_id="a1")
    decision = memory.record_pregrasp_failed(material, attempt_id="a1")
    assert decision.state == PREGRASP_FAILED
    assert decision.allowed is True
    assert memory.last_outcome(material) == "PREGRASP_ARM_CONVERGENCE_TIMEOUT"


def test_attempt_count_never_creates_failed_hard():
    memory = StrictReplayOutcomeMemory("run-a")
    material = key(candidate(attempt_number=10_000))
    assert memory.decision(material).state == UNTRIED
    assert "FAILED_HARD" not in {
        UNTRIED, BACKUP_AVAILABLE, SPREAD_RECOVERY_AVAILABLE,
        EXHAUSTED_UNTIL_MATERIAL_CHANGE,
    }


def test_untried_priority_is_ahead_of_exhausted():
    assert strict_state_priority(UNTRIED) < strict_state_priority(
        EXHAUSTED_UNTIL_MATERIAL_CHANGE)
    common = dict(
        source_index=0, attempts=0, candidate_state="READY_PROVISIONAL",
        candidate_count=1, marker_id=41, cost_components={},
        estimated_completion_s=10.0, deadline_feasible=True,
        strict_control_active=True)
    exhausted = OrderOption(
        order_id="a", strict_memory_state=EXHAUSTED_UNTIL_MATERIAL_CHANGE,
        **common)
    untried = OrderOption(
        order_id="z", strict_memory_state=UNTRIED, **common)
    assert select_order([exhausted, untried]) == untried


def test_scheduler_priority_is_kind_agnostic_and_deterministic():
    states = [UNTRIED, REACTIVATED, SPREAD_RECOVERY_AVAILABLE,
              BACKUP_AVAILABLE, EXHAUSTED_UNTIL_MATERIAL_CHANGE]
    assert [strict_state_priority(item) for item in states] == [0, 1, 2, 3, 99]
    assert key(candidate(kind="alpha")).as_dict().keys() == key(
        candidate(kind="omega")).as_dict().keys()
    common = dict(
        source_index=0, attempts=0, candidate_state="READY_PROVISIONAL",
        candidate_count=1, marker_id=41, cost_components={},
        estimated_completion_s=10.0, deadline_feasible=True,
        strict_control_active=True, strict_memory_state=UNTRIED)
    first = OrderOption(order_id="first", fingerprint_digest="aaa", **common)
    second = OrderOption(order_id="second", fingerprint_digest="bbb", **common)
    assert select_order([second, first]) == first


def test_new_backup_pose_is_material_change():
    memory = StrictReplayOutcomeMemory("run-a")
    old_candidate = candidate(replay_backup_pose_ids=[])
    old = key(old_candidate)
    memory.record_no_association(old, attempt_id="a1", pose_ids=["overview_mid"])
    # With no defined backup the primary result is exhausted.
    assert memory.state(old) == EXHAUSTED_UNTIL_MATERIAL_CHANGE
    changed = candidate(replay_backup_pose_ids=["overview_high"])
    decision = memory.decision(key(changed))
    assert decision.state == REACTIVATED
    assert "new_backup_pose" in decision.reactivation_reason


def test_stage_e_source_stamp_family_remains_exhausted():
    memory = StrictReplayOutcomeMemory("run-a")
    material = key()
    memory.record_no_association(
        material, attempt_id="a1", pose_ids=["overview_mid", "overview_high"])
    for attempt, stamp in enumerate((200, 300, 400, 500), start=2):
        churned = candidate(
            source_yolo_stamp_ns=stamp, source_aruco_stamp_ns=stamp,
            confirmations=attempt * 3, confidence=0.97 + attempt / 10000)
        assert memory.decision(key(churned)).allowed is False


def test_strict_threshold_literals_remain_frozen_in_protected_source():
    protected = pathlib.Path(__file__).parents[1] / "examples" / "supermarket_sorting" / "yolo_aruco_shelf_pick.py"
    if not protected.exists():
        pytest.skip("protected source is outside the isolated staging tree")
    source = protected.read_text(encoding="utf-8")
    for literal in (
        "ARUCO_SYNC_TOLERANCE_NS = 200_000_000",
        "ASSOCIATION_CONFIRMATIONS_REQUIRED = 3",
        "MARKER_SAMPLES_REQUIRED = 5",
        "MARKER_SAMPLE_SPREAD_MAX_M = 0.04",
        "DEPTH_TARGET_MIN_SAMPLES = 5",
        "DEPTH_TARGET_SPREAD_MAX_M = 0.04",
    ):
        assert literal in source
