from __future__ import annotations

from dataclasses import FrozenInstanceError
import math
import pathlib
import sys

import numpy as np
import pytest


HERE = pathlib.Path(__file__).resolve().parents[1] / "examples" / "supermarket_sorting"
sys.path.insert(0, str(HERE))

from source_state_history import (  # noqa: E402
    COMPLETE,
    PARTIAL,
    UNAVAILABLE,
    BoundedSourceStateHistory,
    build_candidate_source_state_evidence,
)


def add_complete(history, stamp, *, x=1.0, slide=0.1):
    history.append_odom(source_stamp_ns=stamp,
                        callback_receipt_monotonic_ns=stamp + 10,
                        x=x, y=2.0, yaw=0.3)
    history.append_joint(source_stamp_ns=stamp,
                         callback_receipt_monotonic_ns=stamp + 11,
                         slide=slide, head_yaw=0.0, head_pitch=-0.2)


def evidence(history, yolo=150, aruco=150):
    return build_candidate_source_state_evidence(
        history=history, run_prefix="run", candidate_id="candidate-7",
        kind="maidong", marker_id=7, confirmation_count=3,
        yolo_source_stamp_ns=yolo, aruco_source_stamp_ns=aruco,
        callback_latest_base_pose=(1.1, 2.0, 0.31),
        callback_latest_head_pose=(0.11, 0.01, -0.19))


def test_histories_are_bounded_and_detached_from_numpy_inputs():
    history = BoundedSourceStateHistory(odom_maxlen=2, joint_maxlen=2)
    source = np.array([1.0, 2.0, 0.3])
    for stamp in (10, 20, 30):
        history.append_odom(source_stamp_ns=stamp,
                            callback_receipt_monotonic_ns=stamp,
                            x=source[0], y=source[1], yaw=source[2])
        history.append_joint(source_stamp_ns=stamp,
                             callback_receipt_monotonic_ns=stamp,
                             slide=0.1, head_yaw=0.0, head_pitch=-0.2)
    source[:] = 99.0
    assert len(history.odom_samples) == len(history.joint_samples) == 2
    assert history.odom_samples[-1].pose == (1.0, 2.0, 0.3)


def test_lookup_is_past_only_and_never_uses_future():
    history = BoundedSourceStateHistory()
    add_complete(history, 100, x=1.0)
    add_complete(history, 200, x=2.0)
    selected, delta = history.odom_at(150)
    assert selected.x == 1.0 and delta == 50


def test_lookup_over_500ms_is_unavailable():
    history = BoundedSourceStateHistory()
    add_complete(history, 100)
    assert history.odom_at(500_000_101) is None


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_nonfinite_state_is_rejected(bad):
    history = BoundedSourceStateHistory()
    with pytest.raises(ValueError):
        history.append_odom(source_stamp_ns=1,
                            callback_receipt_monotonic_ns=2,
                            x=bad, y=0, yaw=0)


def test_stamp_regression_is_recorded_without_dropping_latest_update():
    history = BoundedSourceStateHistory()
    add_complete(history, 200)
    add_complete(history, 100)
    assert history.odom_stamp_regression_count == 1
    assert history.joint_stamp_regression_count == 1
    assert history.odom_samples[-1].source_stamp_ns == 100


def test_complete_evidence_is_immutable_and_has_past_source_deltas():
    history = BoundedSourceStateHistory()
    add_complete(history, 100)
    item = evidence(history)
    assert item.as_dict()["source_context_availability"] == COMPLETE
    assert item.as_dict()["odom_lookup_delta_ns"] == {"yolo": 50, "aruco": 50}
    with pytest.raises(FrozenInstanceError):
        item.fields = ()


def test_partial_and_unavailable_fail_closed():
    partial = BoundedSourceStateHistory()
    partial.append_odom(source_stamp_ns=100,
                        callback_receipt_monotonic_ns=101,
                        x=1, y=2, yaw=0.3)
    assert evidence(partial).as_dict()["source_context_availability"] == PARTIAL
    assert evidence(BoundedSourceStateHistory()).as_dict()[
        "source_context_availability"] == UNAVAILABLE


def test_evidence_contains_no_control_authority_or_target_world():
    history = BoundedSourceStateHistory()
    add_complete(history, 100)
    saved = evidence(history).as_dict()
    assert "target_world" not in saved
    assert "motion_authority" not in saved
    assert "grasp_authority" not in saved


def test_missing_callback_pose_does_not_claim_source_delta_or_crash():
    history = BoundedSourceStateHistory()
    add_complete(history, 100)
    item = build_candidate_source_state_evidence(
        history=history, run_prefix="run", candidate_id="candidate-7",
        kind="maidong", marker_id=7, confirmation_count=3,
        yolo_source_stamp_ns=150, aruco_source_stamp_ns=150,
        callback_latest_base_pose=None, callback_latest_head_pose=None,
    ).as_dict()
    assert item["source_context_availability"] == COMPLETE
    assert item["callback_yolo_pose_delta"] is None


def test_outcome_join_vocabulary():
    from source_state_history import classify_candidate_outcome
    assert classify_candidate_outcome({
        "validated_marker_id": 7}) == "LOCALIZATION_VALIDATED"
    assert classify_candidate_outcome({
        "candidate_first_failure_reason": "no_target_kind"
    }) == "NO_TARGET_KIND"
    assert classify_candidate_outcome({
        "candidate_first_failure_reason": "no_association"
    }) == "NO_ASSOCIATION"
