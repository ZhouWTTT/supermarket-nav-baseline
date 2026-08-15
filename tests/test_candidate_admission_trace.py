import pathlib
import sys


HERE = pathlib.Path(__file__).resolve().parents[1] / "examples" / "supermarket_sorting"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from candidate_admission_trace import (  # noqa: E402
    INVENTORY_SYNC_TOLERANCE_NS,
    CandidateAdmissionTrace,
    first_loss_stage,
    nearest_synchronized_frame,
)


def base_summary(**updates):
    value = {
        "target_task_kind_detection_count": 1,
        "aruco_detection_count": 1,
        "synchronized_frame_pair_count": 1,
        "association_success_count": 1,
        "confirmation_reset_count": 0,
        "candidate_constructed_count": 1,
        "candidate_received_by_runner_count": 1,
        "candidate_inserted_into_inventory_count": 1,
    }
    value.update(updates)
    return value


def test_first_loss_is_earliest_failed_stage():
    assert first_loss_stage(base_summary(
        target_task_kind_detection_count=0)) == "NO_TARGET_DETECTION"
    assert first_loss_stage(base_summary(
        aruco_detection_count=0)) == "NO_ARUCO"
    assert first_loss_stage(base_summary(
        synchronized_frame_pair_count=0)) == "NO_SYNCHRONIZED_PAIR"
    assert first_loss_stage(base_summary(
        association_success_count=0)) == "NO_ASSOCIATION"
    assert first_loss_stage(base_summary(
        confirmation_reset_count=2)) == "CONFIRMATION_RESET"
    assert first_loss_stage(base_summary(
        candidate_constructed_count=0)) == "CANDIDATE_NOT_CONSTRUCTED"
    assert first_loss_stage(base_summary()) == "ADMITTED"


def test_trace_is_bounded_and_primitive_summary():
    trace = CandidateAdmissionTrace("run-a", max_poses=1)
    pose = trace.start_pose(
        attempt_id="a1", station_id=0, pose_name="overview_high",
        shelf_band="top", pending_kinds=["maidong", "heweidao"])
    pose.fresh_rgb_stamps.add(1)
    pose.yolo_detection_count_by_kind["maidong"] += 2
    summary = trace.end_pose("a1")
    assert summary["fresh_rgb_frame_count"] == 1
    assert summary["target_task_kind_detection_count"] == 2
    assert trace.start_pose(
        attempt_id="a2", station_id=1, pose_name="overview_mid",
        shelf_band="middle", pending_kinds=[]) is None


def test_nearest_sync_frame_uses_history_without_relaxing_gate():
    target = 1_000_000_000
    frames = [
        (target - 350_000_000, ["stale"]),
        (target + 125_000_000, ["nearest"]),
        (target + 180_000_000, ["newer"]),
    ]
    assert nearest_synchronized_frame(
        target, frames, tolerance_ns=INVENTORY_SYNC_TOLERANCE_NS
    ) == frames[1]
    assert nearest_synchronized_frame(
        target, [(target + 200_000_001, [])],
        tolerance_ns=INVENTORY_SYNC_TOLERANCE_NS,
    ) is None
