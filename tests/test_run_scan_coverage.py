import pathlib
import sys


HERE = pathlib.Path(__file__).resolve().parents[1] / "examples" / "supermarket_sorting"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from run_scan_coverage import (  # noqa: E402
    COVERED_VALID,
    PARTIAL,
    CoverageKey,
    RunScanCoverage,
    deadline_action,
    stable_attempt_id,
)


ROUTE = ((0, "overview_high", "top"),
         (0, "overview_mid", "middle"),
         (1, "overview_high", "top"))


def test_new_run_empty_and_new_prefix_isolated():
    first = RunScanCoverage("run-a", ROUTE)
    second = RunScanCoverage("run-b", ROUTE)
    assert first.snapshot()["metrics"]["covered_pose_count"] == 0
    foreign = CoverageKey("run-a", 0, "overview_high", "top")
    assert not second.needs_scan(foreign)


def test_valid_completion_survives_worker_rebuild_and_is_skipped():
    coverage = RunScanCoverage("run-a", ROUTE)
    key = coverage.next_uncovered_key()
    assert coverage.start(key, stamp=1.0, resumed=False)
    assert coverage.complete(
        key, stamp=2.0, fresh_rgb_frame_count=2,
        fresh_aruco_frame_count=3, completion_reason="elapsed",
        camera_settled=True, pose_completed=True)
    rebuilt = RunScanCoverage.from_snapshot(coverage.snapshot(), ROUTE)
    assert rebuilt.records[key].state == COVERED_VALID
    assert not rebuilt.needs_scan(key)
    assert rebuilt.next_uncovered_key() != key


def test_partial_and_no_fresh_frame_are_resumed():
    coverage = RunScanCoverage("run-a", ROUTE)
    key = coverage.next_uncovered_key()
    coverage.start(key, stamp=1.0, resumed=False)
    assert not coverage.complete(
        key, stamp=2.0, fresh_rgb_frame_count=0,
        fresh_aruco_frame_count=10, completion_reason="elapsed",
        camera_settled=True, pose_completed=True)
    assert coverage.records[key].state == PARTIAL
    assert coverage.needs_scan(key)
    assert coverage.next_uncovered_key() == key


def test_cursor_resumes_after_covered_pose():
    coverage = RunScanCoverage("run-a", ROUTE)
    first = coverage.next_uncovered_key()
    coverage.start(first, stamp=1.0, resumed=False)
    coverage.complete(
        first, stamp=2.0, fresh_rgb_frame_count=1,
        fresh_aruco_frame_count=1, completion_reason="elapsed",
        camera_settled=True, pose_completed=True)
    assert coverage.next_uncovered_key().pose_name == "overview_mid"


def test_resume_cursor_never_wraps_to_route_start():
    coverage = RunScanCoverage("run-a", ROUTE)
    coverage.cursor_index = len(coverage.route)
    assert coverage.next_uncovered_key() is None
    assert coverage.uncovered_keys_from_cursor() == ()


def test_segment_records_estimated_and_actual_duration():
    coverage = RunScanCoverage("run-a", ROUTE)
    key = coverage.next_uncovered_key()
    coverage.start(key, stamp=10.0, resumed=True, estimated_duration_s=2.0)
    coverage.complete(
        key, stamp=11.25, fresh_rgb_frame_count=1,
        fresh_aruco_frame_count=1, completion_reason="elapsed",
        camera_settled=True, pose_completed=True)
    assert coverage.records[key].estimated_duration_s == 2.0
    assert coverage.records[key].actual_duration_s == 1.25


def test_later_completion_cannot_skip_an_earlier_partial():
    coverage = RunScanCoverage("run-a", ROUTE)
    first, second = coverage.route[:2]
    coverage.start(first, stamp=1.0, resumed=False)
    coverage.complete(
        first, stamp=2.0, fresh_rgb_frame_count=0,
        fresh_aruco_frame_count=1, completion_reason="no_rgb",
        camera_settled=True, pose_completed=True)
    coverage.start(second, stamp=3.0, resumed=False)
    coverage.complete(
        second, stamp=4.0, fresh_rgb_frame_count=1,
        fresh_aruco_frame_count=1, completion_reason="elapsed",
        camera_settled=True, pose_completed=True)
    assert coverage.cursor_index == 0
    assert coverage.next_uncovered_key() == first


def test_attempt_id_is_stable_and_deadlines_distinguish_inflight():
    assert stable_attempt_id("r", "o", 1) == stable_attempt_id("r", "o", 1)
    assert deadline_action(569.0, "DISCOVERY") == "ALLOW_NEW"
    assert deadline_action(570.0, "DISCOVERY") == "BLOCK_NEW"
    assert deadline_action(570.0, "GRASPED") == "ALLOW_INFLIGHT"
    assert deadline_action(570.0, "NAV_TO_DELIVERY") == "ALLOW_INFLIGHT"
    assert deadline_action(570.0, "PLACING") == "ALLOW_INFLIGHT"
    assert deadline_action(600.0, "PLACING") == "HARD_STOP"


def test_coverage_has_no_motion_or_grasp_authority():
    snapshot = RunScanCoverage("run-a", ROUTE).snapshot()
    forbidden = {"target_world", "marker_id", "grasp_authorized", "motion_goal"}
    assert forbidden.isdisjoint(snapshot)
    assert all(forbidden.isdisjoint(record) for record in snapshot["records"])
