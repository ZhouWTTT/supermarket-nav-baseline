import pathlib
import sys


MODULE_DIR = (pathlib.Path(__file__).resolve().parents[1]
              / "examples" / "supermarket_sorting")
sys.path.insert(0, str(MODULE_DIR))

from replay_viewpoint_convergence import (  # noqa: E402
    ReplayViewpointConvergenceController,
    select_replay_head_poses,
)


def controller():
    return ReplayViewpointConvergenceController(
        base_position_tolerance_m=0.055,
        base_yaw_tolerance_rad=0.035,
        slide_tolerance_m=0.015,
        head_tolerance_rad=0.030,
        stable_window_s=0.15)


def test_base_not_reached_blocks_strict_scan():
    item = controller()
    item.start_pose(pose_id="observed:0", now_s=1.0)
    result = item.observe(
        now_s=2.0, base_position_error_m=0.064,
        base_yaw_error_rad=0.01, head_error=(0.0, 0.0, 0.0))
    assert not result.base_target_reached
    assert not result.strict_scan_allowed


def test_head_not_reached_blocks_strict_scan():
    item = controller()
    item.start_pose(pose_id="observed:0", now_s=1.0)
    result = item.observe(
        now_s=2.0, base_position_error_m=0.01,
        base_yaw_error_rad=0.01, head_error=(0.0, 0.031, 0.0))
    assert result.base_target_reached
    assert not result.head_target_reached
    assert not result.strict_scan_allowed


def test_settle_restarts_after_both_targets_are_reached():
    item = controller()
    item.start_pose(pose_id="observed:0", now_s=1.0)
    item.observe(now_s=1.10, base_position_error_m=0.06,
                 base_yaw_error_rad=0.0, head_error=(0.0, 0.0, 0.0))
    item.observe(now_s=1.20, base_position_error_m=0.01,
                 base_yaw_error_rad=0.0, head_error=(0.0, 0.0, 0.0))
    assert not item.observe(
        now_s=1.34, base_position_error_m=0.01,
        base_yaw_error_rad=0.0, head_error=(0.0, 0.0, 0.0)
    ).strict_scan_allowed
    assert item.observe(
        now_s=1.35, base_position_error_m=0.01,
        base_yaw_error_rad=0.0, head_error=(0.0, 0.0, 0.0)
    ).strict_scan_allowed


def test_only_post_convergence_source_stamp_is_fresh():
    item = controller()
    item.start_pose(pose_id="observed:0", now_s=1.0)
    item.observe(now_s=1.0, base_position_error_m=0.0,
                 base_yaw_error_rad=0.0, head_error=(0.0, 0.0, 0.0))
    item.observe(now_s=1.2, base_position_error_m=0.0,
                 base_yaw_error_rad=0.0, head_error=(0.0, 0.0, 0.0))
    item.set_source_stamp_boundary(17, 19)
    assert not item.frame_is_fresh(19)
    assert item.frame_is_fresh(20)


def test_pose_change_creates_epoch_and_resets_convergence():
    item = controller()
    assert item.start_pose(pose_id="observed:0", now_s=1.0) == 1
    item.observe(now_s=1.0, base_position_error_m=0.0,
                 base_yaw_error_rad=0.0, head_error=(0.0, 0.0, 0.0))
    item.observe(now_s=1.2, base_position_error_m=0.0,
                 base_yaw_error_rad=0.0, head_error=(0.0, 0.0, 0.0))
    assert item.start_pose(pose_id="backup:1", now_s=2.0) == 2
    result = item.snapshot()
    assert result.pose_id == "backup:1"
    assert not result.strict_scan_allowed


def test_candidate_budget_exhaustion_remains_fail_closed():
    item = controller()
    item.start_pose(pose_id="observed:0", now_s=0.0)
    result = item.observe(
        now_s=45.0, base_position_error_m=0.056,
        base_yaw_error_rad=0.0, head_error=(0.0, 0.0, 0.0))
    assert not result.strict_scan_allowed


def test_exact_observed_context_precedes_named_pose_default():
    poses = (
        ("overview_high", 0.11, 0.0, -0.20),
        ("overview_mid", 0.11, 0.0, -0.45),
        ("overview_down", 0.11, 0.0, -0.65),
        ("lower_center", 0.45, 0.0, -0.45),
        ("lower_yaw_minus", 0.45, -0.15, -0.45),
        ("lower_yaw_plus", 0.45, 0.15, -0.45),
    )
    observed = ["overview_mid", 0.1192, -0.0031, -0.4414]
    result = select_replay_head_poses({
        "context_type": "OBSERVED_CONTEXT",
        "head_pose_hint": observed,
        "provisional_marker_world": [1.5, 3.1, 0.84],
    }, poses, top_shelf_z_m=1.0, middle_shelf_z_min_m=0.5)
    assert result[0] == tuple(observed)
    assert result[0] != poses[1]
    assert result[0][0] == "overview_mid"
    assert result[1][0] == "overview_high"
