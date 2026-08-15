import pathlib
import sys


HERE = pathlib.Path(__file__).resolve().parents[1] / "examples" / "supermarket_sorting"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from candidate_observation_context import (  # noqa: E402
    CONTEXT_COMPLETE,
    CONTEXT_INCOMPLETE,
    DERIVED_CONTEXT,
    OBSERVED_CONTEXT,
    make_observed_context,
    normalize_kind,
)


POSES = (
    ("overview_high", 0.32, 0.0, 0.45),
    ("overview_down", 0.34, 0.0, 0.73),
)


def test_observed_context_copies_exact_primitives_and_geometry():
    bbox = [10, 20, 50, 80]
    pixel = [31, 86]
    context = make_observed_context(
        base_pose=[1.801, 2.412, 1.569],
        head_pose=[0.321, 0.002, 0.451],
        camera_poses=POSES,
        station_index=0,
        station_x=1.8,
        station_y=2.41,
        yolo_stamp=101,
        aruco_stamp=102,
        detection={"bbox_xyxy": bbox},
        marker={"pixel_center": pixel},
        controller_state="scan",
        scan_index=0,
        pitch_index=0,
        camera_settled=True,
    )
    bbox[0] = 999
    pixel[0] = 999
    saved = context.as_dict()
    assert saved["context_type"] == OBSERVED_CONTEXT
    assert saved["context_quality"] == CONTEXT_COMPLETE
    assert saved["observed_pose_name"] == "overview_high"
    assert saved["observed_head_pose"] == [
        "overview_high", 0.321, 0.002, 0.451]
    assert saved["target_bbox_summary"] == [10.0, 20.0, 50.0, 80.0]
    assert saved["marker_pixel_summary"] == [31.0, 86.0]


def test_missing_actual_pose_is_never_labeled_observed():
    context = make_observed_context(
        base_pose=None,
        head_pose=None,
        camera_poses=POSES,
        station_index=None,
        station_x=None,
        station_y=None,
        yolo_stamp=101,
        aruco_stamp=102,
        detection={"bbox_xyxy": [10, 20, 50, 80]},
        marker={"pixel_center": [31, 86]},
    )
    assert context.context_type == DERIVED_CONTEXT
    assert context.context_quality == CONTEXT_INCOMPLETE
    assert context.context_source == "DERIVED"


def test_kind_normalization_changes_spelling_only():
    assert normalize_kind(" MaiDong ") == "maidong"
    assert normalize_kind("maidong-zero") == "maidong-zero"
