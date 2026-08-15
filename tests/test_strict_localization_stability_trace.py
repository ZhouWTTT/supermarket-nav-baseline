from __future__ import annotations

import importlib.util
import math
import pathlib
import sys

import numpy as np
import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = (ROOT / "examples" / "supermarket_sorting"
               / "strict_localization_stability_trace.py")
SPEC = importlib.util.spec_from_file_location(
    "strict_localization_stability_trace", MODULE_PATH)
trace = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = trace
SPEC.loader.exec_module(trace)


PICK_PATH = (ROOT / "examples" / "supermarket_sorting"
             / "yolo_aruco_shelf_pick.py")


def _load_association_functions_without_ros():
    source = PICK_PATH.read_text(encoding="utf-8")
    tree = __import__("ast").parse(source)
    names = {"association_decision", "marker_below_yolo"}
    selected = [node for node in tree.body
                if isinstance(node, (__import__("ast").FunctionDef,))
                and node.name in names]
    module = __import__("ast").Module(body=selected, type_ignores=[])
    namespace = {
        "AssociationDecision": trace.AssociationDecision,
        "FrozenRecord": trace.FrozenRecord,
        "np": np,
        "math": math,
        "PRODUCT_CENTER_ABOVE_MARKER_M": {"shupian": 0.054},
        "ARUCO_MAX_VERTICAL_GAP_MIN_PX": 65.0,
        "ARUCO_MAX_VERTICAL_GAP_BOX_HEIGHTS": 1.50,
        "ARUCO_MAX_HORIZONTAL_MARGIN_BOX_WIDTHS": 0.35,
        "ARUCO_PRODUCT_LEVEL_TOLERANCE_M": 0.18,
        "DEPTH_TARGET_MARKER_XY_MAX_M": 0.35,
    }
    exec(compile(module, str(PICK_PATH), "exec"), namespace)
    return namespace["association_decision"], namespace["marker_below_yolo"]


def _sample(index=1, *, epoch=1, pose="p1", source="aruco_rgbd", marker=7,
            xyz=(1.0, 2.0, 3.0)):
    identity = np.eye(4)
    return {
        "sample_index": index, "accepted": True, "duplicate": False,
        "rejection_reason": None, "marker_id": marker,
        "scan_epoch_id": epoch, "pose_id": pose,
        "position_source": source, "raw_marker_camera_xyz_m": xyz,
        "marker_world_xyz_m": xyz, "base_pose_used_for_transform": (0, 0, 0),
        "_base_yaw_scalar": 0, "_head_slide_scalar": 0,
        "_head_yaw_scalar": 0, "_head_pitch_scalar": 0,
        "_runtime_camera_world_tmat": identity,
        "_source_time_camera_world_tmat": identity,
    }


def test_per_pair_ledger_is_immutable_and_bounded():
    ledger = trace.BoundedLedger(2)
    source = {"corners": np.array([[1.0, 2.0]])}
    first = ledger.append(source)
    source["corners"][0, 0] = 99.0
    ledger.append({"n": 2})
    ledger.append({"n": 3})
    assert len(ledger) == 2
    assert first.to_dict()["corners"] == [[1.0, 2.0]]
    with pytest.raises(Exception):
        first.fields[0] = ("bad", 1)


def test_per_sample_ledger_is_immutable_bounded_and_detached():
    ledger = trace.StabilityAttemptLedger(sample_maxlen=2)
    point = np.array([1.0, 2.0, 3.0])
    sample = _sample(xyz=point)
    ledger.append_sample(sample)
    point[0] = 9.0
    ledger.append_sample(_sample(2, xyz=(1.1, 2, 3)))
    ledger.append_sample(_sample(3, xyz=(1.2, 2, 3)))
    assert len(ledger.samples) == 2
    assert all(isinstance(record, trace.FrozenRecord)
               for record in ledger.samples.snapshot())


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_nonfinite_data_fails_closed(bad):
    ledger = trace.BoundedLedger(3)
    with pytest.raises(ValueError):
        ledger.append({"value": bad})
    assert trace.transform_xyz(np.eye(4), [bad, 0, 0]) is None


def test_duplicate_pair_does_not_create_new_sample():
    pairs = trace.BoundedLedger(4)
    samples = trace.BoundedLedger(4)
    seen = set()
    for pair in [(1, 2), (1, 2)]:
        pairs.append({"pair": pair})
        if pair not in seen:
            samples.append({"pair": pair})
            seen.add(pair)
    assert len(pairs) == 2
    assert len(samples) == 1


def test_different_pose_and_epoch_do_not_share_samples():
    ledger = trace.StabilityAttemptLedger()
    ledger.append_sample(_sample(1, epoch=1, pose="a", xyz=(0, 0, 0)))
    _, second = ledger.append_sample(
        _sample(1, epoch=2, pose="b", xyz=(5, 5, 5)))
    assert second["accepted_sample_count"] == 1
    assert second["runtime_world_spread_m"] == 0.0


def test_source_time_selection_never_uses_future_state():
    history = [{"source_stamp_ns": 100, "x": 1},
               {"source_stamp_ns": 200, "x": 2}]
    selected, delta = trace.latest_not_after(history, 150)
    assert selected["x"] == 1 and delta == 50


def test_state_delta_over_limit_rejects_without_latest_fallback():
    history = [{"source_stamp_ns": 100, "x": "past"},
               {"source_stamp_ns": 300, "x": "future"}]
    assert trace.latest_not_after(history, 250, max_delta_ns=100) is None


def test_fixed_transform_is_diagnostic_only():
    ledger = trace.StabilityAttemptLedger()
    shifted = np.eye(4)
    shifted[0, 3] = 10.0
    sample = _sample()
    sample["_runtime_camera_world_tmat"] = shifted
    record, _ = ledger.append_sample(sample)
    assert record["marker_world_xyz_m"] == [1.0, 2.0, 3.0]
    assert record["world_xyz_using_fixed_first_sample_transform"][0] == 11.0


def test_no_outlier_removal_or_subset_selection():
    ledger = trace.StabilityAttemptLedger()
    for index, x in enumerate((0.0, 0.01, 0.02, 0.03, 0.50), start=1):
        _, summary = ledger.append_sample(_sample(index, xyz=(x, 0, 0)))
    assert summary["accepted_sample_count"] == 5
    assert summary["runtime_world_spread_m"] == pytest.approx(0.50)


def test_identity_and_source_variation_are_reported_not_hidden():
    ledger = trace.StabilityAttemptLedger()
    ledger.append_sample(_sample(1, marker=7, source="aruco_rgbd"))
    _, summary = ledger.append_sample(
        _sample(2, marker=8, source="aruco_pnp"))
    assert summary["marker_id_unique_values"] == [7, 8]
    assert summary["position_source_unique_values"] == [
        "aruco_pnp", "aruco_rgbd"]


def test_position_source_change_requires_new_identity_epoch():
    guard = trace.IdentityIsolationGuard()
    assert guard.observe(7, "aruco_rgbd") is False
    assert guard.observe(7, "aruco_pnp") is True


def test_marker_id_change_requires_isolation_without_preferred_lock():
    guard = trace.IdentityIsolationGuard()
    assert guard.observe(7, "aruco_rgbd") is False
    assert guard.observe(8, "aruco_rgbd") is True
    assert guard.identity == (8, "aruco_rgbd")


def test_marker_geometry_is_finite_and_detached():
    corners = np.array([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=float)
    geometry = trace.marker_geometry(corners, image_width=10, image_height=10)
    corners[:] = 99
    assert geometry["marker_area_px"] == 4.0
    assert geometry["marker_corner_skew_metric"] == 0.0


def test_association_diagnostic_boolean_matches_unique_wrapper():
    decision_fn, wrapper = _load_association_functions_without_ros()
    detections = [
        {"bbox_xyxy": [10, 10, 30, 50], "class": "shupian"},
        {"bbox_xyxy": [10, 10, 11, 50], "class": "shupian"},
        {"bbox_xyxy": [float("nan"), 10, 30, 50], "class": "shupian"},
    ]
    marker_sets = [
        [{"id": 7, "pixel_center": [20, 55],
          "position_world": [1, 2, 3]}],
        [{"id": 7, "pixel_center": [200, 55],
          "position_world": [1, 2, 3]}],
        [],
    ]
    for detection in detections:
        for markers in marker_sets:
            try:
                decision = decision_fn(detection, markers)
            except ValueError:
                decision = None
            assert (False if decision is None else decision.accepted) == (
                wrapper(detection, markers) is not None)


def test_strict_threshold_literals_remain_frozen():
    source = (ROOT / "examples" / "supermarket_sorting"
              / "yolo_aruco_shelf_pick.py").read_text(encoding="utf-8")
    assert "ARUCO_SYNC_TOLERANCE_NS = 200_000_000" in source
    assert "ASSOCIATION_CONFIRMATIONS_REQUIRED = 3" in source
    assert "MARKER_SAMPLES_REQUIRED = 5" in source
    assert "MARKER_SAMPLE_SPREAD_MAX_M = 0.04" in source
    assert "DEPTH_TARGET_MIN_SAMPLES = 5" in source
    assert "DEPTH_TARGET_SPREAD_MAX_M = 0.04" in source
    assert "np.median(samples, axis=0)" in source
    assert "best 5" not in source.lower()


def test_trace_default_is_off(monkeypatch):
    monkeypatch.delenv("SUPERMARKET_STRICT_STABILITY_TRACE", raising=False)
    assert trace.strict_trace_mode() == trace.TRACE_OFF


def test_trace_off_builds_no_records_and_emits_nothing():
    runtime = trace.StrictTraceRuntime("off")
    sink = []
    runtime.observe_pair()
    runtime.observe_sample(accepted=True, spread_m=0.01)
    assert runtime.emit(lambda *args, **kwargs: sink.append((args, kwargs)),
                        "strict_localization_pair") is False
    assert runtime.ledger is None
    assert runtime.counters() == {
        "strict_trace_pair_records_built": 0,
        "strict_trace_sample_records_built": 0,
        "strict_trace_events_emitted": 0,
    }
    assert sink == []


def test_trace_summary_emits_terminal_only():
    runtime = trace.StrictTraceRuntime("summary")
    events = []
    runtime.observe_pair()
    runtime.observe_sample(accepted=True, spread_m=0.02)
    assert runtime.emit(events.append, "strict_localization_pair") is False
    assert runtime.emit_summary(
        lambda event, **payload: events.append((event, payload)),
        first_failure_stage="association", final_outcome="NO_ASSOCIATION")
    assert [event[0] for event in events] == [
        "strict_localization_trace_summary"]


def test_trace_full_retains_pair_and_sample_ledgers():
    runtime = trace.StrictTraceRuntime("full")
    events = []
    runtime.observe_pair()
    pair = runtime.append_pair({"accepted": True})
    runtime.emit(lambda event, **payload: events.append(event),
                 "strict_localization_pair", **pair)
    runtime.observe_sample(accepted=True, spread_m=0.0)
    sample, _ = runtime.append_sample(_sample())
    runtime.emit(lambda event, **payload: events.append(event),
                 "strict_localization_sample", **sample)
    assert len(runtime.ledger.pairs) == 1
    assert len(runtime.ledger.samples) == 1
    assert runtime.counters() == {
        "strict_trace_pair_records_built": 1,
        "strict_trace_sample_records_built": 1,
        "strict_trace_events_emitted": 2,
    }


def test_trace_mode_does_not_own_or_change_strict_decisions():
    decisions = []
    for mode in ("off", "summary", "full"):
        runtime = trace.StrictTraceRuntime(mode)
        # The production decision is computed before the trace observes it.
        accepted = (3 >= 3 and 5 >= 5 and 0.039 <= 0.04)
        runtime.observe_pair()
        if mode != "off":
            runtime.observe_sample(accepted=accepted, spread_m=0.039)
        decisions.append(accepted)
    assert decisions == [True, True, True]


def test_sink_none_never_falls_back_to_unrelated_telemetry():
    runtime = trace.StrictTraceRuntime("full")
    assert runtime.emit(None, "strict_localization_pair", n=1) is False
    assert runtime.counters()["strict_trace_events_emitted"] == 0


def test_heavy_records_are_built_only_behind_full_mode_guards():
    source = PICK_PATH.read_text(encoding="utf-8")
    pair_helper = source.index("def _strict_trace_record_pair")
    sample_helper = source.index("def _strict_trace_record_sample")
    association = source.index("def try_association_locked")
    assert source.count("pair_record = {") == 1
    assert pair_helper < source.index("pair_record = {") < sample_helper
    assert source.count("sample_record = {") == 1
    assert sample_helper < source.index("sample_record = {") < association
    assert ("if self.strict_trace_runtime.pair_records_enabled:\n"
            "            self._strict_trace_record_pair(") in source
    assert ("if self.strict_trace_runtime.sample_records_enabled:\n"
            "            self._strict_trace_record_sample(") in source


def test_trace_emitter_has_no_telemetry_fallback():
    source = PICK_PATH.read_text(encoding="utf-8")
    start = source.index("def _strict_trace_emit(")
    end = source.index("def strict_trace_summary(", start)
    emitter_source = source[start:end]
    assert "telemetry" not in emitter_source
    assert "strict_trace_sink" in emitter_source
