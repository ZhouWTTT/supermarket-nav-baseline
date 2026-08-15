#!/usr/bin/env python3
"""Small append-only telemetry helpers for five-order score experiments.

The module deliberately has no ROS, NumPy, or simulator dependency so the
runner and its isolated worker processes can share one JSONL event stream.
All durations are derived from ``time.monotonic()`` values.
"""

from __future__ import annotations

import csv
import argparse
import json
import math
from pathlib import Path
import threading
import time
from typing import Any, Iterable


SCHEMA_VERSION = 1


class EventLog:
    """Append compact canonical events with process-safe append semantics."""

    def __init__(self, path: str | Path | None, **context: Any):
        self.path = None if path is None else Path(path)
        self.context = {key: value for key, value in context.items()
                        if value is not None}
        self._lock = threading.Lock()

    def emit(self, event: str, **payload: Any) -> dict[str, Any]:
        record = {
            "schema_version": SCHEMA_VERSION,
            "event": str(event),
            "monotonic_s": round(time.monotonic(), 6),
            **self.context,
            **payload,
        }
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            encoded = json.dumps(
                record, ensure_ascii=False, separators=(",", ":")) + "\n"
            # Each event is emitted by one write on an O_APPEND descriptor.
            # The competition runner supervises only one worker at a time.
            with self._lock:
                with self.path.open("a", encoding="utf-8") as stream:
                    stream.write(encoded)
                    stream.flush()
        return record


def read_events(path: str | Path | None) -> list[dict[str, Any]]:
    if path is None or not Path(path).is_file():
        return []
    events = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and isinstance(value.get("event"), str):
            events.append(value)
    return sorted(events, key=lambda item: float(item.get("monotonic_s", 0.0)))


def _durations(
        events: Iterable[dict[str, Any]], start_name: str,
        end_name: str) -> list[float]:
    events = list(events)
    active: dict[str, list[float]] = {}
    result: list[float] = []
    for event in events:
        name = event.get("event")
        order_id = str(event.get("order_id", "match"))
        stamp = float(event.get("monotonic_s", 0.0))
        if name == start_name:
            active.setdefault(order_id, []).append(stamp)
        elif name == end_name and active.get(order_id):
            start = active[order_id].pop(0)
            result.append(max(0.0, stamp - start))
        elif (name in {"order_completed", "order_failed"}
              and active.get(order_id)):
            result.extend(max(0.0, stamp - start)
                          for start in active.pop(order_id))
    match_ends = [float(event.get("monotonic_s", 0.0))
                  for event in events if event.get("event") == "match_end"]
    if match_ends:
        final_stamp = match_ends[-1]
        for starts in active.values():
            result.extend(max(0.0, final_stamp - start) for start in starts)
    return result


def _order_durations(events: list[dict[str, Any]]) -> list[float]:
    selected: dict[str, float] = {}
    result: list[float] = []
    match_end = next((
        float(event["monotonic_s"]) for event in reversed(events)
        if event.get("event") == "match_end"), None)
    for event in events:
        order_id = event.get("order_id")
        if not isinstance(order_id, str):
            continue
        stamp = float(event.get("monotonic_s", 0.0))
        if event.get("event") == "order_selected":
            selected.setdefault(order_id, stamp)
        elif (event.get("event") in {"order_completed", "order_failed"}
              and order_id in selected):
            result.append(max(0.0, stamp - selected.pop(order_id)))
    if match_end is not None:
        result.extend(max(0.0, match_end - stamp)
                      for stamp in selected.values())
    return result


def build_summary(
        events: list[dict[str, Any]], legacy: dict[str, Any], *, mode: str,
        product_seed: str | int | None, obstacle_seed: str | int | None,
        task_kinds: list[str], inventory: list[dict[str, Any]] | None = None,
        scan_coverage: list[int] | None = None) -> dict[str, Any]:
    """Build the score-oriented summary without inventing official points."""
    starts = [float(item["monotonic_s"]) for item in events
              if item.get("event") == "match_start"]
    ends = [float(item["monotonic_s"]) for item in events
            if item.get("event") == "match_end"]
    match_elapsed = max(0.0, ends[-1] - starts[0]) if starts and ends else None

    def total(start: str, end: str) -> float:
        return round(sum(_durations(events, start, end)), 3)

    def count(name: str) -> int:
        return sum(item.get("event") == name for item in events)

    def time_to_first(name: str) -> float | None:
        if not starts:
            return None
        stamps = [float(item.get("monotonic_s", 0.0)) for item in events
                  if item.get("event") == name]
        return None if not stamps else round(max(0.0, min(stamps) - starts[0]), 3)

    idle_durations = [
        float(item.get("duration_s", 0.0)) for item in events
        if item.get("event") == "idle_gap"
        and math.isfinite(float(item.get("duration_s", 0.0)))
    ]
    delivered = int(legacy.get("delivered", 0))
    candidate_attempt_count = count("candidate_attempt_start")
    candidate_success_count = count("localization_validated")
    candidate_revisit_durations = _durations(
        events, "candidate_attempt_start", "candidate_attempt_end")
    delivery_stamps = [
        float(item.get("monotonic_s", 0.0)) for item in events
        if item.get("event") == "order_completed"
    ]
    first_delivery_stamp = min(delivery_stamps) if delivery_stamps else None
    post_delivery_continuation_count = sum(
        item.get("event") == "attempt_started"
        and first_delivery_stamp is not None
        and float(item.get("monotonic_s", 0.0)) > first_delivery_stamp
        for item in events)
    successful_grasps = [
        float(item.get("monotonic_s", 0.0)) for item in events
        if item.get("event") == "grasp_end" and item.get("success") is True
    ]
    candidate_created = {
        str(item.get("candidate_id")): float(item.get("monotonic_s", 0.0))
        for item in events if item.get("event") == "candidate_created"
        and item.get("candidate_id") is not None
    }
    successful_candidate_id = None
    for item in events:
        if item.get("event") == "candidate_attempt_outcome" and item.get(
                "delivered") is True:
            successful_candidate_id = (item.get("fingerprint") or {}).get(
                "candidate_id")
            break
    successful_candidate_selected = [
        float(item.get("monotonic_s", 0.0)) for item in events
        if item.get("event") == "candidate_attempt_fingerprint"
        and (item.get("fingerprint") or {}).get("candidate_id")
        == successful_candidate_id
    ]
    successful_candidate_created_s = (
        None if not starts or successful_candidate_id not in candidate_created
        else round(candidate_created[successful_candidate_id] - starts[0], 3))
    successful_candidate_selected_s = (
        None if not starts or not successful_candidate_selected else round(
            min(successful_candidate_selected) - starts[0], 3))
    strict_failure_memory_mode = next((
        str(item.get("strict_failure_memory_mode")) for item in events
        if item.get("event") == "match_start"
        and item.get("strict_failure_memory_mode") is not None), "off")
    result = dict(legacy)
    result.update({
        "telemetry_schema_version": SCHEMA_VERSION,
        "mode": mode,
        "strict_failure_memory_mode": strict_failure_memory_mode,
        "product_seed": product_seed,
        "obstacle_seed": obstacle_seed,
        "task_kinds": list(task_kinds),
        "delivered_count": delivered,
        "match_elapsed_s": None if match_elapsed is None else round(match_elapsed, 3),
        "actual_match_elapsed_s": (
            None if match_elapsed is None else round(match_elapsed, 3)),
        "order_elapsed_s": [round(value, 3) for value in _order_durations(events)],
        "search_s_total": total("search_start", "search_end"),
        "full_scan_s_total": total("full_scan_start", "full_scan_end"),
        "discovery_scan_s": total("full_scan_start", "full_scan_end"),
        "candidate_revisit_s": round(sum(candidate_revisit_durations), 3),
        "strict_localization_s": total(
            "strict_localization_start", "strict_localization_end"),
        "fallback_rescan_s": total(
            "fallback_rescan_start", "fallback_rescan_end"),
        "candidate_created_count": count("candidate_created"),
        "candidate_attempt_count": candidate_attempt_count,
        "candidate_localization_success_count": candidate_success_count,
        "candidate_localization_success_rate": (
            0.0 if candidate_attempt_count == 0 else round(
                candidate_success_count / candidate_attempt_count, 6)),
        "candidate_revisit_s_total": round(
            sum(candidate_revisit_durations), 3),
        "candidate_revisit_s_values": [
            round(value, 3) for value in candidate_revisit_durations],
        "strict_localization_s_total": total(
            "strict_localization_start", "strict_localization_end"),
        "fallback_rescan_s_total": total(
            "fallback_rescan_start", "fallback_rescan_end"),
        "time_to_first_localization_validated_s": time_to_first(
            "localization_validated"),
        "time_to_first_grasp_s": time_to_first("grasp_start"),
        "time_to_first_delivery_s": time_to_first("order_completed"),
        "full_rescan_count_after_first_order": sum(
            item.get("event") == "full_scan_start"
            and int(item.get("order_sequence", 1)) > 1
            for item in events),
        "memory_hit_count": count("target_memory_hit"),
        "memory_miss_count": count("target_memory_miss"),
        "revalidation_s_total": total(
            "local_revalidation_start", "local_revalidation_end"),
        "navigation_empty_s_total": total(
            "navigation_to_shelf_start", "navigation_to_shelf_end"),
        "navigation_carrying_s_total": total(
            "navigation_to_delivery_start", "navigation_to_delivery_end"),
        "grasp_s_total": total("grasp_start", "grasp_end"),
        "place_s_total": total("place_start", "place_end"),
        "max_idle_gap_s": round(max(idle_durations, default=0.0), 3),
        "idle_gap_over_15_count": sum(value > 15.0 for value in idle_durations),
        "replan_count": count("replan"),
        "no_path_count": count("no_path"),
        "localization_failure_count": count("localization_failure"),
        "grasp_failure_count": count("grasp_failure"),
        "drop_count": count("drop_suspected"),
        "human_intervention_count": count("human_intervention"),
        "ros_logger_exception_count": count("ros_logger_exception"),
        "runner_unhandled_exception_count": count(
            "runner_unhandled_exception"),
        "post_delivery_continuation_count": (
            post_delivery_continuation_count),
        "same_fingerprint_repeat_count": sum(
            item.get("event") == "candidate_attempt_fingerprint"
            and item.get("fingerprint_status") == "REPEAT"
            for item in events),
        "no_new_evidence_repeat_count": sum(
            item.get("event") == "candidate_attempt_fingerprint"
            and item.get("fingerprint_status") == "SUPPRESSED"
            for item in events),
        "suppressed_repeat_count": count("candidate_retry_suppressed"),
        "actual_suppressed_attempt_count": count(
            "candidate_retry_suppressed"),
        "equivalent_no_association_retry_count": sum(
            item.get("event") == "strict_equivalent_retry_started"
            and item.get("strict_failure_outcome") == "NO_ASSOCIATION"
            for item in events),
        "equivalent_spread_retry_count": sum(
            item.get("event") == "strict_equivalent_retry_started"
            and item.get("strict_failure_outcome") == "SPREAD_REJECT"
            for item in events),
        "equivalent_strict_retry_count": count(
            "strict_equivalent_retry_started"),
        "spread_recovery_attempt_count": count(
            "strict_spread_recovery_attempt"),
        "spread_recovery_success_count": count(
            "strict_spread_recovery_success"),
        "strict_retry_suppressed_count": count("strict_retry_suppressed"),
        "untried_context_selected_count": sum(
            item.get("event") == "strict_candidate_selected"
            and item.get("strict_failure_memory_state") == "UNTRIED"
            for item in events),
        "exhausted_context_skipped_count": count(
            "strict_exhausted_context_skipped"),
        "pregrasp_arm_timeout_count": sum(
            item.get("event") == "strict_failure_outcome"
            and item.get("strict_failure_outcome")
            == "PREGRASP_ARM_CONVERGENCE_TIMEOUT"
            for item in events),
        "actual_strict_suppressed_elapsed_s": round(sum(
            float(item.get("actual_suppressed_elapsed_s", 0.0))
            for item in events
            if item.get("event") == "strict_retry_suppressed"), 3),
        "offline_strict_counterfactual_estimate_s": round(sum(
            float(item.get("offline_counterfactual_estimate_s", 0.0))
            for item in events
            if item.get("event") == "strict_retry_suppressed"), 3),
        "model_estimated_avoided_time_s": round(sum(
            float(item.get(
                "model_estimated_avoided_time_s",
                item.get("avoided_repeat_time_s", 0.0))) for item in events
            if item.get("event") == "candidate_retry_suppressed"), 3),
        "deadline_infeasible_attempt_blocked_count": sum(
            item.get("event") == "candidate_deadline_feasibility"
            and item.get("deadline_feasible") is False for item in events),
        "deadline_feasible_attempt_started_count": len({
            item.get("attempt_id") for item in events
            if item.get("event") == "candidate_attempt_fingerprint"
            and item.get("deadline_feasible", True) is True
            and item.get("attempt_id") is not None}),
        "successful_candidate_created_s": successful_candidate_created_s,
        "successful_candidate_selected_s": successful_candidate_selected_s,
        "successful_candidate_selection_delay_s": (
            None if successful_candidate_created_s is None
            or successful_candidate_selected_s is None else round(
                successful_candidate_selected_s
                - successful_candidate_created_s, 3)),
        "time_to_first_grasp_success_s": (
            None if not starts or not successful_grasps else round(
                min(successful_grasps) - starts[0], 3)),
        "time_to_first_order_completed_s": time_to_first("order_completed"),
        "actual_time_to_first_candidate_s": time_to_first("candidate_created"),
        "actual_time_to_first_grasp_s": time_to_first("grasp_start"),
        "actual_time_to_first_order_completed_s": time_to_first(
            "order_completed"),
        "unused_hard_deadline_time_at_stop_s": (
            None if match_elapsed is None else round(max(
                0.0, float(legacy.get("hard_deadline_s", 600.0))
                - match_elapsed), 3)),
        "estimated_raw_task_score": {
            "label": "INTERNAL_PROXY",
            "completed_order_points": delivered * 12,
        },
        "inventory": [] if inventory is None else inventory,
        "scan_coverage": [] if scan_coverage is None else scan_coverage,
        "order_selected_count": count("order_selected"),
        "attempt_started_count": count("attempt_started"),
        "worker_started_count": count("worker_started"),
        "attempt_terminal_count": count("attempt_terminal"),
        "order_delivered_count": count("order_completed"),
        "unique_orders_selected": len({
            item.get("order_id") for item in events
            if item.get("event") == "order_selected"}),
        "unique_orders_attempt_started": len({
            item.get("order_id") for item in events
            if item.get("event") == "attempt_started"}),
        "unique_orders_terminal": len({
            item.get("order_id") for item in events
            if item.get("event") == "attempt_terminal"}),
        "unique_orders_delivered": len({
            item.get("order_id") for item in events
            if item.get("event") == "order_completed"}),
        "candidate_local_revisit_count": count("candidate_attempt_start"),
        "repeated_covered_pose_count": count("repeated_covered_pose_scan"),
        "coverage_reuse_count": count("covered_pose_reused"),
        "discovery_segments_started": count("discovery_segment_started"),
        "discovery_segments_completed": count("discovery_segment_completed"),
        "discovery_segments_producing_candidates": sum(
            item.get("event") == "discovery_segment_completed"
            and int(item.get("candidate_created_count", 0)) > 0
            for item in events),
        "full_route_restart_count": count("full_route_restart"),
        "initial_discovery_segments": sum(
            item.get("event") == "scan_pose_started"
            and not (item.get("coverage_key") or {}).get("resumed", False)
            for item in events),
        "resumed_uncovered_scan_segments": sum(
            item.get("event") == "scan_pose_started"
            and (item.get("coverage_key") or {}).get("resumed", False)
            for item in events),
        "covered_pose_count": sum(
            item.get("event") == "scan_pose_completed"
            and item.get("covered_valid") is True for item in events),
        "partial_pose_count": sum(
            item.get("event") == "scan_pose_completed"
            and item.get("covered_valid") is False for item in events),
        "soft_deadline_reached": count("soft_deadline_reached") > 0,
        "soft_deadline_phase": next((
            item.get("worker_phase") for item in events
            if item.get("event") == "soft_deadline_reached"), None),
        "new_attempt_blocked": count("new_attempt_blocked") > 0,
        "inflight_completion_allowed": count(
            "inflight_completion_allowed") > 0,
        "inflight_attempt_id": next((
            item.get("attempt_id") for item in events
            if item.get("event") == "inflight_completion_allowed"), None),
        "inflight_order_id": next((
            item.get("order_id") for item in events
            if item.get("event") == "inflight_completion_allowed"), None),
        "terminal_result_observed": count("terminal_result_observed") > 0,
        "terminal_result_status": next((
            item.get("terminal_result_status") for item in reversed(events)
            if item.get("event") == "terminal_result_observed"), None),
        "terminal_result_completion_s": next((
            item.get("terminal_result_completion_s")
            for item in reversed(events)
            if item.get("event") == "terminal_result_observed"), None),
        "terminal_outcome_accepted": count(
            "terminal_outcome_accepted") > 0,
        "terminal_outcome_rejected_reason": next((
            item.get("terminal_outcome_rejected_reason")
            for item in reversed(events)
            if item.get("event") == "terminal_outcome_rejected"), None),
        "inflight_completed_after_soft_deadline": count(
            "inflight_completed_after_soft_deadline") > 0,
        "delivered_count_after_acceptance": next((
            item.get("delivered_count_after_acceptance")
            for item in reversed(events)
            if item.get("event") == "terminal_outcome_accepted"), None),
        "hard_deadline_reached": count("hard_deadline_reached") > 0,
    })
    return result


def atomic_write_json(path: str | Path, document: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(destination)


def write_summary_csv(path: str | Path, document: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    flattened = {
        key: (json.dumps(value, ensure_ascii=False, separators=(",", ":"))
              if isinstance(value, (dict, list)) else value)
        for key, value in document.items()
    }
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flattened))
        writer.writeheader()
        writer.writerow(flattened)
    temporary.replace(destination)


class IdleGapObserver:
    """Detect a genuine lack of base/joint/state progress, not long travel."""

    def __init__(self, threshold_s: float = 15.0):
        self.threshold_s = float(threshold_s)
        self.last_progress_s: float | None = None
        self.last_signature: tuple[Any, ...] | None = None
        self.idle_reported = False

    def update(self, now: float, signature: tuple[Any, ...]) -> float | None:
        if self.last_progress_s is None:
            self.last_progress_s = now
            self.last_signature = signature
            return None
        if signature != self.last_signature:
            duration = now - self.last_progress_s
            self.last_progress_s = now
            self.last_signature = signature
            if self.idle_reported:
                self.idle_reported = False
                return duration
            return None
        if now - self.last_progress_s > self.threshold_s:
            self.idle_reported = True
        return None

    def flush(self, now: float) -> float | None:
        if self.idle_reported and self.last_progress_s is not None:
            duration = now - self.last_progress_s
            self.idle_reported = False
            self.last_progress_s = now
            return duration
        return None


def rebuild_summary_file(path: str | Path) -> dict[str, Any]:
    """Rebuild a generated summary deterministically from its raw JSONL."""
    summary_path = Path(path)
    old = json.loads(summary_path.read_text(encoding="utf-8"))
    rebuilt = build_summary(
        read_events(summary_path.with_name("events.jsonl")),
        old,
        mode=str(old["mode"]),
        product_seed=old.get("product_seed"),
        obstacle_seed=old.get("obstacle_seed"),
        task_kinds=list(old.get("task_kinds", [])),
        inventory=list(old.get("inventory", [])),
        scan_coverage=list(old.get("scan_coverage", [])),
    )
    atomic_write_json(summary_path, rebuilt)
    write_summary_csv(summary_path.with_suffix(".csv"), rebuilt)
    return rebuilt


def main() -> None:
    parser = argparse.ArgumentParser(description="score telemetry utilities")
    parser.add_argument("--rebuild-summary", type=Path, required=True)
    args = parser.parse_args()
    rebuilt = rebuild_summary_file(args.rebuild_summary)
    print(json.dumps(rebuilt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
