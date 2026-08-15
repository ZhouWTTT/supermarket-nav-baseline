import json
import pathlib
import sys
import tempfile
import unittest


MODULE_DIR = (
    pathlib.Path(__file__).resolve().parents[1]
    / "examples" / "supermarket_sorting")
sys.path.insert(0, str(MODULE_DIR))

from score_telemetry import (  # noqa: E402
    EventLog,
    IdleGapObserver,
    build_summary,
    read_events,
    rebuild_summary_file,
)


def event(name, stamp, **payload):
    return {
        "schema_version": 1,
        "event": name,
        "monotonic_s": float(stamp),
        "run_prefix": "run_test",
        "mode": "off",
        **payload,
    }


class ScoreTelemetryTests(unittest.TestCase):
    def test_event_log_is_jsonl_with_monotonic_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "events.jsonl"
            record = EventLog(
                path, run_prefix="run_a", mode="off").emit(
                    "match_start", product_seed="11")
            self.assertEqual(record["schema_version"], 1)
            self.assertEqual(record["run_prefix"], "run_a")
            self.assertEqual(record["mode"], "off")
            self.assertIsInstance(record["monotonic_s"], float)
            self.assertEqual(read_events(path)[0]["event"], "match_start")
            json.loads(path.read_text(encoding="utf-8").strip())

    def test_summary_has_required_score_fields_and_open_stage_closes_at_match(self):
        events = [
            event("match_start", 100.0),
            event("task_received", 100.1),
            event("order_selected", 101.0, order_id="a"),
            event("target_memory_miss", 101.1, order_id="a"),
            event("search_start", 101.0, order_id="a"),
            event("full_scan_start", 101.0, order_id="a", order_sequence=1),
            event("navigation_to_shelf_start", 101.0, order_id="a"),
            event("candidate_created", 105.0, candidate_id="c1"),
            event("candidate_attempt_start", 106.0, order_id="a",
                  candidate_id="c1"),
            event("strict_localization_start", 108.0, order_id="a",
                  candidate_id="c1"),
            event("localization_validated", 110.0, order_id="a",
                  candidate_id="c1"),
            event("strict_localization_end", 110.0, order_id="a",
                  candidate_id="c1"),
            event("candidate_attempt_end", 110.0, order_id="a",
                  candidate_id="c1"),
            event("search_end", 121.0, order_id="a"),
            event("full_scan_end", 121.0, order_id="a"),
            event("navigation_to_shelf_end", 121.0, order_id="a"),
            event("local_revalidation_start", 122.0, order_id="a"),
            event("local_revalidation_end", 125.0, order_id="a"),
            event("grasp_start", 125.0, order_id="a"),
            event("grasp_end", 135.0, order_id="a"),
            event("navigation_to_delivery_start", 136.0, order_id="a"),
            event("no_path", 145.0, order_id="a"),
            event("idle_gap", 150.0, order_id="a", duration_s=16.0),
            event("order_failed", 160.0, order_id="a"),
            event("match_end", 160.0),
        ]
        summary = build_summary(
            events,
            {"run_prefix": "run_test", "delivered": 0, "count": 5},
            mode="off", product_seed="11", obstacle_seed="22",
            task_kinds=["kele"] * 5)
        required = {
            "run_prefix", "mode", "product_seed", "obstacle_seed",
            "task_kinds", "delivered_count", "match_elapsed_s",
            "order_elapsed_s", "search_s_total", "full_scan_s_total",
            "full_rescan_count_after_first_order", "memory_hit_count",
            "memory_miss_count", "revalidation_s_total",
            "navigation_empty_s_total", "navigation_carrying_s_total",
            "grasp_s_total", "place_s_total", "max_idle_gap_s",
            "idle_gap_over_15_count", "replan_count", "no_path_count",
            "localization_failure_count", "grasp_failure_count",
            "drop_count", "human_intervention_count",
            "estimated_raw_task_score",
            "candidate_created_count", "candidate_attempt_count",
            "candidate_localization_success_count",
            "candidate_localization_success_rate",
            "candidate_revisit_s_total", "strict_localization_s_total",
            "fallback_rescan_s_total",
            "time_to_first_localization_validated_s",
            "time_to_first_grasp_s", "time_to_first_delivery_s",
            "discovery_scan_s", "candidate_revisit_s",
            "strict_localization_s", "fallback_rescan_s",
        }
        self.assertTrue(required.issubset(summary))
        self.assertEqual(summary["match_elapsed_s"], 60.0)
        self.assertEqual(summary["search_s_total"], 20.0)
        self.assertEqual(summary["navigation_carrying_s_total"], 24.0)
        self.assertEqual(summary["no_path_count"], 1)
        self.assertEqual(summary["idle_gap_over_15_count"], 1)
        self.assertEqual(summary["candidate_created_count"], 1)
        self.assertEqual(summary["candidate_attempt_count"], 1)
        self.assertEqual(summary["candidate_localization_success_count"], 1)
        self.assertEqual(summary["candidate_localization_success_rate"], 1.0)
        self.assertEqual(summary["candidate_revisit_s_total"], 4.0)
        self.assertEqual(summary["strict_localization_s_total"], 2.0)
        self.assertEqual(
            summary["time_to_first_localization_validated_s"], 10.0)
        self.assertEqual(summary["time_to_first_grasp_s"], 25.0)
        self.assertIsNone(summary["time_to_first_delivery_s"])
        self.assertEqual(
            summary["estimated_raw_task_score"]["label"], "INTERNAL_PROXY")

    def test_idle_observer_reports_only_after_progress_resumes(self):
        observer = IdleGapObserver(15.0)
        self.assertIsNone(observer.update(0.0, ("scan", 0)))
        self.assertIsNone(observer.update(16.0, ("scan", 0)))
        self.assertEqual(observer.update(18.0, ("scan", 1)), 18.0)

    def test_deadline_terminal_outcome_fields_are_summarized(self):
        events = [
            event("match_start", 0.0),
            event("soft_deadline_reached", 570.087,
                  worker_phase="PLACING"),
            event("inflight_completion_allowed", 570.088,
                  attempt_id="a1", order_id="o1"),
            event("terminal_result_observed", 577.400,
                  attempt_id="a1", order_id="o1",
                  terminal_result_status="delivered",
                  terminal_result_completion_s=577.386),
            event("terminal_outcome_accepted", 577.401,
                  attempt_id="a1", order_id="o1",
                  delivered_count_after_acceptance=1),
            event("inflight_completed_after_soft_deadline", 577.402,
                  attempt_id="a1", order_id="o1"),
            event("new_attempt_blocked", 577.403),
            event("match_end", 579.480),
        ]
        summary = build_summary(
            events, {"run_prefix": "run_test", "delivered": 1},
            mode="off", product_seed="1", obstacle_seed="2",
            task_kinds=["kele"])
        self.assertEqual(summary["soft_deadline_phase"], "PLACING")
        self.assertTrue(summary["new_attempt_blocked"])
        self.assertEqual(summary["inflight_attempt_id"], "a1")
        self.assertEqual(summary["inflight_order_id"], "o1")
        self.assertTrue(summary["terminal_result_observed"])
        self.assertEqual(summary["terminal_result_status"], "delivered")
        self.assertEqual(summary["terminal_result_completion_s"], 577.386)
        self.assertTrue(summary["terminal_outcome_accepted"])
        self.assertTrue(summary["inflight_completed_after_soft_deadline"])
        self.assertEqual(summary["delivered_count_after_acceptance"], 1)

    def test_order_failure_closes_only_that_orders_open_stage(self):
        events = [
            event("match_start", 0.0),
            event("search_start", 1.0, order_id="a"),
            event("order_failed", 11.0, order_id="a"),
            event("search_start", 12.0, order_id="b"),
            event("match_end", 20.0),
        ]
        summary = build_summary(
            events, {"run_prefix": "run_test", "delivered": 0},
            mode="off", product_seed="1", obstacle_seed="2",
            task_kinds=["kele", "maidong"])
        self.assertEqual(summary["search_s_total"], 18.0)

    def test_rebuild_summary_file_updates_json_and_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            summary_path = root / "summary.json"
            summary_path.write_text(json.dumps({
                "run_prefix": "run_test",
                "mode": "off",
                "product_seed": "1",
                "obstacle_seed": "2",
                "task_kinds": ["kele"],
                "delivered": 0,
                "inventory": [],
                "scan_coverage": [],
            }), encoding="utf-8")
            records = [
                event("match_start", 1.0),
                event("match_end", 6.0),
            ]
            (root / "events.jsonl").write_text(
                "".join(json.dumps(item) + "\n" for item in records),
                encoding="utf-8")
            rebuilt = rebuild_summary_file(summary_path)
            self.assertEqual(rebuilt["match_elapsed_s"], 5.0)
            self.assertTrue((root / "summary.csv").is_file())

    def test_logger_fault_and_post_delivery_continuation_metrics(self):
        events = [
            event("match_start", 0.0),
            event("attempt_started", 1.0, order_id="a"),
            event("attempt_terminal", 2.0, order_id="a"),
            event("order_completed", 2.1, order_id="a"),
            event("attempt_started", 3.0, order_id="b"),
            event("ros_logger_exception", 3.1),
            event("attempt_terminal", 4.0, order_id="b"),
        ]
        summary = build_summary(
            events, {"run_prefix": "run_test", "delivered": 1},
            mode="off", product_seed="1", obstacle_seed="2",
            task_kinds=["kele", "maidong"])
        self.assertEqual(summary["ros_logger_exception_count"], 1)
        self.assertEqual(summary["runner_unhandled_exception_count"], 0)
        self.assertEqual(summary["post_delivery_continuation_count"], 1)

    def test_r9_retry_deadline_and_success_timing_metrics(self):
        events = [
            event("match_start", 0.0),
            event("candidate_created", 10.0, candidate_id="candidate-7"),
            event("candidate_attempt_fingerprint", 20.0,
                  attempt_id="a1",
                  fingerprint={"candidate_id": "candidate-7"},
                  fingerprint_status="UNTRIED", deadline_feasible=True),
            event("grasp_end", 50.0, success=True),
            event("candidate_retry_suppressed", 60.0,
                  avoided_repeat_time_s=233.0),
            event("candidate_deadline_feasibility", 61.0,
                  deadline_feasible=False),
            event("candidate_attempt_outcome", 70.0, delivered=True,
                  fingerprint={"candidate_id": "candidate-7"}),
            event("order_completed", 71.0, order_id="a"),
        ]
        summary = build_summary(
            events, {"run_prefix": "run_test", "delivered": 1},
            mode="off", product_seed="1", obstacle_seed="2",
            task_kinds=["maidong"])
        self.assertEqual(summary["suppressed_repeat_count"], 1)
        self.assertEqual(summary["model_estimated_avoided_time_s"], 233.0)
        self.assertEqual(summary["deadline_infeasible_attempt_blocked_count"], 1)
        self.assertEqual(summary["deadline_feasible_attempt_started_count"], 1)
        self.assertEqual(summary["successful_candidate_created_s"], 10.0)
        self.assertEqual(summary["successful_candidate_selected_s"], 20.0)
        self.assertEqual(summary["successful_candidate_selection_delay_s"], 10.0)
        self.assertEqual(summary["time_to_first_grasp_success_s"], 50.0)
        self.assertEqual(summary["time_to_first_order_completed_s"], 71.0)

    def test_model_estimate_is_not_reported_as_actual_time(self):
        summary = build_summary([
            event("match_start", 0.0),
            event("candidate_created", 10.0, candidate_id="c"),
            event("candidate_retry_suppressed", 20.0,
                  model_estimated_avoided_time_s=233.0),
            event("match_end", 30.0),
        ], {"delivered": 0, "hard_deadline_s": 600.0}, mode="m",
           product_seed=None, obstacle_seed=None, task_kinds=[])
        self.assertEqual(summary["model_estimated_avoided_time_s"], 233.0)
        self.assertEqual(summary["actual_suppressed_attempt_count"], 1)
        self.assertEqual(summary["actual_match_elapsed_s"], 30.0)
        self.assertEqual(summary["actual_time_to_first_candidate_s"], 10.0)
        self.assertEqual(summary["unused_hard_deadline_time_at_stop_s"], 570.0)
        self.assertNotIn("avoided_repeat_time_s", summary)

    def test_strict_failure_summary_counters_are_separate(self):
        events = [
            event("match_start", 1.0, strict_failure_memory_mode="control"),
            event("strict_candidate_selected", 2.0,
                  strict_failure_memory_state="UNTRIED"),
            event("strict_spread_recovery_attempt", 3.0),
            event("strict_spread_recovery_success", 4.0),
            event("strict_retry_suppressed", 5.0,
                  actual_suppressed_elapsed_s=0.0,
                  offline_counterfactual_estimate_s=233.0),
            event("strict_failure_outcome", 6.0,
                  strict_failure_outcome=(
                      "PREGRASP_ARM_CONVERGENCE_TIMEOUT")),
            event("match_end", 7.0),
        ]
        summary = build_summary(
            events, {"delivered": 0}, mode="run_inventory",
            product_seed=41003, obstacle_seed=51003, task_kinds=[])
        self.assertEqual(summary["strict_failure_memory_mode"], "control")
        self.assertEqual(summary["spread_recovery_attempt_count"], 1)
        self.assertEqual(summary["spread_recovery_success_count"], 1)
        self.assertEqual(summary["strict_retry_suppressed_count"], 1)
        self.assertEqual(summary["untried_context_selected_count"], 1)
        self.assertEqual(summary["pregrasp_arm_timeout_count"], 1)
        self.assertEqual(summary["actual_strict_suppressed_elapsed_s"], 0.0)
        self.assertEqual(
            summary["offline_strict_counterfactual_estimate_s"], 233.0)


if __name__ == "__main__":
    unittest.main()
