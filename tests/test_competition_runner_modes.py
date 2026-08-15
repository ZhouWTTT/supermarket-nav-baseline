import ast
import inspect
import json
import pathlib
import sys
import tempfile
import textwrap
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
from unittest.mock import patch


MODULE_DIR = (
    pathlib.Path(__file__).resolve().parents[1]
    / "examples" / "supermarket_sorting")
sys.path.insert(0, str(MODULE_DIR))

from competition_runner import (  # noqa: E402
    INVALIDATED,
    PROVISIONAL_VIEW_HINT,
    CompetitionRunner,
    _safe_log_error,
    _safe_log_info,
    derive_candidate_view_hint,
    runner_deadline_action,
)
from competition_task import CompetitionTask  # noqa: E402
from candidate_attempt_memory import CandidateAttemptMemory  # noqa: E402
from strict_replay_outcome_memory import (  # noqa: E402
    StrictReplayOutcomeMemory,
)
from score_first_order_policy import select_order as score_first_select_order  # noqa: E402
from score_telemetry import EventLog, build_summary, read_events  # noqa: E402
from integrated_nav_pick_place import (  # noqa: E402
    candidate_failure_stage_reason,
    candidate_replay_poses,
    preferred_local_scan_exhausted,
)
import yolo_aruco_shelf_pick as pick  # noqa: E402


def make_task():
    return CompetitionTask.from_json(json.dumps({
        "schema_version": 1,
        "run_prefix": "run_modes",
        "count": 2,
        "targets": [
            {"id": "tissue", "kind": "zhijin"},
            {"id": "bottle", "kind": "kele"},
        ],
    }))


def make_three_order_task():
    return CompetitionTask.from_json(json.dumps({
        "schema_version": 1,
        "run_prefix": "run_continuation",
        "count": 3,
        "targets": [
            {"id": "order_a", "kind": "zhijin"},
            {"id": "order_b", "kind": "kele"},
            {"id": "order_c", "kind": "maidong"},
        ],
    }))


def bare_runner(memory_mode, *, strict_failure_memory_mode):
    if strict_failure_memory_mode not in {"off", "shadow", "control"}:
        raise ValueError("strict_failure_memory_mode must be off, shadow, or control")
    runner = object.__new__(CompetitionRunner)
    runner.task = make_task()
    runner.args = SimpleNamespace(
        memory_mode=memory_mode, max_attempts=2, inventory_confirmations=3)
    runner.strict_failure_memory_mode = strict_failure_memory_mode
    runner.strict_failure_outcome_memory = (
        None if strict_failure_memory_mode == "off"
        else StrictReplayOutcomeMemory(runner.task.run_prefix))
    runner.candidate_attempt_memory = CandidateAttemptMemory()
    runner.strict_suppression_events = set()
    runner.strict_recovery_attempt_events = set()
    runner.event_log = Mock()
    runner.scan_coverage = set()
    runner.inventory = {
        7: {
            "marker_id": 7,
            "kind": "zhijin",
            "position_world": [-1.8, 3.2, 0.8],
            "confidence": 0.9,
            "confirmations": 3,
            "state": PROVISIONAL_VIEW_HINT,
            "candidate_id": "candidate-7",
            "provisional_marker_id": 7,
            "provisional_marker_world": [-1.8, 3.2, 0.8],
        }
    }
    runner.last_selection = {}
    runner.current_candidate = None
    runner.fallback_rescan_open = False
    return runner


def terminal_runner(completion_s=577.386):
    runner = bare_runner(
        "run_inventory", strict_failure_memory_mode="off")
    runner.current_order = runner.task.orders[0]
    runner.current_attempt_id = "run_modes:tissue:attempt-1"
    runner.task_generation = 3
    runner.current_worker_binding = {
        "run_prefix": "run_modes",
        "generation": 3,
        "order_id": "tissue",
        "attempt_id": runner.current_attempt_id,
        "worker_pid": 123,
        "result_path": "exclusive-result.json",
    }
    runner.args.match_timeout = 600.0
    runner.args.max_attempts = 2
    runner.worker_result_path = None
    runner.worker_stop_reason = None
    runner.worker = Mock(pid=123)
    runner.worker_started_at = 1.0
    runner.worker_terminate_at = None
    runner.current_order_sequence = 1
    runner.current_candidate = None
    runner.preferred_marker_id = None
    runner.event_log = Mock()
    runner.event_file = None
    runner.task_started_at = 0.0
    runner.attempt_terminal_ids = set()
    runner.terminal_result_fingerprints = {}
    runner.terminal_outcome_accepted_ids = set()
    runner.terminal_outcome_rejections = set()
    runner.soft_deadline_emitted = False
    runner.inflight_allowed_emitted = False
    runner.hard_deadline_emitted = False
    runner.new_attempt_blocked_emitted = False
    runner._write_summary = Mock()
    runner._publish_stop = Mock()
    runner.get_logger = Mock(return_value=SimpleNamespace(
        info=Mock(), error=Mock()))
    result = {
        "run_prefix": "run_modes",
        "generation": 3,
        "attempt_id": runner.current_attempt_id,
        "worker_pid": 123,
        "order_id": "tissue",
        "kind": "zhijin",
        "status": "delivered",
        "marker_id": 7,
        "validated_marker_id": 7,
        "terminal_result_completion_s": completion_s,
    }
    return runner, result


class CompetitionRunnerModeTests(unittest.TestCase):
    def test_bare_runner_fixture_requires_complete_explicit_strict_mode(self):
        runner = bare_runner(
            "run_inventory", strict_failure_memory_mode="off")
        self.assertEqual(runner.strict_failure_memory_mode, "off")
        self.assertIsNone(runner.strict_failure_outcome_memory)
        self.assertIsInstance(
            runner.candidate_attempt_memory, CandidateAttemptMemory)
        self.assertEqual(runner.strict_suppression_events, set())
        self.assertEqual(runner.strict_recovery_attempt_events, set())
        with self.assertRaises(TypeError):
            bare_runner("run_inventory")
        with self.assertRaises(ValueError):
            bare_runner(
                "run_inventory", strict_failure_memory_mode="authoritative")

    def test_production_init_validates_strict_failure_memory_mode(self):
        def construct(mode=None):
            environment = {"SUPERMARKET_SCAN_COVERAGE_MODE": "shadow"}
            if mode is not None:
                environment["SUPERMARKET_STRICT_FAILURE_MEMORY_MODE"] = mode
            with patch.dict(
                    "competition_runner.os.environ", environment, clear=True), \
                    patch("competition_runner.Node.__init__", return_value=None), \
                    patch.object(
                        CompetitionRunner, "create_subscription",
                        return_value=Mock()), \
                    patch.object(
                        CompetitionRunner, "create_publisher",
                        return_value=Mock()), \
                    patch.object(
                        CompetitionRunner, "create_timer", return_value=Mock()), \
                    patch.object(
                        CompetitionRunner, "get_logger",
                        return_value=SimpleNamespace(info=Mock())):
                return CompetitionRunner(SimpleNamespace())

        self.assertEqual(construct().strict_failure_memory_mode, "shadow")
        for mode in ("off", "shadow", "control"):
            self.assertEqual(
                construct(mode).strict_failure_memory_mode, mode)
        with self.assertRaises(ValueError):
            construct("authoritative")

    def test_shadow_advice_preserves_selection_and_control_is_authoritative(self):
        def exhaust_candidate(runner):
            key, decision = runner._strict_key_and_decision(
                runner.task.orders[0], runner.inventory[7])
            self.assertTrue(decision.allowed)
            runner.strict_failure_outcome_memory.record_no_association(
                key, attempt_id="fixture-primary-and-backup",
                pose_ids=(key.primary_pose_id, *key.backup_pose_ids))

        off = bare_runner(
            "run_inventory", strict_failure_memory_mode="off")
        off_selection = off._select_order()

        shadow = bare_runner(
            "run_inventory", strict_failure_memory_mode="shadow")
        exhaust_candidate(shadow)
        shadow_selection = shadow._select_order()
        self.assertEqual(shadow_selection, off_selection)
        self.assertEqual(shadow_selection[0].id, "tissue")
        self.assertEqual(shadow_selection[1], 7)
        self.assertFalse(shadow.last_selection["strict_retry_allowed"])
        self.assertEqual(
            shadow.last_selection["strict_failure_memory_mode"], "shadow")
        shadow.event_log.emit.assert_not_called()

        control = bare_runner(
            "run_inventory", strict_failure_memory_mode="control")
        exhaust_candidate(control)
        control_selection = control._select_order()
        self.assertEqual(control_selection[0].id, "bottle")
        self.assertIsNone(control_selection[1])
        self.assertEqual(
            control.last_selection["strict_failure_memory_mode"], "control")
        self.assertEqual(len(control.last_selection["candidates"]), 2)
        self.assertEqual(
            [item.args[0] for item in control.event_log.emit.call_args_list],
            ["strict_retry_suppressed", "strict_exhausted_context_skipped"])

    def test_terminal_logging_has_fixed_severity_callsites(self):
        finish_source = inspect.getsource(CompetitionRunner._finish_worker)
        finish_tree = ast.parse(textwrap.dedent(finish_source))
        forbidden = []
        for node in ast.walk(finish_tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if (node.func.id == "getattr" and node.args
                        and "logger" in ast.unparse(node.args[0]).lower()):
                    forbidden.append("dynamic logger getattr")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr == "log":
                    forbidden.append("logger.log")
            if isinstance(node, ast.IfExp):
                attrs = {
                    child.attr for child in ast.walk(node)
                    if isinstance(child, ast.Attribute)
                }
                if {"info", "error"}.issubset(attrs):
                    forbidden.append("conditional logger callable")
        self.assertEqual(forbidden, [])

        info_source = inspect.getsource(_safe_log_info)
        error_source = inspect.getsource(_safe_log_error)
        self.assertIn("logger.info(message)", info_source)
        self.assertNotIn("logger.error", info_source)
        self.assertNotIn("logger.log", info_source)
        self.assertIn("logger.error(message)", error_source)
        self.assertNotIn("logger.info", error_source)
        self.assertNotIn("logger.log", error_source)

    def test_actual_rclpy_info_error_info_uses_distinct_callsites(self):
        import rclpy

        context = rclpy.context.Context()
        rclpy.init(context=context)
        node = rclpy.create_node("r8c_fixed_severity_test", context=context)
        event_log = Mock()
        try:
            self.assertTrue(_safe_log_info(
                node.get_logger(), "success", event_log=event_log,
                context="test_success_1"))
            self.assertTrue(_safe_log_error(
                node.get_logger(), "failure", event_log=event_log,
                context="test_failure"))
            self.assertTrue(_safe_log_info(
                node.get_logger(), "success", event_log=event_log,
                context="test_success_2"))
            event_log.emit.assert_not_called()
        finally:
            node.destroy_node()
            rclpy.shutdown(context=context)

    def test_delivered_persists_before_info_logger_failure(self):
        runner, result = terminal_runner(270.0)
        call_order = []
        runner._write_summary.side_effect = lambda reason: call_order.append(
            ("summary", reason, runner.task.summary()["delivered"],
             len(runner.attempt_terminal_ids)))
        logger = runner.get_logger.return_value

        def fail_info(_message):
            call_order.append(("info",))
            raise ValueError("synthetic logger failure")

        logger.info.side_effect = fail_info
        runner._finish_worker(
            0, result=result, inspected=runner._inspect_terminal_result(result))

        self.assertEqual(runner.task.summary()["delivered"], 1)
        self.assertIn(result["attempt_id"], runner.attempt_terminal_ids)
        self.assertEqual(call_order[0], ("summary", "worker_finished", 1, 1))
        self.assertEqual(call_order[1], ("info",))
        self.assertEqual(call_order[2], ("summary", "worker_finished", 1, 1))
        runner.event_log.emit.assert_any_call(
            "ros_logger_exception",
            ros_logger_exception_type="ValueError",
            ros_logger_exception_message="synthetic logger failure",
            ros_logger_exception_context="finish_worker_delivered",
            ros_logger_fallback_used=True)

    def test_failed_terminal_persists_before_error_logger_failure(self):
        runner, result = terminal_runner(271.0)
        result.update(status="failed", error="localization_failed")
        call_order = []
        runner._write_summary.side_effect = lambda reason: call_order.append(
            ("summary", reason, runner.task.orders[0].status,
             len(runner.attempt_terminal_ids)))
        logger = runner.get_logger.return_value

        def fail_error(_message):
            call_order.append(("error",))
            raise ValueError("synthetic logger failure")

        logger.error.side_effect = fail_error
        runner._finish_worker(
            1, result=result, inspected=runner._inspect_terminal_result(result))

        self.assertIn(result["attempt_id"], runner.attempt_terminal_ids)
        self.assertEqual(call_order[0][0:2], ("summary", "worker_finished"))
        self.assertEqual(call_order[0][3], 1)
        self.assertEqual(call_order[1], ("error",))
        self.assertEqual(call_order[2][0:2], ("summary", "worker_finished"))
        runner.finished = False
        runner.worker = None
        runner._select_order = Mock(
            return_value=(runner.task.orders[1], None))
        runner._start_worker = Mock()
        with patch("competition_runner.time.monotonic", return_value=10.0):
            runner._tick()
        runner._start_worker.assert_called_once_with(
            runner.task.orders[1], None)

    def test_success_failure_success_and_post_delivery_continuation(self):
        runner, _ = terminal_runner(100.0)
        runner.task = make_three_order_task()
        runner.inventory = {}
        runner.attempt_terminal_ids = set()
        runner.terminal_outcome_accepted_ids = set()
        runner.terminal_result_fingerprints = {}
        runner.current_order_sequence = 0
        runner.task_generation = 7

        with tempfile.TemporaryDirectory() as directory:
            event_file = pathlib.Path(directory) / "events.jsonl"
            runner.event_file = event_file
            runner.event_log = EventLog(
                event_file, run_prefix=runner.task.run_prefix,
                mode="run_inventory")
            runner._write_summary = Mock()

            def finish(index, status, completion_s):
                order = runner.task.orders[index]
                attempt_id = (
                    f"{runner.task.run_prefix}:{order.id}:attempt-1")
                runner.current_order = order
                runner.current_attempt_id = attempt_id
                runner.current_order_sequence = index + 1
                runner.worker = Mock(pid=321 + index)
                runner.worker_started_at = 1.0
                runner.worker_stop_reason = None
                runner.worker_terminate_at = None
                runner.current_candidate = None
                runner.preferred_marker_id = None
                runner.current_worker_binding = {
                    "run_prefix": runner.task.run_prefix,
                    "generation": 7,
                    "order_id": order.id,
                    "attempt_id": attempt_id,
                    "worker_pid": 321 + index,
                    "result_path": f"order_{index}.json",
                }
                runner.event_log.emit(
                    "attempt_started", attempt_id=attempt_id,
                    order_id=order.id)
                result = {
                    "run_prefix": runner.task.run_prefix,
                    "generation": 7,
                    "attempt_id": attempt_id,
                    "worker_pid": 321 + index,
                    "order_id": order.id,
                    "kind": order.kind,
                    "status": status,
                    "marker_id": 40 + index,
                    "validated_marker_id": 40 + index,
                    "terminal_result_completion_s": completion_s,
                }
                runner._finish_worker(
                    0 if status == "delivered" else 1,
                    result=result,
                    inspected=runner._inspect_terminal_result(result))

            finish(0, "delivered", 100.0)
            finish(1, "failed", 110.0)

            runner.finished = False
            runner.worker = None
            runner.task_started_at = 0.0
            runner._select_order = Mock(
                return_value=(runner.task.orders[2], None))
            runner._start_worker = Mock()
            with patch("competition_runner.time.monotonic", return_value=120.0):
                runner._tick()
            runner._start_worker.assert_called_once_with(
                runner.task.orders[2], None)

            finish(2, "delivered", 130.0)
            events = read_events(event_file)
            summary = build_summary(
                events, runner.task.summary(), mode="run_inventory",
                product_seed=41003, obstacle_seed=51003,
                task_kinds=[order.kind for order in runner.task.orders])

        self.assertEqual(runner.task.orders[0].status, "delivered")
        self.assertNotEqual(runner.task.orders[1].status, "delivered")
        self.assertEqual(summary["unique_orders_terminal"], 3)
        self.assertEqual(summary["delivered_count"], 2)
        self.assertEqual(summary["post_delivery_continuation_count"], 2)
        self.assertEqual(summary["ros_logger_exception_count"], 0)
        self.assertEqual(summary["runner_unhandled_exception_count"], 0)

    def test_soft_deadline_blocks_only_new_or_pregrasp_work(self):
        self.assertEqual(
            runner_deadline_action(570.0, None, hard_deadline_s=600.0),
            "BLOCK_NEW")
        self.assertEqual(
            runner_deadline_action(570.0, "DISCOVERY", hard_deadline_s=600.0),
            "BLOCK_NEW")
        for phase in (
                "GRASPED", "BACKUP", "NAV_TO_DELIVERY", "DELIVERING",
                "PLACE_APPROACH", "PLACING", "PLACE_RETRY", "DONE",
                "DONE_AWAITING_RESULT"):
            self.assertEqual(
                runner_deadline_action(570.0, phase, hard_deadline_s=600.0),
                "ALLOW_INFLIGHT")
        self.assertEqual(
            runner_deadline_action(600.0, "PLACING", hard_deadline_s=600.0),
            "HARD_STOP")

    def test_r8a_timeline_delivered_is_accepted_after_soft_deadline(self):
        runner, result = terminal_runner(577.386)
        inspected = runner._inspect_terminal_result(result)
        runner._finish_worker(1, result=result, inspected=inspected)
        self.assertEqual(runner.task.summary()["delivered"], 1)
        self.assertEqual(runner.task.orders[0].attempts, 1)
        self.assertIn(
            "run_modes:tissue:attempt-1",
            runner.terminal_outcome_accepted_ids)

    def test_tick_polls_terminal_before_soft_gating_and_starts_no_next_work(self):
        runner, result = terminal_runner(577.386)
        runner.finished = False
        runner.worker_runtime_phase = "PLACING"
        runner.worker.poll.side_effect = [None, None, 0]
        runner.worker.terminate = Mock()
        worker = runner.worker
        runner._read_worker_result = Mock(side_effect=[{}, result, result])
        runner._start_worker = Mock()
        with patch("competition_runner.time.monotonic", side_effect=[570.087,
                                                                     577.400,
                                                                     579.000]):
            runner._tick()
            runner._tick()
            runner._tick()
        self.assertEqual(runner.task.summary()["delivered"], 1)
        worker.terminate.assert_not_called()
        runner._start_worker.assert_not_called()
        self.assertEqual(runner.task.orders[0].attempts, 1)

    def test_hard_deadline_boundary_is_inclusive(self):
        runner, result = terminal_runner(600.0)
        runner._finish_worker(
            0, result=result,
            inspected=runner._inspect_terminal_result(result))
        self.assertEqual(runner.task.summary()["delivered"], 1)

    def test_completion_after_hard_deadline_does_not_score(self):
        runner, result = terminal_runner(600.001)
        runner._finish_worker(
            0, result=result,
            inspected=runner._inspect_terminal_result(result))
        self.assertEqual(runner.task.summary()["delivered"], 0)
        self.assertEqual(runner.task.orders[0].errors[-1],
                         "completion_after_hard_deadline")

    def test_duplicate_delivered_is_idempotent(self):
        runner, result = terminal_runner()
        inspected = runner._inspect_terminal_result(result)
        runner._finish_worker(0, result=result, inspected=inspected)
        attempts = runner.task.orders[0].attempts
        delivered = runner.task.summary()["delivered"]
        runner.current_order = runner.task.orders[0]
        runner.current_attempt_id = result["attempt_id"]
        runner.current_worker_binding = {
            "run_prefix": "run_modes", "generation": 3,
            "order_id": "tissue", "attempt_id": result["attempt_id"],
            "worker_pid": 123, "result_path": "exclusive-result.json"}
        runner._finish_worker(
            0, result=result,
            inspected=runner._inspect_terminal_result(result))
        self.assertEqual(runner.task.orders[0].attempts, attempts)
        self.assertEqual(runner.task.summary()["delivered"], delivered)

    def test_conflicting_same_attempt_result_fails_closed(self):
        runner, result = terminal_runner()
        self.assertTrue(runner._inspect_terminal_result(result)["valid"])
        conflict = dict(result, status="failed")
        inspected = runner._inspect_terminal_result(conflict)
        self.assertFalse(inspected["valid"])
        self.assertEqual(inspected["reason"],
                         "same_attempt_conflicting_result")

    def test_score_first_defer_does_not_claim_terminal_attempt_identity(self):
        runner, result = terminal_runner()
        runner.worker_stop_reason = "score_first_defer"
        runner.deferred_orders = {}
        runner.orders_deferred_once = set()
        runner.hint_sent_for_order = None
        runner.terminal_result_fingerprints[result["attempt_id"]] = "old"
        runner._finish_worker(
            1, result=result,
            inspected={"valid": False,
                       "reason": "worker_deferred_nonterminal"})
        self.assertEqual(runner.task.orders[0].attempts, 0)
        self.assertNotIn(result["attempt_id"],
                         runner.terminal_result_fingerprints)
        self.assertNotIn(result["attempt_id"], runner.attempt_terminal_ids)

    def test_stale_attempt_and_prior_generation_results_are_rejected(self):
        runner, result = terminal_runner()
        stale = dict(result, attempt_id="run_modes:tissue:attempt-0")
        inspected = runner._inspect_terminal_result(stale)
        self.assertFalse(inspected["valid"])
        self.assertEqual(inspected["reason"], "stale_attempt_result")

        runner, result = terminal_runner()
        old = dict(result, generation=2)
        inspected = runner._inspect_terminal_result(old)
        self.assertFalse(inspected["valid"])
        self.assertEqual(inspected["reason"], "prior_generation_result")
    def test_preferred_scan_fails_fast_only_after_last_local_pose(self):
        self.assertTrue(preferred_local_scan_exhausted(
            7, pick.STATE_SCAN, 5, 6, 0, pick.STATE_GO_SCAN, None))
        self.assertFalse(preferred_local_scan_exhausted(
            7, pick.STATE_SCAN, 4, 6, 5, pick.STATE_GO_SCAN, None))
        self.assertFalse(preferred_local_scan_exhausted(
            None, pick.STATE_SCAN, 5, 6, 0, pick.STATE_GO_SCAN, None))
        self.assertFalse(preferred_local_scan_exhausted(
            7, pick.STATE_SCAN, 5, 6, 0, pick.STATE_ALIGN,
            [1.0, 2.0, 3.0]))

    def test_candidate_view_is_derived_from_public_marker_world(self):
        hint = derive_candidate_view_hint([0.02, 3.18, 0.83])
        self.assertEqual(hint["hint_source"], "DERIVED_VIEW_HINT")
        self.assertEqual(hint["scan_station_hint"]["index"], 2)
        self.assertEqual(hint["head_pose_hint"][0], "overview_down")
        self.assertIsNone(hint["observation_base_pose_hint"])

    def test_candidate_replay_uses_one_primary_and_one_backup_pose(self):
        poses = candidate_replay_poses({
            "provisional_marker_world": [0.02, 3.18, 0.83],
            "head_pose_hint": ["overview_down", 0.11, 0.0, -0.65],
        })
        self.assertEqual([pose[0] for pose in poses], [
            "overview_down", "overview_mid"])

    def test_trace_failure_is_classified_at_unchanged_sample_gate(self):
        stage, reason = candidate_failure_stage_reason({
            "target_kind_detection_count": 8,
            "aruco_detection_count": 10,
            "association_candidate_count": 3,
            "max_association_confirmations": 3,
            "max_marker_samples": 1,
        })
        self.assertEqual(stage, "sample_collection")
        self.assertEqual(
            reason, "insufficient_marker_samples_before_pose_advance")

    def test_off_never_authorizes_inventory_reuse(self):
        runner = bare_runner("off", strict_failure_memory_mode="off")
        order, marker = runner._select_order()
        self.assertEqual(order.id, "bottle")
        self.assertIsNone(marker)
        self.assertEqual(
            runner.last_selection["strategy"], "baseline_grasp_cost")

    def test_run_inventory_selects_minimum_estimated_completion(self):
        runner = bare_runner(
            "run_inventory", strict_failure_memory_mode="off")
        order, marker = runner._select_order()
        self.assertEqual(order.id, "tissue")
        self.assertEqual(marker, 7)
        self.assertEqual(
            runner.last_selection["strategy"],
            "minimum_estimated_completion_time")

    def test_r9_all_candidates_are_globally_ranked_not_one_per_order(self):
        runner = bare_runner(
            "run_inventory", strict_failure_memory_mode="off")
        runner.args.match_timeout = 600.0
        runner.task_started_at = 0.0
        runner.scan_coverage = set()
        runner.candidate_attempt_memory = CandidateAttemptMemory()
        runner.inventory[8] = dict(
            runner.inventory[7], marker_id=8, provisional_marker_id=8,
            candidate_id="candidate-8", candidate_created_monotonic_s=2.0)
        runner.inventory[7]["candidate_created_monotonic_s"] = 1.0
        with patch("competition_runner.time.monotonic", return_value=100.0):
            options = runner._score_first_options()
            selected = score_first_select_order(options)
        tissue = [item for item in options if item.order_id == "tissue"]
        self.assertEqual({item.marker_id for item in tissue}, {7, 8})
        self.assertEqual(selected.marker_id, 7)

    def test_r9_attempt_count_alone_does_not_remove_pending_order(self):
        runner = bare_runner(
            "run_inventory", strict_failure_memory_mode="off")
        runner.args.match_timeout = 600.0
        runner.task_started_at = 0.0
        runner.scan_coverage = set()
        runner.candidate_attempt_memory = CandidateAttemptMemory()
        runner.task.orders[0].attempts = 99
        with patch("competition_runner.time.monotonic", return_value=100.0):
            options = runner._score_first_options()
        self.assertTrue(any(item.order_id == "tissue" for item in options))

    def test_non_available_candidate_is_not_reused(self):
        runner = bare_runner(
            "run_inventory", strict_failure_memory_mode="off")
        runner.inventory[7]["state"] = "DELIVERED"
        _, marker = runner._select_order()
        self.assertIsNone(marker)

    def test_process_failure_does_not_punish_candidate_semantics(self):
        runner = bare_runner(
            "run_inventory", strict_failure_memory_mode="off")
        runner.current_order = runner.task.orders[0]
        runner.preferred_marker_id = 7
        runner.current_candidate = runner.inventory[7]
        runner.worker_result_path = None
        runner.worker_stop_reason = None
        runner.worker = Mock()
        runner.worker_started_at = 1.0
        runner.worker_terminate_at = None
        runner.current_order_sequence = 1
        runner.event_log = Mock()
        runner._write_summary = Mock()
        runner._publish_stop = Mock()
        runner.get_logger = Mock(return_value=SimpleNamespace(
            info=Mock(), error=Mock()))

        runner._finish_worker(2)

        self.assertEqual(
            runner.inventory[7]["state"], PROVISIONAL_VIEW_HINT)
        # A process failure never turns a provisional view hint into an
        # excluded semantic identity.
        self.assertEqual(runner.task.excluded_markers("zhijin"), [])
        self.assertIsNone(runner.preferred_marker_id)


if __name__ == "__main__":
    unittest.main()
