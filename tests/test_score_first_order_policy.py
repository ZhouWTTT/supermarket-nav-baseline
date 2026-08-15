import pathlib
import sys
import unittest


MODULE_DIR = (
    pathlib.Path(__file__).resolve().parents[1]
    / "examples" / "supermarket_sorting")
sys.path.insert(0, str(MODULE_DIR))

from score_first_order_policy import (  # noqa: E402
    DEFERRED,
    PENDING,
    READY_PROVISIONAL,
    READY_VALIDATED,
    OrderOption,
    candidate_state,
    select_order,
    stronger_ready_order,
)


def option(order_id, source_index, state, cost=100.0, count=1, attempts=0,
           **updates):
    return OrderOption(
        order_id=order_id,
        source_index=source_index,
        attempts=attempts,
        candidate_state=state,
        candidate_count=count,
        marker_id=None,
        cost_components={},
        estimated_completion_s=cost,
        candidate_id=updates.get("candidate_id", f"candidate-{order_id}"),
        fingerprint_digest=updates.get("fingerprint_digest", order_id),
        fingerprint_status=updates.get("fingerprint_status", "UNTRIED"),
        new_evidence=updates.get("new_evidence", False),
        context_complete=updates.get("context_complete", True),
        deadline_feasible=updates.get("deadline_feasible", True),
        deadline_slack_s=updates.get("deadline_slack_s", 100.0),
        evidence_created_s=updates.get("evidence_created_s", 0.0),
        association_pair_count=updates.get("association_pair_count", 0),
        sync_delta_ns=updates.get("sync_delta_ns", 2**63 - 1),
        confirmation_count=updates.get("confirmation_count", 0),
    )


class ScoreFirstOrderPolicyTests(unittest.TestCase):
    def test_validated_beats_cheaper_provisional(self):
        selected = select_order([
            option("provisional", 0, READY_PROVISIONAL, cost=10.0),
            option("validated", 1, READY_VALIDATED, cost=90.0),
        ])
        self.assertEqual(selected.order_id, "validated")

    def test_provisional_beats_uncovered_discovery(self):
        selected = select_order([
            option("discovery", 0, PENDING, cost=5.0),
            option("provisional", 1, READY_PROVISIONAL, cost=80.0),
        ])
        self.assertEqual(selected.order_id, "provisional")

    def test_no_evidence_tie_is_stable_task_order_not_product_kind(self):
        selected = select_order([
            option("second", 1, PENDING, cost=296.0),
            option("first", 0, PENDING, cost=296.0),
        ])
        self.assertEqual(selected.order_id, "first")

    def test_new_evidence_reactivates_deferred_order(self):
        self.assertEqual(candidate_state(
            validated=False, provisional=False, deferred=True), DEFERRED)
        self.assertEqual(candidate_state(
            validated=False, provisional=True, deferred=True),
            READY_PROVISIONAL)

    def test_ready_alternative_defers_only_candidate_less_current(self):
        ready = option("ready", 1, READY_PROVISIONAL)
        self.assertEqual(stronger_ready_order(
            "current", False, [option("current", 0, PENDING), ready]), ready)
        self.assertIsNone(stronger_ready_order(
            "current", True, [option("current", 0, PENDING), ready]))

    def test_deadline_infeasible_candidate_is_skipped_for_feasible(self):
        selected = select_order([
            option("infeasible", 0, READY_VALIDATED, cost=10.0,
                   deadline_feasible=False),
            option("feasible", 1, READY_PROVISIONAL, cost=100.0),
        ])
        self.assertEqual(selected.order_id, "feasible")

    def test_all_deadline_infeasible_returns_none(self):
        self.assertIsNone(select_order([
            option("a", 0, READY_PROVISIONAL, deadline_feasible=False),
            option("b", 1, PENDING, deadline_feasible=False),
        ]))

    def test_suppressed_fingerprint_is_not_selected(self):
        selected = select_order([
            option("repeat", 0, READY_VALIDATED, cost=1.0,
                   fingerprint_status="SUPPRESSED"),
            option("fresh", 1, READY_PROVISIONAL, cost=100.0),
        ])
        self.assertEqual(selected.order_id, "fresh")

    def test_stable_tie_break_does_not_use_kind_or_list_order(self):
        selected = select_order([
            option("z", 0, READY_PROVISIONAL, fingerprint_digest="b"),
            option("a", 9, READY_PROVISIONAL, fingerprint_digest="a"),
        ])
        self.assertEqual(selected.order_id, "a")

    def test_context_complete_provisional_beats_derived(self):
        complete = option("complete", 1, READY_PROVISIONAL,
                          context_complete=True)
        derived = option("derived", 0, READY_PROVISIONAL,
                         context_complete=False)
        self.assertEqual(select_order([derived, complete]).order_id,
                         "complete")

    def test_strict_rejection_is_never_overridden_by_quality_rank(self):
        rejected = option(
            "rejected", 0, READY_VALIDATED, fingerprint_status="SUPPRESSED",
            context_complete=True, association_pair_count=99,
            confirmation_count=99, sync_delta_ns=0)
        eligible = option("eligible", 1, READY_PROVISIONAL)
        self.assertEqual(select_order([rejected, eligible]).order_id,
                         "eligible")

    def test_association_sync_and_confirmation_quality_order(self):
        weak = option("weak", 0, READY_PROVISIONAL,
                      association_pair_count=2, sync_delta_ns=100,
                      confirmation_count=3)
        strong = option("strong", 1, READY_PROVISIONAL,
                        association_pair_count=3, sync_delta_ns=150,
                        confirmation_count=4)
        self.assertEqual(select_order([weak, strong]).order_id, "strong")


if __name__ == "__main__":
    unittest.main()
