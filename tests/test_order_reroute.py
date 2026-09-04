"""Host-side tests for the dynamic order reroute scheduling helper.

``competition_task`` is deliberately ROS-free, so this file can import it
directly on the host without the simulator stack.
"""

from __future__ import annotations

from pathlib import Path
import sys


sys.path.insert(
    0,
    str(Path(__file__).parents[1] / "examples" / "supermarket_sorting"))

from competition_task import (  # noqa: E402
    CompetitionTask,
    Order,
    prefer_non_rerouted,
    select_nearest_candidate,
)


def _orders():
    return [
        Order("o1", "kele", 0),
        Order("o2", "chengzi", 1),
        Order("o3", "shupian", 2),
    ]


def test_non_rerouted_orders_are_preferred():
    orders = _orders()
    result = prefer_non_rerouted(orders, {"o1"})
    assert [order.id for order in result] == ["o2", "o3"]


def test_mislocalised_order_is_deferred_not_dropped():
    orders = _orders()
    result = prefer_non_rerouted(orders, {"o1", "o2"})
    assert [order.id for order in result] == ["o3"]


def test_all_rerouted_falls_back_to_full_candidate_set():
    orders = _orders()
    result = prefer_non_rerouted(orders, {"o1", "o2", "o3"})
    assert [order.id for order in result] == ["o1", "o2", "o3"]


def test_empty_reroute_set_keeps_candidates():
    orders = _orders()
    assert prefer_non_rerouted(orders, set()) is orders
    assert prefer_non_rerouted(orders, None) is orders


def test_navigation_only_reroute_preserves_grasp_attempt_budget():
    orders = _orders()
    task = CompetitionTask("run", orders)

    task.defer_order(orders[0], "reroute:direct_slot_missing")

    assert orders[0].status == "pending"
    assert orders[0].attempts == 0
    assert orders[0].errors == ["reroute:direct_slot_missing"]


def test_nearest_visible_pending_product_wins_over_confidence():
    candidates = [
        {
            "kind": "shupian",
            "target_xy": (1.5, 2.5),
            "confidence": 0.99,
        },
        {
            "kind": "chengzi",
            "target_xy": (0.2, 2.5),
            "confidence": 0.72,
        },
    ]

    selected = select_nearest_candidate(candidates, (0.0, 2.5))

    assert selected is candidates[1]


def test_nearest_candidate_rejects_non_finite_motion_targets():
    valid = {"kind": "kele", "target_xy": (0.4, 2.5), "confidence": 0.8}
    candidates = [
        {"kind": "bad", "target_xy": (float("nan"), 2.5)},
        {"kind": "missing"},
        valid,
    ]

    assert select_nearest_candidate(candidates, (0.0, 2.5)) is valid
    assert select_nearest_candidate(candidates, (float("nan"), 0.0)) is None


if __name__ == "__main__":
    test_non_rerouted_orders_are_preferred()
    test_mislocalised_order_is_deferred_not_dropped()
    test_all_rerouted_falls_back_to_full_candidate_set()
    test_empty_reroute_set_keeps_candidates()
    test_navigation_only_reroute_preserves_grasp_attempt_budget()
    test_nearest_visible_pending_product_wins_over_confidence()
    test_nearest_candidate_rejects_non_finite_motion_targets()
    print("test_order_reroute OK")
