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

from competition_task import Order, prefer_non_rerouted  # noqa: E402


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


if __name__ == "__main__":
    test_non_rerouted_orders_are_preferred()
    test_mislocalised_order_is_deferred_not_dropped()
    test_all_rerouted_falls_back_to_full_candidate_set()
    test_empty_reroute_set_keeps_candidates()
    print("test_order_reroute OK")
