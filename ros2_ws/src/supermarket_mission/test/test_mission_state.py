import pytest

from supermarket_mission.mission_state import (
    FirstOrderScanPolicy,
    MissionPhase,
    MissionState,
)


def test_complete_order_has_a_finite_non_overlapping_sequence():
    state = MissionState()
    state.new_run("run_one")
    state.select("order_one")
    for phase in (
        MissionPhase.VERIFY_SLOT,
        MissionPhase.PICK,
        MissionPhase.NAV_DELIVERY,
        MissionPhase.PLACE,
        MissionPhase.UPDATE_MEMORY,
        MissionPhase.SELECT_ORDER,
    ):
        state.transition(phase)
    assert state.phase is MissionPhase.SELECT_ORDER


def test_failed_order_must_be_released_before_another_is_selected():
    state = MissionState()
    state.new_run("run_one")
    state.select("first")
    with pytest.raises(RuntimeError):
        state.select("second")
    state.release_order()
    state.select("second")
    assert state.order_id == "second"


def test_place_cannot_loop_back_to_navigation():
    state = MissionState()
    state.new_run("run_one")
    state.select("first")
    state.transition(MissionPhase.PICK)
    state.transition(MissionPhase.NAV_DELIVERY)
    state.transition(MissionPhase.PLACE)
    with pytest.raises(RuntimeError):
        state.transition(MissionPhase.NAV_SHELF)


def test_first_order_retries_stay_east_then_later_no_hint_scans_start_west():
    policy = FirstOrderScanPolicy()
    assert not policy.prefer_west("order-1", has_memory_hint=False)
    assert not policy.prefer_west("order-1", has_memory_hint=False)
    assert policy.prefer_west("order-2", has_memory_hint=False)
    assert not policy.prefer_west("order-3", has_memory_hint=True)


def test_scan_policy_is_reset_between_run_prefixes():
    policy = FirstOrderScanPolicy()
    policy.prefer_west("old-1", has_memory_hint=False)
    assert policy.prefer_west("old-2", has_memory_hint=False)
    policy.reset()
    assert not policy.prefer_west("new-1", has_memory_hint=False)
