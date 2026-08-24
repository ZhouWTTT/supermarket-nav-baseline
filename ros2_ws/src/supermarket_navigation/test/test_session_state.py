import math
import random

import pytest

from supermarket_navigation.session_state import (
    NavigationSessionState,
    SessionEvent,
    SessionPhase,
)


def make_state(**kwargs):
    return NavigationSessionState(
        session_id="run/order/leg",
        route_candidates=("east", "west", "north"),
        started_at=100.0,
        **kwargs,
    )


def test_happy_path_reaches_only_exact_arrival():
    state = make_state()
    assert state.transition(SessionEvent.START) is SessionPhase.SELECT_ROUTE
    assert state.select_next_route() == "east"
    assert state.transition(SessionEvent.ROUTE_SELECTED) is SessionPhase.PLAN
    assert state.transition(SessionEvent.PLAN_READY) is SessionPhase.TRACK
    assert state.transition(SessionEvent.APPROACH_STARTED) is SessionPhase.APPROACH
    assert state.transition(SessionEvent.EXACT_GOAL_REACHED) is SessionPhase.ARRIVED
    assert state.terminal


def test_replan_preserves_backup_budget_and_block_anchors():
    state = make_state()
    state.transition(SessionEvent.START)
    state.select_next_route()
    state.transition(SessionEvent.ROUTE_SELECTED)
    state.transition(SessionEvent.PLAN_READY)
    anchor = state.record_block(1.0, 2.0, "blocked")
    state.spend_backup(anchor)

    assert state.request_same_route_replan()
    assert not state.request_same_route_replan()
    assert state.recovery_count == 1
    assert state.record_block(1.2, 2.0, "same place") == anchor
    assert not state.may_backup_at(anchor)


def test_new_route_resets_route_local_progress_clock(monkeypatch):
    state = make_state(no_progress_timeout_s=5.0)
    state.transition(SessionEvent.START)
    state.last_remaining_path_m = 0.5
    state.last_progress_at = 10.0
    monkeypatch.setattr(
        "supermarket_navigation.session_state.time.monotonic", lambda: 120.0
    )

    state.select_next_route()

    assert math.isinf(state.last_remaining_path_m)
    assert state.last_progress_at == 120.0
    assert not state.no_progress(now=124.9)


def test_odometry_displacement_counts_as_progress_between_path_updates():
    state = make_state(no_progress_timeout_s=5.0)
    state.update_displacement(1.0, 2.0, now=10.0)
    assert not state.update_displacement(1.009, 2.0, now=14.0)
    assert state.update_displacement(1.011, 2.0, now=14.5)
    assert not state.no_progress(now=19.4)
    assert state.no_progress(now=19.5)


def test_same_anchor_can_never_backup_twice():
    state = make_state()
    first = state.record_block(0.0, 0.0, "box")
    state.spend_backup(first)
    same = state.record_block(0.29, 0.0, "box still present")
    assert same == first
    assert not state.may_backup_at(same)
    with pytest.raises(RuntimeError):
        state.spend_backup(same)


def test_recovery_budget_is_global_across_anchors():
    state = make_state(max_recoveries=3)
    for x in (0.0, 1.0, 2.0):
        anchor = state.record_block(x, 0.0, "box")
        assert state.may_backup_at(anchor)
        state.spend_backup(anchor)
    fourth = state.record_block(3.0, 0.0, "box")
    assert state.recovery_count == 3
    assert not state.may_backup_at(fourth)


def test_illegal_transition_fails_closed():
    state = make_state()
    with pytest.raises(RuntimeError):
        state.transition(SessionEvent.PLAN_READY)


def test_500_randomized_recovery_sequences_always_terminate_with_bounded_budget():
    for seed in range(500):
        rng = random.Random(seed)
        state = make_state()
        state.transition(SessionEvent.START)
        for _route in range(3):
            assert state.select_next_route() is not None
            state.transition(SessionEvent.ROUTE_SELECTED)
            if rng.random() < 0.25:
                state.transition(SessionEvent.ALTERNATE_AVAILABLE)
                state.transition(SessionEvent.ALTERNATE_AVAILABLE)
                continue
            state.transition(SessionEvent.PLAN_READY)
            if rng.random() < 0.20:
                state.transition(SessionEvent.APPROACH_STARTED)
                state.transition(SessionEvent.EXACT_GOAL_REACHED)
                break
            state.transition(SessionEvent.NO_PROGRESS)
            anchor = state.record_block(
                float(rng.randrange(4)), float(rng.randrange(4)), "fuzz block"
            )
            if state.may_backup_at(anchor):
                state.transition(SessionEvent.BACKUP_ALLOWED)
                state.spend_backup(anchor)
                state.transition(SessionEvent.BACKUP_DONE)
            else:
                state.transition(SessionEvent.BACKUP_SKIPPED)
            state.transition(SessionEvent.ALTERNATE_AVAILABLE)
        if not state.terminal:
            assert state.select_next_route() is None
            state.transition(SessionEvent.EXHAUSTED)
        assert state.terminal
        assert state.recovery_count <= 3
        assert len(state.backed_up_anchor_indexes) == state.recovery_count
        assert len(state.history) <= 24
