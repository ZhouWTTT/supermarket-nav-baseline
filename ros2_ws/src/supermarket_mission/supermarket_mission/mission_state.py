"""Finite mission-level state independent from navigation recovery state."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto


class MissionPhase(Enum):
    WAIT_TASK = auto()
    SELECT_ORDER = auto()
    NAV_SHELF = auto()
    VERIFY_SLOT = auto()
    PICK = auto()
    NAV_DELIVERY = auto()
    PLACE = auto()
    UPDATE_MEMORY = auto()


ALLOWED = {
    MissionPhase.WAIT_TASK: {MissionPhase.SELECT_ORDER},
    MissionPhase.SELECT_ORDER: {MissionPhase.NAV_SHELF, MissionPhase.WAIT_TASK},
    MissionPhase.NAV_SHELF: {
        MissionPhase.VERIFY_SLOT, MissionPhase.PICK, MissionPhase.SELECT_ORDER,
    },
    MissionPhase.VERIFY_SLOT: {MissionPhase.PICK, MissionPhase.SELECT_ORDER},
    MissionPhase.PICK: {MissionPhase.NAV_DELIVERY, MissionPhase.SELECT_ORDER},
    MissionPhase.NAV_DELIVERY: {MissionPhase.PLACE, MissionPhase.SELECT_ORDER},
    MissionPhase.PLACE: {MissionPhase.UPDATE_MEMORY, MissionPhase.SELECT_ORDER},
    MissionPhase.UPDATE_MEMORY: {MissionPhase.SELECT_ORDER, MissionPhase.WAIT_TASK},
}


@dataclass
class MissionState:
    phase: MissionPhase = MissionPhase.WAIT_TASK
    run_prefix: str = ""
    order_id: str = ""
    history: list[tuple[MissionPhase, MissionPhase]] = field(default_factory=list)

    def transition(self, phase: MissionPhase) -> None:
        if phase is self.phase:
            return
        if phase not in ALLOWED[self.phase]:
            raise RuntimeError(
                f"illegal mission transition {self.phase.name} -> {phase.name}"
            )
        self.history.append((self.phase, phase))
        self.phase = phase

    def new_run(self, run_prefix: str) -> None:
        self.phase = MissionPhase.WAIT_TASK
        self.run_prefix = str(run_prefix)
        self.order_id = ""
        self.history.clear()
        self.transition(MissionPhase.SELECT_ORDER)

    def select(self, order_id: str) -> None:
        if self.phase is not MissionPhase.SELECT_ORDER:
            raise RuntimeError("orders can only be selected in SELECT_ORDER")
        self.order_id = str(order_id)
        self.transition(MissionPhase.NAV_SHELF)

    def release_order(self) -> None:
        if self.phase not in {
            MissionPhase.NAV_SHELF, MissionPhase.VERIFY_SLOT, MissionPhase.PICK,
            MissionPhase.NAV_DELIVERY, MissionPhase.PLACE,
        }:
            raise RuntimeError("no active order to release")
        self.order_id = ""
        self.transition(MissionPhase.SELECT_ORDER)


@dataclass
class FirstOrderScanPolicy:
    """Per-run owner of wxj's E-first then A-first full-scan policy."""

    first_order_id: str | None = None
    later_order_started: bool = False

    def reset(self) -> None:
        self.first_order_id = None
        self.later_order_started = False

    def prefer_west(self, order_id: str, has_memory_hint: bool) -> bool:
        if self.first_order_id is None:
            self.first_order_id = str(order_id)
        elif str(order_id) != self.first_order_id:
            self.later_order_started = True
        return self.later_order_started and not bool(has_memory_hint)
