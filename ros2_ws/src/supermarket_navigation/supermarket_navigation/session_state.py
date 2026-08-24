"""Pure, deterministic state for one semantic navigation request.

Keeping recovery accounting out of ROS callbacks makes the key invariant
testable: replanning the same session must never restore a spent recovery.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
import math
import time
from typing import Iterable


class SessionPhase(Enum):
    NEW = auto()
    SELECT_ROUTE = auto()
    PLAN = auto()
    TRACK = auto()
    APPROACH = auto()
    BLOCKED = auto()
    BACKUP_ONCE = auto()
    ALTERNATE_ROUTE = auto()
    ARRIVED = auto()
    FAILED = auto()
    CANCELED = auto()


class SessionEvent(Enum):
    START = auto()
    ROUTE_SELECTED = auto()
    PLAN_READY = auto()
    PATH_INVALID = auto()
    NO_PROGRESS = auto()
    BACKUP_ALLOWED = auto()
    BACKUP_SKIPPED = auto()
    BACKUP_DONE = auto()
    ALTERNATE_AVAILABLE = auto()
    APPROACH_STARTED = auto()
    EXACT_GOAL_REACHED = auto()
    EXHAUSTED = auto()
    CANCEL = auto()


TERMINAL_PHASES = {
    SessionPhase.ARRIVED,
    SessionPhase.FAILED,
    SessionPhase.CANCELED,
}


@dataclass(frozen=True)
class BlockAnchor:
    x: float
    y: float

    def distance(self, x: float, y: float) -> float:
        return math.hypot(self.x - x, self.y - y)


@dataclass
class NavigationSessionState:
    session_id: str
    route_candidates: tuple[str, ...]
    started_at: float = field(default_factory=time.monotonic)
    no_progress_timeout_s: float = 5.0
    total_timeout_s: float = 120.0
    max_recoveries: int = 3
    block_anchor_radius_m: float = 0.30
    phase: SessionPhase = SessionPhase.NEW
    selected_route: str = ""
    route_index: int = -1
    recovery_count: int = 0
    same_route_replans: int = 0
    block_reason: str = ""
    last_progress_at: float | None = None
    last_remaining_path_m: float = math.inf
    last_progress_pose: tuple[float, float] | None = None
    displacement_progress_m: float = 0.01
    block_anchors: list[BlockAnchor] = field(default_factory=list)
    backed_up_anchor_indexes: set[int] = field(default_factory=set)
    history: list[tuple[SessionPhase, SessionEvent]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.session_id:
            raise ValueError("session_id must not be empty")
        if not self.route_candidates:
            raise ValueError("at least one route candidate is required")
        if self.no_progress_timeout_s <= 0.0 or self.total_timeout_s <= 0.0:
            raise ValueError("timeouts must be positive")
        self.last_progress_at = self.started_at

    @property
    def terminal(self) -> bool:
        return self.phase in TERMINAL_PHASES

    def transition(self, event: SessionEvent) -> SessionPhase:
        if self.terminal:
            if event is SessionEvent.CANCEL and self.phase is not SessionPhase.ARRIVED:
                self.phase = SessionPhase.CANCELED
                return self.phase
            raise RuntimeError(f"terminal session cannot handle {event.name}")

        current = self.phase
        if event is SessionEvent.CANCEL:
            next_phase = SessionPhase.CANCELED
        else:
            next_phase = self._next_phase(current, event)
        self.history.append((current, event))
        self.phase = next_phase
        return next_phase

    def _next_phase(self, phase: SessionPhase, event: SessionEvent) -> SessionPhase:
        transitions = {
            (SessionPhase.NEW, SessionEvent.START): SessionPhase.SELECT_ROUTE,
            (SessionPhase.SELECT_ROUTE, SessionEvent.ROUTE_SELECTED): SessionPhase.PLAN,
            (SessionPhase.PLAN, SessionEvent.PLAN_READY): SessionPhase.TRACK,
            (SessionPhase.PLAN, SessionEvent.ALTERNATE_AVAILABLE): SessionPhase.ALTERNATE_ROUTE,
            (SessionPhase.TRACK, SessionEvent.PATH_INVALID): SessionPhase.PLAN,
            (SessionPhase.TRACK, SessionEvent.NO_PROGRESS): SessionPhase.BLOCKED,
            (SessionPhase.TRACK, SessionEvent.APPROACH_STARTED): SessionPhase.APPROACH,
            (SessionPhase.APPROACH, SessionEvent.EXACT_GOAL_REACHED): SessionPhase.ARRIVED,
            (SessionPhase.BLOCKED, SessionEvent.BACKUP_ALLOWED): SessionPhase.BACKUP_ONCE,
            (SessionPhase.BLOCKED, SessionEvent.BACKUP_SKIPPED): SessionPhase.ALTERNATE_ROUTE,
            (SessionPhase.BACKUP_ONCE, SessionEvent.BACKUP_DONE): SessionPhase.ALTERNATE_ROUTE,
            (SessionPhase.ALTERNATE_ROUTE, SessionEvent.ALTERNATE_AVAILABLE): SessionPhase.SELECT_ROUTE,
        }
        if event is SessionEvent.EXHAUSTED:
            return SessionPhase.FAILED
        try:
            return transitions[(phase, event)]
        except KeyError as exc:
            raise RuntimeError(
                f"illegal navigation transition {phase.name} + {event.name}"
            ) from exc

    def select_next_route(self) -> str | None:
        next_index = self.route_index + 1
        if next_index >= len(self.route_candidates):
            return None
        self.route_index = next_index
        self.selected_route = self.route_candidates[next_index]
        self.same_route_replans = 0
        # Remaining arc length is route-local.  Carrying it into a longer
        # alternate route makes the fresh route look stalled immediately.
        self.last_remaining_path_m = math.inf
        self.last_progress_pose = None
        self.last_progress_at = time.monotonic()
        return self.selected_route

    def request_same_route_replan(self) -> bool:
        """Spend the one in-place replan without resetting any other budget."""
        if self.same_route_replans >= 1:
            return False
        self.same_route_replans += 1
        return True

    def update_progress(self, remaining_path_m: float, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else float(now)
        remaining = max(0.0, float(remaining_path_m))
        improved = remaining + 0.05 < self.last_remaining_path_m
        if improved:
            self.last_remaining_path_m = remaining
            self.last_progress_at = now
        return improved

    def update_displacement(
        self, x: float, y: float, now: float | None = None
    ) -> bool:
        """Use measured motion as progress between quantized path updates."""

        now = time.monotonic() if now is None else float(now)
        pose = (float(x), float(y))
        if self.last_progress_pose is None:
            self.last_progress_pose = pose
            return False
        distance = math.hypot(
            pose[0] - self.last_progress_pose[0],
            pose[1] - self.last_progress_pose[1],
        )
        if distance < self.displacement_progress_m:
            return False
        self.last_progress_pose = pose
        self.last_progress_at = now
        return True

    def no_progress(self, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else float(now)
        return now - float(self.last_progress_at) >= self.no_progress_timeout_s

    def timed_out(self, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else float(now)
        return now - self.started_at >= self.total_timeout_s

    def record_block(self, x: float, y: float, reason: str) -> int:
        self.block_reason = str(reason)
        for index, anchor in enumerate(self.block_anchors):
            if anchor.distance(x, y) <= self.block_anchor_radius_m:
                return index
        self.block_anchors.append(BlockAnchor(float(x), float(y)))
        return len(self.block_anchors) - 1

    def may_backup_at(self, anchor_index: int) -> bool:
        return (
            self.recovery_count < self.max_recoveries
            and anchor_index not in self.backed_up_anchor_indexes
        )

    def spend_backup(self, anchor_index: int) -> None:
        if not self.may_backup_at(anchor_index):
            raise RuntimeError("backup budget already exhausted for this anchor")
        self.backed_up_anchor_indexes.add(anchor_index)
        self.recovery_count += 1

    def routes_remaining(self) -> Iterable[str]:
        return self.route_candidates[self.route_index + 1 :]
