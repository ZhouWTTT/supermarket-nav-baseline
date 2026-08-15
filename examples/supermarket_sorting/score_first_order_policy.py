#!/usr/bin/env python3
"""Pure evidence-first order scheduling for a bounded competition match."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable

from strict_replay_outcome_memory import strict_state_priority


PENDING = "PENDING"
READY_VALIDATED = "READY_VALIDATED"
READY_PROVISIONAL = "READY_PROVISIONAL"
IN_PROGRESS = "IN_PROGRESS"
DEFERRED = "DEFERRED"
DELIVERED = "DELIVERED"
FAILED = "FAILED"

_EVIDENCE_PRIORITY = {
    READY_VALIDATED: 0,
    READY_PROVISIONAL: 1,
    PENDING: 2,
    DEFERRED: 3,
}


@dataclass(frozen=True)
class OrderOption:
    """One pending order described only by current-run evidence and cost."""

    order_id: str
    source_index: int
    attempts: int
    candidate_state: str
    candidate_count: int
    marker_id: int | None
    cost_components: dict[str, float]
    estimated_completion_s: float
    candidate_id: str = "DISCOVERY"
    fingerprint_digest: str = ""
    fingerprint_status: str = "UNTRIED"
    new_evidence: bool = False
    reactivation_reason: str | None = None
    context_complete: bool = False
    deadline_feasible: bool = True
    remaining_hard_s: float = math.inf
    deadline_slack_s: float = math.inf
    evidence_created_s: float = math.inf
    association_pair_count: int = 0
    sync_delta_ns: int = 2**63 - 1
    confirmation_count: int = 0
    strict_memory_state: str = "UNTRIED"
    strict_retry_allowed: bool = True
    strict_control_active: bool = False
    strict_candidate_selection_reason: str = "strict_memory_not_applicable"
    strict_failure_equivalence_digest: str | None = None
    strict_failure_material_revision: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "order_id": self.order_id,
            "candidate_state": self.candidate_state,
            "candidate_count": self.candidate_count,
            "marker_id": self.marker_id,
            "candidate_id": self.candidate_id,
            "fingerprint_digest": self.fingerprint_digest,
            "fingerprint_status": self.fingerprint_status,
            "new_evidence": self.new_evidence,
            "reactivation_reason": self.reactivation_reason,
            "context_complete": self.context_complete,
            "deadline_feasible": self.deadline_feasible,
            "remaining_hard_s": self.remaining_hard_s,
            "deadline_slack_s": self.deadline_slack_s,
            "evidence_created_s": self.evidence_created_s,
            "association_pair_count": self.association_pair_count,
            "sync_delta_ns": self.sync_delta_ns,
            "confirmation_count": self.confirmation_count,
            "strict_memory_state": self.strict_memory_state,
            "strict_retry_allowed": self.strict_retry_allowed,
            "strict_control_active": self.strict_control_active,
            "strict_candidate_selection_reason": (
                self.strict_candidate_selection_reason),
            "strict_failure_equivalence_digest": (
                self.strict_failure_equivalence_digest),
            "strict_failure_material_revision": (
                self.strict_failure_material_revision),
            "cost_components": dict(self.cost_components),
            "estimated_completion_s": self.estimated_completion_s,
        }


def candidate_state(*, validated: bool, provisional: bool,
                    deferred: bool) -> str:
    """Return the state with new evidence reactivating a deferred order."""
    if validated:
        return READY_VALIDATED
    if provisional:
        return READY_PROVISIONAL
    return DEFERRED if deferred else PENDING


def select_order(options: Iterable[OrderOption]) -> OrderOption | None:
    """Select a feasible unfailed fingerprint using an interpretable order."""
    options = [
        option for option in options
        if option.deadline_feasible
        and option.fingerprint_status != "SUPPRESSED"
    ]
    if not options:
        return None

    def key(option: OrderOption) -> tuple:
        cost = float(option.estimated_completion_s)
        if not math.isfinite(cost):
            cost = math.inf
        if option.strict_control_active:
            policy_prefix = (
                strict_state_priority(option.strict_memory_state),
                0 if option.context_complete else 1,
                _EVIDENCE_PRIORITY[option.candidate_state],
            )
        else:
            policy_prefix = (
                0,
                _EVIDENCE_PRIORITY[option.candidate_state],
                0 if option.context_complete else 1,
            )
        return (
            # Shadow/off modes retain the established ordering byte-for-byte.
            # Control mode inserts only the frozen, kind-agnostic strict tier.
            *policy_prefix,
            0 if option.new_evidence else 1,
            0 if option.fingerprint_status == "UNTRIED" else 1,
            -max(0, int(option.association_pair_count)),
            max(0, int(option.sync_delta_ns)),
            -max(0, int(option.confirmation_count)),
            cost,
            float(option.evidence_created_s),
            str(option.fingerprint_digest),
        )

    return min(options, key=key)


def stronger_ready_order(
        current_order_id: str, current_has_candidate: bool,
        options: Iterable[OrderOption]) -> OrderOption | None:
    """Return a ready alternative that justifies a scheduler-only defer."""
    if current_has_candidate:
        return None
    ready = [
        option for option in options
        if option.order_id != current_order_id
        and option.candidate_state in {READY_VALIDATED, READY_PROVISIONAL}
    ]
    return select_order(ready)
