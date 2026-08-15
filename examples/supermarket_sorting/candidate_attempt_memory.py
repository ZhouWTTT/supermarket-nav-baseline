#!/usr/bin/env python3
"""Pure run-level candidate attempt memory and deadline feasibility.

All identities are canonical JSON values.  No ROS messages, numpy arrays, or
process-local object identities enter a fingerprint.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence


STRICT_POLICY_VERSION = "strict-localization-r8"
DISCOVERY_CANDIDATE_ID = "DISCOVERY"


def _canonical(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canonical(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    return str(value)


def stable_digest(value: Any) -> str:
    encoded = json.dumps(
        _canonical(value), ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _pose_plan(candidate: Mapping[str, Any] | None) -> tuple[str, tuple[str, ...]]:
    if candidate is None:
        return "full_scan", ()
    requested = candidate.get("head_pose_hint")
    primary = (str(requested[0]) if isinstance(requested, (list, tuple))
               and len(requested) == 4 else "derived")
    backup = {
        "overview_high": "overview_mid",
        "overview_mid": "overview_high",
        "overview_down": "overview_mid",
        "lower_center": "lower_yaw_minus",
        "lower_yaw_minus": "lower_center",
        "lower_yaw_plus": "lower_center",
    }.get(primary, "derived_backup")
    return primary, (backup,)


def _source_pair(candidate: Mapping[str, Any] | None) -> tuple[int, int] | None:
    if candidate is None:
        return None
    try:
        return (int(candidate["source_yolo_stamp_ns"]),
                int(candidate["source_aruco_stamp_ns"]))
    except (KeyError, TypeError, ValueError):
        observed = candidate.get("observed_source_stamps")
        if isinstance(observed, (list, tuple)) and len(observed) == 2:
            try:
                return int(observed[0]), int(observed[1])
            except (TypeError, ValueError):
                pass
        return None


def _context_document(candidate: Mapping[str, Any] | None) -> dict[str, Any]:
    if candidate is None:
        return {"context_type": "DISCOVERY", "coverage_revision": 0}
    keys = (
        "context_type", "context_source", "context_quality",
        "observed_base_pose", "observed_head_pose", "observed_scan_station",
        "observed_pose_name", "observed_source_stamps", "target_bbox_summary",
        "marker_pixel_summary", "association_summary", "controller_state",
        "scan_index", "pitch_index", "camera_settled",
    )
    return {key: candidate.get(key) for key in keys}


@dataclass(frozen=True)
class CandidateAttemptFingerprint:
    run_prefix: str
    order_id: str
    candidate_id: str
    product_kind: str
    marker_id: int | None
    candidate_evidence_revision: str
    candidate_context_digest: str
    candidate_source_stamp_pair: tuple[int, int] | None
    replay_primary_pose_id: str
    replay_backup_pose_ids: tuple[str, ...]
    strict_policy_version: str
    navigation_request_revision: int

    @property
    def digest(self) -> str:
        return stable_digest(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return _canonical(asdict(self))


def make_fingerprint(
        *, run_prefix: str, order_id: str, product_kind: str,
        candidate: Mapping[str, Any] | None,
        navigation_request_revision: int = 0,
        coverage_revision: int = 0) -> CandidateAttemptFingerprint:
    primary, backups = _pose_plan(candidate)
    context = _context_document(candidate)
    if candidate is None:
        context["coverage_revision"] = int(coverage_revision)
    pair = _source_pair(candidate)
    candidate_id = (DISCOVERY_CANDIDATE_ID if candidate is None else str(
        candidate.get("candidate_id") or
        f"candidate-{candidate.get('marker_id', 'unknown')}"))
    try:
        marker_id = None if candidate is None else int(candidate["marker_id"])
    except (KeyError, TypeError, ValueError):
        marker_id = None
    evidence_document = {
        "candidate_id": candidate_id,
        "source_pair": pair,
        "context": context,
        "confirmations": (0 if candidate is None else int(
            candidate.get("confirmations", 0))),
        "coverage_revision": int(coverage_revision),
    }
    return CandidateAttemptFingerprint(
        run_prefix=str(run_prefix), order_id=str(order_id),
        candidate_id=candidate_id, product_kind=str(product_kind),
        marker_id=marker_id,
        candidate_evidence_revision=stable_digest(evidence_document),
        candidate_context_digest=stable_digest(context),
        candidate_source_stamp_pair=pair,
        replay_primary_pose_id=primary,
        replay_backup_pose_ids=backups,
        strict_policy_version=STRICT_POLICY_VERSION,
        navigation_request_revision=int(navigation_request_revision),
    )


@dataclass(frozen=True)
class CandidateAttemptOutcome:
    fingerprint: CandidateAttemptFingerprint
    failure_stage: str
    failure_reason: str
    terminal_s: float
    evidence_revision: str
    candidate_state_after_failure: str
    reactivation_requirements: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["fingerprint_digest"] = self.fingerprint.digest
        return _canonical(value)


@dataclass(frozen=True)
class RetryDecision:
    allowed: bool
    reason: str
    new_evidence: bool
    previous_outcome: CandidateAttemptOutcome | None = None


def reactivation_requirements(stage: str, reason: str) -> tuple[str, ...]:
    stage, reason = str(stage), str(reason)
    if reason in {"no_fresh_rgb", "rgb_not_processed_by_yolo"}:
        return ("sensor_recovered",)
    if reason == "no_target_kind" or stage == "target_kind_detection":
        return ("new_context", "new_source_pair", "new_backup_pose",
                "evidence_revision_increased")
    if reason in {
            "no_association", "insufficient_confirmations",
            "insufficient_marker_samples_before_pose_advance",
            "scan_timeout_before_required_samples", "spread_reject"}:
        return ("new_independent_pair", "new_context", "new_backup_pose",
                "new_validated_candidate")
    if stage in {"navigation", "delivery_navigation"} or reason in {
            "no_path", "navigation_timeout"}:
        return ("navigation_revision_changed", "new_path_result",
                "recovery_completed")
    if stage in {"worker", "process"} or reason in {
            "worker_crash", "process_error"}:
        return ("new_attempt_identity",)
    return ("new_evidence_revision", "new_candidate")


class CandidateAttemptMemory:
    """Idempotent memory keyed by canonical candidate fingerprints."""

    def __init__(self) -> None:
        self._outcomes: dict[str, CandidateAttemptOutcome] = {}
        self._latest_by_candidate: dict[tuple[str, str], CandidateAttemptOutcome] = {}

    @property
    def outcomes(self) -> tuple[CandidateAttemptOutcome, ...]:
        return tuple(self._outcomes[key] for key in sorted(self._outcomes))

    def decision(self, fingerprint: CandidateAttemptFingerprint) -> RetryDecision:
        exact = self._outcomes.get(fingerprint.digest)
        if exact is not None:
            if exact.failure_stage in {"worker", "process"} or exact.failure_reason in {
                    "worker_crash", "process_error"}:
                return RetryDecision(True, "process_failure_new_attempt", False, exact)
            return RetryDecision(False, "same_fingerprint_no_new_evidence",
                                 False, exact)
        previous = self._latest_by_candidate.get(
            (fingerprint.order_id, fingerprint.candidate_id))
        if previous is None:
            return RetryDecision(True, "fingerprint_untried", False)
        old = previous.fingerprint
        changes = []
        if fingerprint.candidate_source_stamp_pair != old.candidate_source_stamp_pair:
            changes.append("new_source_pair")
        if fingerprint.candidate_context_digest != old.candidate_context_digest:
            changes.append("new_context")
        if fingerprint.replay_backup_pose_ids != old.replay_backup_pose_ids:
            changes.append("new_backup_pose")
        if (fingerprint.candidate_evidence_revision
                != old.candidate_evidence_revision):
            changes.append("evidence_revision_increased")
        if (fingerprint.navigation_request_revision
                != old.navigation_request_revision):
            changes.append("navigation_revision_changed")
        if changes:
            return RetryDecision(True, "+".join(changes), True, previous)
        return RetryDecision(False, "candidate_changed_without_required_evidence",
                             False, previous)

    def record(self, outcome: CandidateAttemptOutcome) -> bool:
        key = outcome.fingerprint.digest
        if key in self._outcomes:
            return False
        self._outcomes[key] = outcome
        self._latest_by_candidate[(outcome.fingerprint.order_id,
                                   outcome.fingerprint.candidate_id)] = outcome
        return True


@dataclass(frozen=True)
class CompletionEstimate:
    estimated_to_reacquire_localize_grasp_s: float
    estimated_delivery_s: float
    estimated_place_s: float
    safety_margin_s: float
    estimate_sample_count: int
    estimate_source: str

    @property
    def estimated_completion_s(self) -> float:
        return round(
            self.estimated_to_reacquire_localize_grasp_s
            + self.estimated_delivery_s + self.estimated_place_s
            + self.safety_margin_s, 3)

    def feasibility(self, remaining_hard_s: float) -> dict[str, Any]:
        remaining = max(0.0, float(remaining_hard_s))
        slack = round(remaining - self.estimated_completion_s, 3)
        return {
            **asdict(self),
            "estimated_completion_s": self.estimated_completion_s,
            "remaining_hard_s": round(remaining, 3),
            "deadline_slack_s": slack,
            "deadline_feasible": slack >= 0.0,
        }


CALIBRATED_CANDIDATE_ESTIMATE = CompletionEstimate(
    estimated_to_reacquire_localize_grasp_s=75.0,
    estimated_delivery_s=111.0,
    estimated_place_s=41.0,
    safety_margin_s=6.0,
    estimate_sample_count=3,
    estimate_source=(
        "R8-B/R8-C successful attempt-to-grasp upper bound; "
        "R6/R8-B successful delivery/place; censored stages excluded"),
)

CALIBRATED_DISCOVERY_ESTIMATE = CompletionEstimate(
    estimated_to_reacquire_localize_grasp_s=255.0,
    estimated_delivery_s=111.0,
    estimated_place_s=41.0,
    safety_margin_s=6.0,
    estimate_sample_count=3,
    estimate_source=(
        "R6 successful discovery chain plus R8-B/R8-C stage telemetry; "
        "censored stages excluded"),
)


def completion_estimate(candidate_available: bool) -> CompletionEstimate:
    return (CALIBRATED_CANDIDATE_ESTIMATE if candidate_available
            else CALIBRATED_DISCOVERY_ESTIMATE)
