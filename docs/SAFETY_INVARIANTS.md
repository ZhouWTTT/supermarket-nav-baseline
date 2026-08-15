# Safety invariants

These values are release-blocking contracts, not tuning suggestions.

```text
WORLD_MEMORY_CAN_AUTHORIZE_MOTION=false
WORLD_MEMORY_CAN_AUTHORIZE_GRASP=false
ASSOCIATION_CONFIRMATIONS_REQUIRED=3
MARKER_SAMPLES_REQUIRED=5
DEPTH_SAMPLES_REQUIRED=5
MARKER_SAMPLE_SPREAD_MAX_M=0.04
DEPTH_SAMPLE_SPREAD_MAX_M=0.04
exact pregrasp revalidation=true
preferred_marker_id hard lock=false
```

The implementation names the depth contracts
`DEPTH_TARGET_MIN_SAMPLES` and `DEPTH_TARGET_SPREAD_MAX_M`. A preferred marker
is a replay/viewpoint hint; final motion and grasp still require current-frame
kind/marker association, strict confirmation and sample gates, bounded spread,
and the close pregrasp recheck. Failure is closed: missing, stale, dispersed,
or mismatched evidence does not create a target or authorize an action.

WM0/WM1 may retain observations and outcomes only within the active run.
WM2-A may report formal evidence but is not a control authority. Cross-run
state, evidence artifacts, or audit-tool availability cannot authorize motion
or grasp.

The default strict stability trace is `off`. Summary/full trace is diagnostic
only and owns no acceptance threshold. Logging failure must be contained by
safe logger wrappers or best-effort telemetry paths and must not terminate the
robot control thread.
