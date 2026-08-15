# Delta from the common baseline

## Base and scope

- Upload base: `efa6d15bef3e77bd90ccaec3a5bcdefbd7e956d5`.
- Upload branch: `feature/run-scoped-memory-matrix-r15b`.
- No upstream merge or rebase is included.
- Runtime evidence, snapshots, inverse patches, raw identity, and local audit
  tools are excluded from Git.

## Production delta

The new pure modules under `examples/supermarket_sorting/` provide immutable
observation context, candidate attempt fingerprints, source-state history,
scan coverage and cursor state, replay freshness and viewpoint convergence,
strict outcome memory, score-first scheduling, anytime discovery, bounded
telemetry, carrying-navigation diagnostics, and fail-closed place retry.

The integration changes are limited to:

- `competition_runner.py`: run inventory, replay context, scheduling,
  completion reserve, logger containment, and terminal accounting;
- `integrated_nav_pick_place.py`: fresh-frame replay, viewpoint/epoch handling,
  exact pregrasp result semantics, diagnostics, and place retry;
- `supermarket_navigation.py`: navigation behavior used by the execution path;
- `yolo_aruco_shelf_pick.py`: strict localization, samples, spread gates, and
  preferred-marker hint behavior;
- `scripts/run_baseline.sh`: explicit memory and candidate-budget settings.

## Compatibility delta

Normal mode has no dependency on Git HEAD, dirty fileset, formal Docker image,
external evidence, a local Windows path, GUI, or GPU-only audit tooling. Audit
output is best-effort and does not become a robot-control prerequisite.

DEV_CI covers pure Python memory, safety contracts, freshness, terminal
accounting, compilation, and diff checks. FORMAL_CI/manual validation adds the
exact image, the historical 224-test command, runtime evidence, seed ledgers,
formal runs, and identity auditing. Default PR CI does not require a simulator,
competition server, GUI, GPU, pre-existing container, or evidence directory.
