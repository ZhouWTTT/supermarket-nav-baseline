# Run-scoped memory matrix

This branch adds bounded, run-scoped memory to the supermarket competition
runner while retaining the original repository history and upstream authorship.
The feature is based on commit `efa6d15bef3e77bd90ccaec3a5bcdefbd7e956d5`.
The later upstream reference `f09ab31fea4f12cbc30095922393e5d34321badf`
was reviewed as context only and was not merged or rebased.

## Memory layers

- **WM0 — observed inventory:** multiple candidates are retained per run with
  source stamps, fresh-frame context, observed base/head viewpoint, pose and
  scan epoch, shelf coverage, and a deterministic coverage cursor.
- **WM1 — replay and outcome memory:** attempt fingerprints distinguish
  material changes from stamp churn. Replay outcomes, strict failures,
  viewpoint convergence, reactivation, place retry, and accepted terminal
  results are isolated by run, candidate, pose, epoch, and attempt.
- **WM2-A — formal evidence view:** formal ledgers may summarize WM0/WM1 for
  audit. WM2-A is advisory evidence and is not a runtime control authority.

The runner uses score-first selection, bounded anytime discovery, a completion
reserve, and terminal accounting. Memory can rank or suppress work only within
the explicit policy modes; it cannot bypass perception, freshness, strict
localization, pregrasp revalidation, motion safety, or grasp safety.

## Normal development

`SUPERMARKET_MEMORY_MODE` defaults to `off`. Runtime output is configurable
with `--runtime-dir` and defaults outside the repository. Heavy strict trace,
GUI, video, and ground-truth diagnostics are off unless explicitly requested.
Normal imports and tests do not read Git identity, require a formal image, or
require `runtime_evidence/`. The self-contained R9 regression test uses a
minimal event fixture rather than an external evidence directory.

Formal identity, exact-image, seed-ledger, and evidence gates belong in a
separate manual/FORMAL_CI workflow. They are intentionally absent from default
PR and DEV_CI operation.
