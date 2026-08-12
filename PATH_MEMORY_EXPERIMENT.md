# Path Memory Experiment

This branch contains an isolated path-memory experiment for repeated shelf and delivery navigation.

## Scope

Only the navigation-memory path is changed. The main experiment files are:

- `examples/supermarket_sorting/path_memory.py`
- `examples/supermarket_sorting/supermarket_navigation.py`
- `examples/supermarket_sorting/integrated_nav_pick_place.py`

## What It Does

The experiment adds a lightweight path cache for repeated navigation goals.

- Save a successful navigation path after reaching a goal.
- Reuse a cached path on a later run when the start/goal match closely enough.
- Emit runtime diagnostics so cache hits can be verified from logs.

## Important Fixes Included

This experiment also includes fixes for three issues discovered during testing:

1. Path-memory keys must be saved with the true navigation start pose.
2. Cached paths must be normalized to `start -> goal` direction.
3. The saved path must use the first full route snapshot, not a short near-goal replan fragment.

## Runtime Verification

Use the following log lines to verify a real cache hit:

```bash
grep -E "path_memory_runtime|cache_hit|cached_path_active" memory_maidong_run2.log
```

A successful runtime hit should show:

- `cache_hit: true`
- `cached_path_active: true`

## Suggested Validation Flow

1. Delete `/root/.cache/supermarket_path_memory.json`
2. Run `memory_run1` to build the cache
3. Run `memory_run2` with the same fixed target and seeds
4. Check the runtime hit logs

## Current Validation Result

For the fixed single-target `maidong` test, `memory_run2` successfully reported a runtime cache hit on `nav->delivery`.
