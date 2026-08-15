# R15-B verification summary

## Accepted measured result

R15-B passed a fresh control smoke and two formal runs under the frozen image
and seed contract. Each formal run delivered one of five orders:

- Run 1: 1/5 delivered; first localization 73.057 s; first completion
  294.924 s; match end 385.105 s.
- Run 2: 1/5 delivered; first localization 122.954 s; first completion
  354.143 s; match end 459.663 s.

This is the current stable statement: **two R15-B runs at 1/5 each**. It is not
a claim of stable 2/5 performance. The measured formal-run collision, drop,
and human-intervention counts were 0/0/0.

## Static and test verification

- Historical preflight command: 224 passed, 0 failed.
- Full repository collection in upload staging: 229 passed, 0 failed; the five
  additional items are example-level obstacle-layout tests outside `tests/`.
- Strict association/sample/spread thresholds and exact pregrasp revalidation
  remained unchanged.
- Trace was off; GUI, video, and server ground truth were off.
- Formal code identity was unchanged across the accepted R15-B runs.

The complete event streams, manifests, container identity, and raw test output
remain external. See `AUDIT_EVIDENCE_INDEX.md`.
