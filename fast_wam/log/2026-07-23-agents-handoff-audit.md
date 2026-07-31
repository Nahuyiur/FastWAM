# Megatron-Wan AGENTS handoff audit

Date: 2026-07-23
Last updated: 2026-07-23

## Objective and final status

Review and reorganize the root `AGENTS.md` from the perspective of ongoing
`fast_wam` development, remove stale/duplicated Wan material from the root
handoff, delegate Wan-specific instructions to `wan/AGENTS.override.md`, and
add a durable task completion contract.

Status: complete. The root handoff is now Fast-WAM-first and the Wan override
is canonical for Wan video training work.

## Files changed

- `AGENTS.md`: replaced the 899-line mixed Wan/Fast-WAM handoff with a
  substantially shorter Fast-WAM-first repository handoff.
- `wan/AGENTS.override.md`: made the override canonical for `wan/`, labeled
  the `/aifs4su` DGX/H800/NVCR material as historical, and separated Wan
  capabilities from Fast-WAM support.
- `fast_wam/README.md`: documented the `[0,1]` visual-input contract,
  double-normalization hazard, and resumable 2,000-episode runner.
- `README.md`: updated the repository identity and quick smoke for the current
  Fast-WAM focus.
- `fast_wam/log/2026-07-23-agents-handoff-audit.md`: recorded this audit.

## Result

The completion contract now requires:

- validation proportional to the change;
- a dated work log with commands, results, artifacts, limitations, and
  follow-up;
- large artifacts to remain outside Git;
- a final handoff review after every task;
- selective updates of durable, reusable conclusions to `AGENTS.md`;
- transient progress and detailed experiment output to stay in work logs or
  result artifacts;
- stale conclusions to be replaced instead of contradicted by append-only
  notes.
- every created or modified Markdown file to carry an updated
  `Last updated: YYYY-MM-DD` line near its title.

The root handoff now keeps the durable Fast-WAM information needed for current
development:

- Megatron baseline and no-core-modification invariant;
- inference-only scope and TP/DP/DCP architecture;
- camera, `[0,1]` image, MIN_MAX, gripper, attention, and action contracts;
- active Ruibin paths and immutable PPU software stack;
- CPU, fixed 8-episode, BF16 50-episode, and full 2,000-episode gates;
- resumable-output semantics and artifact policy;
- links to detailed implementation and result logs.

Wan data schemas, preprocessing, training, inference, H800 performance, and
historical pitfalls now live only in the Wan override and dated Wan logs.

## Validation

- Inspected the full 874-line pre-edit `AGENTS.md`, the current `fast_wam`
  README, evaluation scripts, work logs, manifests, and local result artifacts.
- Read the complete 494-line pre-edit `wan/AGENTS.override.md` before changing
  its precedence and verified that it already covered the Wan-specific content
  removed from the root handoff.
- Confirmed the documented Megatron tree hash remains
  `fd317ec854371f5c5f1ca260579e887d59630d7b` and that `megatron/` has no
  worktree changes.
- Ran:

```bash
FAST_WAM_DISABLE_MCORE=1 python -m pytest -q fast_wam/tests
```

Result: 5 passed. Two expected warnings reported that accelerator devices are
not exposed in the default command sandbox.

- `git diff --check`: passed.
- Verified the canonical code, checkpoint, DCP, manifest, runner, 8-episode
  summary, and 50-episode summary paths referenced by the handoff all exist.

## Known limitations

- The in-progress 2,000-episode result must not be reported as final until its
  output directory contains a complete `summary.json`.
- The historical Wan override intentionally retains detailed May 2026
  commands/results. Its `/aifs4su` paths require the old DGX/H800 environment
  or explicit remapping and revalidation.
