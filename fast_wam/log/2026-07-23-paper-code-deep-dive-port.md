# Fast-WAM paper/code deep-dive port

Last updated: 2026-07-23

## Objective and final status

Port the Chinese Fast-WAM paper/code deep dive from
`Fast-WAM/docs/fast_wam_paper_code_deep_dive_zh.md` into
`Megatron-Wan/fast_wam/docs/`, preserving the official paper/code analysis
while adapting links and adding the Megatron inference-overlay mapping.

Status: complete.

## Files changed

- Added `fast_wam/docs/fast_wam_paper_code_deep_dive_zh.md`.
- Updated `fast_wam/README.md` with deep-dive and BF16 report links.
- Updated root `AGENTS.md` references.
- Added this work log.

## Porting decisions

- The paper argument, architecture, MoT mask, FlowMatch schedule, inference
  cache, controlled variants, training recipe, benchmarks, caveats, insights,
  and official code-reading map were retained.
- Official paper/code statements remain explicitly attributed to the sibling
  `Fast-WAM` archive at `main@45d8e14`.
- Images, paper source, and official code are reused through verified relative
  links into the sibling `Fast-WAM/` tree instead of duplicating binary assets.
- Added a Megatron-specific mapping for model, scheduler, checkpoint,
  preprocessing, policy, distributed runtime, LIBERO rollout, acceptance, and
  DCP conversion.
- Added explicit boundaries: the Megatron overlay is inference-only and does
  not implement training, future-video loss, Joint/IDM, PP/CP/SP, or RoboTwin.
- Added the `[0,1]` visual-input contract, serialized
  `VISUAL=MEAN_STD(.5,.5)` double-normalization hazard, FP32 SDPA parity
  requirement, and incomplete-2,000-episode semantics.

## Validation

- Confirmed the source document has 941 lines and read it completely.
- Confirmed all referenced source images, paper files, official code files,
  Megatron implementation files, result logs, and local reports exist.
- Markdown links were rewritten for the destination directory rather than
  retaining the source document's now-invalid relative paths.
- Resolved all 59 Markdown/HTML links from the 1,176-line migrated document;
  zero targets were missing.
- `git diff --check` passed.

## Known limitations

- The migrated document reuses sibling-repository assets; moving
  `Megatron-Wan` away from the current workspace layout would require
  relinking or copying those assets.
- Paper benchmark numbers remain paper-reported results. Local Megatron
  8/50/2,000-episode evaluation status is labeled separately and must not be
  conflated with paper reproduction.
