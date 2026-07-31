# Optional Megatron linear bias return for Fast-WAM fusion

Last updated: 2026-07-25

## Objective and status

Add an opt-in `skip_bias_add` capability to the shared Wan `_Linear` wrapper so
the Fast-WAM overlay can call Megatron's fused bias-GELU without changing
parameter keys. Complete.

## Change

`wan/model/wan_dit.py::_Linear` now accepts `skip_bias_add=False`. The default
path is unchanged. When explicitly enabled:

- Megatron Column/RowParallelLinear receives `skip_bias_add=True`;
- the wrapper returns `(output_without_bias, bias)`;
- the replicated fallback uses `F.linear(..., bias=None)` and returns the same
  pair.

No existing Wan module enables the option, so Wan training/inference behavior
and checkpoint layout are unchanged.

## Validation

- Python compilation passed.
- Fast-WAM CPU suite: 14 passed.
- Real PPU MCore BF16 optimized forward/backward passed, including fused
  bias-GELU.
- Full 6.021B-model TP1+DP8 optimized training completed 12 steps without
  non-finite loss or backend failure.
