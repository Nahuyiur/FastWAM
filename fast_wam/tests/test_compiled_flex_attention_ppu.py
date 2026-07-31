"""Host-device smoke test for compiled FlexAttention forward and backward."""

from __future__ import annotations

import pytest
import torch
from torch.nn.attention.flex_attention import flex_attention


def test_compiled_flex_attention_forward_backward() -> None:
    if not torch.cuda.is_available():
        pytest.skip("compiled FlexAttention smoke test requires a CUDA/PPU device")

    q = torch.randn(2, 8, 1024, 64, device="cuda", requires_grad=True)
    k = torch.randn(2, 8, 1024, 64, device="cuda", requires_grad=True)
    v = torch.randn(2, 8, 1024, 64, device="cuda", requires_grad=True)

    compiled_flex = torch.compile(flex_attention, fullgraph=False)
    out = compiled_flex(q, k, v)
    loss = out.sum()
    loss.backward()
    torch.cuda.synchronize()

    assert q.grad is not None
    assert k.grad is not None
    assert v.grad is not None


if __name__ == "__main__":
    test_compiled_flex_attention_forward_backward()
    print("OK")
