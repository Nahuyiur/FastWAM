"""Optimizer policy adapters for baseline-equivalent Fast-WAM training."""

from __future__ import annotations

import functools
from collections.abc import Callable


def all_parameter_adamw_config(original: Callable) -> Callable:
    """Keep AdamW weight decay on every trainable parameter.

    The original RoboCasa trainer passes one flat parameter list to
    ``torch.optim.AdamW``. Megatron's standard optimizer builder instead
    creates a zero-weight-decay group for biases and one-dimensional tensors.
    Returning an explicit empty override mapping preserves the original flat
    AdamW contract because Megatron only installs its defaults for ``None``.
    """

    @functools.wraps(original)
    def wrapped(args):
        config, _ = original(args)
        if not bool(config.decoupled_weight_decay):
            raise ValueError("RoboCasa baseline parity requires AdamW weight decay")
        return config, {}

    return wrapped
