"""Fast-WAM's seed-42 epoch sampler expressed as Megatron DP batches."""

from __future__ import annotations

import math
from collections.abc import Iterator

import torch
from torch.utils.data import Sampler


class OfficialEpochBatchSampler(Sampler[list[int]]):
    """Reproduce Accelerate's sharding of the upstream ResumableEpochSampler.

    Upstream first creates one seed-42+epoch global permutation, groups it into
    local microbatches, and Accelerate assigns consecutive microbatches to DP
    ranks.  Its even-batch behavior pads the final global step from the start of
    the same permutation.
    """

    def __init__(
        self,
        *,
        dataset_size: int,
        consumed_samples: int,
        micro_batch_size: int,
        data_parallel_rank: int,
        data_parallel_size: int,
        seed: int = 42,
    ):
        self.dataset_size = int(dataset_size)
        self.consumed_samples = int(consumed_samples)
        self.micro_batch_size = int(micro_batch_size)
        self.data_parallel_rank = int(data_parallel_rank)
        self.data_parallel_size = int(data_parallel_size)
        self.seed = int(seed)
        if self.dataset_size <= 0 or self.micro_batch_size <= 0:
            raise ValueError("dataset_size and micro_batch_size must be positive")
        if not 0 <= self.data_parallel_rank < self.data_parallel_size:
            raise ValueError("invalid data-parallel rank")
        if self.consumed_samples % self.global_batch_size:
            raise ValueError(
                f"consumed_samples={self.consumed_samples} is not divisible by "
                f"global_batch_size={self.global_batch_size}"
            )

    @property
    def global_batch_size(self) -> int:
        return self.micro_batch_size * self.data_parallel_size

    @property
    def steps_per_epoch(self) -> int:
        return math.ceil(self.dataset_size / self.global_batch_size)

    @property
    def padded_epoch_size(self) -> int:
        return self.steps_per_epoch * self.global_batch_size

    def __len__(self) -> int:
        return self.steps_per_epoch

    def __iter__(self) -> Iterator[list[int]]:
        epoch = self.consumed_samples // self.padded_epoch_size
        offset = self.consumed_samples % self.padded_epoch_size
        generator = torch.Generator(device="cpu").manual_seed(self.seed + epoch)
        permutation = torch.randperm(
            self.dataset_size,
            generator=generator,
        ).tolist()
        padding = self.padded_epoch_size - self.dataset_size
        if padding:
            permutation.extend(permutation[:padding])

        first_step = offset // self.global_batch_size
        for step in range(first_step, self.steps_per_epoch):
            global_start = step * self.global_batch_size
            local_start = (
                global_start
                + self.data_parallel_rank * self.micro_batch_size
            )
            batch = permutation[local_start : local_start + self.micro_batch_size]
            if len(batch) != self.micro_batch_size:
                raise RuntimeError("Fast-WAM sampler produced an incomplete microbatch")
            self.consumed_samples += self.global_batch_size
            yield batch


class OfficialValidationBatchSampler(Sampler[list[int]]):
    """Yield the one deterministic per-rank sample used by upstream eval.

    ``Wan22Trainer.evaluate`` does not iterate a validation DataLoader.  At
    every ``eval_every`` optimizer steps it seeds a CPU generator with
    ``global_step + process_index`` and draws one dataset index.  Megatron's
    validation loop is DataLoader based, so this sampler expresses that exact
    sequence while keeping all ranks in one TP group on the same sample.
    """

    def __init__(
        self,
        *,
        dataset_size: int,
        consumed_samples: int,
        data_parallel_rank: int,
        data_parallel_size: int,
        eval_interval: int,
    ):
        self.dataset_size = int(dataset_size)
        self.data_parallel_rank = int(data_parallel_rank)
        self.data_parallel_size = int(data_parallel_size)
        self.eval_interval = int(eval_interval)
        consumed_samples = int(consumed_samples)
        if self.dataset_size <= 0:
            raise ValueError("dataset_size must be positive")
        if not 0 <= self.data_parallel_rank < self.data_parallel_size:
            raise ValueError("invalid data-parallel rank")
        if self.eval_interval <= 0:
            raise ValueError("eval_interval must be positive")
        if consumed_samples % self.data_parallel_size:
            raise ValueError(
                f"consumed validation samples={consumed_samples} is not divisible by "
                f"DP={self.data_parallel_size}"
            )
        self.evaluation_ordinal = consumed_samples // self.data_parallel_size

    def __len__(self) -> int:
        # The surrounding Megatron cyclic iterator makes a new iterator after
        # this finite sequence.  One item keeps the ordinal state explicit.
        return 1

    def __iter__(self) -> Iterator[list[int]]:
        global_step = (self.evaluation_ordinal + 1) * self.eval_interval
        generator = torch.Generator(device="cpu").manual_seed(
            global_step + self.data_parallel_rank
        )
        index = int(
            torch.randint(
                0,
                self.dataset_size,
                (1,),
                generator=generator,
            ).item()
        )
        self.evaluation_ordinal += 1
        yield [index]
