import logging
import json
import inspect
import os
import re
import shutil
import warnings
from math import ceil
from pathlib import Path
import time
from contextlib import nullcontext
from typing import Any

import numpy as np
import torch
from accelerate import Accelerator
from omegaconf import DictConfig
from PIL import Image
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from .utils.fs import ensure_dir
from .utils.logging_config import get_logger, setup_logging
from .utils.pytorch_utils import set_global_seed
from .utils.samplers import ResumableEpochSampler
from .utils.video_io import save_mp4
from .utils.video_metrics import pil_frames_to_video_tensor, video_psnr, video_ssim
from .evaluation.open_loop_wam import run_autoregressive_open_loop_wam_eval

logger = get_logger(__name__)


class Wan22Trainer:
    def __init__(self, model, train_dataset, val_dataset=None, *, cfg: DictConfig):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.cfg = cfg
        self.output_dir = str(cfg.output_dir)
        self.learning_rate = float(cfg.learning_rate)
        self.weight_decay = float(cfg.weight_decay)
        self.batch_size = int(cfg.batch_size)
        self.num_workers = int(cfg.num_workers)
        self.prefetch_factor = cfg.get("prefetch_factor", None)
        self.persistent_workers = bool(cfg.get("persistent_workers", False)) and self.num_workers > 0
        self.num_epochs = int(cfg.num_epochs)
        max_steps = cfg.max_steps
        self.max_steps = int(max_steps) if max_steps is not None else None
        self.log_every = int(cfg.log_every)
        self.save_every = int(cfg.save_every)
        self.save_final_checkpoint = bool(cfg.get("save_final_checkpoint", True))
        self.eval_every = int(cfg.eval_every)
        self.eval_num_inference_steps = int(cfg.eval_num_inference_steps)
        self.gradient_accumulation_steps = int(cfg.gradient_accumulation_steps)
        self.max_grad_norm = float(cfg.max_grad_norm)
        self.seed = int(cfg.seed)

        self.resume = cfg.get("resume", None)
        self.checkpoint_cfg = cfg.get("checkpoint", {}) or {}
        self.checkpoint_save_full_state = self._as_bool(self.checkpoint_cfg.get("save_full_state", True))
        self.checkpoint_require_full_state = self._as_bool(self.checkpoint_cfg.get("require_full_state", True))
        self.checkpoint_weight_min_free_gb = float(self.checkpoint_cfg.get("weight_min_free_gb", 30))
        self.checkpoint_full_state_min_free_gb = float(self.checkpoint_cfg.get("full_state_min_free_gb", 120))
        keep_last_full_states = self.checkpoint_cfg.get("keep_last_full_states", 2)
        self.checkpoint_keep_last_full_states = (
            0 if self._is_nullish(keep_last_full_states) else int(keep_last_full_states)
        )
        self.mixed_precision = str(cfg.mixed_precision).strip().lower()
        if self.mixed_precision not in {"no", "fp16", "bf16"}:
            raise ValueError(
                f"Unsupported mixed_precision: {cfg.mixed_precision}. "
                "Expected one of: ['no', 'fp16', 'bf16']."
            )
        self.wandb_enabled = bool(cfg.wandb.enabled)
        profile_cfg = cfg.get("profile", {}) or {}
        self.profile_enabled = bool(profile_cfg.get("enabled", False))
        self.profile_warmup_steps = int(profile_cfg.get("warmup_steps", 20))
        self.profile_sync_cuda = bool(profile_cfg.get("sync_cuda", True))
        self.profile_output_jsonl = profile_cfg.get("output_jsonl", None)
        self.profile_torch_cfg = profile_cfg.get("torch_profiler", {}) or {}
        self._profile_step_t0 = None
        self._profile_accum = {}
        self.open_loop_wam_eval_cfg = cfg.get("open_loop_wam_eval", {}) or {}

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            mixed_precision=self.mixed_precision,
            step_scheduler_with_optimizer=False,
        )

        logger.info(
            "Accelerate training: distributed_type=%s zero_stage=%s world_size=%d process_index=%d cfg_mixed_precision=%s accelerator_mixed_precision=%s grad_accum=%d grad_clip=%.4f",
            self.accelerator.distributed_type,
            self.accelerator.state.deepspeed_plugin.deepspeed_config.get("zero_optimization", {}).get("stage", "unknown"),
            self.accelerator.num_processes,
            self.accelerator.process_index,
            self.mixed_precision,
            self.accelerator.mixed_precision,
            self.gradient_accumulation_steps,
            self.max_grad_norm,
        )
        logger.info("using accelerator.device=%s", self.accelerator.device)
        worker_init_fn = set_global_seed(self.seed, get_worker_init_fn=True)
        self._assert_dataset_length_consistent(self.train_dataset, "train_dataset")
        if self.val_dataset is not None:
            self._assert_dataset_length_consistent(self.val_dataset, "val_dataset")

        # Freeze non-trainable modules before optimizer/deepspeed initialization.
        # This keeps DiT (+ optional proprio encoder) as trainable when ZeRO builds optimizer state.
        self._apply_dit_only_train_mode(self.model)
        trainable_params = list(self.model.dit.parameters())
        proprio_encoder = getattr(self.model, "proprio_encoder", None)
        if proprio_encoder is not None:
            trainable_params.extend(list(proprio_encoder.parameters()))
        optimizer_kwargs = {
            "lr": self.learning_rate,
            "weight_decay": self.weight_decay,
            "betas": (0.9, 0.95),
        }
        optimizer_foreach = cfg.get("optimizer_foreach", None)
        if optimizer_foreach is not None:
            optimizer_kwargs["foreach"] = bool(optimizer_foreach)
        optimizer_fused = cfg.get("optimizer_fused", None)
        if optimizer_fused is not None:
            optimizer_kwargs["fused"] = bool(optimizer_fused)
        self.optimizer = torch.optim.AdamW(trainable_params, **optimizer_kwargs)

        self.train_loader = self._build_loader(self.train_dataset, worker_init_fn=worker_init_fn)
        total_train_steps = self._estimate_total_train_steps()
        self.max_steps = total_train_steps
        warmup_steps = int(total_train_steps * 0.05)
        self.scheduler = self._build_scheduler(
            scheduler_type=cfg.lr_scheduler_type,
            total_train_steps=total_train_steps,
            warmup_steps=warmup_steps,
        )
        self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0

        self.checkpoint_root = os.path.join(self.output_dir, "checkpoints")
        self.weights_dir = os.path.join(self.checkpoint_root, "weights")
        self.state_dir = os.path.join(self.checkpoint_root, "state")
        self.eval_dir = os.path.join(self.output_dir, "eval")
        self.open_loop_wam_eval_dir = os.path.join(self.output_dir, "eval_open_loop")

        ensure_dir(self.output_dir)
        ensure_dir(self.checkpoint_root)
        ensure_dir(self.weights_dir)
        ensure_dir(self.state_dir)
        ensure_dir(self.eval_dir)

        self.model, self.optimizer, self.train_loader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.scheduler
        )
        self.optimizer.zero_grad(set_to_none=True)
        self.wandb_run = None
        self._init_wandb()
        self._resume_or_load_checkpoint()

        val_size = len(self.val_dataset) if self.val_dataset is not None else len(self.train_dataset)
        logger.info("Train/val dataset size: %d/%d", len(self.train_dataset), val_size)

    def _init_wandb(self):
        if not self.wandb_enabled or not self.accelerator.is_main_process:
            return
        try:
            import wandb
        except ImportError as e:
            raise ImportError(
                "wandb logging is enabled in config (`wandb.enabled=true`) but wandb is not installed."
            ) from e

        wandb_group = None if self.cfg.wandb.group in (None, "null", "") else str(self.cfg.wandb.group)
        wandb_job_type = self.cfg.wandb.get("job_type", None)
        wandb_job_type = None if wandb_job_type in (None, "null", "") else str(wandb_job_type)
        wandb_subproject = self.cfg.wandb.get("subproject", None)
        wandb_subproject = None if wandb_subproject in (None, "null", "") else str(wandb_subproject)
        wandb_id = self.cfg.wandb.get("id", None)
        wandb_id = None if wandb_id in (None, "null", "") else str(wandb_id)
        wandb_resume = self.cfg.wandb.get("resume", None)
        wandb_resume = None if wandb_resume in (None, "null", "") else str(wandb_resume)
        wandb_tags = self.cfg.wandb.get("tags", None)
        if wandb_tags in (None, "null", ""):
            wandb_tags = None
        elif isinstance(wandb_tags, str):
            wandb_tags = [tag.strip() for tag in wandb_tags.split(",") if tag.strip()]
        else:
            wandb_tags = [str(tag) for tag in wandb_tags]

        self.wandb_run = wandb.init(
            entity=self.cfg.wandb.workspace,
            project=self.cfg.wandb.project,
            name=self.cfg.wandb.name,
            id=wandb_id,
            resume=wandb_resume,
            group=wandb_group,
            job_type=wandb_job_type,
            tags=wandb_tags,
            config={
                "subproject": wandb_subproject,
                "output_dir": self.output_dir,
                "max_steps": self.max_steps,
                "batch_size": self.batch_size,
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
                "learning_rate": self.learning_rate,
                "checkpoint_resume_from": self._none_if_nullish(self.checkpoint_cfg.get("resume_from", None)),
                "checkpoint_init_from_weights": self._none_if_nullish(
                    self.checkpoint_cfg.get("init_from_weights", None)
                ),
                "checkpoint_load_step_from_weights": self._as_bool(
                    self.checkpoint_cfg.get("load_step_from_weights", False)
                ),
                "checkpoint_initial_step": self._none_if_nullish(self.checkpoint_cfg.get("initial_step", None)),
                "checkpoint_advance_scheduler_to_step": self._as_bool(
                    self.checkpoint_cfg.get("advance_scheduler_to_step", True)
                ),
                "legacy_resume": self._none_if_nullish(self.resume),
                "wandb_id": wandb_id,
                "wandb_resume": wandb_resume,
            },
            mode=self.cfg.wandb.mode,
            dir=self.output_dir,
        )
        logger.info(
            "Initialized wandb run: workspace=%s project=%s group=%s name=%s id=%s resume=%s subproject=%s",
            self.cfg.wandb.workspace,
            self.cfg.wandb.project,
            wandb_group,
            self.cfg.wandb.name,
            wandb_id,
            wandb_resume,
            wandb_subproject,
        )

    def _wandb_log(self, payload: dict):
        if self.wandb_run is None:
            return
        self.wandb_run.log(payload, step=self.global_step)

    def _finish_wandb(self):
        if self.wandb_run is None:
            return
        self.wandb_run.finish()
        self.wandb_run = None

    def _profile_now(self) -> float:
        if self.profile_enabled and self.profile_sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter()

    def _profile_start_step_if_needed(self) -> None:
        if not self.profile_enabled or self._profile_step_t0 is not None:
            return
        self._profile_accum = {
            "data_wait_s": 0.0,
            "forward_loss_s": 0.0,
            "backward_s": 0.0,
            "grad_clip_s": 0.0,
            "optimizer_step_s": 0.0,
            "scheduler_step_s": 0.0,
            "zero_grad_s": 0.0,
            "log_gather_s": 0.0,
        }
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self._profile_step_t0 = self._profile_now()

    def _profile_add(self, key: str, elapsed: float) -> None:
        if not self.profile_enabled:
            return
        self._profile_accum[key] = float(self._profile_accum.get(key, 0.0) + elapsed)

    def _profile_measure_start(self) -> float | None:
        if not self.profile_enabled:
            return None
        return self._profile_now()

    def _profile_measure_end(self, key: str, start: float | None) -> None:
        if not self.profile_enabled or start is None:
            return
        self._profile_add(key, self._profile_now() - start)

    def _profile_output_path(self) -> Path:
        if self.profile_output_jsonl not in (None, "", "null"):
            return Path(str(self.profile_output_jsonl))
        return Path(self.output_dir) / "profile" / "step_times.jsonl"

    def _profile_complete_optimizer_step(self, *, loss: float, loss_metrics: dict, lr: float, grad_norm: float) -> None:
        if not self.profile_enabled or self._profile_step_t0 is None:
            return
        step_total_s = self._profile_now() - self._profile_step_t0
        self._profile_accum["step_total_s"] = float(step_total_s)
        peak_memory = float(torch.cuda.max_memory_allocated() / (1024 ** 3)) if torch.cuda.is_available() else 0.0

        metric_keys = [
            "step_total_s",
            "data_wait_s",
            "forward_loss_s",
            "backward_s",
            "grad_clip_s",
            "optimizer_step_s",
            "scheduler_step_s",
            "zero_grad_s",
            "log_gather_s",
        ]
        local_values = [self._profile_accum.get(key, 0.0) for key in metric_keys] + [peak_memory]
        local_tensor = torch.tensor(local_values, device=self.accelerator.device, dtype=torch.float32).unsqueeze(0)
        gathered = self.accelerator.gather_for_metrics(local_tensor)

        if self.global_step > self.profile_warmup_steps and self.accelerator.is_main_process:
            means = gathered.mean(dim=0).detach().cpu().tolist()
            maxes = gathered.max(dim=0).values.detach().cpu().tolist()
            payload = {
                "step": int(self.global_step),
                "epoch": int(self.epoch),
                "batch_in_epoch": int(self.batch_in_epoch),
                "loss": float(loss),
                "lr": float(lr),
                "grad_norm": float(grad_norm),
                "profile_warmup_steps": int(self.profile_warmup_steps),
            }
            for idx, key in enumerate(metric_keys):
                payload[f"{key}_rank_avg"] = float(means[idx])
                payload[f"{key}_rank_max"] = float(maxes[idx])
            payload["peak_memory_gb_rank_avg"] = float(means[-1])
            payload["peak_memory_gb_rank_max"] = float(maxes[-1])
            for key, value in sorted(loss_metrics.items()):
                payload[key] = float(value)
            path = self._profile_output_path()
            ensure_dir(str(path.parent))
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, sort_keys=True) + "\n")
            self._wandb_log({
                "profile/step_total_s": payload["step_total_s_rank_avg"],
                "profile/forward_loss_s": payload["forward_loss_s_rank_avg"],
                "profile/backward_s": payload["backward_s_rank_avg"],
                "profile/optimizer_step_s": payload["optimizer_step_s_rank_avg"],
                "profile/peak_memory_gb": payload["peak_memory_gb_rank_max"],
            })

        self._profile_step_t0 = None
        self._profile_accum = {}

    def _torch_profiler_context(self):
        torch_profiler_cfg = self.profile_torch_cfg or {}
        if not bool(torch_profiler_cfg.get("enabled", False)):
            return nullcontext(None)
        write_trace = bool(torch_profiler_cfg.get("write_trace", True))
        on_trace_ready = None
        if write_trace:
            trace_dir = torch_profiler_cfg.get("trace_dir", None)
            if trace_dir in (None, "", "null"):
                trace_dir = str(Path(self.output_dir) / "profile" / "torch_profiler")
            ensure_dir(str(trace_dir))
            on_trace_ready = torch.profiler.tensorboard_trace_handler(str(trace_dir))
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        schedule = torch.profiler.schedule(
            wait=int(torch_profiler_cfg.get("wait", 1)),
            warmup=int(torch_profiler_cfg.get("warmup", 1)),
            active=int(torch_profiler_cfg.get("active", 3)),
            repeat=int(torch_profiler_cfg.get("repeat", 1)),
        )
        return torch.profiler.profile(
            activities=activities,
            schedule=schedule,
            on_trace_ready=on_trace_ready,
            record_shapes=bool(torch_profiler_cfg.get("record_shapes", True)),
            profile_memory=bool(torch_profiler_cfg.get("profile_memory", True)),
            with_stack=bool(torch_profiler_cfg.get("with_stack", False)),
        )

    def _write_torch_profiler_summary(self, torch_profiler) -> None:
        if torch_profiler is None or not self.accelerator.is_main_process:
            return
        torch_profiler_cfg = self.profile_torch_cfg or {}
        table_path = torch_profiler_cfg.get("table_path", None)
        if table_path in (None, "", "null"):
            table_path = str(Path(self.output_dir) / "profile" / "torch_profiler_key_averages.txt")
        sort_by = str(torch_profiler_cfg.get("sort_by", "self_cuda_time_total"))
        row_limit = int(torch_profiler_cfg.get("row_limit", 80))
        group_by_input_shape = bool(torch_profiler_cfg.get("group_by_input_shape", False))
        try:
            table = torch_profiler.key_averages(
                group_by_input_shape=group_by_input_shape,
            ).table(sort_by=sort_by, row_limit=row_limit)
        except Exception as exc:
            table = f"Failed to render torch profiler table sorted by {sort_by}: {exc}\n"
        path = Path(str(table_path))
        ensure_dir(str(path.parent))
        path.write_text(table, encoding="utf-8")
        logger.info("Wrote torch profiler key averages to %s", path)

    def _build_loader(self, dataset, worker_init_fn=None):
        self.train_sampler = ResumableEpochSampler(
            dataset=dataset,
            seed=self.seed,
            batch_size=self.batch_size,
            num_processes=self.accelerator.num_processes,
        )
        loader_kwargs = {
            "batch_size": self.batch_size,
            "shuffle": False,
            "sampler": self.train_sampler,
            "num_workers": self.num_workers,
            "pin_memory": torch.cuda.is_available(),
            "worker_init_fn": worker_init_fn,
        }
        if self.num_workers > 0:
            loader_kwargs["persistent_workers"] = self.persistent_workers
            if self.prefetch_factor not in (None, "", "null"):
                loader_kwargs["prefetch_factor"] = int(self.prefetch_factor)
        return DataLoader(dataset, **loader_kwargs)

    def _assert_dataset_length_consistent(self, dataset, dataset_name: str):
        if not hasattr(dataset, "__len__"):
            raise TypeError(f"`{dataset_name}` must implement __len__ for rank consistency checks.")

        local_length = len(dataset)
        gathered_lengths = self.accelerator.gather(
            torch.tensor([local_length], device=self.accelerator.device, dtype=torch.int64)
        ).reshape(-1)
        if torch.all(gathered_lengths == gathered_lengths[0]):
            return

        if self.accelerator.is_main_process:
            print(f"[dataset-check] {dataset_name} length mismatch across ranks after initialization:")
            for rank, rank_length in enumerate(gathered_lengths.cpu().tolist()):
                print(f"rank {rank}: {rank_length}")
        self.accelerator.wait_for_everyone()
        raise RuntimeError(
            f"{dataset_name} length mismatch across ranks: {gathered_lengths.cpu().tolist()}"
        )

    def _estimate_total_train_steps(self) -> int:
        if self.max_steps is not None:
            return max(int(self.max_steps), 1)

        if not hasattr(self.train_dataset, "__len__"):
            raise TypeError("`train_dataset` must implement __len__ when `max_steps` is None.")

        num_processes = max(int(self.accelerator.num_processes), 1)
        global_batch_size = max(self.batch_size * num_processes, 1)
        micro_steps_per_epoch = max(ceil(len(self.train_dataset) / global_batch_size), 1)
        opt_steps_per_epoch = max(
            ceil(micro_steps_per_epoch / self.gradient_accumulation_steps),
            1,
        )
        return max(opt_steps_per_epoch * self.num_epochs, 1)

    def _build_scheduler(self, scheduler_type, total_train_steps: int, warmup_steps: int = 0):
        scheduler_type = str(scheduler_type).strip().lower()
        total_train_steps = max(int(total_train_steps), 1)
        warmup_steps = min(max(int(warmup_steps), 0), total_train_steps - 1)

        remaining_steps = max(total_train_steps - warmup_steps, 1)
        if scheduler_type == "cosine":
            main_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=remaining_steps,
                eta_min=self.learning_rate * 0.01,
            )
        elif scheduler_type == "constant":
            main_scheduler = ConstantLR(self.optimizer, factor=1.0, total_iters=remaining_steps)
        else:
            raise ValueError(
                f"Unsupported lr_scheduler_type: {scheduler_type}. "
                "Expected one of: ['cosine', 'constant']."
            )

        if warmup_steps <= 0:
            return main_scheduler

        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=1.0 / warmup_steps,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        return SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps],
        )

    def _estimate_eta(self):
        elapsed = max(time.perf_counter() - self.run_start_time, 1e-6)
        done_steps = max(self.global_step - self.run_start_step, 1)
        steps_per_sec = done_steps / elapsed
        remaining_steps = max(self.max_steps - self.global_step, 0)
        eta_seconds = int(remaining_steps / max(steps_per_sec, 1e-9))
        eta_h, eta_rem = divmod(eta_seconds, 3600)
        eta_m, eta_s = divmod(eta_rem, 60)
        return f"{eta_h:02d}:{eta_m:02d}:{eta_s:02d}", steps_per_sec

    @staticmethod
    def _is_nullish(value) -> bool:
        if value is None:
            return True
        text = str(value).strip()
        return text == "" or text.lower() in {"none", "null", "false"}

    @classmethod
    def _none_if_nullish(cls, value):
        return None if cls._is_nullish(value) else value

    @classmethod
    def _as_bool(cls, value) -> bool:
        if isinstance(value, bool):
            return value
        if cls._is_nullish(value):
            return False
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "y", "on"}:
            return True
        if text in {"0", "no", "n", "off"}:
            return False
        return bool(value)

    @staticmethod
    def _step_from_path(path: Path) -> int | None:
        match = re.fullmatch(r"step[_-](\d+)(?:\.pt)?", path.name)
        if match:
            return int(match.group(1))
        return None

    @staticmethod
    def _is_visible_step_dir(path: Path) -> bool:
        return path.is_dir() and re.fullmatch(r"step[_-]\d+", path.name) is not None

    def _is_complete_state_checkpoint(self, path: Path) -> bool:
        if not self._is_visible_step_dir(path):
            return False
        if not (path / "trainer_state.json").is_file():
            return False

        manifest_file = path / "checkpoint_manifest.json"
        if not manifest_file.exists():
            return True
        try:
            with open(manifest_file, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except (OSError, json.JSONDecodeError):
            return False
        return bool(manifest.get("complete", False))

    @classmethod
    def _latest_checkpoint_candidate(cls, candidates: list[Path], *, kind: str) -> Path:
        candidates = [path for path in candidates if path.exists()]
        if not candidates:
            raise FileNotFoundError(f"No {kind} checkpoint candidates found.")

        def sort_key(path: Path):
            step = cls._step_from_path(path)
            step_key = -1 if step is None else step
            try:
                mtime = path.stat().st_mtime
            except OSError:
                mtime = 0.0
            return step_key, mtime

        return max(candidates, key=sort_key)

    def _resolve_state_checkpoint_path(self, source) -> Path:
        path = Path(str(source)).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Resume state path not found: {source}")
        if not path.is_dir():
            raise ValueError(
                f"`checkpoint.resume_from` must point to an Accelerate state directory or run directory, got: {source}"
            )

        if self._is_complete_state_checkpoint(path):
            if not (path / "checkpoint_manifest.json").exists() and self.accelerator.is_main_process:
                logger.warning(
                    "Resume state %s has no checkpoint manifest; treating it as a legacy complete state.",
                    path,
                )
            return path
        if self._is_visible_step_dir(path):
            raise FileNotFoundError(
                f"State directory {path} is missing `trainer_state.json`; it is not a complete exact-resume "
                "checkpoint. Use `checkpoint.init_from_weights` for weights-only continuation."
            )

        candidates: list[Path] = []
        for root in (path / "checkpoints" / "state", path / "state", path):
            if not root.is_dir():
                continue
            candidates.extend(
                child
                for child in root.iterdir()
                if self._is_complete_state_checkpoint(child)
            )
        if not candidates:
            raise FileNotFoundError(
                "Could not find any `step_*` Accelerate state directories under "
                f"{source}. Expected either `<run>/checkpoints/state/step_XXXXXX` "
                "or a run directory containing `checkpoints/state/step_*`."
            )
        resolved = self._latest_checkpoint_candidate(candidates, kind="state")
        logger.info("Resolved latest resume state under %s -> %s", source, resolved)
        return resolved

    def _resolve_weights_checkpoint_path(self, source) -> Path:
        path = Path(str(source)).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Weights checkpoint path not found: {source}")
        if path.is_file():
            return path

        candidates: list[Path] = []
        for root in (path / "checkpoints" / "weights", path / "weights", path):
            if not root.is_dir():
                continue
            candidates.extend(child for child in root.iterdir() if child.is_file() and child.suffix == ".pt")
        if not candidates:
            raise FileNotFoundError(
                "Could not find any `.pt` weight checkpoints under "
                f"{source}. Expected either `<run>/checkpoints/weights/step_XXXXXX.pt` "
                "or a run directory containing `checkpoints/weights/*.pt`."
            )
        resolved = self._latest_checkpoint_candidate(candidates, kind="weights")
        logger.info("Resolved latest weights checkpoint under %s -> %s", source, resolved)
        return resolved

    def _checkpoint_request(self):
        resume_from = self._none_if_nullish(self.checkpoint_cfg.get("resume_from", None))
        init_from_weights = self._none_if_nullish(self.checkpoint_cfg.get("init_from_weights", None))
        legacy_resume = self._none_if_nullish(self.resume)

        if legacy_resume is not None:
            if resume_from is not None or init_from_weights is not None:
                raise ValueError(
                    "Use either legacy `resume` or the new `checkpoint.*` fields, not both."
                )
            legacy_path = Path(str(legacy_resume)).expanduser()
            if legacy_path.suffix == ".pt":
                init_from_weights = legacy_resume
                logger.warning(
                    "`resume=/path/to/step.pt` is deprecated; use `checkpoint.init_from_weights=/path/to/step.pt`."
                )
            else:
                resume_from = legacy_resume
                logger.warning(
                    "`resume=/path/to/state` is deprecated; use `checkpoint.resume_from=/path/to/state`."
                )

        if resume_from is not None and init_from_weights is not None:
            raise ValueError(
                "`checkpoint.resume_from` and `checkpoint.init_from_weights` are mutually exclusive. "
                "Use `resume_from` for exact state resume, or `init_from_weights` when changing training settings."
            )
        return resume_from, init_from_weights

    def _set_step_after_weights_load(self, payload: dict, weights_path: Path) -> None:
        initial_step = self._none_if_nullish(self.checkpoint_cfg.get("initial_step", None))
        load_step_from_weights = self._as_bool(self.checkpoint_cfg.get("load_step_from_weights", False))
        if initial_step is None and not load_step_from_weights:
            return

        if initial_step is not None:
            step = int(initial_step)
            source = "checkpoint.initial_step"
        else:
            raw_step = payload.get("step", None)
            if raw_step is None:
                raw_step = self._step_from_path(weights_path)
            if raw_step is None:
                raise ValueError(
                    "`checkpoint.load_step_from_weights=true` was requested, but the checkpoint has no `step` "
                    f"payload and the filename does not contain `step_XXXXXX`: {weights_path}"
                )
            step = int(raw_step)
            source = "checkpoint payload"

        if step < 0:
            raise ValueError(f"Checkpoint initial step must be non-negative, got {step}.")
        self.global_step = step
        self.epoch = 0
        self.batch_in_epoch = 0
        self.train_sampler.clear_resume_batch_offset()
        logger.info("Initialized training step from %s: global_step=%d", source, self.global_step)

    def _advance_scheduler_to_global_step(self) -> None:
        if self.global_step <= 0:
            return
        if not self._as_bool(self.checkpoint_cfg.get("advance_scheduler_to_step", True)):
            logger.warning(
                "Keeping scheduler at its initial state despite global_step=%d because "
                "`checkpoint.advance_scheduler_to_step=false`.",
                self.global_step,
            )
            return

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Detected call of `lr_scheduler.step\(\)` before `optimizer.step\(\)`",
                category=UserWarning,
            )
            for _ in range(self.global_step):
                self.scheduler.step()
        current_lr = float(self.optimizer.param_groups[0]["lr"])
        logger.info(
            "Advanced LR scheduler to global_step=%d for weights-only continuation; current_lr=%.6e",
            self.global_step,
            current_lr,
        )

    def _resume_or_load_checkpoint(self):
        resume_from, init_from_weights = self._checkpoint_request()
        if resume_from is None and init_from_weights is None:
            return

        if resume_from is not None:
            resume_path = self._resolve_state_checkpoint_path(resume_from)
            logger.info("Resuming full training state from directory: %s", resume_path)
            self.load_training_state(str(resume_path))
            logger.info(
                "Exact resume restored optimizer/scheduler/dataloader state. "
                "Use `checkpoint.init_from_weights` instead if changing LR, batch size, grad accumulation, or schedule."
            )
            return

        weights_path = self._resolve_weights_checkpoint_path(init_from_weights)
        logger.info("Initializing model weights from checkpoint: %s", weights_path)
        payload = self.accelerator.unwrap_model(self.model).load_checkpoint(str(weights_path), optimizer=None)
        self._set_step_after_weights_load(payload=payload, weights_path=weights_path)
        self._advance_scheduler_to_global_step()
        logger.info(
            "Loaded weights only. Optimizer, scheduler, and dataloader are freshly built from the current config."
        )
        if self.max_steps is not None and self.global_step >= self.max_steps:
            logger.warning(
                "Current global_step=%d is already >= max_steps=%d. Increase `max_steps` if this run should continue.",
                self.global_step,
                self.max_steps,
            )

    def _set_dit_only_train_mode(self):
        # Match DiffSynth's freeze_except("dit"): only DiT stays trainable/in-train-mode.
        logger.info("Setting DiT to train mode and freezing other model components.")
        model = self.accelerator.unwrap_model(self.model)
        self._apply_dit_only_train_mode(model)

    @staticmethod
    def _apply_dit_only_train_mode(model):
        model.eval()
        model.requires_grad_(False)
        model.dit.train()
        model.dit.requires_grad_(True)
        proprio_encoder = getattr(model, "proprio_encoder", None)
        if proprio_encoder is not None:
            proprio_encoder.train()
            proprio_encoder.requires_grad_(True)

    @staticmethod
    def _to_batched_eval_sample(sample):
        video = sample["video"]
        prompt = sample["prompt"]
        action = sample.get("action", None)
        proprio = sample.get("proprio", None)
        context = sample.get("context", None)
        context_mask = sample.get("context_mask", None)

        if not isinstance(video, torch.Tensor):
            raise TypeError(
                f"Expected tensor video for evaluation, got {type(video)}. "
                "Evaluation now expects `video` with shape [3,T,H,W] or [B,3,T,H,W]."
            )
        if video.ndim == 4:
            video = video.unsqueeze(0)
        if video.ndim != 5:
            raise ValueError(f"Expected video shape [3,T,H,W] or [B,3,T,H,W], got {tuple(video.shape)}")
        num_video_frames = video.shape[2]
        if num_video_frames <= 1:
            raise ValueError(f"`sample['video']` must have at least 2 frames for action evaluation, got {num_video_frames}")

        if isinstance(prompt, str):
            prompt = [prompt]
        elif isinstance(prompt, tuple):
            prompt = list(prompt)
        elif not isinstance(prompt, list):
            raise TypeError(f"Expected prompt type str/list[str], got {type(prompt)}")
        if len(prompt) != video.shape[0]:
            raise ValueError(f"Prompt batch mismatch: len(prompt)={len(prompt)} vs video batch={video.shape[0]}")

        action_horizon = None
        action = None
        if "action" in sample:
            action = sample["action"]
            if not isinstance(action, torch.Tensor):
                raise TypeError(
                    f"`sample['action']` must be a torch.Tensor, got {type(action)}"
                )
            if action.ndim == 2:
                action = action.unsqueeze(0)
            if action.ndim != 3:
                raise ValueError(f"`sample['action']` must be 3D [B, T, a_dim], got shape {tuple(action.shape)}")
            if action.shape[1] % (num_video_frames - 1) != 0:
                raise ValueError(f"`sample['action']` temporal dimension must be divisible by video frames-1={num_video_frames - 1}, got {action.shape[1]}")
            action_horizon = int(action.shape[1])

        policy_action = None
        if "policy_action" in sample:
            policy_action = sample["policy_action"]
            if not isinstance(policy_action, torch.Tensor):
                raise TypeError(f"`sample['policy_action']` must be a torch.Tensor, got {type(policy_action)}")
            if policy_action.ndim == 2:
                policy_action = policy_action.unsqueeze(0)
            if policy_action.ndim != 3:
                raise ValueError(
                    f"`sample['policy_action']` must be 3D [B, T, a_dim], got shape {tuple(policy_action.shape)}"
                )
            if policy_action.shape[0] != video.shape[0]:
                raise ValueError(
                    f"`sample['policy_action']` batch mismatch: {tuple(policy_action.shape)} vs video batch={video.shape[0]}"
                )
            if int(policy_action.shape[1]) != 1:
                raise ValueError(f"`sample['policy_action']` must be [B,1,D], got {tuple(policy_action.shape)}")
            action_horizon = int(policy_action.shape[1])

        proprio = None
        if "proprio" in sample:
            proprio = sample["proprio"]
            if not isinstance(proprio, torch.Tensor):
                raise TypeError(f"`sample['proprio']` must be a torch.Tensor, got {type(proprio)}")
            if proprio.ndim == 2:
                proprio = proprio.unsqueeze(0)
            if proprio.ndim != 3:
                raise ValueError(f"`sample['proprio']` must be 3D [B, T, d], got shape {tuple(proprio.shape)}")

        if context is not None or context_mask is not None:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must both exist in eval sample.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )

        return {
            "video": video,
            "prompt": prompt,
            "action": action,
            "policy_action": policy_action,
            "proprio": proprio,
            "context": context,
            "context_mask": context_mask,
            "action_horizon": action_horizon,
        }

    @torch.no_grad()
    def evaluate(self):
        if self.val_dataset is None:
            return None

        model = self.accelerator.unwrap_model(self.model)
        was_dit_training = model.dit.training
        model.eval()

        # eval_index = (self.global_step + self.accelerator.process_index) % len(self.val_dataset)
        rng = torch.Generator(device="cpu").manual_seed(self.global_step + self.accelerator.process_index)
        eval_index = torch.randint(0, len(self.val_dataset), (1,), generator=rng).item()
        sample = self._to_batched_eval_sample(self.val_dataset[eval_index])

        # 1. training loss
        with self.accelerator.autocast():
            val_loss, _ = model.training_loss(sample)
            val_loss = val_loss.float().item()

        prompt = sample["prompt"][0]
        video0 = sample["video"][0] # Tensor [3, T, H, W] in (-1, 1)
        action = sample["action"][0] if "action" in sample and sample["action"] is not None else None
        proprio = sample["proprio"][0, 0] if "proprio" in sample and sample["proprio"] is not None else None # from [1, T, d] to [d]
        input_image = video0[:, 0].unsqueeze(0)
        _, num_frames, _, _ = video0.shape

        # 2. inference and video saving
        infer_kwargs = {
            "input_image": input_image,
            "num_frames": num_frames,
            "action": action,
            "action_horizon": sample['action_horizon'],
            "proprio": proprio,
            "text_cfg_scale": 1.0,
            "action_cfg_scale": 1.0,
            "num_inference_steps": self.eval_num_inference_steps,
            "seed": 42,
            "tiled": False,
        }
        if sample["context"] is not None:
            infer_kwargs["prompt"] = None
            infer_kwargs["context"] = sample["context"][0]
            infer_kwargs["context_mask"] = sample["context_mask"][0]
        else:
            infer_kwargs["prompt"] = prompt

        pred = model.infer(
            **infer_kwargs,
        )

        pred_video = pred["video"]
        pred_action = pred.get("action", None)

        # 3. inference metrics against GT video
        pred_video_tensor = pil_frames_to_video_tensor(pred_video)
        gt_video_tensor = ((video0.detach().float().cpu().clamp(-1.0, 1.0) + 1.0) * 0.5).contiguous()

        assert pred_video_tensor.shape == gt_video_tensor.shape, (
            "Eval infer prediction/GT shape mismatch: "
            f"pred={tuple(pred_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
        )

        psnr_rollout_vs_gt = video_psnr(pred=pred_video_tensor, target=gt_video_tensor)
        ssim_rollout_vs_gt = video_ssim(pred=pred_video_tensor, target=gt_video_tensor)

        action_l1 = None
        action_l2 = None
        metric_action = sample["policy_action"][0] if sample.get("policy_action") is not None else action
        if metric_action is not None and pred_action is not None:
            if sample["proprio"] is None:
                raise ValueError("Eval sample must contain `proprio` for action denormalization.")
            proprio = sample["proprio"].detach().to(device="cpu", dtype=torch.float32)

            processor = self.val_dataset.lerobot_dataset.processor

            denorm_actions = {}
            action_meta = processor.shape_meta["action"]
            state_meta = processor.shape_meta["state"]
            for action_name, raw_action in (("pred", pred_action), ("gt", metric_action)):
                if not isinstance(raw_action, torch.Tensor):
                    raise TypeError(f"{action_name} action must be a torch.Tensor, got {type(raw_action)}")
                if raw_action.ndim == 2:
                    action_btd = raw_action.unsqueeze(0)
                elif raw_action.ndim == 3 and raw_action.shape[0] == 1:
                    action_btd = raw_action
                else:
                    raise ValueError(
                        f"{action_name} action must have shape [T, D] or [1, T, D], got {tuple(raw_action.shape)}"
                    )
                action_btd = action_btd.detach().to(device="cpu", dtype=torch.float32)

                batch = {
                    "action": action_btd,
                    "state": proprio,
                }
                batch = processor.action_state_merger.backward(batch)
                batch = processor.normalizer.backward(batch)
                merged_batch = {
                    "action": {meta["key"]: batch["action"][meta["key"]].squeeze(0) for meta in action_meta},
                    "state": {meta["key"]: batch["state"][meta["key"]].squeeze(0) for meta in state_meta},
                }
                merged_batch = processor.action_state_merger.forward(merged_batch)
                denorm_action = merged_batch["action"].unsqueeze(0)
                if denorm_action.ndim != 3 or denorm_action.shape[0] != 1:
                    raise ValueError(
                        f"Denormalized {action_name} action must have shape [1, T, D], got {tuple(denorm_action.shape)}"
                    )
                denorm_actions[action_name] = denorm_action

            pred_action_denorm = denorm_actions["pred"]
            gt_action_denorm = denorm_actions["gt"]

            if pred_action_denorm.shape != gt_action_denorm.shape:
                raise ValueError(
                    "Predicted action/GT action shape mismatch after denormalization: "
                    f"pred={tuple(pred_action_denorm.shape)} vs gt={tuple(gt_action_denorm.shape)}"
                )
            action_diff = pred_action_denorm - gt_action_denorm
            action_l1 = action_diff.abs().mean().item()
            action_l2 = action_diff.pow(2).mean().item()

        # 4. VAE reconstruction metrics against GT video
        gt_video_batch = video0.unsqueeze(0).to(device=model.device, dtype=model.torch_dtype)
        vae_latents = model._encode_video_latents(gt_video_batch, tiled=False)
        vae_recon_video = model._decode_latents(vae_latents, tiled=False)
        vae_video_tensor = pil_frames_to_video_tensor(vae_recon_video)

        assert vae_video_tensor.shape == gt_video_tensor.shape, (
            "Eval VAE reconstruction/GT shape mismatch: "
            f"vae={tuple(vae_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
        )

        psnr_decode_vs_gt = video_psnr(pred=vae_video_tensor, target=gt_video_tensor)
        ssim_decode_vs_gt = video_ssim(pred=vae_video_tensor, target=gt_video_tensor)

        psnr_rollout_vs_decode = video_psnr(pred=pred_video_tensor, target=vae_video_tensor)
        ssim_rollout_vs_decode = video_ssim(pred=pred_video_tensor, target=vae_video_tensor)

        stitched_video_tensor = torch.cat(
            [pred_video_tensor, vae_video_tensor, gt_video_tensor],
            dim=2,
        ).contiguous()
        stitched_frames = []
        for t in range(stitched_video_tensor.shape[1]):
            frame = (stitched_video_tensor[:, t].permute(1, 2, 0).clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)
            stitched_frames.append(Image.fromarray(frame))

        video_path = os.path.join(
            self.eval_dir,
            f"step_{self.global_step:06d}_rank_{self.accelerator.process_index:03d}.mp4",
        )
        save_mp4(stitched_frames, video_path, fps=8)

        local_metrics = torch.tensor(
            [
                float(val_loss),
                float(psnr_rollout_vs_gt),
                float(ssim_rollout_vs_gt),
                float(psnr_rollout_vs_decode),
                float(ssim_rollout_vs_decode),
                float(psnr_decode_vs_gt),
                float(ssim_decode_vs_gt),
                float(action_l2) if action_l2 is not None else -1.0,
                float(action_l1) if action_l1 is not None else -1.0,
            ],
            device=self.accelerator.device,
            dtype=torch.float32,
        ).unsqueeze(0)
        gathered_metrics = self.accelerator.gather_for_metrics(local_metrics)
        mean_metrics = gathered_metrics[:, :7].mean(dim=0)
        action_l2_mean = gathered_metrics[:, 7].mean().item() if action_l2 is not None else None
        action_l1_mean = gathered_metrics[:, 8].mean().item() if action_l1 is not None else None

        if was_dit_training:
            self._set_dit_only_train_mode()

        result = {
            "val_loss": float(mean_metrics[0].item()),
            "psnr_rg": float(mean_metrics[1].item()),
            "ssim_rg": float(mean_metrics[2].item()),
            "psnr_rd": float(mean_metrics[3].item()),
            "ssim_rd": float(mean_metrics[4].item()),
            "psnr_dg": float(mean_metrics[5].item()),
            "ssim_dg": float(mean_metrics[6].item()),
            "video_path": video_path,
        }
        if action_l2_mean is not None:
            result["action_l2"] = float(action_l2_mean)
        if action_l1_mean is not None:
            result["action_l1"] = float(action_l1_mean)
        return result

    def _should_run_open_loop_wam_eval(self) -> bool:
        cfg = self.open_loop_wam_eval_cfg
        enabled = bool(cfg.get("enabled", False))
        every = int(cfg.get("every", 0))
        return (
            enabled
            and every > 0
            and self.val_dataset is not None
            and self.global_step > 0
            and self.global_step % every == 0
        )

    def _run_open_loop_wam_eval_if_due(self):
        if not self._should_run_open_loop_wam_eval():
            return None

        metrics = None
        error_to_raise = None
        if self.accelerator.is_main_process:
            cfg = self.open_loop_wam_eval_cfg
            model = self.accelerator.unwrap_model(self.model)
            was_dit_training = model.dit.training
            try:
                model.eval()
                with self.accelerator.autocast():
                    metrics = run_autoregressive_open_loop_wam_eval(
                        model=model,
                        dataset=self.val_dataset,
                        output_dir=self.open_loop_wam_eval_dir,
                        global_step=self.global_step,
                        num_samples=int(cfg.get("num_samples", 1)),
                        rollout_chunks=int(cfg.get("rollout_chunks", 4)),
                        chunk_stride=int(cfg.get("chunk_stride", 32)),
                        num_inference_steps=int(cfg.get("num_inference_steps", self.eval_num_inference_steps)),
                        seed=int(cfg.get("seed", self.seed)),
                        save_video=bool(cfg.get("save_video", True)),
                        video_fps=int(cfg.get("video_fps", 8)),
                        tiled=bool(cfg.get("tiled", False)),
                    )
            except Exception as exc:
                logger.exception("[wam_open_loop] step=%d failed: %s", self.global_step, exc)
                error_to_raise = exc
                metrics = {
                    "error": 1.0,
                    "error_message": str(exc),
                }
            finally:
                if was_dit_training:
                    self._set_dit_only_train_mode()

        self.accelerator.wait_for_everyone()
        if (
            self.accelerator.is_main_process
            and error_to_raise is not None
            and bool(self.open_loop_wam_eval_cfg.get("fail_on_error", False))
        ):
            raise error_to_raise
        return metrics

    def _log_open_loop_wam_eval(self, metrics: dict[str, Any]) -> None:
        if not self.accelerator.is_main_process:
            return
        payload = {
            "wam_open_loop/error": float(metrics.get("error", 0.0)),
        }
        for key in ("num_samples", "rollout_chunks", "frames", "psnr_gt_mean", "ssim_gt_mean"):
            if key in metrics:
                payload[f"wam_open_loop/{key}"] = float(metrics[key])

        video_paths = list(metrics.get("video_paths", []) or [])
        if video_paths and self.wandb_run is not None:
            try:
                import wandb

                max_videos = int(self.open_loop_wam_eval_cfg.get("max_wandb_videos", 1))
                video_fps = int(self.open_loop_wam_eval_cfg.get("video_fps", 8))
                for video_idx, video_path in enumerate(video_paths[:max(0, max_videos)]):
                    payload[f"wam_open_loop/video_{video_idx}"] = wandb.Video(video_path, fps=video_fps, format="mp4")
            except ImportError:
                logger.warning("wandb is not installed; skipping WAM open-loop video upload.")
        self._wandb_log(payload)

        if float(metrics.get("error", 0.0)) > 0:
            logger.warning("[wam_open_loop] step=%d error=%s", self.global_step, metrics.get("error_message", "unknown"))
        else:
            logger.info(
                "[wam_open_loop] step=%d samples=%d frames=%d psnr_gt=%.4f ssim_gt=%.4f summary=%s",
                self.global_step,
                int(metrics.get("num_samples", 0)),
                int(metrics.get("frames", 0)),
                float(metrics.get("psnr_gt_mean", 0.0)),
                float(metrics.get("ssim_gt_mean", 0.0)),
                metrics.get("summary_path", ""),
            )

    def _save_weights_checkpoint(self, step_tag: str):
        model = self.accelerator.unwrap_model(self.model)
        ckpt_path = os.path.join(self.weights_dir, f"{step_tag}.pt")
        tmp_path = os.path.join(self.weights_dir, f".{step_tag}.pt.tmp")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        try:
            model.save_checkpoint(tmp_path, optimizer=None, step=self.global_step)
            os.replace(tmp_path, ckpt_path)
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise
        return ckpt_path

    def _save_trainer_state(self, state_path: str):
        state_file = os.path.join(state_path, "trainer_state.json")
        payload = {
            "global_step": int(self.global_step),
            "epoch": int(self.epoch),
            "batch_in_epoch": int(self.batch_in_epoch),
        }
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)

    def _write_checkpoint_manifest(self, state_path: str, step_tag: str, weights_path: str | None):
        manifest_file = os.path.join(state_path, "checkpoint_manifest.json")
        payload = {
            "checkpoint_version": 1,
            "complete": True,
            "step_tag": step_tag,
            "global_step": int(self.global_step),
            "epoch": int(self.epoch),
            "batch_in_epoch": int(self.batch_in_epoch),
            "weights_path": weights_path,
            "world_size": int(self.accelerator.num_processes),
            "zero_stage": self.accelerator.state.deepspeed_plugin.deepspeed_config.get("zero_optimization", {}).get(
                "stage", "unknown"
            ),
        }
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)

    def _collective_error_count(self, failed: bool) -> int:
        flag = torch.tensor([1 if failed else 0], device=self.accelerator.device, dtype=torch.int32)
        return int(self.accelerator.gather(flag).sum().item())

    def _collective_disk_preflight(self, path: str, min_free_gb: float, *, label: str) -> tuple[bool, float]:
        if min_free_gb <= 0:
            return True, float("inf")
        free_gb = float(shutil.disk_usage(path).free / 1024**3)
        ok = free_gb >= float(min_free_gb)
        ok_flag = torch.tensor([1 if ok else 0], device=self.accelerator.device, dtype=torch.int32)
        free_value = torch.tensor([free_gb], device=self.accelerator.device, dtype=torch.float32)
        all_ok = int(self.accelerator.gather(ok_flag).min().item()) == 1
        min_seen_free_gb = float(self.accelerator.gather(free_value).min().item())
        if not all_ok and self.accelerator.is_main_process:
            logger.error(
                "[ckpt] %s checkpoint preflight failed at step=%d: min free disk %.2f GiB < required %.2f GiB.",
                label,
                self.global_step,
                min_seen_free_gb,
                min_free_gb,
            )
        return all_ok, min_seen_free_gb

    def _should_save_full_state(self) -> tuple[bool, str]:
        if not self.checkpoint_save_full_state:
            return False, "full Accelerate/DeepSpeed state save disabled by config"
        min_free_gb = float(self.checkpoint_full_state_min_free_gb)
        ok, min_seen_free_gb = self._collective_disk_preflight(
            self.state_dir,
            min_free_gb,
            label="full-state",
        )
        if not ok:
            return False, f"free disk {min_seen_free_gb:.2f} GiB < required {min_free_gb:.2f} GiB"
        return True, ""

    def _prune_old_full_state_checkpoints(self):
        keep = int(self.checkpoint_keep_last_full_states)
        if keep <= 0 or not self.accelerator.is_main_process:
            return
        state_root = Path(self.state_dir)
        candidates = [path for path in state_root.iterdir() if self._is_complete_state_checkpoint(path)]
        candidates.sort(key=lambda path: self._step_from_path(path) or -1)
        for stale in candidates[:-keep]:
            deleting = stale.with_name(f".deleting_{stale.name}_{int(time.time())}")
            try:
                os.replace(stale, deleting)
                shutil.rmtree(deleting)
                logger.info("[ckpt] pruned old full state checkpoint: %s", stale)
            except Exception:
                logger.exception("[ckpt] failed to prune old full state checkpoint: %s", stale)

    def _promote_full_state_checkpoint(self, tmp_state_path: str, state_path: str, step_tag: str, ckpt_path: str | None):
        final_state = Path(state_path)
        if self._is_complete_state_checkpoint(final_state):
            logger.info("[ckpt] full state already complete for %s; skipping duplicate state promotion.", step_tag)
            shutil.rmtree(tmp_state_path, ignore_errors=True)
            return

        self._save_trainer_state(tmp_state_path)
        self._write_checkpoint_manifest(tmp_state_path, step_tag=step_tag, weights_path=ckpt_path)

        backup_path = None
        if os.path.exists(state_path):
            backup_path = os.path.join(self.state_dir, f".replacing_{step_tag}_{int(time.time())}")
            os.replace(state_path, backup_path)
        try:
            os.replace(tmp_state_path, state_path)
        except Exception:
            if backup_path is not None and os.path.exists(backup_path) and not os.path.exists(state_path):
                os.replace(backup_path, state_path)
            raise
        if backup_path is not None:
            shutil.rmtree(backup_path, ignore_errors=True)

    def save_checkpoint(self):
        step_tag = f"step_{self.global_step:06d}"

        self.accelerator.wait_for_everyone()
        weights_ok, weights_free_gb = self._collective_disk_preflight(
            self.weights_dir,
            float(self.checkpoint_weight_min_free_gb),
            label="weights",
        )
        if not weights_ok:
            raise RuntimeError(
                f"[ckpt] refusing to save weights checkpoint at step={self.global_step}: "
                f"free disk {weights_free_gb:.2f} GiB < required {float(self.checkpoint_weight_min_free_gb):.2f} GiB"
            )

        ckpt_path = None
        weight_error = None
        if self.accelerator.is_main_process:
            try:
                ckpt_path = self._save_weights_checkpoint(step_tag=step_tag)
            except Exception as exc:
                weight_error = exc
        weight_error_count = self._collective_error_count(weight_error is not None)
        if weight_error_count > 0:
            if weight_error is not None:
                logger.exception("[ckpt] main rank failed to save weights checkpoint.")
            raise RuntimeError(f"[ckpt] weights checkpoint save failed on {weight_error_count} rank(s)")
        self.accelerator.wait_for_everyone()

        state_path = os.path.join(self.state_dir, step_tag)
        should_save_state, skip_reason = self._should_save_full_state()
        if not should_save_state:
            message = (
                f"[ckpt] full state checkpoint unavailable at step={self.global_step}: {skip_reason}. "
                "Exact optimizer/dataloader/RNG resume would not be available."
            )
            if self.checkpoint_require_full_state:
                raise RuntimeError(message)
            if self.accelerator.is_main_process:
                logger.warning("%s Continuing because checkpoint.require_full_state=false.", message)
            self.accelerator.wait_for_everyone()
            return {"weights_path": ckpt_path, "state_path": None, "state_skipped": True}

        tmp_state_path = os.path.join(self.state_dir, f".{step_tag}.tmp")
        if self.accelerator.is_main_process:
            shutil.rmtree(tmp_state_path, ignore_errors=True)
        self.accelerator.wait_for_everyone()
        ensure_dir(tmp_state_path)

        state_error = None
        try:
            self.accelerator.save_state(output_dir=tmp_state_path)
        except Exception as exc:
            state_error = exc

        error_flag = torch.tensor([1 if state_error is not None else 0], device=self.accelerator.device)
        error_count = int(self.accelerator.gather(error_flag).sum().item())
        if error_count > 0:
            if state_error is not None:
                logger.exception("[ckpt] local rank failed to save full state checkpoint.")
            if self.accelerator.is_main_process:
                logger.error(
                    "[ckpt] full state checkpoint failed on %d rank(s); deleting temporary state directory %s. "
                    "Weights checkpoint remains available for weights-only continuation.",
                    error_count,
                    tmp_state_path,
                )
                shutil.rmtree(tmp_state_path, ignore_errors=True)
            self.accelerator.wait_for_everyone()
            if self.checkpoint_require_full_state:
                raise RuntimeError(f"[ckpt] full state checkpoint save failed on {error_count} rank(s)")
            return {"weights_path": ckpt_path, "state_path": None, "state_skipped": True}

        final_error = None
        if self.accelerator.is_main_process:
            try:
                self._promote_full_state_checkpoint(
                    tmp_state_path,
                    state_path,
                    step_tag=step_tag,
                    ckpt_path=ckpt_path,
                )
                self._prune_old_full_state_checkpoints()
            except Exception as exc:
                final_error = exc
        final_error_count = self._collective_error_count(final_error is not None)
        if final_error_count > 0:
            if final_error is not None:
                logger.exception("[ckpt] main rank failed to finalize full state checkpoint.")
            raise RuntimeError(f"[ckpt] full state checkpoint finalization failed on {final_error_count} rank(s)")
        self.accelerator.wait_for_everyone()

        return {"weights_path": ckpt_path, "state_path": state_path}

    def load_training_state(self, state_dir: str):
        state_path = Path(state_dir)
        state_file = state_path / "trainer_state.json"
        if not self._is_complete_state_checkpoint(state_path):
            raise FileNotFoundError(
                f"State directory {state_dir} is not a complete exact-resume checkpoint. "
                "Use `checkpoint.init_from_weights` for weights-only continuation."
            )
        self.accelerator.load_state(input_dir=state_dir)
        with open(state_file, "r", encoding="utf-8") as f:
            payload = json.load(f)
        self.global_step = int(payload["global_step"])

        if "epoch" in payload and "batch_in_epoch" in payload:
            self.epoch = int(payload["epoch"])
            self.batch_in_epoch = int(payload["batch_in_epoch"])
            self.train_sampler.set_epoch_offset(self.epoch)
            self.train_sampler.set_resume_batch_offset(self.batch_in_epoch)
            logger.info(
                "Restored dataloader progress: epoch=%d batch_in_epoch=%d sample_offset=%d",
                self.epoch,
                self.batch_in_epoch,
                self.batch_in_epoch * self.batch_size * self.accelerator.num_processes,
            )
        else:
            raise FileNotFoundError(
                f"State file {state_file} is missing dataloader progress. "
                "It cannot provide exact resume."
            )
        self.accelerator.wait_for_everyone()
        logger.info("Loaded accelerate training state from %s at step=%d", state_dir, self.global_step)

    def train(self):
        self._set_dit_only_train_mode()

        unwrapped_model = self.accelerator.unwrap_model(self.model)

        if self.max_steps is None:
            raise ValueError("`max_steps` must be set before entering the while-step training loop.")

        logger.info("Starting training with max_steps=%d.", self.max_steps)
        data_iter = iter(self.train_loader)
        self.run_start_step = self.global_step
        self.run_start_time = time.perf_counter()

        stop_after_max_steps = False
        torch_profiler_for_summary = None
        with self._torch_profiler_context() as torch_profiler:
            torch_profiler_for_summary = torch_profiler
            while self.global_step < self.max_steps:
                self._profile_start_step_if_needed()
                data_t0 = self._profile_measure_start()
                try:
                    sample = next(data_iter)
                    self.batch_in_epoch += 1
                except StopIteration:
                    self.epoch += 1
                    self.batch_in_epoch = 0
                    self.train_sampler.clear_resume_batch_offset()
                    data_iter = iter(self.train_loader)
                    continue
                self._profile_measure_end("data_wait_s", data_t0)

                with self.accelerator.accumulate(self.model):
                    train_model = self.model if hasattr(self.model, "training_loss") else self.accelerator.unwrap_model(self.model)

                    forward_t0 = self._profile_measure_start()
                    with self.accelerator.autocast():
                        loss, loss_dict = train_model.training_loss(sample)
                    self._profile_measure_end("forward_loss_s", forward_t0)

                    backward_t0 = self._profile_measure_start()
                    self.accelerator.backward(loss)
                    self._profile_measure_end("backward_s", backward_t0)

                    if self.accelerator.sync_gradients:
                        clip_t0 = self._profile_measure_start()
                        grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        self._profile_measure_end("grad_clip_s", clip_t0)

                        optimizer_t0 = self._profile_measure_start()
                        self.optimizer.step()
                        self._profile_measure_end("optimizer_step_s", optimizer_t0)

                        scheduler_t0 = self._profile_measure_start()
                        if not self.accelerator.optimizer_step_was_skipped:
                            self.scheduler.step()
                        self._profile_measure_end("scheduler_step_s", scheduler_t0)

                        zero_t0 = self._profile_measure_start()
                        self.optimizer.zero_grad(set_to_none=True)
                        self._profile_measure_end("zero_grad_s", zero_t0)

                        self.global_step += 1
                        log_t0 = self._profile_measure_start()
                        global_loss = float(
                            self.accelerator.gather(loss.detach().float().reshape(1)).mean().item()
                        )
                        global_loss_metrics = {}
                        for key, value in loss_dict.items():
                            metric_tensor = torch.tensor(float(value), device=loss.device, dtype=torch.float32).reshape(1)
                            global_loss_metrics[key] = float(
                                self.accelerator.gather(metric_tensor).mean().item()
                            )
                        grad_norm_tensor = torch.tensor(grad_norm, device=loss.device, dtype=torch.float32)
                        global_grad_norm = float(self.accelerator.gather(grad_norm_tensor).mean().item())
                        self._profile_measure_end("log_gather_s", log_t0)

                        current_lr = float(self.optimizer.param_groups[0]["lr"])
                        self._profile_complete_optimizer_step(
                            loss=global_loss,
                            loss_metrics=global_loss_metrics,
                            lr=current_lr,
                            grad_norm=global_grad_norm,
                        )
                        if torch_profiler is not None:
                            torch_profiler.step()

                        if self.log_every > 0 and self.global_step % self.log_every == 0 and self.accelerator.is_main_process:
                            eta_str, steps_per_sec = self._estimate_eta()
                            description = "[train] epoch=%d step=%d/%d loss=%.4f " % (
                                self.epoch,
                                self.global_step,
                                self.max_steps,
                                global_loss,
                            )
                            if global_loss_metrics:
                                detail_str = " ".join([f"{k}={v:.4f}" for k, v in sorted(global_loss_metrics.items())])
                                description += detail_str + " "
                            description += "lr=%.2e speed=%.2f step/s, %.2f samples/s eta=%s" % (
                                current_lr,
                                steps_per_sec,
                                steps_per_sec * self.batch_size * self.accelerator.num_processes,
                                eta_str,
                            )
                            logger.info(description)

                            wandb_payload = {
                                "train/loss": global_loss,
                                "train/grad_norm": global_grad_norm,
                                "train/lr": current_lr,
                                "performance/steps_per_sec": steps_per_sec,
                                "performance/samples_per_sec": steps_per_sec * self.batch_size * self.accelerator.num_processes,
                            }
                            for key, value in global_loss_metrics.items():
                                wandb_payload[f"train/{key}"] = value
                            self._wandb_log(wandb_payload)

                        if (
                            self.eval_every > 0
                            and self.val_dataset is not None
                            and self.global_step % self.eval_every == 0
                        ):
                            metrics = self.evaluate()
                            self.accelerator.wait_for_everyone()
                            if metrics is not None and self.accelerator.is_main_process:
                                description = "[eval] step=%d val_loss=%.4f infer_psnr=%.4f infer_ssim=%.4f" % (
                                    self.global_step,
                                    metrics["val_loss"],
                                    metrics["psnr_rd"],
                                    metrics["ssim_rd"],
                                )
                                if "action_l2" in metrics:
                                    description += " action_l2=%.4f" % metrics["action_l2"]
                                if "action_l1" in metrics:
                                    description += " action_l1=%.4f" % metrics["action_l1"]
                                logger.info(description)
                                eval_payload = {
                                    "eval/val_loss": float(metrics["val_loss"]),
                                    "eval/psnr_rg": float(metrics["psnr_rg"]),
                                    "eval/ssim_rg": float(metrics["ssim_rg"]),
                                    "eval/psnr_rd": float(metrics["psnr_rd"]),
                                    "eval/ssim_rd": float(metrics["ssim_rd"]),
                                    "eval/psnr_dg": float(metrics["psnr_dg"]),
                                    "eval/ssim_dg": float(metrics["ssim_dg"]),
                                }
                                if "action_l2" in metrics:
                                    eval_payload["eval/action_l2"] = float(metrics["action_l2"])
                                if "action_l1" in metrics:
                                    eval_payload["eval/action_l1"] = float(metrics["action_l1"])
                                self._wandb_log(eval_payload)

                        open_loop_metrics = self._run_open_loop_wam_eval_if_due()
                        if open_loop_metrics is not None:
                            self._log_open_loop_wam_eval(open_loop_metrics)

                        if self.save_every > 0 and self.global_step % self.save_every == 0:
                            ckpt_info = self.save_checkpoint()
                            if self.accelerator.is_main_process:
                                logger.info(
                                    "[ckpt] step=%d weights=%s state=%s",
                                    self.global_step,
                                    ckpt_info["weights_path"],
                                    ckpt_info["state_path"],
                                )

                        if self.global_step >= self.max_steps:
                            if self.save_final_checkpoint:
                                ckpt_info = self.save_checkpoint()
                                if self.accelerator.is_main_process:
                                    logger.info(
                                        "[done] max_steps reached step=%d weights=%s state=%s",
                                        self.global_step,
                                        ckpt_info["weights_path"],
                                        ckpt_info["state_path"],
                                    )
                            elif self.accelerator.is_main_process:
                                logger.info(
                                    "[done] max_steps reached step=%d; final checkpoint skipped.",
                                    self.global_step,
                                )
                            stop_after_max_steps = True
                            break

        self._write_torch_profiler_summary(torch_profiler_for_summary)
        if stop_after_max_steps:
            return
        if self.save_final_checkpoint:
            ckpt_info = self.save_checkpoint()
            if self.accelerator.is_main_process:
                logger.info(
                    "[done] training finished step=%d weights=%s state=%s",
                    self.global_step,
                    ckpt_info["weights_path"],
                    ckpt_info["state_path"],
                )
        elif self.accelerator.is_main_process:
            logger.info("[done] training finished step=%d; final checkpoint skipped.", self.global_step)
