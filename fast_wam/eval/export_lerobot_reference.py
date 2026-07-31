#!/usr/bin/env python3
"""Export LeRobot action chunks and closed-loop outcomes for the fixed manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from fast_wam.components import prepare_camera_image
from fast_wam.config import FastWAMConfig
from fast_wam.libero import make_env, preprocess_observation
from fast_wam.policy import MinMaxStats


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--assets", required=True, help="Local Wan2.2 Diffusers snapshot")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--manifest", default=str(Path(__file__).with_name("manifest.json")))
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-action-steps", type=int, default=10)
    return parser.parse_args()


def build_reference(args):
    from lerobot.policies.fastwam.configuration_fastwam import FastWAMConfig as LeRobotConfig
    from lerobot.policies.fastwam.modeling_fastwam import FastWAMPolicy

    # Keep this exporter fully offline and point both frozen components at the
    # exact local Diffusers snapshot used by the Megatron runner.
    import lerobot.policies.fastwam.wan.components as components

    components.WAN22_DIFFUSERS_MODEL_ID = str(args.assets)
    cfg = LeRobotConfig.from_pretrained(args.checkpoint)
    cfg.device = "cuda"
    cfg.torch_dtype = "float32"
    cfg.n_action_steps = args.n_action_steps
    cfg.tokenizer_model_id = str(args.tokenizer)
    cfg.text_encoder_model_id = str(args.assets)
    return FastWAMPolicy.from_pretrained(args.checkpoint, config=cfg, local_files_only=True)


@torch.no_grad()
def predict(policy, cfg, stats, images, state, task):
    image = prepare_camera_image(images, cfg.image_size)
    proprio = stats.normalize_state(state)
    prompt = cfg.prompt_template.format(task=task)
    normalized = policy.model.infer_action(
        prompt=prompt,
        input_image=image,
        action_horizon=cfg.action_horizon,
        proprio=proprio,
        num_inference_steps=cfg.num_inference_steps,
        sigma_shift=cfg.sigma_shift,
        seed=cfg.inference_seed,
        rand_device="cpu",
    )["action"]
    return stats.unnormalize_action(normalized)


def main():
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(args.manifest).read_text())
    cfg = FastWAMConfig.from_pretrained(args.checkpoint)
    stats = MinMaxStats.from_pretrained(args.checkpoint)
    policy = build_reference(args)
    results = []
    for case in manifest["cases"]:
        env = make_env(case)
        observation, _ = env.reset(seed=0)
        records = []
        success = False
        step = 0
        try:
            while step < env._max_episode_steps and not success:
                images, state = preprocess_observation(observation)
                task = env.task_description
                chunk = predict(policy, cfg, stats, images, state, task)
                records.append(
                    {
                        "images": {key: (value * 255).round().byte() for key, value in images.items()},
                        "state": state,
                        "task": task,
                        "action": chunk,
                    }
                )
                for action in chunk[: args.n_action_steps]:
                    observation, _, terminated, _, info = env.step(action.numpy())
                    step += 1
                    success = bool(info.get("is_success", False))
                    if terminated or success or step >= env._max_episode_steps:
                        break
        finally:
            env.close()
        torch.save({"case": case, "records": records, "success": success}, output / f"{case['id']}.pt")
        results.append({**case, "success": success, "steps": step, "replans": len(records)})
        print(json.dumps(results[-1], ensure_ascii=False), flush=True)
    summary = {
        "engine": "lerobot",
        "precision": "fp32",
        "n_action_steps": args.n_action_steps,
        "success_vector": [item["success"] for item in results],
        "results": results,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
