# -*- coding: utf-8 -*-
"""
Offline evaluation of a Flow Matching Policy checkpoint on LIBERO tasks.

Usage (from repo root):
  python -m scripts.eval \
      --checkpoint path/to/after_task_09_ema.pt \
      --benchmark libero_object \
      --num-episodes 20 \
      [--task-indices 0 1 2]  # defaults to all tasks
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

from model.flow_policy import FlowPolicy
from scripts.evaluation import (
    evaluate_checkpoint_on_all_tasks,
    compute_nbt,
    compute_average_sr,
    save_results_json,
    save_results_csv,
    plot_performance_matrix,
)
from libero.libero.benchmark import get_benchmark


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    action_mean = ckpt.get("action_mean")
    action_std = ckpt.get("action_std")
    wandb_run_id = ckpt.get("wandb_run_id")

    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"  Task idx: {ckpt.get('task_idx', 'unknown')}")

    # wandb resume
    wandb_cfg = cfg.get("wandb", {})
    use_wandb = wandb_cfg.get("enabled", False) and wandb_run_id is not None
    if use_wandb:
        try:
            import wandb
            date_str = datetime.now().strftime("%m%d")
            run_name = f"{wandb_cfg.get('name', 'eval')}_{date_str}"
            wandb.init(
                entity=wandb_cfg["entity"],
                project=wandb_cfg["project"],
                group=wandb_cfg.get("group"),
                name=run_name,
                resume="must",
                id=wandb_run_id,
            )
            print(f"wandb resumed run id={wandb_run_id}")
        except ImportError:
            print("wandb not available, disabling wandb logging")
            use_wandb = False
        except Exception as e:
            print(f"wandb resume failed: {e}")
            use_wandb = False

    # Build model
    model = FlowPolicy(cfg).to(device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # Benchmark
    benchmark_name = args.benchmark or cfg.get("benchmark", {}).get("name", "libero_object")
    task_order = args.task_order or cfg.get("benchmark", {}).get("task_order_index", 0)
    benchmark = get_benchmark(benchmark_name)(task_order_index=task_order)
    n_tasks = benchmark.get_num_tasks()
    task_names = benchmark.get_task_names()

    data_root = args.data_root or cfg.get("benchmark", {}).get("data_root", "/workspace/data")
    data_cfg = cfg["data"]
    eval_cfg = cfg.get("evaluation", {})
    low_dim_keys = [k for k in data_cfg["obs_keys"] if "image" not in k]

    task_indices = args.task_indices if args.task_indices else list(range(n_tasks))

    print(f"\nEvaluating on {len(task_indices)} task(s) from benchmark '{benchmark_name}'...")

    results = evaluate_checkpoint_on_all_tasks(
        model=model,
        benchmark=benchmark,
        task_indices=task_indices,
        num_episodes=args.num_episodes or eval_cfg.get("num_episodes", 20),
        max_steps=eval_cfg.get("max_steps_per_episode", 600),
        action_execution_horizon=eval_cfg.get("action_execution_horizon", 8),
        action_mean=action_mean,
        action_std=action_std,
        obs_horizon=data_cfg["obs_horizon"],
        image_size=tuple(data_cfg.get("image_size", [128, 128])),
        use_eye_in_hand=data_cfg.get("use_eye_in_hand", True),
        low_dim_keys=low_dim_keys,
        device=device,
        use_ddim=False,   # FM: Euler integration via model.sample_action()
        ddim_steps=eval_cfg.get("num_flow_steps", model.num_flow_steps),
        seed=args.seed,
    )

    print("\n=== Results ===")
    for task_idx, sr in results.items():
        print(f"  Task {task_idx:2d} ({task_names[task_idx][:40]}): SR = {sr:.4f}")
    avg_sr = np.mean(list(results.values()))
    print(f"\n  Average SR: {avg_sr:.4f}")

    if use_wandb:
        import wandb as _wandb
        eval_log = {f"eval/task{k}_sr": float(sr) for k, sr in results.items()}
        eval_log["eval/avg_sr"] = float(avg_sr)
        _wandb.log(eval_log)
        _wandb.finish()

    # Save results
    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        results_path = out_dir / "eval_results.json"
        with open(results_path, "w") as f:
            json.dump({
                "checkpoint": str(args.checkpoint),
                "benchmark": benchmark_name,
                "task_indices": task_indices,
                "results": {str(k): float(v) for k, v in results.items()},
                "avg_sr": float(avg_sr),
            }, f, indent=2)
        print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--benchmark", type=str, default=None)
    parser.add_argument("--task-order", type=int, default=None)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--task-indices", type=int, nargs="+", default=None)
    parser.add_argument("--num-episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()
    main(args)
