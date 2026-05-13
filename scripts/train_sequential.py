# -*- coding: utf-8 -*-
"""
Sequential Continual Learning with Experience Replay (ER) — FM version.

Trains FlowPolicy sequentially across N tasks with an optional replay buffer.
No SDFT; for SDFT use train_sequential_sdft.py.

Usage (from repo root):
  python -m scripts.train_sequential \
      --config configs/cl_object.yaml [--skip-eval]
"""

import os
import math
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

import yaml
import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from model.flow_policy import FlowPolicy, EMAModel
from scripts.datasets import (
    SingleTaskDataset,
    create_single_task_dataloader,
    compute_global_action_stats,
)
from scripts.utils_er import ReplayMemory, cycle, merge_batches, split_batch_size
from scripts.evaluation import (
    evaluate_checkpoint_on_all_tasks,
    compute_nbt,
    compute_average_sr,
    compute_average_sr_per_stage,
    save_results_json,
    save_results_csv,
    plot_performance_matrix,
    plot_forgetting_summary,
)

from libero.libero.benchmark import get_benchmark


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _checkpoint_step(path: Path) -> int:
    digits = "".join(ch if ch.isdigit() else " " for ch in path.stem).split()
    return int(digits[-1]) if digits else -1


def _prepare_run_dirs(cfg: dict) -> tuple:
    log_cfg = cfg["logging"]
    exp_name = log_cfg.get("exp_name") or cfg.get("exp_name")

    if exp_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path("output") / f"{exp_name}_{timestamp}"
        ckpt_dir = run_dir / "checkpoints"
        results_dir = run_dir / "results"
        cfg["run_name"] = run_dir.name
        cfg["run_dir"] = str(run_dir.resolve())
        cfg["logging"]["checkpoint_dir"] = str(ckpt_dir.resolve())
        cfg["logging"]["results_dir"] = str(results_dir.resolve())
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "config_resolved.yaml", "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
    else:
        run_dir = None
        ckpt_dir = Path(log_cfg["checkpoint_dir"])
        results_dir = Path(log_cfg["results_dir"])

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, ckpt_dir, results_dir


def _init_tensorboard_writer(cfg: dict, results_dir: Path):
    log_cfg = cfg["logging"]
    if not log_cfg.get("use_tensorboard", False):
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        print("TensorBoard not available")
        return None
    tb_dir = log_cfg.get("tensorboard_dir")
    if tb_dir:
        tb_dir = Path(tb_dir)
    elif cfg.get("run_dir"):
        tb_dir = Path(cfg["run_dir"]) / "tensorboard"
    else:
        tb_dir = results_dir / "tensorboard"
    tb_dir.mkdir(parents=True, exist_ok=True)
    cfg["logging"]["tensorboard_dir"] = str(tb_dir.resolve())
    return SummaryWriter(log_dir=str(tb_dir))


def _resolve_weights_path(weights_dir: str):
    if not weights_dir:
        return None
    path = Path(weights_dir).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"weights_dir does not exist: {path}")
    if path.is_file():
        return path
    for candidate in [
        path / "checkpoints" / "best_ema.pt", path / "checkpoints" / "best.pt",
        path / "best_ema.pt", path / "best.pt",
    ]:
        if candidate.exists():
            return candidate
    ema_candidates = sorted(path.rglob("*_ema.pt"), key=_checkpoint_step)
    if ema_candidates:
        return ema_candidates[-1]
    ckpt_candidates = sorted(path.rglob("*.pt"), key=_checkpoint_step)
    if ckpt_candidates:
        return ckpt_candidates[-1]
    raise FileNotFoundError(f"No checkpoint found under: {path}")


def _load_initial_weights(model: FlowPolicy, cfg: dict, device: torch.device):
    weights_path = _resolve_weights_path(cfg.get("weights_dir"))
    if weights_path is None:
        print("Training mode: scratch")
        return None
    checkpoint = torch.load(weights_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model_state = model.state_dict()
    compatible = {k: v for k, v in state_dict.items()
                  if k in model_state and model_state[k].shape == v.shape}
    if not compatible:
        raise ValueError(f"No compatible parameters in checkpoint: {weights_path}")
    model_state.update(compatible)
    model.load_state_dict(model_state)
    print(f"Training mode: finetune from {weights_path} ({len(compatible)} tensors loaded)")
    return weights_path


def _save_checkpoint(path: Path, payload: dict, model: FlowPolicy, ema: EMAModel, use_ema: bool):
    checkpoint = dict(payload)
    checkpoint["checkpoint_kind"] = "ema" if use_ema else "raw"
    checkpoint["model_state_dict"] = ema.state_dict() if use_ema else model.state_dict()
    checkpoint["ema_state_dict"] = ema.state_dict()
    torch.save(checkpoint, path)


def verify_task_names(benchmark, benchmark_name: str):
    task_names = benchmark.get_task_names()
    n = benchmark.get_num_tasks()
    print("\n" + "=" * 70)
    print(f"Benchmark: {benchmark_name}  |  Tasks: {n}  |  Order: {benchmark.task_order_index}")
    print("=" * 70)
    for i, name in enumerate(task_names):
        print(f"  Task {i:2d}: {name}")
    print("=" * 70 + "\n")
    return task_names


def verify_data_files(data_root: str, benchmark):
    print("Verifying data files...")
    missing = []
    for i in range(benchmark.get_num_tasks()):
        demo_rel = benchmark.get_task_demonstration(i)
        demo_path = os.path.join(data_root, demo_rel)
        exists = os.path.exists(demo_path)
        print(f"  Task {i}: {demo_rel} [{'OK' if exists else 'MISSING'}]")
        if not exists:
            missing.append(demo_path)
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} demo file(s):\n"
                                + "\n".join(f"  - {p}" for p in missing))
    print("All data files verified.\n")


# ---------------------------------------------------------------------------
# Per-task training
# ---------------------------------------------------------------------------

def train_on_task(
    model: FlowPolicy,
    task_idx: int,
    task_name: str,
    demo_path: str,
    cfg: dict,
    action_mean: np.ndarray,
    action_std: np.ndarray,
    device: torch.device,
    tb_writer=None,
    tb_global_step_offset: int = 0,
    replay_memory: ReplayMemory = None,
    use_wandb: bool = False,
) -> tuple:
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    cl_cfg = cfg["continual_learning"]
    log_cfg = cfg["logging"]
    replay_cfg = cfg.get("replay", {})

    epochs = cl_cfg["epochs_per_task"]
    current_batch_size = data_cfg["batch_size"]
    replay_batch_size = 0
    replay_iterator = None

    if replay_memory is not None and replay_memory.has_samples():
        mix_ratio = float(replay_cfg.get("mix_ratio", 0.5))
        current_batch_size, replay_batch_size = split_batch_size(data_cfg["batch_size"], mix_ratio)
        replay_loader = replay_memory.build_loader(
            cfg=cfg, action_mean=action_mean, action_std=action_std, batch_size=replay_batch_size,
        )
        if replay_loader is not None:
            replay_iterator = cycle(replay_loader)
            print(f"  Replay enabled: {replay_memory.num_samples()} samples / "
                  f"{replay_memory.num_tasks()} task(s)  "
                  f"[current={current_batch_size}, replay={replay_batch_size}]")

    print(f"  Loading dataset: {demo_path}")
    loader, dataset = create_single_task_dataloader(
        hdf5_path=demo_path,
        batch_size=current_batch_size,
        num_workers=data_cfg["num_workers"],
        obs_horizon=data_cfg["obs_horizon"],
        action_horizon=data_cfg["action_horizon"],
        action_mean=action_mean if data_cfg.get("normalize_action", True) else None,
        action_std=action_std if data_cfg.get("normalize_action", True) else None,
        obs_keys=data_cfg["obs_keys"],
        use_eye_in_hand=data_cfg.get("use_eye_in_hand", True),
        image_size=tuple(data_cfg.get("image_size", [128, 128])),
    )
    print(f"  Dataset: {len(dataset)} samples, {len(loader)} batches/epoch")

    lr = train_cfg.get("_effective_lr", train_cfg["learning_rate"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=train_cfg.get("weight_decay", 1e-6)
    )

    total_steps = epochs * len(loader)
    warmup_steps = train_cfg.get("lr_warmup_steps", 500)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    ema = EMAModel(model, decay=train_cfg.get("ema_decay", 0.995))
    use_amp = train_cfg.get("mixed_precision", True)
    scaler = GradScaler(enabled=use_amp)

    epoch_losses = []
    global_step = 0

    for epoch in range(epochs):
        model.train()
        batch_losses = []
        pbar = tqdm(loader, desc=f"  Task {task_idx} | Epoch {epoch+1}/{epochs}", leave=False)
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            if replay_iterator is not None:
                replay_batch = next(replay_iterator)
                replay_batch = {k: v.to(device) for k, v in replay_batch.items()}
                batch = merge_batches(batch, replay_batch)

            with autocast(enabled=use_amp):
                loss = model.compute_loss(batch)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            if train_cfg.get("gradient_clip", 0) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["gradient_clip"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema.update(model)

            loss_val = loss.item()
            batch_losses.append(loss_val)
            global_step += 1
            pbar.set_postfix(loss=f"{loss_val:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

            if use_wandb and global_step % log_cfg.get("log_interval", 50) == 0:
                import wandb as _wandb
                _wandb.log({
                    "train/loss": loss_val,
                    "train/lr": scheduler.get_last_lr()[0],
                }, step=tb_global_step_offset + global_step)
            if tb_writer and global_step % log_cfg.get("log_interval", 50) == 0:
                tb_step = tb_global_step_offset + global_step
                tb_writer.add_scalar("train/loss", loss_val, tb_step)
                tb_writer.add_scalar("train/lr", scheduler.get_last_lr()[0], tb_step)

        avg_loss = np.mean(batch_losses)
        epoch_losses.append(avg_loss)
        print(f"  Task {task_idx} | Epoch {epoch+1:3d}/{epochs} | "
              f"loss={avg_loss:.4f} | lr={scheduler.get_last_lr()[0]:.2e}")

    return model, ema, epoch_losses, global_step, dataset


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(cfg, skip_eval=False, pretrain_ckpt=None, start_task: int = 0):
    device = torch.device(cfg.get("device", "cuda"))
    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    benchmark_cfg = cfg["benchmark"]
    data_cfg = cfg["data"]
    eval_cfg = cfg["evaluation"]
    replay_cfg = cfg.get("replay", {})
    run_dir, ckpt_dir, results_dir = _prepare_run_dirs(cfg)
    tb_writer = _init_tensorboard_writer(cfg, results_dir)

    # Load resume checkpoint before wandb init to reuse saved run_id
    resume_ckpt_data = None
    if start_task > 0:
        resume_ckpt_path = ckpt_dir / f"after_task_{start_task - 1:02d}_ema.pt"
        if not resume_ckpt_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_ckpt_path}")
        resume_ckpt_data = torch.load(resume_ckpt_path, map_location=device, weights_only=False)
        print(f"Resuming from: {resume_ckpt_path}  (start_task={start_task})")

    wandb_cfg = cfg.get("wandb", {})
    use_wandb = wandb_cfg.get("enabled", False)
    wandb_run_id = resume_ckpt_data.get("wandb_run_id") if resume_ckpt_data else None
    if use_wandb:
        try:
            import wandb
            date_str = datetime.now().strftime("%m%d")
            run_name = f"{wandb_cfg['name']}_{date_str}"
            wandb.init(
                entity=wandb_cfg["entity"],
                project=wandb_cfg["project"],
                group=wandb_cfg.get("group"),
                name=run_name,
                tags=wandb_cfg.get("tags", []),
                config=cfg,
                resume="must" if wandb_run_id else wandb_cfg.get("resume", "allow"),
                id=wandb_run_id if wandb_run_id else None,
            )
            wandb_run_id = wandb.run.id
            print(f"wandb run: {run_name}  (id={wandb_run_id})")
        except ImportError:
            print("wandb not available, disabling wandb logging")
            use_wandb = False

    replay_memory = None
    if replay_cfg.get("enabled", False):
        buffer_size = int(replay_cfg.get("buffer_size", 0))
        if buffer_size > 0:
            replay_memory = ReplayMemory(capacity=buffer_size, seed=seed)
            print(f"Replay enabled: buffer_size={buffer_size}, "
                  f"mix_ratio={replay_cfg.get('mix_ratio', 0.5):.2f}\n")
        else:
            print("Replay requested but buffer_size <= 0, disabling.\n")
    else:
        print("Replay disabled.\n")

    benchmark = get_benchmark(benchmark_cfg["name"])(
        task_order_index=benchmark_cfg.get("task_order_index", 0)
    )
    n_tasks = benchmark.get_num_tasks()
    task_names = verify_task_names(benchmark, benchmark_cfg["name"])

    data_root = benchmark_cfg["data_root"]
    verify_data_files(data_root, benchmark)

    print("Computing global action normalization stats...")
    action_mean, action_std = compute_global_action_stats(data_root, benchmark)

    print("\nBuilding Flow Matching Policy...")
    model = FlowPolicy(cfg).to(device)
    init_weights_path = _load_initial_weights(model, cfg, device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {param_count:,}")

    if resume_ckpt_data is not None:
        model.load_state_dict(resume_ckpt_data["model_state_dict"], strict=True)
        cfg["training"]["_effective_lr"] = cfg["training"]["learning_rate"]
        print(f"Model weights loaded from resume checkpoint.")
    elif pretrain_ckpt is not None:
        print(f"\nLoading pretrained weights from: {pretrain_ckpt}")
        ckpt = torch.load(pretrain_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        if "finetune_learning_rate" in cfg.get("training", {}):
            cfg["training"]["_effective_lr"] = cfg["training"]["finetune_learning_rate"]
        else:
            cfg["training"]["_effective_lr"] = cfg["training"]["learning_rate"]
    else:
        cfg["training"]["_effective_lr"] = cfg["training"]["learning_rate"]
    print()

    perf_matrix = np.full((n_tasks, n_tasks), np.nan)
    training_log = []
    tb_global_step = resume_ckpt_data.get("tb_global_step", 0) if resume_ckpt_data else 0
    low_dim_keys = [k for k in data_cfg["obs_keys"] if "image" not in k]

    if start_task > 0:
        inter_path = results_dir / "perf_matrix_intermediate.npy"
        if inter_path.exists():
            saved = np.load(inter_path)
            perf_matrix[:saved.shape[0], :saved.shape[1]] = saved
            print(f"Loaded perf_matrix from {inter_path}")
        log_path = results_dir / "training_log.json"
        if log_path.exists():
            with open(log_path) as f:
                training_log = json.load(f).get("training_log", [])
            print(f"Loaded training_log ({len(training_log)} entries)")

        # Backfill eval for any previously trained task whose eval row is missing
        if not skip_eval:
            for k in range(start_task):
                if not np.all(np.isnan(perf_matrix[k, :k + 1])):
                    continue  # already have results
                past_ckpt = ckpt_dir / f"after_task_{k:02d}_ema.pt"
                if not past_ckpt.exists():
                    continue
                print(f"\n  [resume] Backfilling missed eval for task {k} ({past_ckpt.name})")
                past_data = torch.load(past_ckpt, map_location=device, weights_only=False)
                eval_model = FlowPolicy(cfg).to(device)
                eval_model.load_state_dict(past_data["model_state_dict"], strict=True)
                eval_model.eval()
                eval_results = evaluate_checkpoint_on_all_tasks(
                    model=eval_model, benchmark=benchmark,
                    task_indices=list(range(k + 1)),
                    num_episodes=eval_cfg.get("num_episodes", 20),
                    max_steps=eval_cfg.get("max_steps_per_episode", 600),
                    action_execution_horizon=eval_cfg.get("action_execution_horizon", 8),
                    action_mean=action_mean if data_cfg.get("normalize_action", True) else None,
                    action_std=action_std if data_cfg.get("normalize_action", True) else None,
                    obs_horizon=data_cfg["obs_horizon"],
                    image_size=tuple(data_cfg.get("image_size", [128, 128])),
                    use_eye_in_hand=data_cfg.get("use_eye_in_hand", True),
                    low_dim_keys=low_dim_keys, device=device,
                    use_ddim=False,
                    ddim_steps=eval_cfg.get("num_flow_steps", 10),
                    seed=seed,
                )
                del eval_model
                for j, sr in eval_results.items():
                    perf_matrix[k, j] = sr
                avg_sr_k = float(np.nanmean(perf_matrix[k, :k + 1]))
                nbt_k = float(compute_nbt(perf_matrix[:k + 1, :k + 1]))
                print(f"  [resume] Task {k} eval done — Avg SR: {avg_sr_k:.4f}  NBT: {nbt_k:.4f}")
                if use_wandb:
                    import wandb as _wandb
                    eval_log = {f"eval/task{j}_sr": float(sr) for j, sr in eval_results.items()}
                    eval_log["eval/avg_sr"] = avg_sr_k
                    eval_log["eval/nbt"] = nbt_k
                    _wandb.log(eval_log, step=past_data.get("tb_global_step", 0))
            _save_intermediate(perf_matrix, task_names, training_log, cfg, results_dir, start_task - 1)

    for task_k in range(start_task, n_tasks):
        print("\n" + "=" * 70)
        print(f"STAGE {task_k + 1}/{n_tasks}: Task {task_k}  —  {task_names[task_k]}")
        print("=" * 70)

        demo_rel = benchmark.get_task_demonstration(task_k)
        demo_path = os.path.join(data_root, demo_rel)

        t_start = time.time()
        model, ema, epoch_losses, task_steps, task_dataset = train_on_task(
            model=model, task_idx=task_k, task_name=task_names[task_k],
            demo_path=demo_path, cfg=cfg, action_mean=action_mean, action_std=action_std,
            device=device, tb_writer=tb_writer, tb_global_step_offset=tb_global_step,
            replay_memory=replay_memory, use_wandb=use_wandb,
        )
        tb_global_step += task_steps

        if replay_memory is not None:
            replay_memory.add_task(demo_path, task_dataset.index)
            print(f"  Replay buffer: {replay_memory.num_samples()} samples / "
                  f"{replay_memory.num_tasks()} task(s)")
        del task_dataset

        train_time = time.time() - t_start
        print(f"\n  Training time: {train_time:.1f}s | Final loss: {epoch_losses[-1]:.4f}")

        ckpt_path = ckpt_dir / f"after_task_{task_k:02d}.pt"
        ema_ckpt_path = ckpt_dir / f"after_task_{task_k:02d}_ema.pt"
        payload = {
            "task_idx": task_k, "task_name": task_names[task_k],
            "config": cfg, "action_mean": action_mean, "action_std": action_std,
            "epoch_losses": epoch_losses, "wandb_run_id": wandb_run_id,
            "tb_global_step": tb_global_step,
        }
        _save_checkpoint(ckpt_path, payload, model, ema, use_ema=False)
        _save_checkpoint(ema_ckpt_path, payload, model, ema, use_ema=True)
        print(f"  Checkpoint: {ckpt_path}")

        stage_log = {
            "task_idx": task_k, "task_name": task_names[task_k],
            "train_time_s": train_time, "final_train_loss": float(epoch_losses[-1]),
        }

        if not skip_eval:
            print(f"\n  Evaluating on tasks 0..{task_k}:")
            eval_model = ema.model
            eval_model.eval()

            t_eval = time.time()
            eval_results = evaluate_checkpoint_on_all_tasks(
                model=eval_model, benchmark=benchmark,
                task_indices=list(range(task_k + 1)),
                num_episodes=eval_cfg.get("num_episodes", 20),
                max_steps=eval_cfg.get("max_steps_per_episode", 600),
                action_execution_horizon=eval_cfg.get("action_execution_horizon", 8),
                action_mean=action_mean if data_cfg.get("normalize_action", True) else None,
                action_std=action_std if data_cfg.get("normalize_action", True) else None,
                obs_horizon=data_cfg["obs_horizon"],
                image_size=tuple(data_cfg.get("image_size", [128, 128])),
                use_eye_in_hand=data_cfg.get("use_eye_in_hand", True),
                low_dim_keys=low_dim_keys, device=device,
                use_ddim=False,  # FM uses Euler integration via sample_action()
                ddim_steps=eval_cfg.get("num_flow_steps", 10),
                seed=seed,
                save_video=eval_cfg.get("save_video", False),
                video_dir=str(results_dir / "videos" / f"stage_{task_k:02d}"),
            )
            eval_time = time.time() - t_eval

            for task_j, sr in eval_results.items():
                perf_matrix[task_k, task_j] = sr

            avg_sr_stage = np.nanmean(perf_matrix[task_k, : task_k + 1])
            nbt_so_far = compute_nbt(perf_matrix[: task_k + 1, : task_k + 1])
            stage_log.update({"eval_time_s": eval_time, "avg_sr": float(avg_sr_stage),
                               "nbt": float(nbt_so_far),
                               "eval_results": {str(k): float(v) for k, v in eval_results.items()}})

            if use_wandb:
                import wandb as _wandb
                eval_log = {f"eval/task{j}_sr": float(sr) for j, sr in eval_results.items()}
                eval_log["eval/avg_sr"] = float(avg_sr_stage)
                eval_log["eval/nbt"] = float(nbt_so_far)
                _wandb.log(eval_log, step=tb_global_step)

            print(f"\n  --- Stage {task_k + 1} ---  Avg SR: {avg_sr_stage:.4f}  NBT: {nbt_so_far:.4f}")
        else:
            print("\n  [skip-eval] evaluation skipped")

        training_log.append(stage_log)
        _save_intermediate(perf_matrix, task_names, training_log, cfg, results_dir, task_k)

    # Final metrics
    run_meta = {
        "end_time": datetime.now().isoformat(), "config": cfg,
        "n_tasks": n_tasks, "task_names": task_names, "param_count": param_count,
        "training_log": training_log,
    }

    if not skip_eval:
        nbt_final = compute_nbt(perf_matrix)
        avg_sr_final = compute_average_sr(perf_matrix)
        print(f"\nFINAL: Avg SR = {avg_sr_final:.4f}  |  NBT = {nbt_final:.4f}")
        run_meta["nbt"] = float(nbt_final)
        run_meta["avg_sr_final"] = float(avg_sr_final)
        save_results_json(perf_matrix, task_names, nbt_final, avg_sr_final, cfg,
                          str(results_dir / "results.json"))
        save_results_csv(perf_matrix, task_names, str(results_dir / "perf_matrix.csv"))
        np.save(results_dir / "perf_matrix.npy", perf_matrix)
        plot_performance_matrix(perf_matrix, task_names, str(results_dir / "heatmap.png"),
                                benchmark_name=benchmark_cfg.get("name"))
        plot_forgetting_summary(perf_matrix, task_names, str(results_dir / "forgetting_summary.png"))

    with open(results_dir / "run_meta.json", "w") as f:
        json.dump(run_meta, f, indent=2, default=str)

    print(f"\nAll results saved to: {results_dir}")
    if tb_writer:
        tb_writer.flush()
        tb_writer.close()
    if use_wandb:
        import wandb as _wandb
        _wandb.finish()


def _save_intermediate(perf_matrix, task_names, training_log, cfg, results_dir, task_k):
    np.save(results_dir / "perf_matrix_intermediate.npy", perf_matrix)
    nbt = compute_nbt(perf_matrix[: task_k + 1, : task_k + 1])
    avg_sr = np.nanmean(perf_matrix[task_k, : task_k + 1])
    save_results_json(perf_matrix, task_names, nbt, avg_sr, cfg,
                      str(results_dir / "results_intermediate.json"))
    with open(results_dir / "training_log.json", "w") as f:
        json.dump({"completed_tasks": task_k + 1, "training_log": training_log},
                  f, indent=2, default=str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--pretrain-ckpt", type=str, default=None)
    parser.add_argument("--start-task", type=int, default=0,
                        help="Resume from this task index (loads after_task_{N-1}_ema.pt)")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    main(cfg, skip_eval=args.skip_eval, pretrain_ckpt=args.pretrain_ckpt,
         start_task=args.start_task)
