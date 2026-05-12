# IngChicken-FM

Flow Matching Policy for Continual Learning on LIBERO, migrated from the DDPM-based [IngChicken](https://github.com/chynl63/IngChicken) codebase.

## What changed from DDPM

| Component | DDPM (before) | FM (this repo) |
|---|---|---|
| Forward process | `x_t = √ᾱ_t·x0 + √(1-ᾱ_t)·ε` | `x_t = (1-t)·x0 + t·x1` |
| Training target | noise `ε` | velocity `v = x1 - x0` |
| Timestep | integer `[0, T)` | continuous `[0, 1]`, scaled ×1000 for embedding |
| Inference | DDIM reverse chain | Euler integration |
| SDFT loss | ε-space MSE | velocity-space MSE |

## Structure

```
IngChicken-FM/
├── model/
│   └── flow_policy.py          # FlowPolicy + EMAModel
├── scripts/
│   ├── train_pretrain.py       # LIBERO-90 pretraining
│   ├── train_sequential.py     # CL baseline (ER)
│   ├── train_sequential_sdft.py # CL + FM-SDFT
│   ├── eval.py                 # offline evaluation
│   ├── datasets/               # data loading (from yerincho)
│   ├── evaluation/             # rollout evaluation & CL metrics
│   └── utils_er.py             # experience replay utilities
├── SDFT/
│   └── fm.py                   # FM-SDFT: euler_integration + compute_fm_sdft_loss
└── configs/
    ├── pretrain.yaml
    ├── cl_{object,spatial,goal,long}.yaml       # CL baseline
    └── cl_{object,spatial,goal,long}_sdft.yaml  # CL + FM-SDFT
```

## Usage

### Pretraining on LIBERO-90
```bash
python -m scripts.train_pretrain --config configs/pretrain.yaml
```

### Sequential CL (ER baseline)
```bash
python -m scripts.train_sequential \
    --config configs/cl_object.yaml \
    [--pretrain-ckpt checkpoints/pretrain_fm/best_ema.pt] \
    [--skip-eval]
```

### Sequential CL + FM-SDFT
```bash
python -m scripts.train_sequential_sdft \
    --config configs/cl_object_sdft.yaml \
    [--pretrain-ckpt checkpoints/pretrain_fm/best_ema.pt] \
    [--skip-eval]
```

### Offline evaluation
```bash
python -m scripts.eval \
    --checkpoint checkpoints/cl_object_sdft/after_task_09_ema.pt \
    --benchmark libero_object \
    --num-episodes 20 \
    --output-dir results/eval_object
```

## FM-SDFT loss

```
D_FM = E_{t~U[0,1], x0~N(0,I)} [ |v_student(x_t,t) - v_teacher(x_t,t)|^2 ]
```

where `x_t = (1-t)·x0 + t·x1`, `x1` is teacher's predicted action via Euler integration,
and `v = vector_field_net(x_t, t×1000, obs_cond)`.

## Key config fields

- `flow_matching.num_flow_steps`: Euler integration steps (default 10)
- `evaluation.num_flow_steps`: Euler steps at eval time
- `sdft.num_flow_steps`: Euler steps for teacher rollout during SDFT collection
- `sdft.weight`: FM-SDFT loss weight (λ in `L = L_FM + λ·L_SDFT`)
