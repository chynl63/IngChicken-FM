# 🐔 IngChicken-FM

> **Flow Matching 기반 Diffusion Policy의 Continual Learning**
> DDPM 기반 [IngChicken](https://github.com/chynl63/IngChicken) 코드베이스를 **Flow Matching(FM)** 으로 마이그레이션한 버전입니다.
> FM-SDFT(Self-Distillation via Flow Trajectories)를 통해 순차 학습 시 발생하는 catastrophic forgetting을 완화하는 것이 핵심 연구 목표입니다.

---

## 📌 프로젝트 개요

| 항목 | 내용 |
|------|------|
| **베이스라인** | Flow Matching Policy (DDPM → FM 마이그레이션) |
| **핵심 기법** | FM-SDFT: velocity space self-distillation로 forgetting 완화 |
| **벤치마크** | LIBERO (Object / Spatial / Goal / Long) |
| **CL 방법** | Experience Replay (ER) + Pretraining (PT) + SDFT |

### DDPM과의 차이

| 구성 요소 | DDPM (기존) | FM (이 레포) |
|-----------|------------|-------------|
| Forward process | `x_t = √ᾱ_t·x₀ + √(1-ᾱ_t)·ε` | `x_t = (1-t)·x₀ + t·x₁` |
| 학습 타겟 | noise `ε` | velocity `v = x₁ - x₀` |
| Timestep | 정수 `[0, T)` | 연속 `[0, 1]`, ×1000 스케일링 후 임베딩 |
| Inference | DDIM reverse chain | Euler integration (10 steps) |
| SDFT 손실 | ε-space MSE | velocity-space MSE |

---

## 🛠️ 환경 설정

**이 레포는 기존 [IngChicken](https://github.com/chynl63/IngChicken) 레포와 동일한 환경을 사용합니다.**
IngChicken 환경이 이미 구축되어 있다면 추가 설치 없이 바로 사용 가능합니다.

```bash
# 레포 클론
git clone https://github.com/chynl63/IngChicken-FM.git
cd IngChicken-FM

# (IngChicken 환경이 없는 경우) 패키지 설치
pip install -r requirements.txt
```

> 💡 Python 3.10 + PyTorch 2.x + CUDA 12.x 환경 권장

---

## 📂 데이터 준비

### 디렉토리 구조

```
/your/data/root/
├── libero_90/          ← Pretraining용 (LIBERO-90, 총 90개 태스크)
│   ├── task_0.hdf5
│   └── ...
├── libero_object/      ← CL 실험용
├── libero_spatial/
├── libero_goal/
└── libero_long/
```

### 경로 설정

**Pretraining** — `configs/pretrain.yaml` 수정:

```yaml
data:
  data_dir: "/your/data/root/libero_90"   # ← 이 경로를 본인 환경에 맞게 수정
```

**CL 실험** — 각 `configs/cl_*.yaml` 수정:

```yaml
benchmark:
  data_root: "/your/data/root"            # ← 이 경로를 본인 환경에 맞게 수정
```

---

## 🧪 실험별 사용법

모든 명령어는 **레포 루트**에서 실행합니다.
`{suite}` 자리에는 `object` / `spatial` / `goal` / `long` 중 하나를 입력합니다.

### 실험 조합 요약

| # | 실험 | 스크립트 | PT | ER | SDFT |
|---|------|----------|:--:|:--:|:----:|
| 4-1 | Pretraining | `train_pretrain` | — | — | — |
| 4-2 | PT only | `train_sequential` | ✅ | ❌ | ❌ |
| 4-3 | ER only | `train_sequential` | ❌ | ✅ | ❌ |
| 4-4 | PT + ER | `train_sequential` | ✅ | ✅ | ❌ |
| 4-5 | PT + ER + MSE-SDFT | `train_sequential_sdft` | ✅ | ✅ | MSE |
| 4-6 | PT + ER + FM-SDFT | `train_sequential_sdft` | ✅ | ✅ | FM |

---

### 4-1. 🏋️ Pretraining

LIBERO-90 전체 데이터로 Flow Matching Policy를 사전 학습합니다.
이후 CL 실험(4-2 ~ 4-6)의 초기 가중치로 사용됩니다.

```bash
python -m scripts.train_pretrain --config configs/pretrain.yaml
```

| 설정 | 값 |
|------|-----|
| Config | `configs/pretrain.yaml` |
| 학습 에폭 | 500 |
| 체크포인트 저장 경로 | `checkpoints/pretrain_fm/` |
| 주요 저장 파일 | `best_ema.pt` (최저 loss), `epoch_NNNN_ema.pt` (10 에폭마다) |

---

### 4-2. 🔵 Baseline — PT only (Pretraining + Fine-tuning, No ER)

사전학습된 가중치로 초기화 후, 리플레이 없이 순차 학습합니다.
catastrophic forgetting에 취약한 단순 fine-tuning 베이스라인입니다.

```bash
python -m scripts.train_sequential \
    --config configs/cl_{suite}.yaml \
    --pretrain-ckpt checkpoints/pretrain_fm/best_ema.pt
```

> ⚠️ config 파일에서 `replay.enabled: false` 로 설정되어 있는지 확인하세요.

| 설정 | 값 |
|------|-----|
| Config | `configs/cl_{suite}.yaml` |
| `replay.enabled` | `false` |
| `--pretrain-ckpt` | `checkpoints/pretrain_fm/best_ema.pt` |
| 결과 저장 경로 | `results/cl_{suite}/` |

---

### 4-3. 🟡 Baseline — ER only (No Pretraining)

Experience Replay만 사용하여 forgetting을 완화합니다. 랜덤 초기화에서 시작합니다.

```bash
python -m scripts.train_sequential \
    --config configs/cl_{suite}.yaml
```

> ⚠️ config 파일에서 `replay.enabled: true` 로 설정되어 있는지 확인하세요 (기본값).

| 설정 | 값 |
|------|-----|
| Config | `configs/cl_{suite}.yaml` |
| `replay.enabled` | `true` |
| `replay.buffer_size` | `1000` (전체 리플레이 샘플 수) |
| `replay.mix_ratio` | `0.5` (배치의 50%를 리플레이 버퍼에서 샘플링) |
| `--pretrain-ckpt` | 없음 |
| 결과 저장 경로 | `results/cl_{suite}/` |

**ER 동작 방식:**
- 1000개 샘플 예산을 현재까지 학습한 태스크 수에 따라 균등 분배
- 새 태스크 추가 시 기존 태스크 할당량을 줄여 rebalancing
- 매 학습 스텝에서 현재 태스크 배치(50%)와 리플레이 배치(50%)를 합쳐 학습

---

### 4-4. 🟢 Baseline — PT + ER (Pretraining + Experience Replay)

사전학습 + ER을 결합한 강력한 베이스라인입니다.

```bash
python -m scripts.train_sequential \
    --config configs/cl_{suite}.yaml \
    --pretrain-ckpt checkpoints/pretrain_fm/best_ema.pt
```

| 설정 | 값 |
|------|-----|
| Config | `configs/cl_{suite}.yaml` |
| `replay.enabled` | `true` |
| `--pretrain-ckpt` | `checkpoints/pretrain_fm/best_ema.pt` |
| 결과 저장 경로 | `results/cl_{suite}/` |

> 💡 `--pretrain-ckpt` 제공 시 `training.finetune_learning_rate` 가 `learning_rate` 대신 자동으로 적용됩니다.

---

### 4-5. 🔴 PT + ER + MSE-SDFT

이전 태스크 teacher의 velocity 예측을 student가 MSE로 모방하는 self-distillation을 추가합니다.

```bash
python -m scripts.train_sequential_sdft \
    --config configs/cl_{suite}_sdft.yaml \
    --pretrain-ckpt checkpoints/pretrain_fm/best_ema.pt \
    --loss-type mse
```

| 설정 | 값 |
|------|-----|
| Config | `configs/cl_{suite}_sdft.yaml` |
| `--loss-type` | `mse` |
| `sdft.weight` | `0.3` |
| Teacher rollout | Euler `num_flow_steps=10` steps, `num_episodes=5` 에피소드 |
| 결과 저장 경로 | `results/cl_{suite}_sdft/` |

**MSE-SDFT 손실:**
```
L = L_FM(merged_batch) + 0.3 × MSE(v_student(x_t, t), v_teacher(x_t, t))
```

---

### 4-6. 🟣 PT + ER + FM-SDFT

FM velocity space에서 teacher rollout 궤적을 기반으로 self-distillation을 수행합니다.
FM interpolation 구조를 활용해 teacher의 demonstrated trajectory를 학습 신호로 사용합니다.

```bash
python -m scripts.train_sequential_sdft \
    --config configs/cl_{suite}_sdft.yaml \
    --pretrain-ckpt checkpoints/pretrain_fm/best_ema.pt \
    --loss-type fm
```

| 설정 | 값 |
|------|-----|
| Config | `configs/cl_{suite}_sdft.yaml` |
| `--loss-type` | `fm` |
| Teacher | 태스크 전환 시점에 frozen copy 생성 |
| 결과 저장 경로 | `results/cl_{suite}_sdft/` |

**FM-SDFT 손실 (velocity space distillation):**
```
x₁ = Euler(teacher, x₀, steps=10)      # teacher rollout (no_grad)
x_t = (1-t)·x₀ + t·x₁                 # FM interpolation
L_SDFT = MSE(v_student(x_t, t), v_teacher(x_t, t))
L = L_FM(merged_batch) + 0.3 × L_SDFT
```

---

## 📊 평가

### 체크포인트 평가

```bash
python -m scripts.eval \
    --checkpoint checkpoints/cl_object/after_task_09_ema.pt \
    --benchmark libero_object \
    --num-episodes 20 \
    --output-dir results/eval_object/
```

### 특정 태스크만 평가

```bash
python -m scripts.eval \
    --checkpoint checkpoints/cl_object/after_task_09_ema.pt \
    --benchmark libero_object \
    --task-indices 0 1 2 3 4
```

### 주요 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--checkpoint` | 평가할 `.pt` 파일 경로 | 필수 |
| `--benchmark` | `libero_object` / `libero_spatial` / `libero_goal` / `libero_long` | config에서 자동 |
| `--task-indices` | 평가할 태스크 인덱스 (미지정 시 전체) | 전체 |
| `--num-episodes` | 태스크당 에피소드 수 | 20 |
| `--data-root` | LIBERO 데이터 루트 경로 | config에서 자동 |
| `--output-dir` | 결과 JSON 저장 경로 | 저장 안 함 |

### 결과 확인

```bash
# JSON 결과 확인
cat results/eval_object/eval_results.json

# TensorBoard 학습 로그 확인
tensorboard --logdir checkpoints/cl_object/
```

**주요 CL 지표:**
- **Average Success Rate (ASR):** 마지막 태스크 학습 후 전체 태스크 평균 성공률 — 높을수록 좋음
- **Normalized Backward Transfer (NBT):** forgetting 정도 — 낮을수록 좋음 (0에 가까울수록 forgetting 없음)

---

## 📁 폴더 구조

```
IngChicken-FM/
│
├── model/
│   ├── flow_policy.py          # FlowPolicy 모델 (FM 핵심 구현, EMAModel 포함)
│   └── __init__.py
│
├── SDFT/
│   ├── fm.py                   # FM-SDFT 손실 함수 및 teacher Euler rollout 구현
│   └── __init__.py
│
├── scripts/
│   ├── train_pretrain.py           # LIBERO-90 사전학습 스크립트
│   ├── train_sequential.py         # CL 순차 학습 (PT / ER / PT+ER 지원)
│   ├── train_sequential_sdft.py    # CL 순차 학습 + FM-SDFT (--loss-type mse/fm)
│   ├── eval.py                     # 오프라인 체크포인트 평가
│   ├── utils_er.py                 # Experience Replay 버퍼 (ReplayMemory)
│   │
│   ├── datasets/
│   │   ├── libero_dataset.py               # LIBERO-90 균등 샘플링 데이터셋
│   │   └── libero_single_task_dataset.py   # 단일 태스크 HDF5 데이터셋
│   │
│   └── evaluation/
│       ├── rollout_evaluator.py    # 환경 롤아웃 평가 (OffScreenRenderEnv)
│       └── cl_metrics.py           # ASR, NBT 등 CL 지표 계산
│
├── configs/
│   ├── pretrain.yaml               # LIBERO-90 사전학습 설정
│   ├── cl_object.yaml              # LIBERO-Object CL 설정
│   ├── cl_object_sdft.yaml         # LIBERO-Object + SDFT 설정
│   ├── cl_spatial.yaml             # LIBERO-Spatial CL 설정
│   ├── cl_spatial_sdft.yaml        # LIBERO-Spatial + SDFT 설정
│   ├── cl_goal.yaml                # LIBERO-Goal CL 설정
│   ├── cl_goal_sdft.yaml           # LIBERO-Goal + SDFT 설정
│   ├── cl_long.yaml                # LIBERO-Long CL 설정 (max_steps=1000)
│   └── cl_long_sdft.yaml           # LIBERO-Long + SDFT 설정
│
├── checkpoints/                    # 학습 체크포인트 저장 (자동 생성)
├── results/                        # 평가 결과 저장 (자동 생성)
├── .gitignore
└── README.md
```

---

## 🔧 주요 Config 파라미터

### CL 실험 공통 (`configs/cl_{suite}.yaml`)

```yaml
benchmark:
  data_root: "/your/data/root"      # ← 수정 필요

continual_learning:
  epochs_per_task: 50               # 태스크당 학습 에폭 수

flow_matching:
  num_flow_steps: 10                # Euler integration step 수 (학습/추론 공통)

replay:
  enabled: true                     # ER 활성화 여부
  buffer_size: 1000                 # 전체 리플레이 샘플 수 (태스크 수로 균등 분배)
  mix_ratio: 0.5                    # 배치 내 리플레이 비율

training:
  learning_rate: 1.0e-4
  finetune_learning_rate: 1.0e-4    # --pretrain-ckpt 사용 시 적용되는 LR
```

### SDFT 추가 파라미터 (`configs/cl_{suite}_sdft.yaml`)

```yaml
sdft:
  enabled: true
  weight: 0.3                       # SDFT 손실 가중치 (λ)
  num_flow_steps: 10                # Teacher rollout Euler step 수
  num_episodes: 5                   # On-policy 관찰 수집 에피소드 수
  max_states: 200                   # 수집할 최대 상태 수
  batch_size: 16                    # SDFT 손실 계산용 배치 크기
```

---

## 📝 참고

- **기반 코드:** [IngChicken](https://github.com/chynl63/IngChicken) (yerincho baseline + chaeyoon model + SDFT_FIX)
- **벤치마크:** [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)
- **FM 참고:** Flow Matching for Generative Modeling (Lipman et al., 2022)
