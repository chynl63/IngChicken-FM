#!/bin/bash
# Submit rollout evaluation for CL checkpoints (LIBERO-Long, Naive) on Slurm + Singularity.
#
# Usage:
#   EVAL_TASK=0 bash scripts/submit/submit_eval_cl_long_naive.sh
#   EVAL_ALL=1  bash scripts/submit/submit_eval_cl_long_naive.sh
#
# Notes:
# - Checkpoints: checkpoints/cl_long_pt/
# - Results:     results/cl_long_pt/
# - W&B:         resumes the same run as training (reads wandb_run_id from checkpoint)
set -euo pipefail

BASE="/home/cyhoaoen/IngChicken-FM"
SIF_IMAGE="${SIF_IMAGE:-/home/cyhoaoen/IngChicken/Baseline/chaeyoon/dp_libero.sif}"
PARTITION="${PARTITION:-gigabyte_a5000}"
TIME="${TIME:-24:00:00}"
CPU="${CPU:-8}"
MEM="${MEM:-32G}"
GPU_TYPE="${GPU_TYPE:-A5000}"
GPU_N="${GPU_N:-1}"
EVAL_TASK="${EVAL_TASK:-0}"
EVAL_ALL="${EVAL_ALL:-0}"

if [[ "${EVAL_ALL}" == "1" ]]; then
  EVAL_ARG="--all"
  JOB_SUFFIX="all"
else
  EVAL_ARG="--task ${EVAL_TASK}"
  JOB_SUFFIX="t${EVAL_TASK}"
fi

mkdir -p "${BASE}/logs"

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=fm_cl_lon_eval_${JOB_SUFFIX}
#SBATCH --partition=${PARTITION}
#SBATCH --gres=gpu:${GPU_TYPE}:${GPU_N}
#SBATCH --cpus-per-task=${CPU}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME}
#SBATCH --output=${BASE}/logs/cl_lon_naive_eval_${JOB_SUFFIX}_%j.out
#SBATCH --error=${BASE}/logs/cl_lon_naive_eval_${JOB_SUFFIX}_%j.err

set -euo pipefail
export CUDA_VISIBLE_DEVICES="\${GPU_DEVICE:-0}"

exec singularity exec --nv --writable-tmpfs \\
  --bind ${BASE}:/workspace \\
  --bind /home/cyhoaoen:/home/cyhoaoen \\
  --bind ${BASE}/LIBERO:/workspace/LIBERO \\
  "${SIF_IMAGE}" \\
  bash -lc '
    set -euo pipefail
    cd /workspace
    source /workspace/scripts/singularity/dp_image_env.sh

    export MUJOCO_GL=osmesa
    export PYOPENGL_PLATFORM=osmesa

    python -m pip install -q wandb
    python -m pip install -q bddl easydict cloudpickle
    python -m pip install -q --no-deps gym_notices gym

    python -m scripts.eval_sequential \\
      --config /workspace/configs/cl_long_pt.yaml \\
      ${EVAL_ARG}
  '
EOF

echo "Submitted: FM eval (LIBERO-Long, ${JOB_SUFFIX})"
echo "  Logs:    ${BASE}/logs/cl_lon_naive_eval_${JOB_SUFFIX}_<JOBID>.{out,err}"
echo "  Results: ${BASE}/results/cl_long_pt/"
