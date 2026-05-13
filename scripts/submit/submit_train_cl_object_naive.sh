#!/bin/bash
# Submit Sequential CL Training (LIBERO-Object, Naive — no pretrain, no ER) on Slurm + Singularity.
#
# Usage:
#   bash submit_train_cl_object_naive.sh
#   PARTITION=gigabyte_a6000 GPU_TYPE=A6000 bash submit_train_cl_object_naive.sh
#   SIF_IMAGE=/path/to/custom.sif bash submit_train_cl_object_naive.sh
#
# Notes:
# - Config:       configs/cl_object_pt.yaml  (replay.enabled: false, no pretrain weights)
# - Checkpoints:  checkpoints/cl_object_pt/
# - Results:      results/cl_object_pt/
# - W&B:          enabled in config (entity: ingchicken, project: baseline, group: object)
set -euo pipefail

BASE="/home/cyhoaoen/IngChicken-FM"
SIF_IMAGE="${SIF_IMAGE:-/home/cyhoaoen/IngChicken/Baseline/chaeyoon/dp_libero.sif}"
PARTITION="${PARTITION:-gigabyte_a5000}"
TIME="${TIME:-48:00:00}"
CPU="${CPU:-8}"
MEM="${MEM:-32G}"
GPU_TYPE="${GPU_TYPE:-A5000}"
GPU_N="${GPU_N:-1}"
START_TASK="${START_TASK:-0}"

mkdir -p "${BASE}/logs" "${BASE}/checkpoints/cl_object_pt" "${BASE}/results/cl_object_pt"

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=fm_cl_obj_naive
#SBATCH --partition=${PARTITION}
#SBATCH --gres=gpu:${GPU_TYPE}:${GPU_N}
#SBATCH --cpus-per-task=${CPU}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME}
#SBATCH --output=${BASE}/logs/cl_obj_naive_%j.out
#SBATCH --error=${BASE}/logs/cl_obj_naive_%j.err

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

    export PYOPENGL_PLATFORM=egl
    export MUJOCO_GL=egl

    python -m pip install -q wandb
    python -m pip install -q bddl easydict cloudpickle
    python -m pip install -q --no-deps gym_notices gym

    python -m scripts.train_sequential \\
      --config /workspace/configs/cl_object_pt.yaml \\
      --start-task ${START_TASK}
  '
EOF

echo "Submitted: FM naive CL training (LIBERO-Object)"
echo "  Logs:        ${BASE}/logs/cl_obj_naive_<JOBID>.{out,err}"
echo "  Checkpoints: ${BASE}/checkpoints/cl_object_pt/"
echo "  Results:     ${BASE}/results/cl_object_pt/"
