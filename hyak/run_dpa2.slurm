#!/bin/bash
#SBATCH --job-name=mdsim-dpa2
#SBATCH --account=REPLACE_WITH_HYAK_ACCOUNT
#SBATCH --partition=gpu-l40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=l40:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Production-length DPA-2 MD (uses equil_steps / prod_steps_dpa2 from configs/dac.yaml).
# Submit:
#   sbatch hyak/run_dpa2.slurm
#
# Other systems: set CONFIG below or pass via sbatch --export=ALL,CONFIG=configs/cgpei.yaml

set -euo pipefail

CONFIG="${CONFIG:-configs/dac.yaml}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${MD_SIM_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$ROOT"
mkdir -p logs

if [[ -f "$SCRIPT_DIR/env.sh" ]]; then
  # shellcheck source=/dev/null
  source "$SCRIPT_DIR/env.sh"
fi

if [[ -n "${CONDA_ENV:-}" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

echo "JobID=$SLURM_JOB_ID  Node=$SLURMD_NODENAME  CONFIG=$CONFIG  Start=$(date)"
bash "$SCRIPT_DIR/verify_deepmd.sh"
bash "$SCRIPT_DIR/run_dpa2.sh" "$CONFIG"
echo "Finished $(date)"
