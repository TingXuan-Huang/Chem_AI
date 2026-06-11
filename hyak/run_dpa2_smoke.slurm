#!/bin/bash
#SBATCH --job-name=mdsim-dpa2-smoke
#SBATCH --account=REPLACE_WITH_HYAK_ACCOUNT
#SBATCH --partition=gpu-l40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=l40:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Short DPA-2 smoke test on Hyak (500 equil + 2000 prod steps).
# Submit from repo root:
#   mkdir -p logs
#   sbatch hyak/run_dpa2_smoke.slurm
#
# Before first submit:
#   1. cp hyak/env.example.sh hyak/env.sh  && edit HYAK_ACCOUNT / paths
#   2. Edit #SBATCH --account and --partition above (or use env.sh + wrapper)
#   3. conda env with: conda install -c deepmodeling deepmd-kit lammps
#   4. Place models/dpa2.pb and data/raw_trajectories/DAC/system.data

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${MD_SIM_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$ROOT"
mkdir -p logs

if [[ -f "$SCRIPT_DIR/env.sh" ]]; then
  # shellcheck source=/dev/null
  source "$SCRIPT_DIR/env.sh"
fi

# Conda (adjust if your cluster uses modules instead)
if [[ -n "${CONDA_ENV:-}" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

echo "JobID=$SLURM_JOB_ID  Node=$SLURMD_NODENAME  Start=$(date)"
bash "$SCRIPT_DIR/verify_deepmd.sh"
bash "$SCRIPT_DIR/run_dpa2.sh" configs/dac.yaml smoke
echo "Finished $(date)"
