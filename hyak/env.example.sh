# Copy to hyak/env.sh and edit for your Hyak account (env.sh is gitignored).
#
#   cp hyak/env.example.sh hyak/env.sh
#
# Then: source hyak/env.sh

# Slurm (required on Hyak)
export HYAK_ACCOUNT="your_lab_account"    # from: hyakalloc
export HYAK_PARTITION="gpu-l40"         # e.g. gpu-l40, gpu-a100, gpu-a40
export HYAK_GPU_TYPE="l40"              # must match partition (l40, a100, a40, ...)

# Project root on Hyak (where you cloned the repo)
export MD_SIM_ROOT="${MD_SIM_ROOT:-$HOME/MD_Sim}"

# Conda env with deepmd-kit + lammps (pair_style deepmd)
export CONDA_ENV="md-sim-mlip"

# Optional: limit OpenMP threads (DeepMD/LAMMPS often use GPU; keep CPU low)
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
