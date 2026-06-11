# Running DPA-2 (DeepMD + LAMMPS) on UW Hyak

Scripts to test `scripts/lammps_mlip_run.py` on Hyak GPU nodes.

## One-time setup on Hyak

```bash
# 1. Clone repo (if not already)
cd ~
git clone https://github.com/YOUR_USER/MD_Sim.git
cd MD_Sim

# 2. Hyak account / partition
hyakalloc   # note your account name and GPU partition

cp hyak/env.example.sh hyak/env.sh
# Edit hyak/env.sh: HYAK_ACCOUNT, HYAK_PARTITION, HYAK_GPU_TYPE, MD_SIM_ROOT

# 3. Conda environment (recommended)
module load anaconda   # if needed on Hyak; exact module name may vary
conda create -n md-sim-mlip python=3.11 -y
conda activate md-sim-mlip
conda install -c deepmodeling deepmd-kit lammps -y
pip install -r requirements.txt pyyaml

# 4. Verify LAMMPS + DeepMD before submitting jobs
bash hyak/verify_deepmd.sh

# 5. Inputs (not in git — large / generated)
#    - models/dpa2.pb          (DPA-2 checkpoint from DeepModeling)
#    - data/.../system.data    (from build_system.py or your own structure)
```

Generate `system.data` on Hyak (needs PACKMOL + templates):

```bash
conda install -c conda-forge packmol rdkit -y
pip install -r requirements-mlip.txt
# build molecular PDBs — see templates/molecules/README.md
python scripts/build_system.py --config configs/dac.yaml
```

## Submit jobs

Edit `#SBATCH --account` and `#SBATCH --partition` in the `.slurm` file to match `hyak/env.sh`, then:

```bash
mkdir -p logs

# Short smoke test (~500 equil + 2000 prod steps)
sbatch hyak/run_dpa2_smoke.slurm

# Full run (values from configs/dac.yaml: 20k equil + 200k prod)
sbatch hyak/run_dpa2.slurm

# Another system
sbatch --export=ALL,CONFIG=configs/cgpei.yaml hyak/run_dpa2.slurm
```

Monitor:

```bash
squeue -u "$USER"
tail -f logs/mdsim-dpa2-smoke-<jobid>.out
```

## Run interactively (debug)

Request a GPU shell (example; check [Hyak docs](https://hyak.uw.edu/docs/compute/start-here/) for current syntax):

```bash
srun -A YOUR_ACCOUNT -p gpu-l40 --gpus=l40:1 --mem=32G --time=2:00:00 --pty bash
source hyak/env.sh
conda activate md-sim-mlip
cd "$MD_SIM_ROOT"
bash hyak/run_dpa2.sh configs/dac.yaml smoke
```

## Files

| File | Purpose |
|------|---------|
| `env.example.sh` | Copy to `env.sh` (local account settings) |
| `config_to_env.py` | Reads YAML → shell exports for `run_dpa2.sh` |
| `verify_deepmd.sh` | Pre-flight: `import lammps`, optional `deepmd` |
| `run_dpa2.sh` | Driver calling `lammps_mlip_run.py` |
| `run_dpa2_smoke.slurm` | Slurm: short test |
| `run_dpa2.slurm` | Slurm: production steps from YAML |

## Troubleshooting

- **`LAMMPS Python module not found`** — activate conda env; install `lammps` from `deepmodeling` channel, not plain `conda-forge` alone.
- **`pair_style deepmd` invalid** — LAMMPS build lacks DeepMD plugin; use `conda install -c deepmodeling deepmd-kit lammps`.
- **Missing `system.data`** — run `build_system.py` or upload a pre-built data file.
- **Missing `models/dpa2.pb`** — download DPA-2 model; path must match `mlip_dpa2.model` in your YAML.
- **GPU not used** — DeepMD offloads via LAMMPS; confirm `nvidia-smi` on the compute node and that the job requested `--gpus=`.
