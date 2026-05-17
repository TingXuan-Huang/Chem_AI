# Ion-Transport MD Pipeline

Python workflow for measuring Li⁺ transport in quasi-solid polymer
electrolytes. Three layers, each independent:

1. **Layer A-classic — classical-FF MD** (`scripts/lammps_run.py`)
   LAMMPS Python API + GPU package. User supplies force-field include files.
2. **Layer A-MLIP — MLIP-driven MD** (`scripts/build_system.py`,
   `scripts/lammps_mlip_run.py`, `scripts/ase_mlip_run.py`)
   Build with PACKMOL, run MD with DeepMD DPA-2 (LAMMPS, primary) and
   Meta UMA (ASE, cross-check). No force-field assembly required.
3. **Layer B — trajectory analysis** (`scripts/ion_transport_md_pipeline.py`)
   Li⁺ MSD, `D_Li`, displacement distribution, RDFs, coordination numbers,
   ion-cluster statistics, and a 3D residence-probability grid.

Layer B is engine-agnostic — anything MDAnalysis can read works.

See `memory.md` for design choices and assumptions.

## Install

```bash
# Analysis only (Layer B)
pip install -r requirements.txt

# Classical Layer A (LAMMPS Python API + GPU)
conda install -c conda-forge lammps

# MLIP Layer A (DeepMD DPA-2 + UMA + supporting tools)
pip install -r requirements-mlip.txt
conda install -c deepmodeling deepmd-kit lammps   # LAMMPS with deepmd plugin
conda install -c conda-forge packmol               # for system construction
```

## Quick start — analyse one trajectory

```bash
python scripts/ion_transport_md_pipeline.py analyze \
  --top  data/raw_trajectories/CgBPEI/system.data \
  --traj data/raw_trajectories/CgBPEI/traj.lammpstrj \
  --out  results/CgBPEI \
  --li     "type 1" \
  --o-cell "type 5 6" \
  --o-poly "type 7" \
  --n-poly "type 8" \
  --anion  "type 9" \
  --tmin-ps 500 --tmax-ps 3000 \
  --cluster-cut-A 4.0 \
  --residence-grid \
  --unwrap
```

Outputs (matching the spec) land in `results/CgBPEI/`:

```
diffusion_result.json
li_msd.csv,                  li_msd_fit.png
li_displacement_distance.csv li_displacement_distance.png
rdf_Li-O_cellulose.csv       rdf_Li-O_cellulose.png
rdf_Li-O_PDOL.csv            rdf_Li-O_PDOL.png
rdf_Li-N_polyamine.csv       rdf_Li-N_polyamine.png
rdf_Li-anion_center.csv      rdf_Li-anion_center.png
rdf_coordination_summary.csv
ion_cluster_probability.csv
ion_association_fractions.json
li_residence_probability_grid.npy
li_residence_probability_grid_edges.npz
```

## Quick start — run all systems

Edit the YAML files in `configs/` so the selections and paths match
your topology, then:

```bash
python scripts/run_all_systems.py --configs configs
python scripts/plot_summary.py    --results results
```

`results/summary.csv` and `results/summary_plot.png` will be created.

## Quick start — generate a trajectory with classical LAMMPS + GPU

```bash
python scripts/lammps_run.py \
  --data        data/raw_trajectories/CgBPEI/system.data \
  --pre-include ff/cgbpei_styles.in \
  --post-include ff/cgbpei_coeffs.in \
  --out-traj    data/raw_trajectories/CgBPEI/traj.lammpstrj \
  --out-data    data/raw_trajectories/CgBPEI/final.data \
  --temp 300 --dt-fs 1.0 \
  --equil-steps 100000 --npt-steps 200000 --prod-steps 1000000 \
  --dump-every 1000 \
  --gpu --gpu-n 1
```

`--gpu` opts into the GPU package (`package gpu N` + `suffix gpu`). On a
machine without an NVIDIA GPU (e.g. macOS), drop `--gpu` to run on CPU.

## Quick start — MLIP pipeline (DPA-2 + UMA, no force field)

End-to-end on three systems (DAC / CgPEI / CgBPEI) using the included
configs:

```bash
# 1. Generate molecular templates from SMILES (one-time setup).
#    See templates/molecules/README.md for the full list.
python scripts/build_molecule.py --smiles "F[P-](F)(F)(F)(F)F"  --out templates/molecules/PF6.pdb        --name PF6
python scripts/build_molecule.py --smiles "COCCOC"              --out templates/molecules/DME.pdb        --name DME
python scripts/build_molecule.py --smiles "OCCOCOCCOCOCCOCOCCOCOCCO" \
                                 --out templates/molecules/PDOL_5mer.pdb  --name PDL
# ... and likewise for DAC_unit, PEI_4mer, BPEI_branch.

# 2. Drop a DeepMD DPA-2 model at models/dpa2.pb (or edit the model path
#    in configs/*.yaml). Foundation models live at:
#      https://github.com/deepmodeling/DPA-2

# 3. Run the full pipeline for every config in configs/:
python scripts/run_all_mlip.py --configs configs

# 4. Aggregate results across systems:
python scripts/plot_summary.py --results results
```

Per system this runs four stages:

```
build_system.py      pack initial config         -> system.data
lammps_mlip_run.py   DPA-2 production MD         -> dpa2_traj.lammpstrj
ase_mlip_run.py      UMA cross-check MD          -> uma_traj.dcd
ion_transport_md_pipeline.py  analyse DPA-2 traj -> results/<sys>/
```

Compare `D_Li` and first-shell CN between the DPA-2 and UMA outputs by
re-running step 4 against the UMA trajectory (just change `--traj` to
the `.dcd` path).

## Run a single MLIP-MD step

```bash
# DPA-2 in LAMMPS (production, primary):
python scripts/lammps_mlip_run.py \
  --data data/raw_trajectories/DAC/system.data \
  --mlip-model models/dpa2.pb \
  --out-traj data/raw_trajectories/DAC/dpa2_traj.lammpstrj \
  --temp 300 --dt-ps 0.0005 \
  --equil-steps 20000 --prod-steps 200000 --dump-every 200

# UMA via ASE (cross-check, shorter):
python scripts/ase_mlip_run.py \
  --data data/raw_trajectories/DAC/system.data \
  --type-map "H C N O F P S Li" \
  --uma-size small \
  --out-traj data/raw_trajectories/DAC/uma_traj.dcd \
  --temp 300 --dt-fs 0.5 \
  --equil-steps 2000 --prod-steps 20000 --dump-every 20
```

## Tests

```bash
# Light deps (no DeepMD / UMA / LAMMPS / PACKMOL needed for unit tests)
python -m venv .venv-test
.venv-test/bin/pip install pytest rdkit ase numpy pyyaml
.venv-test/bin/pytest tests/ -v
```

Tests cover: SMILES->PDB generation, PACKMOL input formatting,
PDB->LAMMPS-data conversion, LAMMPS command-list construction (no
LAMMPS/DeepMD needed), ASE atoms construction from data files,
and per-system batch wiring.

## Atom selections (MDAnalysis)

Selection strings follow MDAnalysis syntax. Two common patterns:

- LAMMPS data file with no residue names → use atom **types**:
  `--li "type 1"`, `--anion "type 9"`.
- Topology with residue names (GROMACS, PDB, ...) → use the names:
  `--li "name LI"`, `--anion "resname TFSI PF6 and (name N P S)"`.

## Layer roadmap (status)

- [x] v0.1 load → MSD → D_Li fit
- [x] v0.2 RDFs, coordination numbers, displacement distribution
- [x] v0.3 ion-cluster stats, free-Li fraction, residence grid
- [x] v0.4 batch over DAC / CgPEI / CgBPEI + summary plot
- [ ] v0.5 ML dataset builder (intentionally deferred)
- [x] v0.6 MLIP integration (DPA-2 in LAMMPS + UMA in ASE, no AL)
- [ ] v0.7 active learning (DP-GEN) — only if v0.6 cross-check disagrees
- [ ] v1.0 ML prediction (descriptor-based regression)
- [ ] v2.0 E(3)-equivariant direct structure-to-transport model
