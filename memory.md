# Design Notes — Ion-Transport MD Pipeline

Concise log of design choices, why each was picked, and how it was implemented.
Kept short on purpose so it stays useful as the code evolves.

## Scope decision

- The original spec puts Layer A (MD generation) out-of-scope and Layer B
  (trajectory analysis) in-scope.
- The user prompt explicitly asks for "lammps python api with gpu acceleration".
- **Choice**: ship Layer B in full (sections 6.1–6.10 + batch + summary) AND a
  small Layer A LAMMPS runner using the Python API with the GPU package.
  They are separate scripts — Layer B does not depend on the LAMMPS runner.
- Only the spec roadmap up to v0.4 is implemented. v0.5 (ML dataset),
  section 9–10 advanced labels, and section 11 validation suite are intentionally
  *not* built (karpathy guideline #2: no unrequested features).

## Layer A — `scripts/lammps_run.py` (LAMMPS Python API + GPU)

- **Why GPU package, not KOKKOS**: GPU package is a one-line opt-in
  (`package gpu N` + `suffix gpu`); KOKKOS needs a Kokkos-enabled build, more
  command-line flags, and is overkill for a single-GPU pair-force offload.
- **How**: import `lammps` directly (no subprocess), drive commands through
  `lmp.command(...)`. The script exposes a fixed minimize → NVT equil →
  optional NPT → NVT production schedule. Production dump is written with
  `id type xu yu zu` so MSD analysis sees unwrapped coordinates.
- **Force field is *not* baked in**. Two opt-in include hooks:
  - `--pre-include FILE ...`  (lines pasted *before* `read_data`, e.g.
    `pair_style`, `bond_style`, `kspace_style`).
  - `--post-include FILE ...` (lines pasted *after* `read_data`, e.g.
    `pair_coeff`).
  The user supplies whatever their force field needs.
- **macOS caveat**: GPU package targets CUDA. Mac has no NVIDIA GPUs, so
  `--gpu` is opt-in and the script falls back to CPU when it is omitted.
  Run real GPU jobs on a Linux + CUDA box.

## Layer B — `scripts/ion_transport_md_pipeline.py` (analysis)

CLI entry: `python scripts/ion_transport_md_pipeline.py analyze ...`. Single
file, ~250 lines, one function per analysis step.

- **Trajectory I/O**: MDAnalysis. Reason: it auto-detects topology + format
  (LAMMPS data, lammpstrj, GROMACS xtc/tpr, DCD, ...), so the same code works
  whether the trajectory came from `lammps_run.py` or another engine.
- **Optional `--unwrap`**: applies `MDAnalysis.transformations.NoJump()` for
  trajectories that were written wrapped. Spec §14 task 3 explicitly flags
  this as a manual check; the flag makes it a one-line fix.
- **MSD**: `MDAnalysis.analysis.msd.EinsteinMSD(..., fft=True)`. FFT method
  is O(N log N) per Li and matches the textbook Einstein MSD definition the
  spec writes out. `fft=True` requires `tidynamics` — pinned in
  `requirements.txt`. (Caught by the synthetic smoke test.)
- **D_Li fit**: `scipy.stats.linregress` over user-supplied
  `[--tmin-ps, --tmax-ps]`. If the user does not pick a window the fit
  defaults to the middle 50 % of the trajectory (a sane place for a linear
  region; user is expected to refine after seeing the plot).
- **Displacement-distance distribution**: pre-load Li positions into
  `[T, N_Li, 3]`, then sweep all `t0` for one user-chosen `--disp-lag-ps`.
  Memory ≈ `4 × T × N_Li × 3` bytes — fine for any production-scale Li set.
- **RDF**: `MDAnalysis.analysis.rdf.InterRDF`. Running CN computed locally
  by `cumsum(4πr² ρ_b g(r) dr)`, where `ρ_b = N_b / V_box`. This matches
  the spec formula and gives a CN(r) curve we can read the first-shell value
  off.
- **First-peak / first-min detection**: `argmax` of g(r) restricted to
  r > 1 Å, then `argmin` of g(r) after that index. Crude but robust enough
  for clean RDFs; the user can override by reading the CSV.
- **Cluster stats**: `MDAnalysis.lib.distances.distance_array` with PBC,
  per frame, counts neighbors within `--cluster-cut-A`. Builds a 6×6
  `P[N_Li_neigh, N_anion_neigh]` matrix (cap at 5 for both axes).
  Free-Li / contact-ion-pair / aggregate fractions follow the spec §6.9
  definitions:
    - free Li⁺: `cells[:, 0].sum()` (zero anion neighbors)
    - CIP: `cells[:, 1].sum()` (one anion neighbor)
    - aggregate: `cells[:, 2:].sum()` (≥2 anion neighbors)
- **Residence grid**: `np.histogramdd` on Li positions wrapped into the
  frame-0 box, default 80³ bins. Stored as `.npy` (probability cube) and
  `.npz` (bin edges). Assumes orthogonal box and roughly NVT — non-orthogonal
  or NPT trajectories need a different binning scheme.

## Output naming

Filenames follow spec §8 verbatim so plots/scripts that consume them later
do not need a translation table.

## Batch + summary

- `scripts/run_all_systems.py`: glob `configs/*.yaml`, build the analyzer
  CLI, run as a subprocess. Subprocess (instead of importing `analyze` as a
  function) keeps the standalone CLI as the single source of truth and gives
  one-off failures their own crash without taking the whole batch with them.
- `scripts/plot_summary.py`: reads each `results/<system>/diffusion_result.json`
  + `rdf_coordination_summary.csv` + `ion_association_fractions.json`,
  writes `results/summary.csv` and a two-panel comparison PNG (D_Li bar +
  CN bars per pair).

## Configs

YAML, one file per system. Holds: system name, top/traj paths, atom
selections, fit window, cutoffs, residence-grid flag. The names line up
1:1 with analyzer CLI flags.

## Open assumptions

- LAMMPS `units real` (Å, fs, ps, kcal/mol) and dumps with `xu yu zu`.
  MDAnalysis usually auto-detects units from the trajectory format, but
  verify on your specific system.
- Trajectory has been unwrapped (MSD requires it). If not, pass `--unwrap`.
- Periodic box is orthogonal for the residence grid step.
- A force-field include file (or coeffs in the data file) is provided by
  the user.

## Verification

End-to-end smoke test on a synthetic 200-frame, 50-atom random-walk
trajectory was run during development:

- All 18 spec §8 output files were produced for the analyzer.
- Linear MSD fit recovered the expected diffusion coefficient
  (input random-walk D ≈ 1.25e-5 cm²/s, recovered ≈ 9.3e-6 cm²/s,
  r = 0.998 — same order of magnitude as expected).
- Batch + summary path (`run_all_systems.py` → `plot_summary.py`)
  produced `summary.csv` and `summary_plot.png` correctly.

The smoke-test script and its outputs were removed after passing
(per karpathy guideline #3 — clean up your own mess). The pipeline is
verified at the API level; correctness on real polymer-electrolyte
trajectories still requires user validation per spec §11.

## MLIP Pipeline (v0.6) — Design Decisions

User pivoted off classical force-field assembly. Two pretrained MLIPs drive
MD instead, no active learning loop, foundation models only.

### MLIP choice — DeepMD DPA-2 (primary) + Meta UMA (cross-check)

- **Primary: DPA-2 in LAMMPS** via `pair_style deepmd`. Reasons: native
  LAMMPS plugin (fastest path to ns-scale trajectories on a GPU), DPA-2 is
  trained on a broad materials-and-molecules corpus, mature production stack
  with `dp` CLI for inference and DP-GEN ready if AL becomes necessary.
- **Cross-check: UMA-S via ASE/fairchem**. Reasons: best zero-shot accuracy
  benchmark; fully Python so it works without a custom LAMMPS build; runs
  shorter (~10 ps) trajectories or single-point energies on snapshots from
  the DPA-2 trajectory. If DPA-2 and UMA agree on first-shell CN and `D_Li`
  within ~2× we trust the result; if they diverge, that's the trigger to
  add an AL loop later.

### Why no active learning yet

- DP-GEN style AL adds ~1–2 weeks of work and requires a DFT engine
  (VASP / QE / CP2K) on the server. We start with foundation-only and
  reserve AL as v0.7 if validation fails.

### System-construction strategy

- PACKMOL packs random initial positions from molecular PDB templates.
- No classical pre-equilibration: PACKMOL `tolerance 2.0` is enough; both
  MLIPs handle the relaxation in their own minimization step. (Keeps the
  pipeline FF-free, which was the whole point of switching.)
- Atom typing is element-based: a single integer atom type per element,
  mapped via `type_mapping` in the per-system config. Both DPA-2 and UMA
  consume element symbols; LAMMPS data files use the integer types and
  pair_style `deepmd` reads the type→element map from the model file.

### Why two trajectory file formats

- DPA-2 + LAMMPS dumps `*.lammpstrj` with `id type xu yu zu` (unwrapped).
- UMA + ASE writes `*.dcd`. ASE's LAMMPS-data reader doesn't preserve type
  metadata cleanly, so the UMA runner attaches the `type_map` explicitly.
- MDAnalysis reads both, so Layer B is unaffected.

### MLIP files added (v0.6)

```
requirements-mlip.txt
templates/molecules/Li.pdb        (hand-written single atom)
templates/molecules/DME.pdb       (RDKit-generated example)
templates/molecules/README.md     (SMILES recipes for the rest)
scripts/build_molecule.py         (SMILES -> PDB)
scripts/build_system.py           (PACKMOL -> LAMMPS data)
scripts/lammps_mlip_run.py        (DPA-2 + LAMMPS, primary MD)
scripts/ase_mlip_run.py           (UMA + ASE, cross-check MD)
scripts/run_all_mlip.py           (4-stage batch driver)
configs/{dac,cgpei,cgbpei}.yaml   (extended with build/mlip_*/md sections)
tests/conftest.py                 (sys.path setup for scripts/)
tests/test_build_molecule.py
tests/test_build_system.py
tests/test_lammps_mlip_run.py
tests/test_ase_mlip_run.py
tests/test_run_all_mlip.py
```

### Skipped on purpose

- `scripts/validate_mlip.py` (cross-MLIP energy/force MAE) — dropped per
  "keep it simple". Equivalent cross-check is to re-run the analyzer on
  the UMA `.dcd` and compare `D_Li` / first-shell CN against the DPA-2
  result.

### TDD reset

First pass on `build_molecule.py` was tests-after, which violates the
test-driven-development skill ("write code before the test? Delete it").
Both files were deleted and rewritten test-first. Subsequent tasks
(3, 4, 5, 6) were strict TDD: write failing test, watch it fail with
the expected error (ModuleNotFoundError), implement minimal code, watch
the suite turn green.

### v0.6 verification

- 24 unit tests, all passing on macOS with only the light deps
  (`pytest rdkit ase numpy pyyaml`).
- `--help` works for every new script.
- All 3 configs parse via `yaml.safe_load`.
- `templates/molecules/DME.pdb` generated end-to-end by
  `scripts/build_molecule.py` from SMILES `COCCOC`, 15 atoms with
  correct connectivity (4 C + 2 O + 9 H).
- The actual MD stages (LAMMPS+DeepMD, ASE+UMA) are *not* exercised on
  macOS — they need the GPU server. Their pure helpers (command-list
  builders, type-map mappers, data-file readers) are covered by the
  unit tests.

### Open assumptions for v0.6

- DPA-2 model lives at `models/dpa2.pb` (or override per config).
  Foundation models: https://github.com/deepmodeling/DPA-2.
- UMA model loaded by `fairchem.core.OCPCalculator(model_name="uma-small", ...)`;
  `local_cache="models/uma"` so weights download once into the project.
- LAMMPS build has the deepmd plugin compiled in.
- 0.5 fs time step is appropriate for both MLIPs at 300 K.
- Element-only atom typing means we cannot separate "DAC oxygens" from
  "PDOL oxygens" or "DME oxygens" in the analyzer; the Li-O RDF is over
  *all* oxygens. Acceptable for v0.6; revisit if per-subgroup CNs become
  important.

## Files added

```
memory.md
requirements.txt
README.md
scripts/ion_transport_md_pipeline.py
scripts/lammps_run.py
scripts/run_all_systems.py
scripts/plot_summary.py
configs/dac.yaml
configs/cgpei.yaml
configs/cgbpei.yaml
data/raw_trajectories/{DAC,CgPEI,CgBPEI}/   (empty, drop your trajectories here)
data/processed/                              (empty)
results/{DAC,CgPEI,CgBPEI}/                  (empty, analyzer fills these)
```
