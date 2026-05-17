"""Run MLIP-MD via ASE with the Meta UMA (fairchem) calculator.

Reads a LAMMPS atomic-style data file (built by build_system.py), maps
integer atom types to element symbols using a user-supplied --type-map,
attaches a UMA calculator, runs Langevin NVT, and writes a DCD trajectory
that the existing analyzer can read.

Usage:
    python scripts/ase_mlip_run.py \
        --data data/raw_trajectories/DAC/system.data \
        --type-map "H C N O F P S Li" \
        --uma-size small \
        --out-traj data/raw_trajectories/DAC/uma_traj.dcd
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Pure helpers (testable without ASE / fairchem)
# ---------------------------------------------------------------------------

def parse_type_map(text: str) -> list[str]:
    """'H C N O' -> ['H', 'C', 'N', 'O']."""
    return text.split()


def assign_symbols(types, type_map):
    """Map integer LAMMPS types to element symbols using `type_map[t-1]`."""
    out = []
    for t in types:
        if t < 1 or t > len(type_map):
            sys.exit(f"Type {t} out of range for type_map of length {len(type_map)}")
        out.append(type_map[t - 1])
    return out


def read_lammps_atomic_data(path):
    """Parse a LAMMPS atomic-style data file written by build_system.py.

    Returns (box_A, types, positions) where:
      box_A      : list[float] of length 3
      types      : list[int]
      positions  : list[list[float]]
    """
    text = Path(path).read_text()
    lines = text.splitlines()

    def _find(suffix):
        for line in lines:
            if line.strip().endswith(suffix):
                return line.split()
        sys.exit(f"Data file missing line ending in '{suffix}'")

    xlo, xhi = float(_find("xlo xhi")[0]), float(_find("xlo xhi")[1])
    ylo, yhi = float(_find("ylo yhi")[0]), float(_find("ylo yhi")[1])
    zlo, zhi = float(_find("zlo zhi")[0]), float(_find("zlo zhi")[1])
    box_A = [xhi - xlo, yhi - ylo, zhi - zlo]

    atoms_idx = next((i for i, line in enumerate(lines)
                      if line.strip().startswith("Atoms")), None)
    if atoms_idx is None:
        sys.exit("Data file missing 'Atoms' section")

    types: list[int] = []
    positions: list[list[float]] = []
    for line in lines[atoms_idx + 1:]:
        tokens = line.split()
        if len(tokens) == 5 and tokens[0].isdigit() and tokens[1].isdigit():
            types.append(int(tokens[1]))
            positions.append([float(tokens[2]), float(tokens[3]), float(tokens[4])])
    return box_A, types, positions


def make_atoms(data_path, type_map):
    """Build an ASE Atoms with element symbols and orthogonal periodic box."""
    try:
        from ase import Atoms
    except ImportError as e:
        sys.exit(f"ASE not installed: {e}")

    box_A, types, positions = read_lammps_atomic_data(data_path)
    symbols = assign_symbols(types, type_map)
    atoms = Atoms(
        symbols=symbols,
        positions=positions,
        cell=[box_A[0], box_A[1], box_A[2], 90.0, 90.0, 90.0],
        pbc=[True, True, True],
    )
    return atoms


# ---------------------------------------------------------------------------
# UMA + MD (only exercised at runtime; not unit-tested)
# ---------------------------------------------------------------------------

def setup_uma(atoms, uma_size: str):
    try:
        from fairchem.core import OCPCalculator
    except ImportError as e:
        sys.exit(
            "fairchem not installed. Install via "
            "`pip install -r requirements-mlip.txt`. "
            f"({e})"
        )
    calc = OCPCalculator(model_name=f"uma-{uma_size}", local_cache="models/uma")
    atoms.calc = calc
    return atoms


def run_md(atoms, *, temp, dt_fs, equil_steps, prod_steps, dump_every, out_traj):
    from ase import units
    from ase.io import write
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

    MaxwellBoltzmannDistribution(atoms, temperature_K=temp)
    dyn = Langevin(atoms, timestep=dt_fs * units.fs,
                   temperature_K=temp, friction=0.01 / units.fs)

    if equil_steps > 0:
        dyn.run(equil_steps)

    out_traj = Path(out_traj)
    out_traj.parent.mkdir(parents=True, exist_ok=True)
    snapshots = []
    for step in range(prod_steps):
        dyn.run(1)
        if step % dump_every == 0:
            snapshots.append(atoms.copy())
    write(str(out_traj), snapshots, format="dcd")


def run(args: argparse.Namespace) -> None:
    type_map = parse_type_map(args.type_map)
    atoms = make_atoms(args.data, type_map)
    atoms = setup_uma(atoms, args.uma_size)
    run_md(
        atoms,
        temp=args.temp, dt_fs=args.dt_fs,
        equil_steps=args.equil_steps, prod_steps=args.prod_steps,
        dump_every=args.dump_every, out_traj=args.out_traj,
    )


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="ASE MLIP MD via UMA (fairchem).")
    ap.add_argument("--data", required=True, help="LAMMPS atomic-style data file")
    ap.add_argument("--type-map", required=True,
                    help="space-separated element list, e.g. 'H C N O F P S Li'")
    ap.add_argument("--out-traj", required=True, type=Path,
                    help="output trajectory (DCD)")
    ap.add_argument("--uma-size", default="small",
                    choices=["small", "medium", "large"])
    ap.add_argument("--temp", type=float, default=300.0)
    ap.add_argument("--dt-fs", type=float, default=0.5)
    ap.add_argument("--equil-steps", type=int, default=2_000)
    ap.add_argument("--prod-steps", type=int, default=20_000)
    ap.add_argument("--dump-every", type=int, default=20)
    return ap


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
