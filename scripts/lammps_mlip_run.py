"""Run MLIP-MD via LAMMPS Python API with DeepMD pair_style.

Assumes LAMMPS was built with the deepmd-kit plugin (`pair_style deepmd`),
e.g. via `conda install -c deepmodeling deepmd-kit lammps`.

Reads a system data file built by build_system.py (atom_style atomic,
element-typed). Trajectory is dumped with unwrapped coordinates (xu yu zu)
so the existing analyzer's MSD step works without further unwrapping.

Usage:
    python scripts/lammps_mlip_run.py \
        --data data/raw_trajectories/DAC/system.data \
        --mlip-model models/dpa2.pb \
        --out-traj data/raw_trajectories/DAC/dpa2_traj.lammpstrj
"""

from __future__ import annotations

import argparse
import sys


def build_commands(args: argparse.Namespace) -> list[str]:
    """Construct the LAMMPS command list. Pure function; no side effects."""
    type_map = (args.type_map or "").strip()
    pair_coeff = f"pair_coeff * * {type_map}" if type_map else "pair_coeff * *"
    return [
        "units metal",
        "atom_style atomic",
        "boundary p p p",
        f"read_data {args.data}",
        f"pair_style deepmd {args.mlip_model}",
        pair_coeff,
        "neighbor 2.0 bin",
        "neigh_modify every 10 delay 0 check yes",
        f"velocity all create {args.temp} {args.seed} rot yes dist gaussian",
        "minimize 1.0e-4 1.0e-6 1000 10000",
        f"timestep {args.dt_ps}",
        f"thermo {args.thermo}",
        "thermo_style custom step temp press etotal density",
        f"fix nvt_eq all nvt temp {args.temp} {args.temp} 0.1",
        f"run {args.equil_steps}",
        "unfix nvt_eq",
        f"fix nvt_prod all nvt temp {args.temp} {args.temp} 0.1",
        f"dump traj all custom {args.dump_every} {args.out_traj} id type xu yu zu",
        "dump_modify traj sort id",
        f"run {args.prod_steps}",
        "undump traj",
        "unfix nvt_prod",
        f"write_data {args.out_data}",
    ]


def run(args: argparse.Namespace) -> None:
    try:
        from lammps import lammps
    except ImportError as e:
        sys.exit(
            "LAMMPS Python module not found. Install LAMMPS with the deepmd "
            "plugin via `conda install -c deepmodeling deepmd-kit lammps`. "
            f"({e})"
        )
    lmp = lammps()
    for cmd in build_commands(args):
        lmp.command(cmd)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="LAMMPS MLIP MD via DeepMD.")
    ap.add_argument("--data", required=True,
                    help="LAMMPS data file from build_system.py")
    ap.add_argument("--mlip-model", required=True,
                    help="DeepMD .pb model file (DPA-2 or trained DP)")
    ap.add_argument("--out-traj", default="dpa2_traj.lammpstrj")
    ap.add_argument("--out-data", default="dpa2_final.data")
    ap.add_argument("--temp", type=float, default=300.0, help="K")
    ap.add_argument("--dt-ps", type=float, default=0.0005,
                    help="ps; default 0.5 fs (MLIPs need shorter dt)")
    ap.add_argument("--equil-steps", type=int, default=20_000)
    ap.add_argument("--prod-steps", type=int, default=200_000)
    ap.add_argument("--dump-every", type=int, default=200)
    ap.add_argument("--thermo", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--type-map", default="",
        help='Space-separated element list matching LAMMPS atom types 1..N '
             '(e.g. "H C N O F P S Li").  Appended to `pair_coeff * *` so the '
             "DeepMD model uses the right species ordering.",
    )
    return ap


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
