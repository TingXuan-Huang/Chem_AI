"""Layer A: drive LAMMPS through its Python API with the GPU package.

Runs:  minimize -> NVT equilibration -> (optional) NPT -> NVT production
       -> dump trajectory with unwrapped coordinates (xu yu zu).

The force field is supplied by the user via include files:

    --pre-include  FILE [FILE ...]   pasted *before* read_data
                                     (pair_style, bond_style, kspace_style, ...)
    --post-include FILE [FILE ...]   pasted *after*  read_data
                                     (pair_coeff, bond_coeff, ...)

Usage example:
    python scripts/lammps_run.py \
        --data system.data \
        --pre-include  ff/styles.in \
        --post-include ff/coeffs.in \
        --gpu --gpu-n 1
"""

from __future__ import annotations

import argparse
import sys


def _exec(lmp, line: str) -> None:
    line = line.strip()
    if line and not line.startswith("#"):
        lmp.command(line)


def build_commands(args: argparse.Namespace) -> list[str]:
    """Construct the LAMMPS command list. Pure function; no side effects."""
    commands = [
        "units real",
        "atom_style full",
        "boundary p p p",
    ]

    if args.gpu:
        # LAMMPS requires package/suffix setup before atoms are created.
        commands.extend([
            f"package gpu {args.gpu_n} neigh yes",
            "suffix gpu",
        ])

    commands.extend(f"include {path}" for path in args.pre_include)
    commands.append(f"read_data {args.data}")
    commands.extend(f"include {path}" for path in args.post_include)

    commands.extend([
        "neighbor 2.0 bin",
        "neigh_modify every 1 delay 0 check yes",
        f"velocity all create {args.temp} {args.seed} rot yes dist gaussian",
        "minimize 1.0e-4 1.0e-6 1000 10000",
        f"timestep {args.dt_fs}",
        f"thermo {args.thermo}",
        "thermo_style custom step temp press etotal density",
        f"fix nvt_eq all nvt temp {args.temp} {args.temp} 100.0",
        f"run {args.equil_steps}",
        "unfix nvt_eq",
    ])

    if args.npt_steps > 0:
        commands.extend([
            f"fix npt_eq all npt temp {args.temp} {args.temp} 100.0 "
            f"iso {args.pressure} {args.pressure} 1000.0",
            f"run {args.npt_steps}",
            "unfix npt_eq",
        ])

    commands.extend([
        f"fix nvt_prod all nvt temp {args.temp} {args.temp} 100.0",
        f"dump traj all custom {args.dump_every} {args.out_traj} id type xu yu zu",
        "dump_modify traj sort id",
        f"run {args.prod_steps}",
        "undump traj",
        "unfix nvt_prod",
        f"write_data {args.out_data}",
    ])
    return commands


def run(args: argparse.Namespace) -> None:
    try:
        from lammps import lammps
    except ImportError as e:
        sys.exit(
            "Could not import the LAMMPS Python module. Install via "
            "`conda install -c conda-forge lammps` or build LAMMPS with "
            "-DPKG_PYTHON=on -DBUILD_SHARED_LIBS=on. "
            f"(import error: {e})"
        )

    lmp = lammps()
    for command in build_commands(args):
        _exec(lmp, command)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="LAMMPS Python-API runner.")
    ap.add_argument("--data", required=True, help="LAMMPS data file")
    ap.add_argument("--pre-include", nargs="*", default=[],
                    help="files included BEFORE read_data (styles, kspace)")
    ap.add_argument("--post-include", nargs="*", default=[],
                    help="files included AFTER read_data (coeffs)")
    ap.add_argument("--out-traj", default="traj.lammpstrj",
                    help="production trajectory output (custom xu yu zu)")
    ap.add_argument("--out-data", default="final.data",
                    help="final state written via write_data")
    ap.add_argument("--temp", type=float, default=300.0, help="K")
    ap.add_argument("--pressure", type=float, default=1.0, help="atm (NPT)")
    ap.add_argument("--dt-fs", type=float, default=1.0)
    ap.add_argument("--equil-steps", type=int, default=100_000)
    ap.add_argument("--npt-steps", type=int, default=200_000,
                    help="0 to skip NPT")
    ap.add_argument("--prod-steps", type=int, default=1_000_000)
    ap.add_argument("--dump-every", type=int, default=1000)
    ap.add_argument("--thermo", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu", action="store_true",
                    help="enable LAMMPS GPU package + suffix")
    ap.add_argument("--gpu-n", type=int, default=1, help="number of GPUs")
    return ap


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
