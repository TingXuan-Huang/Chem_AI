"""Build a Li-polymer-electrolyte system: PACKMOL pack -> LAMMPS data file.

Reads a system YAML config (configs/<sys>.yaml). Writes:
  - <build.output.data>: LAMMPS data file (atom_style atomic), element-typed.

The MLIPs (DPA-2 in LAMMPS, UMA in ASE) handle their own minimisation step,
so we deliberately do *not* run a classical pre-equilibration here.

Usage:
    python scripts/build_system.py --config configs/dac.yaml
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml


# Standard atomic masses (g/mol) for the elements we expect to see in
# Li-polymer-electrolyte systems. Add to this dict if you bring in new
# elements via SMILES.
ATOMIC_MASS = {
    "H":  1.008, "Li": 6.94,  "C": 12.011, "N": 14.007,
    "O": 15.999, "F": 18.998, "P": 30.974, "S": 32.06,
}


def packmol_input_text(composition, box_A, output_pdb, templates_dir,
                       tolerance: float = 2.0) -> str:
    """Format a PACKMOL input file for the given composition."""
    templates_dir = Path(templates_dir)
    lines = [
        f"tolerance {tolerance}",
        f"output {output_pdb}",
        "filetype pdb",
        "",
    ]
    for entry in composition:
        pdb = templates_dir / f"{entry['name']}.pdb"
        if not pdb.exists():
            sys.exit(f"Missing molecular template: {pdb}")
        lines += [
            f"structure {pdb}",
            f"  number {entry['count']}",
            f"  inside box 0. 0. 0. {box_A[0]} {box_A[1]} {box_A[2]}",
            "end structure",
            "",
        ]
    return "\n".join(lines)


def run_packmol(input_text: str, packmol_bin: str = "packmol") -> str:
    """Run PACKMOL via subprocess, fail loudly if it errors."""
    proc = subprocess.run(
        [packmol_bin], input=input_text, text=True, capture_output=True
    )
    if proc.returncode != 0:
        sys.exit(f"PACKMOL failed:\n{proc.stdout}\n{proc.stderr}")
    return proc.stdout


def pdb_to_lammps_data(pdb_path: Path, data_path: Path,
                       type_mapping, box_A) -> None:
    """Convert a PDB to a LAMMPS atomic-style data file with element types."""
    try:
        from ase.io import read as ase_read
    except ImportError as e:
        sys.exit(f"ASE not installed: {e}")

    atoms = ase_read(str(pdb_path), format="proteindatabank")
    elem_to_type = {entry["element"]: entry["type"] for entry in type_mapping}
    for sym in atoms.get_chemical_symbols():
        if sym not in elem_to_type:
            sys.exit(f"Element {sym!r} not in type_mapping")

    n_atoms = len(atoms)
    n_types = len(type_mapping)
    types = [elem_to_type[s] for s in atoms.get_chemical_symbols()]
    pos = atoms.get_positions()

    data_path = Path(data_path)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    with open(data_path, "w") as f:
        f.write("# Built by build_system.py\n\n")
        f.write(f"{n_atoms} atoms\n{n_types} atom types\n\n")
        f.write(f"0.0 {box_A[0]:.4f} xlo xhi\n")
        f.write(f"0.0 {box_A[1]:.4f} ylo yhi\n")
        f.write(f"0.0 {box_A[2]:.4f} zlo zhi\n\n")
        f.write("Masses\n\n")
        for entry in type_mapping:
            elem = entry["element"]
            f.write(f"{entry['type']} {ATOMIC_MASS[elem]:.4f}  # {elem}\n")
        f.write("\nAtoms  # atomic\n\n")
        for i, (t, p) in enumerate(zip(types, pos), start=1):
            f.write(f"{i} {t} {p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")


def build(args: argparse.Namespace) -> None:
    cfg = yaml.safe_load(Path(args.config).read_text())
    if "build" not in cfg:
        sys.exit(f"{args.config}: missing top-level 'build' section")
    b = cfg["build"]
    out_data = Path(b["output"]["data"])
    box_A = b["box_A"]

    with tempfile.TemporaryDirectory() as tmp:
        packed = Path(tmp) / "packed.pdb"
        text = packmol_input_text(
            b["composition"], box_A, str(packed),
            templates_dir=args.templates_dir,
        )
        run_packmol(text, packmol_bin=args.packmol_bin)
        pdb_to_lammps_data(packed, out_data, b["type_mapping"], box_A)

    print(f"Wrote {out_data}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Build initial system via PACKMOL.")
    ap.add_argument("--config", required=True, help="system YAML config")
    ap.add_argument("--packmol-bin", default="packmol",
                    help="PACKMOL executable (default: 'packmol' on PATH)")
    ap.add_argument("--templates-dir", default="templates/molecules",
                    type=Path, help="directory holding per-molecule PDBs")
    return ap


def main() -> None:
    build(build_parser().parse_args())


if __name__ == "__main__":
    main()
