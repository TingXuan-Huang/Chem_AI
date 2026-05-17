"""Generate a PDB from a SMILES string via RDKit.

Used to bootstrap molecular templates for PACKMOL packing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def smiles_to_pdb(smiles: str, output: Path, name: str = "MOL", seed: int = 42) -> None:
    """Embed a SMILES string in 3D and write a PDB file.

    `name` is stamped into PDB columns 18-20 of every HETATM/ATOM record so
    PACKMOL has a consistent residue tag.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError as e:
        sys.exit(f"RDKit not installed: {e}")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        sys.exit(f"Could not parse SMILES: {smiles!r}")

    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, randomSeed=seed) != 0:
        if AllChem.EmbedMolecule(mol, randomSeed=seed, useRandomCoords=True) != 0:
            sys.exit(f"RDKit could not embed 3D coordinates for {smiles!r}")
    AllChem.MMFFOptimizeMolecule(mol)

    safe_name = (name + "   ")[:3]
    out_lines = []
    for line in Chem.MolToPDBBlock(mol).splitlines():
        if line.startswith(("HETATM", "ATOM")) and len(line) >= 20:
            line = line[:17] + safe_name + line[20:]
        out_lines.append(line)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(out_lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a PDB from a SMILES string.")
    ap.add_argument("--smiles", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--name", default="MOL", help="3-letter residue name")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    smiles_to_pdb(args.smiles, args.out, args.name, args.seed)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
