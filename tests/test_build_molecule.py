"""TDD tests for scripts/build_molecule.py.

We want a `smiles_to_pdb(smiles, output_path, name)` function that:
  - writes a non-empty PDB file with HETATM/ATOM records,
  - stamps the 3-letter residue name into columns 18-20,
  - exits cleanly on an invalid SMILES.
"""

from __future__ import annotations

import pytest

pytest.importorskip("rdkit")

from build_molecule import smiles_to_pdb


def test_smiles_to_pdb_writes_dme(tmp_path):
    out = tmp_path / "DME.pdb"
    smiles_to_pdb("COCCOC", out, name="DME")
    text = out.read_text()
    assert text, "PDB file should not be empty"
    atom_lines = [ln for ln in text.splitlines()
                  if ln.startswith(("HETATM", "ATOM"))]
    assert atom_lines, "expected at least one HETATM/ATOM line"
    # Residue name is stamped into columns 18-20.
    assert all(ln[17:20] == "DME" for ln in atom_lines), \
        "every ATOM/HETATM line should carry the residue name 'DME'"


def test_smiles_to_pdb_rejects_garbage(tmp_path):
    with pytest.raises(SystemExit):
        smiles_to_pdb("NOT_A_SMILES_$$$", tmp_path / "x.pdb")
