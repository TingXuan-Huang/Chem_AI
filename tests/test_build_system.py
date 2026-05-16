"""TDD tests for scripts/build_system.py.

Two pure helpers we want:

  packmol_input_text(composition, box_A, output_pdb, templates_dir)
      -> str    formatted PACKMOL input file content.

  pdb_to_lammps_data(pdb_path, data_path, type_mapping, box_A)
      -> None   writes a LAMMPS `atomic`-style data file with element-based
                integer atom types matching `type_mapping`.

PACKMOL itself is a binary; we don't run it from tests. The orchestration
function is exercised by the end-to-end smoke test in Task 7.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from build_system import packmol_input_text, pdb_to_lammps_data


# ---------------------------------------------------------------------------
# packmol_input_text
# ---------------------------------------------------------------------------

def test_packmol_input_text_basic(tmp_path):
    # Two minimal template PDBs so the path-existence check passes.
    (tmp_path / "Li.pdb").write_text("HETATM    1 LI  LI  A  1   0  0  0\nEND\n")
    (tmp_path / "DME.pdb").write_text("HETATM    1 C   DME A  1   0  0  0\nEND\n")

    composition = [
        {"name": "Li",  "count": 5},
        {"name": "DME", "count": 10},
    ]
    text = packmol_input_text(
        composition, box_A=[20.0, 20.0, 20.0],
        output_pdb="/tmp/packed.pdb", templates_dir=tmp_path,
    )

    assert text.startswith("tolerance")
    assert "output /tmp/packed.pdb" in text
    assert "filetype pdb" in text
    # One structure block per species, with the right counts.
    assert text.count("structure ") == 2
    assert text.count("end structure") == 2
    assert "number 5" in text
    assert "number 10" in text
    # Box dimensions appear inside the constraint line.
    assert "0. 0. 0. 20.0 20.0 20.0" in text


def test_packmol_input_text_missing_template(tmp_path):
    composition = [{"name": "DoesNotExist", "count": 1}]
    with pytest.raises(SystemExit):
        packmol_input_text(
            composition, box_A=[10.0, 10.0, 10.0],
            output_pdb=str(tmp_path / "x.pdb"), templates_dir=tmp_path,
        )


# ---------------------------------------------------------------------------
# pdb_to_lammps_data
# ---------------------------------------------------------------------------

def _write_minimal_pdb(path: Path) -> None:
    """Three atoms: one H, one C, one Li at known positions."""
    path.write_text(
        "HETATM    1  H   MOL A   1       0.000   0.000   0.000  1.00  0.00           H\n"
        "HETATM    2  C   MOL A   1       1.500   0.000   0.000  1.00  0.00           C\n"
        "HETATM    3 LI   MOL A   1       3.000   0.000   0.000  1.00  0.00          LI\n"
        "END\n"
    )


def test_pdb_to_lammps_data_writes_atoms_and_types(tmp_path):
    pytest.importorskip("ase")

    pdb = tmp_path / "tiny.pdb"
    _write_minimal_pdb(pdb)
    data = tmp_path / "tiny.data"
    type_mapping = [
        {"type": 1, "element": "H"},
        {"type": 2, "element": "C"},
        {"type": 3, "element": "Li"},
    ]
    pdb_to_lammps_data(pdb, data, type_mapping, box_A=[10.0, 10.0, 10.0])

    text = data.read_text()
    assert "3 atoms" in text
    assert "3 atom types" in text
    assert "0.0 10.0000 xlo xhi" in text
    assert "Masses" in text
    # H, C, Li masses appear (rounded to 4 dp).
    assert "1.0080" in text
    assert "12.0110" in text
    assert "6.9400" in text
    # Atom section: 3 lines with the right type indices in column 2.
    atom_lines = [ln for ln in text.splitlines()
                  if ln and ln.split()[0].isdigit() and len(ln.split()) == 5]
    assert len(atom_lines) == 3
    types_seen = sorted(int(ln.split()[1]) for ln in atom_lines)
    assert types_seen == [1, 2, 3]


def test_pdb_to_lammps_data_rejects_unknown_element(tmp_path):
    pytest.importorskip("ase")

    pdb = tmp_path / "tiny.pdb"
    _write_minimal_pdb(pdb)
    type_mapping = [
        {"type": 1, "element": "H"},
        {"type": 2, "element": "C"},
        # Li deliberately absent.
    ]
    with pytest.raises(SystemExit):
        pdb_to_lammps_data(pdb, tmp_path / "x.data", type_mapping,
                           box_A=[10.0, 10.0, 10.0])
