"""TDD tests for scripts/ase_mlip_run.py.

The fairchem (UMA) backend isn't installed in the test venv; we test the
pure helpers that read the data file and assign element symbols, plus
argparse.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ase_mlip_run import (
    assign_symbols,
    build_parser,
    make_atoms,
    parse_type_map,
    read_lammps_atomic_data,
)


# ---------------------------------------------------------------------------
# parse_type_map
# ---------------------------------------------------------------------------

def test_parse_type_map_splits_on_whitespace():
    assert parse_type_map("H C N O F P S Li") == ["H", "C", "N", "O", "F", "P", "S", "Li"]


# ---------------------------------------------------------------------------
# assign_symbols
# ---------------------------------------------------------------------------

def test_assign_symbols_maps_types_to_elements():
    assert assign_symbols([1, 2, 1, 3], ["H", "C", "Li"]) == ["H", "C", "H", "Li"]


def test_assign_symbols_unknown_type_raises():
    with pytest.raises(SystemExit):
        assign_symbols([1, 5], ["H", "C"])


# ---------------------------------------------------------------------------
# read_lammps_atomic_data
# ---------------------------------------------------------------------------

_DATA_TEXT = """\
# tiny system

3 atoms
2 atom types

0.0 12.5000 xlo xhi
0.0 11.0000 ylo yhi
0.0 13.0000 zlo zhi

Masses

1 1.008  # H
2 6.940  # Li

Atoms  # atomic

1 1 0.000 0.000 0.000
2 1 1.500 0.000 0.000
3 2 5.000 6.000 7.000
"""


def test_read_lammps_atomic_data_parses_box_types_positions(tmp_path):
    path = tmp_path / "tiny.data"
    path.write_text(_DATA_TEXT)
    box_A, types, positions = read_lammps_atomic_data(path)
    assert box_A == [12.5, 11.0, 13.0]
    assert types == [1, 1, 2]
    assert positions[0] == [0.0, 0.0, 0.0]
    assert positions[2] == [5.0, 6.0, 7.0]
    assert len(positions) == 3


# ---------------------------------------------------------------------------
# make_atoms
# ---------------------------------------------------------------------------

def test_make_atoms_uses_type_map_for_symbols(tmp_path):
    pytest.importorskip("ase")
    path = tmp_path / "tiny.data"
    path.write_text(_DATA_TEXT)
    atoms = make_atoms(path, ["H", "Li"])
    assert atoms.get_chemical_symbols() == ["H", "H", "Li"]
    assert all(atoms.pbc)
    cell = atoms.get_cell().lengths()
    assert list(cell) == pytest.approx([12.5, 11.0, 13.0])


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------

def test_parser_requires_data_typemap_and_outtraj():
    ap = build_parser()
    with pytest.raises(SystemExit):
        ap.parse_args([])
    args = ap.parse_args([
        "--data", "x.data",
        "--type-map", "H C Li",
        "--out-traj", "uma.dcd",
    ])
    assert args.data == "x.data"
    assert args.type_map == "H C Li"
    assert str(args.out_traj) == "uma.dcd"
    assert args.uma_size == "small"   # default
