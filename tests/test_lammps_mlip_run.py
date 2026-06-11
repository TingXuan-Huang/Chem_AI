"""TDD tests for scripts/lammps_mlip_run.py.

The LAMMPS Python module isn't installed in the test venv, so we test the
pure command-list builder rather than running LAMMPS itself.

We want a `build_commands(args) -> list[str]` function that returns the
exact LAMMPS commands the runner will execute, plus a `build_parser()`
argparse factory.
"""

from __future__ import annotations

import argparse

import pytest

from lammps_mlip_run import build_commands, build_parser


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------

def test_parser_requires_data_and_model():
    ap = build_parser()
    with pytest.raises(SystemExit):
        ap.parse_args([])
    with pytest.raises(SystemExit):
        ap.parse_args(["--data", "x.data"])
    args = ap.parse_args(["--data", "sys.data", "--mlip-model", "m.pb"])
    assert args.data == "sys.data"
    assert args.mlip_model == "m.pb"
    # Defaults appropriate for DPA-2 (metal units, sub-fs MD).
    assert args.dt_ps == 0.0005
    assert args.equil_steps > 0
    assert args.prod_steps > 0


# ---------------------------------------------------------------------------
# command-list builder
# ---------------------------------------------------------------------------

def _make_args(**overrides) -> argparse.Namespace:
    base = dict(
        data="sys.data", mlip_model="dpa2.pb",
        out_traj="prod.lammpstrj", out_data="final.data",
        temp=300.0, dt_ps=0.0005,
        equil_steps=1000, prod_steps=10_000,
        dump_every=100, thermo=500, seed=7,
        type_map="",
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_build_commands_uses_metal_units_and_atomic_style():
    cmds = build_commands(_make_args())
    assert cmds[0] == "units metal"
    assert "atom_style atomic" in cmds


def test_build_commands_uses_deepmd_pair_style():
    cmds = build_commands(_make_args(mlip_model="my_model.pb"))
    assert "pair_style deepmd my_model.pb" in cmds
    # No explicit --type-map -> bare `pair_coeff * *` (model order trusted).
    assert "pair_coeff * *" in cmds


def test_build_commands_appends_type_map_to_pair_coeff():
    """DeepMD docs require an element list on pair_coeff when LAMMPS type
    order may not match the model's species order.  Passing --type-map
    'H C N O F P S Li' must produce `pair_coeff * * H C N O F P S Li`.
    Without this, the model silently uses the wrong species mapping.
    """
    cmds = build_commands(_make_args(type_map="H C N O F P S Li"))
    assert "pair_coeff * * H C N O F P S Li" in cmds
    assert "pair_coeff * *" not in cmds  # bare form must not also appear


def test_parser_accepts_type_map_arg():
    ap = build_parser()
    args = ap.parse_args([
        "--data", "s.data", "--mlip-model", "m.pb",
        "--type-map", "H O Li",
    ])
    assert args.type_map == "H O Li"


def test_build_commands_reads_user_data_file():
    cmds = build_commands(_make_args(data="my_system.data"))
    assert "read_data my_system.data" in cmds


def test_build_commands_includes_minimize_then_md():
    cmds = build_commands(_make_args())
    # minimize must come before NVT runs.
    minimize_idx = next(i for i, c in enumerate(cmds) if c.startswith("minimize"))
    fix_eq_idx = next(i for i, c in enumerate(cmds) if "fix nvt_eq" in c)
    fix_prod_idx = next(i for i, c in enumerate(cmds) if "fix nvt_prod" in c)
    assert minimize_idx < fix_eq_idx < fix_prod_idx


def test_build_commands_dumps_unwrapped():
    cmds = build_commands(_make_args(out_traj="t.lammpstrj", dump_every=42))
    dump = next(c for c in cmds if c.startswith("dump traj"))
    assert "custom 42 t.lammpstrj id type xu yu zu" in dump


def test_build_commands_uses_user_temp_and_step_counts():
    cmds = build_commands(_make_args(temp=350.0, equil_steps=5, prod_steps=99))
    assert any("fix nvt_eq all nvt temp 350.0 350.0" in c for c in cmds)
    assert "run 5" in cmds        # equilibration
    assert "run 99" in cmds       # production
