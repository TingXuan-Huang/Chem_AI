"""TDD tests for scripts/lammps_run.py (classical/traditional MD runner).

The LAMMPS Python module isn't installed in the test venv, so we test the
pure command-list builder rather than running LAMMPS itself.

We want a `build_commands(args) -> list[str]` function that mirrors the
MLIP runner's API: pure, side-effect-free, returns the exact LAMMPS
commands the runner will execute.
"""

from __future__ import annotations

import argparse

import pytest

from lammps_run import build_commands, build_parser


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------

def test_parser_requires_data():
    ap = build_parser()
    with pytest.raises(SystemExit):
        ap.parse_args([])
    args = ap.parse_args(["--data", "sys.data"])
    assert args.data == "sys.data"
    # Defaults appropriate for classical FF (real units, fs timestep).
    assert args.dt_fs == 1.0
    assert args.equil_steps > 0
    assert args.prod_steps > 0


# ---------------------------------------------------------------------------
# command-list builder
# ---------------------------------------------------------------------------

def _make_args(**overrides) -> argparse.Namespace:
    base = dict(
        data="sys.data",
        pre_include=[], post_include=[],
        out_traj="prod.lammpstrj", out_data="final.data",
        temp=300.0, pressure=1.0, dt_fs=1.0,
        equil_steps=1000, npt_steps=0, prod_steps=10_000,
        dump_every=100, thermo=500, seed=7,
        gpu=False, gpu_n=1,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_build_commands_uses_real_units_and_full_atom_style():
    cmds = build_commands(_make_args())
    assert cmds[0] == "units real"
    assert "atom_style full" in cmds


def test_build_commands_inserts_pre_then_data_then_post_includes():
    cmds = build_commands(_make_args(
        pre_include=["ff/styles.in", "ff/kspace.in"],
        post_include=["ff/coeffs.in"],
        data="my.data",
    ))
    pre1 = cmds.index("include ff/styles.in")
    pre2 = cmds.index("include ff/kspace.in")
    data = cmds.index("read_data my.data")
    post = cmds.index("include ff/coeffs.in")
    assert pre1 < pre2 < data < post


def test_build_commands_gpu_off_omits_package_gpu():
    cmds = build_commands(_make_args(gpu=False, gpu_n=2))
    assert not any(c.startswith("package gpu") for c in cmds)
    assert "suffix gpu" not in cmds


def test_build_commands_gpu_on_emits_package_and_suffix():
    cmds = build_commands(_make_args(gpu=True, gpu_n=2))
    assert "package gpu 2 neigh yes" in cmds
    assert "suffix gpu" in cmds
    # Must precede read_data — LAMMPS requires `package` before atoms exist.
    assert cmds.index("package gpu 2 neigh yes") < cmds.index("read_data sys.data")


def test_build_commands_npt_steps_zero_skips_npt():
    cmds = build_commands(_make_args(npt_steps=0))
    assert not any("fix npt_eq" in c for c in cmds)


def test_build_commands_npt_steps_positive_inserts_npt_between_nvt_phases():
    cmds = build_commands(_make_args(equil_steps=10, npt_steps=20, prod_steps=30))
    nvt_eq = next(i for i, c in enumerate(cmds) if "fix nvt_eq" in c)
    npt = next(i for i, c in enumerate(cmds) if "fix npt_eq" in c)
    nvt_prod = next(i for i, c in enumerate(cmds) if "fix nvt_prod" in c)
    assert nvt_eq < npt < nvt_prod


def test_build_commands_dumps_unwrapped():
    cmds = build_commands(_make_args(out_traj="t.lammpstrj", dump_every=42))
    dump = next(c for c in cmds if c.startswith("dump traj"))
    assert "custom 42 t.lammpstrj id type xu yu zu" in dump


def test_build_commands_writes_final_data():
    cmds = build_commands(_make_args(out_data="end.data"))
    assert "write_data end.data" in cmds
