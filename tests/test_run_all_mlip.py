"""TDD tests for scripts/run_all_mlip.py.

We want a `build_steps(cfg, py) -> list[tuple[str, list[str]]]` function
that, given a per-system config dict, returns the (label, command) tuples
to run sequentially: build -> DPA-2 MD -> UMA MD -> analyze.
"""

from __future__ import annotations

import pytest

from run_all_mlip import build_steps


def _sample_cfg():
    return {
        "system": "Test",
        "__path__": "configs/test.yaml",
        "build": {
            "output": {"data": "out/sys.data"},
            "type_mapping": [
                {"type": 1, "element": "H"},
                {"type": 2, "element": "Li"},
            ],
        },
        "md": {
            "temp": 300.0,
            "dt_ps": 0.0005, "dt_fs": 0.5,
            "equil_steps": 1000,
            "prod_steps_dpa2": 5000, "prod_steps_uma": 500,
            "dump_every": 50,
        },
        "mlip_dpa2": {
            "model": "models/dpa2.pb",
            "output": {"traj": "out/dpa2.lammpstrj", "data": "out/final.data"},
        },
        "mlip_uma": {
            "size": "small",
            "output": {"traj": "out/uma.dcd"},
        },
        "top": "out/sys.data",
        "traj": "out/dpa2.lammpstrj",
        "out": "results/Test",
        "selections": {"li": "type 2"},
        "fit": {"tmin_ps": 1.0, "tmax_ps": 5.0},
    }


def test_build_steps_returns_four_named_stages():
    steps = build_steps(_sample_cfg(), py="python3")
    labels = [name for name, _ in steps]
    assert len(steps) == 4
    assert labels == [
        "Test build",
        "Test DPA-2 MD",
        "Test UMA MD",
        "Test analyze",
    ]


def test_build_steps_build_command_uses_config_path():
    steps = build_steps(_sample_cfg(), py="python3")
    _, cmd = steps[0]
    assert cmd[0] == "python3"
    assert cmd[1].endswith("build_system.py")
    assert "--config" in cmd
    assert "configs/test.yaml" in cmd


def test_build_steps_dpa2_command_uses_model_and_out_paths():
    steps = build_steps(_sample_cfg(), py="python3")
    _, cmd = steps[1]
    assert cmd[1].endswith("lammps_mlip_run.py")
    assert "--data" in cmd and "out/sys.data" in cmd
    assert "--mlip-model" in cmd and "models/dpa2.pb" in cmd
    assert "--out-traj" in cmd and "out/dpa2.lammpstrj" in cmd
    assert "--prod-steps" in cmd and "5000" in cmd


def test_build_steps_uma_command_passes_type_map():
    steps = build_steps(_sample_cfg(), py="python3")
    _, cmd = steps[2]
    assert cmd[1].endswith("ase_mlip_run.py")
    type_map_idx = cmd.index("--type-map")
    assert cmd[type_map_idx + 1] == "H Li"
    assert "--uma-size" in cmd and "small" in cmd
    assert "--out-traj" in cmd and "out/uma.dcd" in cmd


def test_build_steps_analyze_uses_dpa2_traj():
    steps = build_steps(_sample_cfg(), py="python3")
    _, cmd = steps[3]
    assert cmd[1].endswith("ion_transport_md_pipeline.py")
    assert "analyze" in cmd
    assert "--top" in cmd and "out/sys.data" in cmd
    assert "--traj" in cmd and "out/dpa2.lammpstrj" in cmd
    assert "--li" in cmd and "type 2" in cmd
    assert "--tmin-ps" in cmd
    assert "--tmax-ps" in cmd
