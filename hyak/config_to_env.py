#!/usr/bin/env python3
"""Print shell `export` lines for lammps_mlip_run.py from a system YAML.

Usage:
    eval "$(python3 hyak/config_to_env.py configs/dac.yaml)"
    eval "$(python3 hyak/config_to_env.py configs/dac.yaml smoke)"
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("usage: config_to_env.py <configs/foo.yaml> [smoke]")

    cfg_path = Path(sys.argv[1])
    smoke = len(sys.argv) > 2 and sys.argv[2] == "smoke"

    cfg = yaml.safe_load(cfg_path.read_text())
    build = cfg["build"]
    dpa2 = cfg["mlip_dpa2"]
    md = cfg["md"]

    equil = 500 if smoke else md["equil_steps"]
    prod = 2_000 if smoke else md["prod_steps_dpa2"]

    exports = {
        "SYSTEM": cfg.get("system", cfg_path.stem),
        "CONFIG_FILE": str(cfg_path),
        "DATA_FILE": build["output"]["data"],
        "MLIP_MODEL": dpa2["model"],
        "OUT_TRAJ": dpa2["output"]["traj"],
        "OUT_DATA": dpa2["output"]["data"],
        "TEMP": str(md["temp"]),
        "DT_PS": str(md["dt_ps"]),
        "EQUIL_STEPS": str(equil),
        "PROD_STEPS": str(prod),
        "DUMP_EVERY": str(md["dump_every"]),
    }
    for key, val in exports.items():
        # Escape double quotes in paths
        safe = str(val).replace('"', '\\"')
        print(f'export {key}="{safe}"')


if __name__ == "__main__":
    main()
