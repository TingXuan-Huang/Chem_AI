"""End-to-end MLIP pipeline driver per YAML config.

For each `configs/*.yaml`:
  1. build_system.py     -> system.data
  2. lammps_mlip_run.py  -> DPA-2 MD trajectory (production)
  3. ase_mlip_run.py     -> UMA MD trajectory   (cross-check)
  4. ion_transport_md_pipeline.py analyze on the DPA-2 trajectory.

Usage:
    python scripts/run_all_mlip.py --configs configs
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent


def build_steps(cfg: dict, py: str = sys.executable) -> list[tuple[str, list[str]]]:
    """Build the (label, command) pairs to run for one system.

    Pure function so tests can verify the wiring without running anything.
    """
    name = cfg.get("system", "system")
    build = cfg["build"]
    md = cfg["md"]
    dpa2 = cfg["mlip_dpa2"]
    uma = cfg["mlip_uma"]
    type_map = " ".join(e["element"] for e in build["type_mapping"])
    sels = cfg.get("selections", {})

    cfg_path = cfg.get("__path__", "")

    build_cmd = [
        py, str(HERE / "build_system.py"),
        "--config", cfg_path,
    ]

    dpa2_cmd = [
        py, str(HERE / "lammps_mlip_run.py"),
        "--data", build["output"]["data"],
        "--mlip-model", dpa2["model"],
        "--out-traj", dpa2["output"]["traj"],
        "--out-data", dpa2["output"]["data"],
        "--temp", str(md["temp"]),
        "--dt-ps", str(md["dt_ps"]),
        "--equil-steps", str(md["equil_steps"]),
        "--prod-steps", str(md["prod_steps_dpa2"]),
        "--dump-every", str(md["dump_every"]),
    ]

    uma_cmd = [
        py, str(HERE / "ase_mlip_run.py"),
        "--data", build["output"]["data"],
        "--type-map", type_map,
        "--uma-size", uma["size"],
        "--out-traj", uma["output"]["traj"],
        "--temp", str(md["temp"]),
        "--dt-fs", str(md["dt_fs"]),
        "--equil-steps", str(max(1, md["equil_steps"] // 10)),
        "--prod-steps", str(md["prod_steps_uma"]),
        "--dump-every", str(md["dump_every"]),
    ]

    analyze_cmd = [
        py, str(HERE / "ion_transport_md_pipeline.py"), "analyze",
        "--top", build["output"]["data"],
        "--traj", dpa2["output"]["traj"],
        "--out", cfg["out"],
        "--li", sels.get("li", ""),
    ]
    for cli, key in [("--o-cell", "o_cell"), ("--o-poly", "o_poly"),
                     ("--n-poly", "n_poly"), ("--anion", "anion")]:
        if sels.get(key):
            analyze_cmd += [cli, sels[key]]
    fit = cfg.get("fit", {})
    if "tmin_ps" in fit:
        analyze_cmd += ["--tmin-ps", str(fit["tmin_ps"])]
    if "tmax_ps" in fit:
        analyze_cmd += ["--tmax-ps", str(fit["tmax_ps"])]
    if cfg.get("residence_grid"):
        analyze_cmd += ["--residence-grid"]
    if cfg.get("unwrap"):
        analyze_cmd += ["--unwrap"]

    return [
        (f"{name} build",     build_cmd),
        (f"{name} DPA-2 MD",  dpa2_cmd),
        (f"{name} UMA MD",    uma_cmd),
        (f"{name} analyze",   analyze_cmd),
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description="End-to-end MLIP pipeline.")
    ap.add_argument("--configs", default="configs",
                    help="directory containing system *.yaml configs")
    ap.add_argument("--continue-on-error", action="store_true",
                    help="don't abort the batch if one stage fails")
    args = ap.parse_args()

    paths = sorted(Path(args.configs).glob("*.yaml"))
    if not paths:
        sys.exit(f"No configs in {args.configs}")

    failures = []
    for path in paths:
        cfg = yaml.safe_load(path.read_text())
        cfg["__path__"] = str(path)
        for label, cmd in build_steps(cfg):
            print(f"\n=== {label} ===\n$ {' '.join(cmd)}")
            rc = subprocess.run(cmd).returncode
            if rc != 0:
                failures.append((label, rc))
                if not args.continue_on_error:
                    sys.exit(f"{label} failed (rc={rc})")
                break  # skip remaining stages of this system

    if failures:
        print("\nFailures:")
        for label, rc in failures:
            print(f"  {label}: rc={rc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
