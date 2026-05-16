"""Run ion_transport_md_pipeline.py on every YAML config in --configs."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ANALYZER = HERE / "ion_transport_md_pipeline.py"


def build_cmd(cfg: dict, py: str = sys.executable) -> list[str]:
    sels = cfg.get("selections", {})
    cmd = [py, str(ANALYZER), "analyze",
           "--top", cfg["top"],
           "--traj", cfg["traj"],
           "--out", cfg["out"],
           "--li", sels["li"]]
    for cli, key in [("--o-cell", "o_cell"), ("--o-poly", "o_poly"),
                     ("--n-poly", "n_poly"), ("--anion", "anion")]:
        val = sels.get(key)
        if val:
            cmd += [cli, val]
    fit = cfg.get("fit", {})
    if "tmin_ps" in fit:
        cmd += ["--tmin-ps", str(fit["tmin_ps"])]
    if "tmax_ps" in fit:
        cmd += ["--tmax-ps", str(fit["tmax_ps"])]
    for cli, key in [("--cluster-cut-A", "cluster_cut_A"),
                     ("--disp-lag-ps", "disp_lag_ps"),
                     ("--rdf-rmax-A", "rdf_rmax_A")]:
        if key in cfg:
            cmd += [cli, str(cfg[key])]
    if cfg.get("residence_grid"):
        cmd += ["--residence-grid"]
    if cfg.get("unwrap"):
        cmd += ["--unwrap"]
    return cmd


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch-run analyzer over YAML configs.")
    ap.add_argument("--configs", default="configs",
                    help="directory containing *.yaml configs")
    ap.add_argument("--continue-on-error", action="store_true",
                    help="don't abort the batch if one system fails")
    args = ap.parse_args()

    paths = sorted(Path(args.configs).glob("*.yaml"))
    if not paths:
        sys.exit(f"No *.yaml configs in {args.configs}")

    failures = []
    for path in paths:
        cfg = yaml.safe_load(path.read_text())
        name = cfg.get("system", path.stem)
        print(f"\n=== {name} ({path.name}) ===")
        cmd = build_cmd(cfg)
        print("$ " + " ".join(cmd))
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            failures.append((name, rc))
            if not args.continue_on_error:
                sys.exit(f"{name} failed with exit code {rc}")

    if failures:
        print("\nFailures:")
        for name, rc in failures:
            print(f"  {name}: rc={rc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
