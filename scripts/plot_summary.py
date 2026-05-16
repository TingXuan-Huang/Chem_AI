"""Aggregate per-system results into summary.csv + summary_plot.png."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def collect(results_dir: Path) -> pd.DataFrame:
    rows = []
    for sub in sorted(results_dir.iterdir()):
        if not sub.is_dir():
            continue
        diff_path = sub / "diffusion_result.json"
        if not diff_path.exists():
            continue
        diff = json.loads(diff_path.read_text())
        row = {"system": sub.name, **diff}

        cn_path = sub / "rdf_coordination_summary.csv"
        if cn_path.exists():
            cn = pd.read_csv(cn_path)
            for _, r in cn.iterrows():
                row[f"CN_{r['pair']}"] = r["CN_first_shell"]
                row[f"r_first_peak_{r['pair']}"] = r["r_first_peak_A"]

        assoc_path = sub / "ion_association_fractions.json"
        if assoc_path.exists():
            row.update(json.loads(assoc_path.read_text()))

        rows.append(row)
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].bar(df["system"], df["D_cm2_per_s"], color="C0")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("D_Li (cm²/s)")
    axes[0].set_title("Li⁺ diffusion coefficient")
    for i, v in enumerate(df["D_cm2_per_s"]):
        axes[0].text(i, v, f"{v:.1e}", ha="center", va="bottom", fontsize=8)

    cn_cols = [c for c in df.columns if c.startswith("CN_")]
    if cn_cols:
        x = list(range(len(df["system"])))
        width = 0.8 / max(1, len(cn_cols))
        for i, c in enumerate(cn_cols):
            offset = i * width
            axes[1].bar([xi + offset for xi in x], df[c].fillna(0.0),
                        width, label=c.replace("CN_", ""))
        axes[1].set_xticks([xi + width * (len(cn_cols) - 1) / 2 for xi in x])
        axes[1].set_xticklabels(df["system"])
        axes[1].set_ylabel("CN (first shell)")
        axes[1].set_title("Coordination numbers")
        axes[1].legend(fontsize=8)
    else:
        axes[1].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarise analyzer outputs across systems.")
    ap.add_argument("--results", default="results",
                    help="root directory containing one folder per system")
    ap.add_argument("--out", default="results/summary.csv")
    ap.add_argument("--plot", default="results/summary_plot.png")
    args = ap.parse_args()

    results_dir = Path(args.results)
    df = collect(results_dir)
    if df.empty:
        print(f"No diffusion_result.json files found under {results_dir}.")
        return

    out_csv = Path(args.out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    plot(df, Path(args.plot))
    print(f"Wrote {out_csv}  ({len(df)} systems)")
    print(f"Wrote {args.plot}")


if __name__ == "__main__":
    main()
