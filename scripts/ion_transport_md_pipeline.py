"""Ion-Transport MD Pipeline (Layer B): compute Li+ transport descriptors.

Reads any MDAnalysis-readable trajectory (LAMMPS data + lammpstrj, GROMACS
xtc/tpr, PDB+DCD, ...) and writes the outputs described in spec section 8.

Usage:
    python ion_transport_md_pipeline.py analyze \
        --top SYSTEM --traj TRAJ --out OUTDIR --li "..." [--o-cell ...] ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import MDAnalysis as mda
from MDAnalysis import transformations
from MDAnalysis.analysis import msd as mda_msd
from MDAnalysis.analysis import rdf as mda_rdf
from MDAnalysis.lib.distances import distance_array

from scipy.stats import linregress


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_universe(top: str, traj: str, unwrap: bool = False) -> mda.Universe:
    u = mda.Universe(top, traj)
    if unwrap:
        # NoJump removes periodic-image jumps so MSD sees true displacement.
        u.trajectory.add_transformations(transformations.NoJump())
    return u


# ---------------------------------------------------------------------------
# Step 4-5: MSD + diffusion fit
# ---------------------------------------------------------------------------

def compute_msd_and_diffusion(
    u: mda.Universe,
    li_sel: str,
    fit_tmin_ps: float,
    fit_tmax_ps: float,
    out_dir: Path,
) -> dict:
    li = u.select_atoms(li_sel)
    if li.n_atoms == 0:
        raise ValueError(f"Li selection '{li_sel}' matched zero atoms.")

    msd_calc = mda_msd.EinsteinMSD(u, select=li_sel, msd_type="xyz", fft=True)
    msd_calc.run()

    dt_ps = float(u.trajectory.dt)
    msd_A2 = np.asarray(msd_calc.results.timeseries)
    times_ps = np.arange(len(msd_A2)) * dt_ps

    df = pd.DataFrame({
        "lag_frame": np.arange(len(times_ps)),
        "lag_time_ps": times_ps,
        "MSD_A2": msd_A2,
        "MSD_nm2": msd_A2 * 1e-2,
    })
    df.to_csv(out_dir / "li_msd.csv", index=False)

    mask = (times_ps >= fit_tmin_ps) & (times_ps <= fit_tmax_ps)
    if mask.sum() < 4:
        # Trajectory shorter than the chosen window; fall back to last 50 %.
        mask = np.zeros_like(times_ps, dtype=bool)
        mask[len(times_ps) // 2:] = True

    res = linregress(times_ps[mask], msd_A2[mask])
    slope = float(res.slope)             # Å^2 / ps
    D_A2_per_ps = slope / 6.0            # 3D Einstein relation
    D_cm2_per_s = D_A2_per_ps * 1e-4
    D_nm2_per_ps = D_A2_per_ps * 1e-2

    diff = {
        "slope_A2_per_ps": slope,
        "intercept_A2": float(res.intercept),
        "D_A2_per_ps": D_A2_per_ps,
        "D_cm2_per_s": D_cm2_per_s,
        "D_nm2_per_ps": D_nm2_per_ps,
        "fit_tmin_ps": float(times_ps[mask][0]),
        "fit_tmax_ps": float(times_ps[mask][-1]),
        "r_value": float(res.rvalue),
        "n_li": int(li.n_atoms),
        "n_frames": int(len(times_ps)),
        "dt_ps": dt_ps,
    }
    (out_dir / "diffusion_result.json").write_text(json.dumps(diff, indent=2))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(times_ps, msd_A2, label="MSD")
    fit_x = times_ps[mask]
    ax.plot(fit_x, slope * fit_x + res.intercept, "r--",
            label=f"linear fit\nD={D_cm2_per_s:.2e} cm²/s\nr={res.rvalue:.3f}")
    ax.axvspan(fit_x[0], fit_x[-1], color="red", alpha=0.08)
    ax.set_xlabel("time (ps)")
    ax.set_ylabel("MSD (Å²)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "li_msd_fit.png", dpi=150)
    plt.close(fig)

    return diff


# ---------------------------------------------------------------------------
# Step 6: displacement-distance distribution at fixed lag
# ---------------------------------------------------------------------------

def compute_displacement_distribution(
    u: mda.Universe,
    li_sel: str,
    lag_ps: float,
    out_dir: Path,
    n_bins: int = 60,
) -> None:
    li = u.select_atoms(li_sel)
    dt_ps = float(u.trajectory.dt)
    lag_frames = max(1, int(round(lag_ps / dt_ps)))

    n_frames = len(u.trajectory)
    if n_frames <= lag_frames:
        return  # not enough frames to evaluate this lag

    # Pre-load positions once (memory ~ 12 * T * N_Li bytes).
    positions = np.empty((n_frames, li.n_atoms, 3), dtype=np.float32)
    for i, _ in enumerate(u.trajectory):
        positions[i] = li.positions

    # Sliding origins.
    diffs = positions[lag_frames:] - positions[:-lag_frames]
    disps = np.linalg.norm(diffs, axis=2).ravel()

    upper = max(float(disps.max()), 1.0)
    edges = np.linspace(0.0, upper, n_bins + 1)
    hist, _ = np.histogram(disps, bins=edges, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])

    pd.DataFrame({"distance_A": centers, "p": hist}).to_csv(
        out_dir / "li_displacement_distance.csv", index=False
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(centers, hist)
    ax.set_xlabel("|Δr| (Å)")
    ax.set_ylabel("p(|Δr|)")
    ax.set_title(f"Li⁺ displacement at lag = {lag_ps:.1f} ps "
                 f"(mean={disps.mean():.2f} Å, median={np.median(disps):.2f} Å)")
    fig.tight_layout()
    fig.savefig(out_dir / "li_displacement_distance.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Step 7-8: RDF + running coordination number + first-shell CN
# ---------------------------------------------------------------------------

def compute_rdf(
    u: mda.Universe,
    sel_a: str,
    sel_b: str,
    label: str,
    out_dir: Path,
    r_max: float = 10.0,
    n_bins: int = 200,
):
    a = u.select_atoms(sel_a)
    b = u.select_atoms(sel_b)
    if a.n_atoms == 0 or b.n_atoms == 0:
        return None

    rdf_calc = mda_rdf.InterRDF(a, b, nbins=n_bins, range=(0.05, r_max))
    rdf_calc.run()
    rs = np.asarray(rdf_calc.results.bins)
    g = np.asarray(rdf_calc.results.rdf)

    # Running CN: integrate 4*pi*r^2 * rho_b * g(r) dr.
    box = u.dimensions[:3]
    vol = float(np.prod(box))
    rho_b = b.n_atoms / vol if vol > 0 else 0.0
    dr = float(rs[1] - rs[0])
    cn_running = np.cumsum(4.0 * np.pi * rs ** 2 * g * rho_b * dr)

    pd.DataFrame({"r_A": rs, "g_r": g, "CN_running": cn_running}).to_csv(
        out_dir / f"rdf_{label}.csv", index=False
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(rs, g, "C0-", label="g(r)")
    ax.set_xlabel("r (Å)")
    ax.set_ylabel("g(r)")
    ax2 = ax.twinx()
    ax2.plot(rs, cn_running, "C3-", alpha=0.6, label="CN(r)")
    ax2.set_ylabel("CN(r)")
    ax.set_title(label)
    fig.tight_layout()
    fig.savefig(out_dir / f"rdf_{label}.png", dpi=150)
    plt.close(fig)

    return rs, g, cn_running


def coordination_number_at_first_min(rs, g, cn):
    """Return (first_peak_r, first_min_r, CN at first min) or all-None."""
    mask = rs > 1.0
    if not mask.any():
        return None, None, None
    peak_rel = int(np.argmax(g[mask]))
    peak_idx = int(mask.nonzero()[0][peak_rel])
    if peak_idx >= len(g) - 1:
        return float(rs[peak_idx]), None, None
    after = g[peak_idx:]
    min_rel = int(np.argmin(after))
    min_idx = peak_idx + min_rel
    return float(rs[peak_idx]), float(rs[min_idx]), float(cn[min_idx])


# ---------------------------------------------------------------------------
# Step 9: ion-cluster probability matrix + free-Li / CIP / aggregate fractions
# ---------------------------------------------------------------------------

def compute_cluster_stats(
    u: mda.Universe,
    li_sel: str,
    anion_sel: str,
    cutoff: float,
    out_dir: Path,
    n_bin: int = 6,
):
    li = u.select_atoms(li_sel)
    an = u.select_atoms(anion_sel)
    if li.n_atoms == 0:
        return None

    cells = np.zeros((n_bin, n_bin), dtype=np.float64)
    for ts in u.trajectory:
        d_ll = distance_array(li.positions, li.positions, box=ts.dimensions)
        np.fill_diagonal(d_ll, np.inf)
        n_li_neigh = (d_ll < cutoff).sum(axis=1)

        if an.n_atoms > 0:
            d_la = distance_array(li.positions, an.positions, box=ts.dimensions)
            n_an_neigh = (d_la < cutoff).sum(axis=1)
        else:
            n_an_neigh = np.zeros(li.n_atoms, dtype=int)

        for nl, na in zip(n_li_neigh, n_an_neigh):
            cells[min(int(nl), n_bin - 1), min(int(na), n_bin - 1)] += 1.0

    total = cells.sum()
    if total > 0:
        cells /= total

    df = pd.DataFrame(cells, columns=[f"n_anion_eq_{j}" for j in range(n_bin)])
    df.insert(0, "n_li_neighbors", [f"n_li_eq_{i}" for i in range(n_bin)])
    df.to_csv(out_dir / "ion_cluster_probability.csv", index=False)

    free_li = float(cells[:, 0].sum())
    cip = float(cells[:, 1].sum())
    agg = float(cells[:, 2:].sum())
    payload = {
        "free_li": free_li,
        "contact_ion_pair": cip,
        "aggregate": agg,
        "cutoff_A": float(cutoff),
    }
    (out_dir / "ion_association_fractions.json").write_text(
        json.dumps(payload, indent=2)
    )
    return payload


# ---------------------------------------------------------------------------
# Step 10: 3D Li+ residence probability grid
# ---------------------------------------------------------------------------

def compute_residence_grid(
    u: mda.Universe,
    li_sel: str,
    out_dir: Path,
    n_bins: int = 80,
) -> None:
    li = u.select_atoms(li_sel)
    if li.n_atoms == 0:
        return
    box = u.trajectory[0].dimensions[:3].astype(float)
    edges = [np.linspace(0.0, box[k], n_bins + 1) for k in range(3)]

    grid = np.zeros((n_bins, n_bins, n_bins), dtype=np.int64)
    for _ in u.trajectory:
        pos = np.mod(li.positions, box)  # wrap into the frame-0 box
        h, _ = np.histogramdd(pos, bins=edges)
        grid += h.astype(np.int64)

    total = grid.sum()
    p = grid / total if total > 0 else grid.astype(float)
    np.save(out_dir / "li_residence_probability_grid.npy", p)
    np.savez(out_dir / "li_residence_probability_grid_edges.npz",
             x=edges[0], y=edges[1], z=edges[2])


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def analyze(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    u = load_universe(args.top, args.traj, unwrap=args.unwrap)
    n_frames = len(u.trajectory)
    dt_ps = float(u.trajectory.dt)
    total_ps = (n_frames - 1) * dt_ps if n_frames > 1 else 0.0

    tmin = args.tmin_ps if args.tmin_ps is not None else 0.25 * total_ps
    tmax = args.tmax_ps if args.tmax_ps is not None else 0.75 * total_ps

    print(f"[loaded] n_frames={n_frames}  dt={dt_ps:.3f} ps  "
          f"total={total_ps:.1f} ps  fit_window=[{tmin:.1f}, {tmax:.1f}] ps")

    diff = compute_msd_and_diffusion(u, args.li, tmin, tmax, out)
    print(f"[D_Li]   {diff['D_cm2_per_s']:.3e} cm^2/s   "
          f"(r={diff['r_value']:.3f}, N_Li={diff['n_li']})")

    compute_displacement_distribution(u, args.li, args.disp_lag_ps, out)

    summary_rows = []
    for label, sel_b in [
        ("Li-O_cellulose", args.o_cell),
        ("Li-O_PDOL", args.o_poly),
        ("Li-N_polyamine", args.n_poly),
        ("Li-anion_center", args.anion),
    ]:
        if not sel_b:
            continue
        out_rdf = compute_rdf(u, args.li, sel_b, label, out,
                              r_max=args.rdf_rmax_A, n_bins=args.rdf_nbins)
        if out_rdf is None:
            continue
        rs, g, cn = out_rdf
        peak, first_min, cn_first = coordination_number_at_first_min(rs, g, cn)
        summary_rows.append({
            "pair": label,
            "r_first_peak_A": peak,
            "r_first_min_A": first_min,
            "CN_first_shell": cn_first,
        })
    pd.DataFrame(summary_rows).to_csv(
        out / "rdf_coordination_summary.csv", index=False
    )

    if args.anion:
        assoc = compute_cluster_stats(u, args.li, args.anion,
                                      args.cluster_cut_A, out)
        if assoc is not None:
            print(f"[assoc]  free={assoc['free_li']:.3f}  "
                  f"CIP={assoc['contact_ion_pair']:.3f}  "
                  f"agg={assoc['aggregate']:.3f}")

    if args.residence_grid:
        compute_residence_grid(u, args.li, out, n_bins=args.residence_bins)
        print(f"[grid]   {args.residence_bins}^3 written")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Trajectory analysis for Li+ transport in polymer electrolytes."
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze", help="run analyses on one trajectory")
    a.add_argument("--top", required=True, help="topology / structure file")
    a.add_argument("--traj", required=True, help="trajectory file")
    a.add_argument("--out", required=True, help="output directory")
    a.add_argument("--li", required=True, help="MDAnalysis selection for Li+")
    a.add_argument("--o-cell", default=None,
                   help="MDAnalysis selection for cellulose/DAC oxygens")
    a.add_argument("--o-poly", default=None,
                   help="MDAnalysis selection for PDOL polymer oxygens")
    a.add_argument("--n-poly", default=None,
                   help="MDAnalysis selection for polyamine nitrogens")
    a.add_argument("--anion", default=None,
                   help="MDAnalysis selection for anion centers (P / N / S)")
    a.add_argument("--tmin-ps", type=float, default=None,
                   help="MSD fit lower bound in ps (default: 25%% of total)")
    a.add_argument("--tmax-ps", type=float, default=None,
                   help="MSD fit upper bound in ps (default: 75%% of total)")
    a.add_argument("--cluster-cut-A", type=float, default=4.0,
                   help="cutoff (Å) for cluster / association statistics")
    a.add_argument("--disp-lag-ps", type=float, default=100.0,
                   help="lag time (ps) for displacement-distance histogram")
    a.add_argument("--rdf-rmax-A", type=float, default=10.0,
                   help="upper bound (Å) for RDFs")
    a.add_argument("--rdf-nbins", type=int, default=200, help="RDF bins")
    a.add_argument("--residence-grid", action="store_true",
                   help="also build the 3D Li residence probability grid")
    a.add_argument("--residence-bins", type=int, default=80,
                   help="bins per axis for the residence grid")
    a.add_argument("--unwrap", action="store_true",
                   help="apply NoJump unwrapping (use for wrapped trajectories)")
    return ap


def main() -> None:
    args = build_parser().parse_args()
    if args.cmd == "analyze":
        analyze(args)


if __name__ == "__main__":
    main()
