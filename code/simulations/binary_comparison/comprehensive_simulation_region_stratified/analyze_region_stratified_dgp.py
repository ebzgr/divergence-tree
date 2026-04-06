"""
DGP diagnostics for region-stratified simulation.
"""

import os
from pathlib import Path
import time
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _repo_root() -> Path:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return next(p for p in Path(script_dir).resolve().parents if (p / "pyproject.toml").exists())


def _save_hist(series: pd.Series, title: str, out_file: str, log_x: bool = False) -> None:
    vals = series.dropna().values
    if len(vals) == 0:
        return
    plt.figure(figsize=(8, 5))
    bins = 30 if len(vals) > 100 else 15
    plt.hist(vals, bins=bins, alpha=0.85, edgecolor="black")
    if log_x:
        plt.xscale("log")
    plt.title(title)
    plt.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()


def _save_treated_vs_untreated_outcome_hist(
    merged: pd.DataFrame,
    out_file: str,
    *,
    title: str = "Outcome distribution by treatment status (default DGP sample)",
) -> None:
    if not {"T", "YF", "YC"}.issubset(set(merged.columns)):
        return
    treated = merged.loc[merged["T"] == 1, "YF"].dropna().astype(float).values
    control = merged.loc[merged["T"] == 0, "YC"].dropna().astype(float).values
    if len(treated) == 0 or len(control) == 0:
        return

    plt.figure(figsize=(9, 5))
    bins = 40
    plt.hist(control, bins=bins, density=True, alpha=0.55, edgecolor="black", label=f"Untreated (n={len(control)})")
    plt.hist(treated, bins=bins, density=True, alpha=0.55, edgecolor="black", label=f"Treated (n={len(treated)})")
    plt.title(title)
    plt.xlabel("Observed outcome (Y)")
    plt.ylabel("Density")
    plt.legend(loc="best")
    plt.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()


def _append_with_cap(
    kept: Optional[np.ndarray],
    new: np.ndarray,
    *,
    cap: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Append new samples while keeping at most cap values (simple downsampling).

    This avoids reading thousands of simulations and holding all outcomes in memory.
    """
    if kept is None or kept.size == 0:
        kept = new
    else:
        kept = np.concatenate([kept, new], axis=0)
    if kept.size <= cap:
        return kept
    idx = rng.choice(kept.size, size=cap, replace=False)
    return kept[idx]


def analyze_dgp(
    results_file: str,
    output_dir: str,
    max_per_sim_plots: int = 200,
    *,
    verbose: bool = True,
    per_sim_progress_every: int = 250,
    outcome_hist_first_n_sims: int = 1000,
    outcome_hist_cap_per_group: int = 1_000_000,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_pickle(results_file)

    # Global aspect distributions
    aspect_cols = ["noise", "data_size", "sparsity", "dispersion", "rareness"]
    for col in aspect_cols:
        if col in df.columns:
            _save_hist(
                df[col],
                title=f"Distribution of {col}",
                out_file=os.path.join(output_dir, f"aspect_distribution_{col}.png"),
                log_x=(col in ("noise", "data_size")),
            )

    # Realized region distributions over simulations
    for prefix in ["dispersion_real_region", "rareness_real_region"]:
        for r in [1, 2, 3, 4]:
            col = f"{prefix}{r}"
            if col in df.columns:
                _save_hist(
                    df[col],
                    title=f"Distribution of {col}",
                    out_file=os.path.join(output_dir, f"{col}.png"),
                    log_x=False,
                )

    # Target vs realized scatter for region 2
    pairs = [
        ("dispersion_target_region2", "dispersion_real_region2"),
        ("rareness_target_region2", "rareness_real_region2"),
    ]
    for x_col, y_col in pairs:
        if x_col not in df.columns or y_col not in df.columns:
            continue
        d = df[[x_col, y_col]].dropna()
        if len(d) == 0:
            continue
        plt.figure(figsize=(6, 6))
        if len(d) > max_per_sim_plots:
            d = d.sample(max_per_sim_plots, random_state=0)
        plt.scatter(d[x_col], d[y_col], alpha=0.5, s=15)
        lo = min(d[x_col].min(), d[y_col].min())
        hi = max(d[x_col].max(), d[y_col].max())
        plt.plot([lo, hi], [lo, hi], linestyle="--", color="black")
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(f"{y_col} vs {x_col}")
        plt.grid(alpha=0.3, linestyle="--")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{y_col}_vs_{x_col}.png"), dpi=300, bbox_inches="tight")
        plt.close()

    # Save numeric summaries
    summary_cols = [c for c in df.columns if c in aspect_cols or c.startswith("dispersion_") or c.startswith("rareness_")]
    if summary_cols:
        desc = df[summary_cols].describe(include="all").T
        desc.to_csv(os.path.join(output_dir, "dgp_summary_statistics.csv"))

    # Outcome histogram using stored train/val/test files from a subset of simulations.
    base_output_dir = os.path.dirname(os.path.dirname(os.path.dirname(results_file)))
    data_root = os.path.join(base_output_dir, "data", "region_stratified_lambda_twostep_comparison")
    if os.path.isdir(data_root):
        sim_ids = sorted(df["simulation_id"].dropna().astype(int).tolist())[: max(0, int(outcome_hist_first_n_sims))]
        if sim_ids:
            rng = np.random.default_rng(0)
            t0 = time.perf_counter()
            processed = 0
            found = 0
            missing = 0
            treated_keep: Optional[np.ndarray] = None
            control_keep: Optional[np.ndarray] = None
            for sim_id in sim_ids:
                processed += 1
                sim_dir = os.path.join(data_root, f"simulation_{sim_id:06d}")
                train_f = os.path.join(sim_dir, "train_data.pkl")
                val_f = os.path.join(sim_dir, "val_data.pkl")
                test_f = os.path.join(sim_dir, "test_data.pkl")
                if not (os.path.exists(train_f) and os.path.exists(val_f) and os.path.exists(test_f)):
                    missing += 1
                else:
                    found += 1
                    train_df = pd.read_pickle(train_f)
                    val_df = pd.read_pickle(val_f)
                    test_df = pd.read_pickle(test_f)
                    merged = pd.concat([train_df, val_df, test_df], axis=0, ignore_index=True)
                    if {"T", "YF", "YC"}.issubset(set(merged.columns)):
                        treated = merged.loc[merged["T"] == 1, "YF"].dropna().astype(float).values
                        control = merged.loc[merged["T"] == 0, "YC"].dropna().astype(float).values
                        if treated.size:
                            treated_keep = _append_with_cap(
                                treated_keep, treated, cap=outcome_hist_cap_per_group, rng=rng
                            )
                        if control.size:
                            control_keep = _append_with_cap(
                                control_keep, control, cap=outcome_hist_cap_per_group, rng=rng
                            )

                if verbose and per_sim_progress_every > 0 and (processed % per_sim_progress_every == 0):
                    dt = time.perf_counter() - t0
                    rate = processed / max(dt, 1e-9)
                    n_t = 0 if treated_keep is None else int(treated_keep.size)
                    n_c = 0 if control_keep is None else int(control_keep.size)
                    print(
                        f"[post-hoc DGP] outcomes: processed={processed}/{len(sim_ids)} found={found} missing={missing} "
                        f"treated_kept={n_t} control_kept={n_c} elapsed={dt:.1f}s ({rate:.1f} sims/s)",
                        flush=True,
                    )

            if verbose:
                dt = time.perf_counter() - t0
                n_t = 0 if treated_keep is None else int(treated_keep.size)
                n_c = 0 if control_keep is None else int(control_keep.size)
                print(
                    f"[post-hoc DGP] outcomes done. processed={processed} found={found} missing={missing} "
                    f"treated_kept={n_t} control_kept={n_c} elapsed={dt:.1f}s",
                    flush=True,
                )

            if treated_keep is not None and control_keep is not None and treated_keep.size and control_keep.size:
                out_png = os.path.join(
                    output_dir, f"outcome_hist_treated_vs_untreated_first_{len(sim_ids)}_sims.png"
                )
                plt.figure(figsize=(9, 5))
                bins = 50
                plt.hist(
                    control_keep,
                    bins=bins,
                    density=True,
                    alpha=0.55,
                    edgecolor="black",
                    label=f"Untreated (kept n={control_keep.size})",
                )
                plt.hist(
                    treated_keep,
                    bins=bins,
                    density=True,
                    alpha=0.55,
                    edgecolor="black",
                    label=f"Treated (kept n={treated_keep.size})",
                )
                plt.title(f"Outcome distribution by treatment status (first {len(sim_ids)} simulations)")
                plt.xlabel("Observed outcome (Y)")
                plt.ylabel("Density")
                plt.legend(loc="best")
                plt.grid(alpha=0.3, linestyle="--")
                plt.tight_layout()
                plt.savefig(out_png, dpi=300, bbox_inches="tight")
                plt.close()

                out_csv = os.path.join(
                    output_dir, f"outcome_hist_treated_vs_untreated_first_{len(sim_ids)}_sims_summary.csv"
                )
                pd.DataFrame(
                    [
                        {
                            "group": "untreated",
                            "kept_n": int(control_keep.size),
                            "mean": float(np.mean(control_keep)),
                            "std": float(np.std(control_keep)),
                        },
                        {
                            "group": "treated",
                            "kept_n": int(treated_keep.size),
                            "mean": float(np.mean(treated_keep)),
                            "std": float(np.std(treated_keep)),
                        },
                    ]
                ).to_csv(out_csv, index=False)


if __name__ == "__main__":
    repo_root = _repo_root()
    base_dir = str(repo_root / "outputs" / "simulations" / "comprehensive_simulation_region_stratified")
    results_file = os.path.join(base_dir, "aggregated", "region_stratified_lambda_twostep_comparison", "all_simulations_results.pkl")
    output_dir = os.path.join(base_dir, "aggregated", "region_stratified_lambda_twostep_comparison", "analysis", "dgp_diagnostics")
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Results file not found: {results_file}")
    analyze_dgp(results_file, output_dir, verbose=True)

