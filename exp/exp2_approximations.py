"""
Experiment 2: Numerical validation of leading-order approximations.

Compares exact numerical solutions from exp_1 with leading-order closed-form
approximations for optimal width, free-entry margin, exclusion margin, and
capital viability.

Outputs: results/exp2_*.png and results/exp2_approximation_errors.csv
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from src.models import (
    H_03, H_005, B1_coeff, sigma_c as compute_sigma_c, optimal_width,
)
from src.approximations import (
    w0_star, s_star, s_excl as compute_s_excl,
    g_max, c_min, approximation_errors,
)

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)

# ── Base parameters ───────────────────────────────────────────────────────────
H      = H_03
F      = 0.003
V_BAR  = 2e7
L_BAR  = 8e8
Q_BAR  = 2000.0
B1     = B1_coeff(F, V_BAR, Q_BAR, H, L_BAR)
SC     = compute_sigma_c(B1, H)

G_GRID         = np.linspace(0.0001, 0.005, 50)          # 1–50 bp
KAPPA_GRID     = [0.00175, 0.0005, 0.000476, 0.0003]
S_GRID         = np.linspace(0.50, 0.999, 60)            # sigma/sigma_c grid
G_BASE         = 0.0005                                    # 5 bp base case
G0_USD         = 20.0                                      # $20 fixed cost
G0_MODEL       = G0_USD / Q_BAR                           # model units
W_MAX          = 200


def section_base_case() -> dict:
    """Base-case analysis: g = 5 bp."""
    print("\n── Base case: g = 5 bp ──────────────────────────────────")
    x_star = (12.0 * G_BASE / 7.0) ** (1.0 / 3.0)
    ss     = s_star(G_BASE)
    print(f"  x* = {x_star:.6f}  (paper ~0.095)")
    print(f"  s* = {ss:.6f}  (paper ~0.992)")
    assert abs(x_star - 0.095) < 0.015, f"x* deviation: {x_star:.4f}"
    assert abs(ss    - 0.992) < 0.015, f"s* deviation: {ss:.4f}"

    # Exact w* vs w0* across s-grid
    w_exact = []
    w_approx = []
    for s_val in S_GRID:
        sigma_val = s_val * SC
        if sigma_val >= SC:
            continue
        opt   = optimal_width(F, V_BAR, L_BAR, Q_BAR, H, sigma_val, G_BASE, W_MAX)
        w_e   = opt["w_star"]
        w_a   = w0_star(G_BASE, sigma_val, H, SC)
        w_exact.append(w_e)
        w_approx.append(w_a)

    return {"s_grid": S_GRID[:len(w_exact)], "w_exact": w_exact,
            "w_approx": w_approx, "x_star": x_star, "s_star_val": ss}


def section_exclusion() -> pd.DataFrame:
    """Compute s_excl for each kappa and verify reference values."""
    print("\n── Exclusion thresholds ─────────────────────────────────")
    ref_map = {0.00175: 0.9823, 0.0005: 0.9922, 0.000476: 0.9924, 0.0003: 0.9944}
    rows    = []
    for kp in KAPPA_GRID:
        se  = compute_s_excl(kp)
        ref = ref_map.get(kp, np.nan)
        err = abs(se - ref) if not np.isnan(ref) else np.nan
        rows.append({"kappa": kp, "s_excl": se, "ref_paper": ref, "error": err})
        print(f"  κ={kp:.5f}  s_excl={se:.6f}  ref={ref:.4f}  err={err:.4f}")
    return pd.DataFrame(rows)


def section_capital_viability() -> pd.DataFrame:
    """c_min(s) for each kappa."""
    print("\n── Capital viability c_min(s) ───────────────────────────")
    s_vals = np.linspace(0.70, 0.999, 60)
    rows   = []
    for kp in KAPPA_GRID:
        for s_val in s_vals:
            gm   = g_max(s_val)
            cm   = c_min(s_val, G0_MODEL, kp)
            rows.append({"kappa": kp, "s": s_val,
                         "g_max": gm, "c_min": cm})
    return pd.DataFrame(rows)


def section_error_table() -> pd.DataFrame:
    """Compute |w* - w0*| and |s_exact - s_star| over parameter grids."""
    print("\n── Approximation error table (may take a moment) ────────")
    g_sample  = np.linspace(0.0001, 0.002, 10)
    s_sample  = np.linspace(0.60, 0.98, 15)
    kp_sample = [0.0005]
    df = approximation_errors(
        B1=B1, h=H, f=F, V_bar=V_BAR, L_bar=L_BAR, q_bar=Q_BAR,
        g_grid=g_sample, kappa_grid=kp_sample,
        sigma_sigma_c_grid=s_sample, W_max=W_MAX,
    )
    print(f"  Rows computed: {len(df)}")
    return df


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_width_comparison(base: dict) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(base["s_grid"], base["w_exact"],  "b-",  linewidth=2, label="w* (exact)")
    ax.plot(base["s_grid"], np.minimum(base["w_approx"], W_MAX),
            "r--", linewidth=1.5, label="w0* (leading-order)")
    ax.set_xlabel("σ / σ_c")
    ax.set_ylabel("Optimal half-width")
    ax.set_title(f"Exp 2: Exact vs approximate optimal width  (g={G_BASE*1e4:.0f} bp)")
    ax.legend()
    ax.set_ylim(0, W_MAX)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "exp2_width_comparison.png"), dpi=120)
    plt.close(fig)
    print("  Saved: exp2_width_comparison.png")


def plot_error_heatmap(df_err: pd.DataFrame) -> None:
    # Pivot: error_w over (g, s)
    pivot = df_err.pivot_table(index="s", columns="g", values="error_w", aggfunc="mean")
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    img = ax.imshow(pivot.values, aspect="auto", origin="lower",
                    cmap="YlOrRd",
                    extent=[pivot.columns.min(), pivot.columns.max(),
                            pivot.index.min(),   pivot.index.max()])
    plt.colorbar(img, ax=ax, label="|w* - w0*|")
    ax.set_xlabel("g (total cost)")
    ax.set_ylabel("σ / σ_c")
    ax.set_title("Exp 2: Width approximation error |w* - w0*|")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "exp2_error_heatmap.png"), dpi=120)
    plt.close(fig)
    print("  Saved: exp2_error_heatmap.png")


def plot_s_excl(df_excl: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    kappas  = df_excl["kappa"].values
    se_vals = df_excl["s_excl"].values
    ref_vals = df_excl["ref_paper"].values
    x = np.arange(len(kappas))
    ax.bar(x - 0.2, se_vals,  0.35, label="Computed s_excl",  color="steelblue")
    ax.bar(x + 0.2, ref_vals, 0.35, label="Paper reference",  color="orange", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"κ={kp:.5f}" for kp in kappas], rotation=15, fontsize=9)
    ax.set_ylabel("s_excl")
    ax.set_ylim(0.97, 1.0)
    ax.set_title("Exp 2: Exclusion thresholds vs paper reference values")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "exp2_s_excl.png"), dpi=120)
    plt.close(fig)
    print("  Saved: exp2_s_excl.png")


def plot_c_min(df_cap: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    colours = plt.cm.tab10(np.linspace(0, 0.4, len(KAPPA_GRID)))
    for kp, colour in zip(KAPPA_GRID, colours):
        subset = df_cap[df_cap["kappa"] == kp].copy()
        # Cap at a finite value for plotting
        cm_plot = np.minimum(subset["c_min"].replace(np.inf, np.nan), 1e6)
        ax.plot(subset["s"], cm_plot, label=f"κ={kp:.5f}", color=colour, linewidth=1.8)
    ax.set_xlabel("σ / σ_c")
    ax.set_ylabel("c_min (model units; g0=$20/q_bar)")
    ax.set_title("Exp 2: Minimum viable capital c_min(s)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "exp2_c_min.png"), dpi=120)
    plt.close(fig)
    print("  Saved: exp2_c_min.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("EXPERIMENT 2: Numerical validation of approximations")
    print("=" * 60)
    print(f"  h={H:.6f}  B1={B1:.6f}  σ_c={SC:.6f}")

    base      = section_base_case()
    df_excl   = section_exclusion()
    df_cap    = section_capital_viability()
    df_err    = section_error_table()

    # Save error table
    out_csv = os.path.join(RESULTS, "exp2_approximation_errors.csv")
    df_err.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}  ({len(df_err)} rows)")

    # Plots
    print("\nGenerating plots …")
    plot_width_comparison(base)
    plot_error_heatmap(df_err)
    plot_s_excl(df_excl)
    plot_c_min(df_cap)

    print("\nExperiment 2 complete.")


if __name__ == "__main__":
    main()
