"""
Experiment 5: Finite-state averaged mean-field game replication.

Numerically solves the coupled HJB-KFE system for the averaged re-centering game
and tests the paper's existence/uniqueness claims.

NOTE: This is NUMERICAL EVIDENCE supporting the paper's theoretical claims,
not a mathematical proof of existence or uniqueness.

Outputs: results/exp5_*.png, results/exp5_mfg_summary.csv
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

from src.mfg import MFGParams, MFGSolver
from src.models import H_03

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)

# ── Default parameters ─────────────────────────────────────────────────────────
DEFAULT_PARAMS = MFGParams(
    M        = 5,
    W_max    = 4,
    f        = 0.003,
    V_base   = 1_000_000.0,
    L_0      = 1e7,
    Gamma    = 1.0,
    g        = 0.0005,
    h        = H_03,
    kappa    = 0.0005,
    max_iter = 150,
    tol      = 1e-7,
    damping  = 0.4,
    q_ref    = 2000.0,
)


# ── Equilibrium existence and payoff-relevant uniqueness ──────────────────────

def test_existence_and_uniqueness(params: MFGParams, n_init: int = 4) -> dict:
    """
    Run solver from multiple initializations and assess uniqueness.

    Records: convergence status, value-function agreement, aggregate-liquidity agreement.
    """
    solver  = MFGSolver(params)
    results = solver.solve(n_init=n_init, seed=42)

    # Payoff-relevant uniqueness: compare aggregate liquidity L(0, mu) across inits
    L_vals  = []
    V_means = []
    for mu, V in zip(results["all_mu_finals"], results["all_V_finals"]):
        L_val = solver._aggregate_liquidity(0, mu)
        L_vals.append(L_val)
        V_means.append(float(np.mean(V)))

    L_arr   = np.array(L_vals)
    L_range = float(np.max(L_arr) - np.min(L_arr))

    # Full-law uniqueness: compare mu across inits
    mu_arr = np.array([r for r in results["all_mu_finals"]])
    mu_max_dev = float(np.max(np.max(np.abs(mu_arr - mu_arr[0:1]), axis=0)))

    print(f"\n── Existence & Uniqueness ───────────────────────────────")
    print(f"  n_inits: {n_init}")
    print(f"  Converged: {results['all_converged']}")
    print(f"  Aggregate L range: {L_range:.4e}  (small → payoff-relevant uniqueness)")
    print(f"  μ max deviation across inits: {mu_max_dev:.4e}")

    return {
        "converged":       results["all_converged"],
        "L_vals":          L_vals,
        "L_range":         L_range,
        "mu_max_dev":      mu_max_dev,
        "iterations":      results["iterations"],
        "residuals":       results["residuals"],
        "V_final":         results["V_final"],
        "mu_final":        results["mu_final"],
        "policy_final":    results["policy_final"],
    }


# ── Comparative statics: congestion via L_0 ──────────────────────────────────

def run_comparative_statics(params: MFGParams) -> pd.DataFrame:
    print("\n── Comparative statics: L_0 (congestion) ───────────────")
    L0_vals = [1e6, 5e6, 1e7, 5e7, 1e8]
    solver  = MFGSolver(params)
    df      = solver.comparative_statics("L_0", L0_vals)
    print(df[["L_0", "agg_L", "mean_fee", "converged"]].to_string(index=False))
    return df


def run_comparative_statics_f(params: MFGParams) -> pd.DataFrame:
    print("\n── Comparative statics: fee rate f ─────────────────────")
    f_vals = [0.0001, 0.0005, 0.001, 0.003, 0.005]
    solver  = MFGSolver(params)
    df      = solver.comparative_statics("f", f_vals)
    print(df[["f", "agg_L", "mean_fee", "converged"]].to_string(index=False))
    return df


# ── Convergence diagnostics ───────────────────────────────────────────────────

def plot_convergence(residuals: list[float]) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(residuals, "b-o", markersize=4)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Fixed-point residual (log scale)")
    ax.set_title("Exp 5: MFG fixed-point convergence")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "exp5_convergence.png"), dpi=120)
    plt.close(fig)
    print("  Saved: exp5_convergence.png")


# ── Stationary population law ─────────────────────────────────────────────────

def plot_population_law(solver: MFGSolver, mu_final: np.ndarray) -> None:
    states   = solver.states
    # Marginal over w (width usage)
    w_vals   = np.array([w for _, w in states])
    w_hist   = np.zeros(solver.p.W_max + 1)
    for i, w in enumerate(w_vals):
        w_hist[w] += mu_final[i]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(np.arange(solver.p.W_max + 1), w_hist, color="steelblue", alpha=0.8)
    ax.set_xlabel("Half-width w (0 = out-of-pool)")
    ax.set_ylabel("μ(w) — stationary mass on width w")
    ax.set_title("Exp 5: Stationary population law (marginal over width)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "exp5_population_law.png"), dpi=120)
    plt.close(fig)
    print("  Saved: exp5_population_law.png")


# ── Comparative statics plot ───────────────────────────────────────────────────

def plot_comparative_statics(df_L0: pd.DataFrame, df_f: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(df_L0["L_0"], df_L0["mean_fee"], "b-o", linewidth=2, markersize=5)
    ax.set_xlabel("Background liquidity L_0")
    ax.set_ylabel("Mean fee income")
    ax.set_title("Exp 5: Congestion effect — fee income vs L_0")
    ax.set_xscale("log")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(df_f["f"] * 1e4, df_f["mean_fee"], "r-o", linewidth=2, markersize=5)
    ax.set_xlabel("Fee rate f (basis points)")
    ax.set_ylabel("Mean fee income")
    ax.set_title("Exp 5: Fee-rate comparative statics")
    ax.grid(alpha=0.3)

    fig.suptitle("Exp 5: MFG comparative statics", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "exp5_comparative_statics.png"), dpi=120)
    plt.close(fig)
    print("  Saved: exp5_comparative_statics.png")


# ── Value function ─────────────────────────────────────────────────────────────

def plot_value_function(solver: MFGSolver, V_final: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(V_final))
    ax.bar(x, V_final, color="steelblue", alpha=0.8)
    ax.set_xlabel("State index i = (j, w)")
    ax.set_ylabel("V*(i)")
    ax.set_title("Exp 5: MFG stationary value function V*(j, w)")
    ax.grid(axis="y", alpha=0.3)
    # Annotate state labels
    labels = [f"j{j}w{w}" for j, w in solver.states]
    if len(labels) <= 30:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=75, fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "exp5_value_function.png"), dpi=120)
    plt.close(fig)
    print("  Saved: exp5_value_function.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("EXPERIMENT 5: Finite-state averaged mean-field game")
    print("=" * 60)

    params = DEFAULT_PARAMS
    solver = MFGSolver(params)
    print(f"  State space size: {solver.n_states} states "
          f"({params.M} ticks × {params.W_max+1} widths)")
    print(f"  Active-tick grid: K={solver.K}, π_k=1/{solver.K}")

    # ── Existence and uniqueness test
    uniq_result = test_existence_and_uniqueness(params, n_init=4)

    V_final  = uniq_result["V_final"]
    mu_final = uniq_result["mu_final"]

    # ── Comparative statics
    df_L0 = run_comparative_statics(params)
    df_f  = run_comparative_statics_f(params)

    # ── Summary table
    summary = {
        "n_states":     solver.n_states,
        "M":            params.M,
        "W_max":        params.W_max,
        "iterations":   uniq_result["iterations"],
        "converged":    all(uniq_result["converged"]),
        "L_range":      uniq_result["L_range"],
        "mu_max_dev":   uniq_result["mu_max_dev"],
        "mean_V":       float(np.mean(V_final)),
        "max_residual": float(uniq_result["residuals"][-1]),
    }
    df_summary = pd.DataFrame([summary])
    out_csv = os.path.join(RESULTS, "exp5_mfg_summary.csv")
    df_summary.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")
    print(df_summary.to_string(index=False))

    # ── Plots
    print("\nGenerating plots …")
    plot_convergence(uniq_result["residuals"])
    plot_population_law(solver, mu_final)
    plot_comparative_statics(df_L0, df_f)
    plot_value_function(solver, V_final)

    print("\nExperiment 5 complete.")


if __name__ == "__main__":
    main()
