"""
Experiment 1: Closed-form reproduction of boundary-triggered LP profitability.

Reproduces the paper's exact stationary closed-form results for a boundary-triggered
liquidity provider under driftless log-price diffusion and fast rebalancing.

Outputs: results/exp1_*.png and results/exp1_summary.csv
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
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

from src.models import (
    CLModel, H_03, H_005,
    epsilon_w_centered, varphi, eta, lambda_eff,
    rho_w_array, R_cyc, R_0, Pi,
    optimal_width, value_function_V_star,
    B1_coeff, sigma_c, run_exp1_scenario,
)

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)
W_MAX = 200


# ── Scenario parameters ───────────────────────────────────────────────────────

# Normalised parameters (q_bar = L_bar = 1) with V_bar calibrated so that
# sigma_c = 1.0 for both tiers.
#   sigma_c = 2*sqrt(h*B_1),  B_1 = f*V_bar/(sqrt(q_bar)*h*L_bar)
#   => sigma_c^2 = 4*f*V_bar/(sqrt(q_bar)*L_bar)
#   => V_bar = sigma_c^2 * sqrt(q_bar) * L_bar / (4*f)
# For sigma_c=1, q_bar=1, L_bar=1:
#   0.3%  tier  (f=0.003):   V_bar = 1/(4*0.003) ≈ 83.33
#   0.05% tier  (f=0.0005):  V_bar = 1/(4*0.0005) = 500.0
TIERS = {
    "Uniswap v3 0.3%":  dict(f=0.003,  h=H_03,  V_bar=83.33,  L_bar=1.0, q_bar=1.0),
    "Uniswap v3 0.05%": dict(f=0.0005, h=H_005, V_bar=500.0,  L_bar=1.0, q_bar=1.0),
}
# SIGMAS expressed in the same units as sigma_c = 1.0 so that both profitable
# (s<1) and unprofitable (s>1) regimes are visible.
SIGMAS   = [0.30, 0.60, 0.85, 0.95, 1.05, 1.20]
G        = 0.0005   # 5 bp
KAPPA    = 0.0005
W_GRID   = np.arange(1, W_MAX + 1, dtype=int)


# ── Assertion helpers ─────────────────────────────────────────────────────────

def run_assertions(tier_name: str, tier: dict, sigma_val: float) -> None:
    h    = tier["h"]
    t_j  = tier["q_bar"]

    # Epsilon positivity
    eps_arr = np.array([epsilon_w_centered(t_j, int(w), h) for w in W_GRID])
    assert np.all(eps_arr > 0), f"[{tier_name}] epsilon_w must be positive"

    # rho_w sums to 1 for representative widths
    for w_test in [1, 5, 10, 20, 50]:
        rho = rho_w_array(w_test)
        total = float(np.sum(rho))
        assert abs(total - 1.0) < 1e-12, (
            f"[{tier_name}] rho_w({w_test}) does not sum to 1: {total}"
        )

    # Compute B1 / sigma_c
    B1 = B1_coeff(tier["f"], tier["V_bar"], tier["q_bar"], h, tier["L_bar"])
    sc = sigma_c(B1, h)

    # Profitability check for low sigma
    if sigma_val < sc:
        opt = optimal_width(tier["f"], tier["V_bar"], tier["L_bar"], t_j,
                            h, sigma_val, G, W_MAX)
        assert opt["Pi_opt"] > 0, (
            f"[{tier_name}] σ<σ_c but Pi(w*)={opt['Pi_opt']:.4e} not positive"
        )
        # Pi becomes negative for large w
        pi_large = Pi(W_MAX, tier["f"], tier["V_bar"], tier["L_bar"], t_j,
                      h, sigma_val, G)
        assert pi_large < opt["Pi_opt"], (
            f"[{tier_name}] Pi should decrease for large w"
        )

    print(f"  ✓ Assertions passed: [{tier_name}] σ={sigma_val:.2f}")


# ── Plot 1: Pi(w) vs w for multiple sigma ─────────────────────────────────────

def plot_profit_curves(summary_rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    tier_names = list(TIERS.keys())
    colours    = plt.cm.viridis(np.linspace(0.1, 0.9, len(SIGMAS)))

    for ax_idx, tier_name in enumerate(tier_names):
        ax    = axes[ax_idx]
        tier  = TIERS[tier_name]
        h     = tier["h"]
        t_j   = tier["q_bar"]
        B1    = B1_coeff(tier["f"], tier["V_bar"], tier["q_bar"], h, tier["L_bar"])
        sc    = sigma_c(B1, h)

        for i, sigma_val in enumerate(SIGMAS):
            pi_arr = np.array([
                Pi(int(w), tier["f"], tier["V_bar"], tier["L_bar"],
                   t_j, h, sigma_val, G)
                for w in W_GRID
            ])
            s_ratio = sigma_val / sc
            ax.plot(W_GRID, pi_arr, color=colours[i],
                    label=f"σ={sigma_val:.2f} (s={s_ratio:.2f})")

        ax.axhline(0, color="k", linewidth=0.8, linestyle="--")
        ax.set_xlim(1, 100)
        ax.set_xlabel("Half-width w")
        ax.set_ylabel("Profit rate Π(w)")
        ax.set_title(f"{tier_name}\nProfit rate vs width")
        ax.legend(fontsize=7, loc="upper right")

    fig.suptitle("Exp 1: Long-run profit rate Π(w) for varying σ", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "exp1_profit_curves.png"), dpi=120)
    plt.close(fig)
    print("  Saved: exp1_profit_curves.png")


# ── Plot 2: Optimal width w* vs sigma/sigma_c ─────────────────────────────────

def plot_optimal_width_vs_s(summary_rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    tier_names = list(TIERS.keys())
    s_grid     = np.linspace(0.1, 0.99, 80)

    for ax_idx, tier_name in enumerate(tier_names):
        ax   = axes[ax_idx]
        tier = TIERS[tier_name]
        h    = tier["h"]
        t_j  = tier["q_bar"]
        B1   = B1_coeff(tier["f"], tier["V_bar"], tier["q_bar"], h, tier["L_bar"])
        sc   = sigma_c(B1, h)

        w_stars = []
        for s_val in s_grid:
            sigma_val = s_val * sc
            opt = optimal_width(tier["f"], tier["V_bar"], tier["L_bar"],
                                t_j, h, sigma_val, G, W_MAX)
            w_stars.append(opt["w_star"])

        ax.plot(s_grid, w_stars, "b-", linewidth=2)
        ax.set_xlabel("σ / σ_c")
        ax.set_ylabel("Optimal half-width w*")
        ax.set_title(f"{tier_name}\nOptimal width vs participation margin")
        ax.grid(alpha=0.3)

    fig.suptitle("Exp 1: Optimal width w* vs σ/σ_c", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "exp1_optimal_width.png"), dpi=120)
    plt.close(fig)
    print("  Saved: exp1_optimal_width.png")


# ── Plot 3: Value function V*(n) ──────────────────────────────────────────────

def plot_value_function(tier_name: str, tier: dict, sigma_val: float) -> None:
    h   = tier["h"]
    t_j = tier["q_bar"]
    opt = optimal_width(tier["f"], tier["V_bar"], tier["L_bar"],
                        t_j, h, sigma_val, G, W_MAX)
    w_star = opt["w_star"]
    n_arr  = np.arange(-w_star, w_star + 1, dtype=float)
    V      = value_function_V_star(n_arr, w_star, h, G)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(n_arr, V, "b-", linewidth=2, label="V*(n)")
    ax.axhline(G / 2.0, color="gray", linestyle="--", alpha=0.5, label="C = g/2")
    ax.set_xlabel("Offset n")
    ax.set_ylabel("V*(n)")
    ax.set_title(
        f"Exp 1: Stationary value function V*(n)\n"
        f"{tier_name}, σ={sigma_val:.2f}, w*={w_star}"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "exp1_value_function.png"), dpi=120)
    plt.close(fig)
    print("  Saved: exp1_value_function.png")


# ── Plot 4: Cycle occupation law rho(n) ──────────────────────────────────────

def plot_rho(w_demo: int = 10) -> None:
    rho  = rho_w_array(w_demo)
    ns   = np.arange(-w_demo, w_demo + 1)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(ns, rho, color="steelblue", alpha=0.8, edgecolor="navy", linewidth=0.5)
    ax.set_xlabel("Offset n")
    ax.set_ylabel("ρ_w(n)")
    ax.set_title(f"Exp 1: Triangular cycle occupation law ρ_w(n), w={w_demo}")
    ax.annotate(f"Sum = {rho.sum():.12f}", xy=(0.02, 0.95),
                xycoords="axes fraction", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "exp1_rho_cycle.png"), dpi=120)
    plt.close(fig)
    print("  Saved: exp1_rho_cycle.png")


# ── Plot 5: Critical sigma frontier ──────────────────────────────────────────

def plot_sigma_frontier() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    tier_names = list(TIERS.keys())
    sigma_range = np.linspace(0.3, 2.0, 100)

    for ax_idx, tier_name in enumerate(tier_names):
        ax   = axes[ax_idx]
        tier = TIERS[tier_name]
        h    = tier["h"]
        t_j  = tier["q_bar"]
        B1   = B1_coeff(tier["f"], tier["V_bar"], tier["q_bar"], h, tier["L_bar"])
        sc   = sigma_c(B1, h)

        max_pis = []
        for sigma_val in sigma_range:
            opt = optimal_width(tier["f"], tier["V_bar"], tier["L_bar"],
                                t_j, h, sigma_val, G, W_MAX)
            max_pis.append(opt["Pi_opt"])

        ax.plot(sigma_range, max_pis, "b-", linewidth=2)
        ax.axhline(0, color="k", linewidth=0.8, linestyle="--")
        ax.axvline(sc, color="r", linewidth=1.5, linestyle=":", label=f"σ_c={sc:.4f}")
        ax.set_xlabel("σ (annualised)")
        ax.set_ylabel("max_w Π(w)")
        ax.set_title(f"{tier_name}\nMax profit vs volatility")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle("Exp 1: Critical volatility frontier max_w Π(w) vs σ", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "exp1_sigma_frontier.png"), dpi=120)
    plt.close(fig)
    print("  Saved: exp1_sigma_frontier.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("EXPERIMENT 1: Closed-form LP profitability reproduction")
    print("=" * 60)

    summary_rows = []

    for tier_name, tier in TIERS.items():
        h   = tier["h"]
        t_j = tier["q_bar"]
        B1  = B1_coeff(tier["f"], tier["V_bar"], tier["q_bar"], h, tier["L_bar"])
        sc  = sigma_c(B1, h)
        print(f"\n[{tier_name}]  h={h:.6f}  B1={B1:.6f}  σ_c={sc:.6f}")

        for sigma_val in SIGMAS:
            s = sigma_val / sc
            run_assertions(tier_name, tier, sigma_val)

            params = CLModel(
                f=tier["f"], h=h, V_bar=tier["V_bar"], L_bar=tier["L_bar"],
                q_bar=tier["q_bar"], sigma=sigma_val, g=G, kappa=KAPPA,
            )
            result = run_exp1_scenario(params, W_max=W_MAX)

            print(
                f"  σ={sigma_val:.2f}  s={s:.3f}  "
                f"w*={result['w_star']}  Π(w*)={result['Pi_opt']:.6e}"
            )
            summary_rows.append({
                "tier":       tier_name,
                "h":          h,
                "f":          tier["f"],
                "sigma":      sigma_val,
                "sigma_c":    sc,
                "s":          s,
                "w_star":     result["w_star"],
                "Pi_opt":     result["Pi_opt"],
                "B1":         B1,
            })

    # Save summary CSV
    df_summary = pd.DataFrame(summary_rows)
    out_csv    = os.path.join(RESULTS, "exp1_summary.csv")
    df_summary.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")
    print(df_summary.to_string(index=False))

    # Plots
    print("\nGenerating plots …")
    plot_profit_curves(summary_rows)
    plot_optimal_width_vs_s(summary_rows)

    # Value function for base case (0.3% tier, sigma=0.85)
    base_tier = TIERS["Uniswap v3 0.3%"]
    plot_value_function("Uniswap v3 0.3%", base_tier, sigma_val=0.85)

    plot_rho(w_demo=10)
    plot_sigma_frontier()

    print("\nExperiment 1 complete.")


if __name__ == "__main__":
    main()
