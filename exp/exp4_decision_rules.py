"""
Experiment 4: Decision-rule validation — exclusion thresholds, capital viability,
and implied-volatility relation.

Classifies pools under boundary-triggered provision; computes minimum viable capital;
and tests the link between critical volatility and on-chain implied volatility.

Outputs: results/exp4_*.png, results/exp4_exclusion_table.csv,
         results/exp4_capital_viability.csv
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

from data.synthetic_pool_data import POOL_CONFIGS, load_pool_data
from src.empirical import compute_full_sample_margins
from src.approximations import s_star as compute_s_star, g_max
from src.decision_rules import (
    compute_exclusion_threshold,
    classify_pool,
    compute_c_min,
    compute_sigma_IV,
    analyze_implied_vol_relation as test_implied_vol_relation,
    build_exclusion_table,
)

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)

# ── Paper reference values ─────────────────────────────────────────────────────
# kappa values as specified in the paper
POOL_META = {
    "ETH_UNIV3_WETH_USDC_030":   {"kappa": 0.00175,  "ref_s_excl": 0.9823},
    "ETH_UNIV3_WETH_USDC_005":   {"kappa": 0.0005,   "ref_s_excl": 0.9922},
    "BASE_UNIV3_WETH_USDC_005":  {"kappa": 0.0005,   "ref_s_excl": 0.9922},
    "BASE_AERO_WETH_USDC":       {"kappa": 0.000476, "ref_s_excl": 0.9924},
    "BASE_PANCAKE_WETH_USDC_001": {"kappa": 0.0003,  "ref_s_excl": 0.9944},
}

G0_USD_VALS   = [10.0, 20.0, 50.0]          # fixed cost scenarios
C_USD_VALS    = [30_000, 56_000, 130_000, 2_200_000]  # capital levels (USD)


def get_pool_margins() -> list[dict]:
    """Generate synthetic data and compute full-sample margins for each pool."""
    rows = []
    for cfg in POOL_CONFIGS:
        pid = cfg["pool_id"]
        df  = load_pool_data(pid)
        m   = compute_full_sample_margins(df)
        kp  = POOL_META[pid]["kappa"]
        rows.append({
            "pool_id":   pid,
            "f":         cfg["f"],
            "h":         cfg["h"],
            "kappa":     kp,
            "sigma":     m["sigma"],
            "sigma_c":   m["sigma_c"],
            "s":         m["s"],
            "L_bar":     cfg["L_mean"],
            "V_mean":    cfg["V_mean"],
            "q_bar":     cfg["q0"],
            "df":        df,
        })
    return rows


# ── Exclusion table ───────────────────────────────────────────────────────────

def build_full_exclusion_table(pool_margins: list[dict]) -> pd.DataFrame:
    pool_dicts = []
    for p in pool_margins:
        pool_dicts.append({
            "pool_id": p["pool_id"],
            "kappa":   p["kappa"],
            "s":       p["s"],
            "f":       p["f"],
            "h":       p["h"],
            "g":       p["kappa"],   # use proportional costs as proxy for g
        })
    df_excl = build_exclusion_table(pool_dicts)

    # Add reference values from paper
    df_excl["ref_s_excl"] = df_excl["pool_id"].map(
        {k: v["ref_s_excl"] for k, v in POOL_META.items()}
    )
    df_excl["error_excl"] = (df_excl["s_excl"] - df_excl["ref_s_excl"]).abs()
    return df_excl


# ── Capital viability ─────────────────────────────────────────────────────────

def compute_capital_table(pool_margins: list[dict]) -> pd.DataFrame:
    rows = []
    for p in pool_margins:
        s  = p["s"]
        kp = p["kappa"]
        q_bar = p["q_bar"]
        for g0_usd in G0_USD_VALS:
            g0_model = g0_usd / q_bar
            for c_usd in C_USD_VALS:
                c_model = c_usd / q_bar
                # total cost = g0/c + kappa
                g_total = g0_model / c_model + kp
                cm = compute_c_min(s, g0_model, kp)
                cm_usd = cm * q_bar if np.isfinite(cm) else np.inf
                rows.append({
                    "pool_id":   p["pool_id"],
                    "kappa":     kp,
                    "s":         s,
                    "g0_usd":    g0_usd,
                    "c_usd":     c_usd,
                    "g_total":   g_total,
                    "c_min_usd": cm_usd,
                    "viable":    np.isfinite(cm_usd) and c_usd >= cm_usd,
                })
    return pd.DataFrame(rows)


# ── Implied volatility relation ───────────────────────────────────────────────

def test_iv_relation(pool_margins: list[dict]) -> pd.DataFrame:
    rows = []
    for p in pool_margins:
        df    = p["df"]
        V_arr = df["V_t"].values
        q_arr = df["q_t"].values
        h     = p["h"]
        f     = p["f"]
        L_bar = p["L_bar"]

        sigma_IV_val = compute_sigma_IV(V_arr, q_arr, L_bar, f, h)
        rel = test_implied_vol_relation(p["sigma_c"], sigma_IV_val, h, f)
        rows.append({
            "pool_id":         p["pool_id"],
            "sigma_c":         rel["sigma_c"],
            "sigma_IV":        rel["sigma_IV"],
            "predicted_ratio": rel["predicted_ratio"],
            "observed_ratio":  rel["observed_ratio"],
            "h_approx_2f":     rel["close_for_h_approx_2f"],
        })
    return pd.DataFrame(rows)


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_thresholds(pool_margins: list[dict], df_excl: pd.DataFrame) -> None:
    pids    = [p["pool_id"] for p in pool_margins]
    s_vals  = [p["s"] for p in pool_margins]
    s_excls = [compute_exclusion_threshold(p["kappa"]) for p in pool_margins]
    s_stars = [compute_s_star(p["kappa"]) for p in pool_margins]
    x       = np.arange(len(pids))

    fig, ax = plt.subplots(figsize=(12, 6))
    bar_w = 0.25
    ax.bar(x - bar_w, s_vals,  bar_w, label="s observed",  color="steelblue")
    ax.bar(x,         s_excls, bar_w, label="s_excl",       color="tomato",   alpha=0.8)
    ax.bar(x + bar_w, s_stars, bar_w, label="s_star (g=κ)", color="forestgreen", alpha=0.8)
    ax.axhline(1.0, color="k", linestyle="--", linewidth=1.0, label="s = 1")
    ax.set_xticks(x)
    ax.set_xticklabels([pid.replace("_", "\n") for pid in pids], fontsize=8)
    ax.set_ylabel("Participation margin s")
    ax.set_title("Exp 4: Observed s vs thresholds s_excl, s_star per pool")
    ax.legend(fontsize=9)
    ax.set_ylim(0.9, 1.1)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "exp4_thresholds.png"), dpi=120)
    plt.close(fig)
    print("  Saved: exp4_thresholds.png")


def plot_c_min_curves(pool_margins: list[dict]) -> None:
    s_range = np.linspace(0.70, 0.995, 100)
    fig, ax = plt.subplots(figsize=(10, 5))
    colours = plt.cm.tab10(np.linspace(0, 0.5, len(POOL_META)))

    for (pid, meta), colour in zip(POOL_META.items(), colours):
        kp    = meta["kappa"]
        q_bar = next(c["q0"] for c in POOL_CONFIGS if c["pool_id"] == pid)
        g0_model = 20.0 / q_bar   # $20 fixed cost
        cms = [compute_c_min(s, g0_model, kp) for s in s_range]
        cms_cap = np.minimum(
            [c * q_bar if np.isfinite(c) else np.nan for c in cms], 1e6
        )
        ax.plot(s_range, cms_cap, label=f"{pid[:20]} κ={kp:.5f}",
                color=colour, linewidth=1.8)

    ax.set_xlabel("σ / σ_c")
    ax.set_ylabel("c_min (USD equivalent, g0=$20)")
    ax.set_title("Exp 4: Minimum viable capital c_min(s)")
    ax.legend(fontsize=7)
    ax.set_ylim(0, 2e5)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "exp4_c_min.png"), dpi=120)
    plt.close(fig)
    print("  Saved: exp4_c_min.png")


def plot_iv_scatter(df_iv: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for _, row in df_iv.iterrows():
        ax.scatter(row["sigma_IV"], row["sigma_c"], s=80, zorder=3)
        ax.annotate(row["pool_id"][-10:], (row["sigma_IV"], row["sigma_c"]),
                    fontsize=7, xytext=(4, 4), textcoords="offset points")

    lims = [
        min(df_iv["sigma_IV"].min(), df_iv["sigma_c"].min()) * 0.95,
        max(df_iv["sigma_IV"].max(), df_iv["sigma_c"].max()) * 1.05,
    ]
    ax.plot(lims, lims, "k--", alpha=0.5, label="σ_c = σ_IV (identity)")
    ax.set_xlabel("σ_IV (on-chain implied vol)")
    ax.set_ylabel("σ_c (critical vol)")
    ax.set_title("Exp 4: σ_c vs σ_IV — implied-volatility relation test")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "exp4_iv_relation.png"), dpi=120)
    plt.close(fig)
    print("  Saved: exp4_iv_relation.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("EXPERIMENT 4: Decision-rule validation")
    print("=" * 60)

    pool_margins = get_pool_margins()

    # Exclusion table
    df_excl = build_full_exclusion_table(pool_margins)
    print("\n── Exclusion table ───────────────────────────────────────")
    print(df_excl[["pool_id", "kappa", "s_excl", "ref_s_excl", "error_excl",
                   "s_observed", "verdict"]].to_string(index=False))
    df_excl.to_csv(os.path.join(RESULTS, "exp4_exclusion_table.csv"), index=False)
    print(f"Saved: exp4_exclusion_table.csv")

    # Capital viability
    df_cap = compute_capital_table(pool_margins)
    print("\n── Capital viability (first 15 rows) ────────────────────")
    print(df_cap.head(15).to_string(index=False))
    df_cap.to_csv(os.path.join(RESULTS, "exp4_capital_viability.csv"), index=False)
    print(f"Saved: exp4_capital_viability.csv")

    # Implied volatility test
    df_iv = test_iv_relation(pool_margins)
    print("\n── Implied volatility relation ──────────────────────────")
    print(df_iv.to_string(index=False))

    # Plots
    print("\nGenerating plots …")
    plot_thresholds(pool_margins, df_excl)
    plot_c_min_curves(pool_margins)
    plot_iv_scatter(df_iv)

    print("\nExperiment 4 complete.")


if __name__ == "__main__":
    main()
