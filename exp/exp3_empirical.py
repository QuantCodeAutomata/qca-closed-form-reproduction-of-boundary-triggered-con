"""
Experiment 3: Empirical reconstruction of rolling critical volatility and
participation margins from (SYNTHETIC) hourly pool data.

*** ALL DATA IS SYNTHETIC — NOT REAL ON-CHAIN DATA ***
This script demonstrates the paper's methodology using simulated pool data
that mimics the statistical properties of the target WETH/USDC pools.

Outputs: results/exp3_*.png and results/exp3_pool_summary.csv
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
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings("ignore")

from data.synthetic_pool_data import POOL_CONFIGS, load_pool_data
from src.empirical import (
    compute_rolling_margins,
    compute_full_sample_margins,
    block_bootstrap_ci,
    run_pool_panel,
)

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)

WINDOW       = 720    # 30-day rolling window (hours)
N_BOOT       = 2000
BLOCK_SIZE   = 168    # one week
ANNUALIZATION = 8760

DISCLAIMER = (
    "⚠  SYNTHETIC DEMONSTRATION: All pool data are simulated.\n"
    "   Results do not reflect actual on-chain pool performance."
)


def interpret_s(s: float) -> str:
    if s < 1.0:
        return "potentially_profitable"
    elif s < 1.05:
        return "near_breakeven"
    elif s < 1.15:
        return "above_breakeven"
    return "high_margin"


# ── Rolling analysis for a single pool ────────────────────────────────────────

def analyse_pool(cfg: dict, seed: int = 42) -> dict:
    """Run full rolling and full-sample analysis for one pool."""
    pool_seed = seed + hash(cfg["pool_id"]) % 10_000
    df = load_pool_data(cfg["pool_id"], seed=seed)
    df_margins = compute_rolling_margins(df, window=WINDOW, annualization=ANNUALIZATION)
    margins_fs = compute_full_sample_margins(df, annualization=ANNUALIZATION)
    try:
        ci = block_bootstrap_ci(df, n_boot=N_BOOT, block_size=BLOCK_SIZE,
                                seed=pool_seed, annualization=ANNUALIZATION)
    except Exception as exc:
        print(f"  [Warning] Bootstrap failed for {cfg['pool_id']}: {exc}")
        ci = {"s_hat": margins_fs["s"], "ci_lower": np.nan, "ci_upper": np.nan}
    return {
        "pool_id":   cfg["pool_id"],
        "cfg":       cfg,
        "df":        df_margins,
        "full":      margins_fs,
        "ci":        ci,
    }


# ── Plot: rolling sigma vs sigma_c for primary pool ──────────────────────────

def plot_rolling_sigma_main(results: dict) -> None:
    r   = results["ETH_UNIV3_WETH_USDC_030"]
    df  = r["df"].dropna(subset=["sigma_hat", "sigma_c_hat"])

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df["timestamp"], df["sigma_hat"],   "b-",  linewidth=1.2, label="σ̂ (realized vol)")
    ax.plot(df["timestamp"], df["sigma_c_hat"], "r--", linewidth=1.2, label="σ̂_c (critical vol)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=30)
    ax.set_ylabel("Annualised volatility")
    ax.set_title("Exp 3 (SYNTHETIC): Rolling 30-day σ vs σ_c — ETH Uniswap v3 0.3% WETH/USDC")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "exp3_rolling_sigma_main.png"), dpi=120)
    plt.close(fig)
    print("  Saved: exp3_rolling_sigma_main.png")


# ── Plot: rolling s across all pools ─────────────────────────────────────────

def plot_rolling_s_all(results: dict) -> None:
    colours = plt.cm.tab10(np.linspace(0, 0.9, len(POOL_CONFIGS)))
    fig, ax = plt.subplots(figsize=(14, 6))

    for (pool_id, r), colour in zip(results.items(), colours):
        df = r["df"].dropna(subset=["s_hat"])
        ax.plot(df["timestamp"], df["s_hat"], linewidth=0.9,
                color=colour, label=pool_id, alpha=0.8)

    ax.axhline(1.0, color="k", linestyle="--", linewidth=1.2, label="s = 1 (break-even)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=30)
    ax.set_ylabel("s = σ / σ_c")
    ax.set_title("Exp 3 (SYNTHETIC): Rolling participation margin s across pools")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "exp3_rolling_s_all.png"), dpi=120)
    plt.close(fig)
    print("  Saved: exp3_rolling_s_all.png")


# ── Plot: full-sample summary bar chart ───────────────────────────────────────

def plot_full_sample_summary(df_summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    x       = np.arange(len(df_summary))
    labels  = [pid.replace("_", "\n") for pid in df_summary["pool_id"]]

    ax.bar(x, df_summary["s_hat"], width=0.5, color="steelblue", alpha=0.8, label="s = σ/σ_c")
    ax.errorbar(
        x, df_summary["s_hat"],
        yerr=[
            df_summary["s_hat"] - df_summary["ci_lower"],
            df_summary["ci_upper"] - df_summary["s_hat"],
        ],
        fmt="none", color="black", capsize=6, linewidth=1.5, label="95% CI"
    )
    ax.axhline(1.0, color="r", linestyle="--", linewidth=1.2, label="Break-even (s=1)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Participation margin s = σ / σ_c")
    ax.set_title("Exp 3 (SYNTHETIC): Full-sample participation margins with 95% CI")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "exp3_full_sample_summary.png"), dpi=120)
    plt.close(fig)
    print("  Saved: exp3_full_sample_summary.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("EXPERIMENT 3: Rolling critical volatility reconstruction")
    print(DISCLAIMER)
    print("=" * 60)

    all_results: dict[str, dict] = {}
    summary_rows = []

    for cfg in POOL_CONFIGS:
        pid = cfg["pool_id"]
        print(f"\nProcessing pool: {pid}")
        r = analyse_pool(cfg)
        all_results[pid] = r

        full = r["full"]
        ci   = r["ci"]
        interp = interpret_s(full["s"])

        print(
            f"  σ̂={full['sigma']:.4f}  σ̂_c={full['sigma_c']:.4f}  "
            f"ŝ={full['s']:.4f}  95% CI=[{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]"
            f"  → {interp}"
        )
        summary_rows.append({
            "pool_id":    pid,
            "f":          cfg["f"],
            "h":          cfg["h"],
            "sigma_hat":  full["sigma"],
            "sigma_c_hat": full["sigma_c"],
            "s_hat":      full["s"],
            "ci_lower":   ci["ci_lower"],
            "ci_upper":   ci["ci_upper"],
            "interpretation": interp,
        })

    df_summary = pd.DataFrame(summary_rows)
    out_csv    = os.path.join(RESULTS, "exp3_pool_summary.csv")
    df_summary.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")
    print(df_summary.to_string(index=False))

    # Plots
    print("\nGenerating plots …")
    plot_rolling_sigma_main(all_results)
    plot_rolling_s_all(all_results)
    plot_full_sample_summary(df_summary)

    print(f"\n{DISCLAIMER}")
    print("Experiment 3 complete.")


if __name__ == "__main__":
    main()
