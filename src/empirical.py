"""
Empirical estimation functions for rolling critical volatility and participation margins.

Since on-chain WETH/USDC pool data is not directly accessible in this environment,
this module provides both the estimator implementations and a synthetic data generator
that mimics the statistical properties of the target pools.

NOTE: All outputs from synthetic data runs are clearly labeled 'SYNTHETIC DEMONSTRATION'.

Uses arch.bootstrap.CircularBlockBootstrap for block bootstrap CI.
# Using arch CircularBlockBootstrap — Context7 confirmed (/bashtage/arch)
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from typing import Optional

from arch.bootstrap import CircularBlockBootstrap

from src.models import H_03, H_005, sigma_c as compute_sigma_c


# ── Synthetic data generation ─────────────────────────────────────────────────

def generate_synthetic_pool_data(
    n_hours: int = 26280,
    seed: int = 42,
    f: float = 0.003,
    h: Optional[float] = None,
    sigma_annualized: float = 0.85,
    q0: float = 2000.0,
    L_mean: float = 1e8,
    V_mean: float = 5e5,
) -> pd.DataFrame:
    """
    Generate synthetic hourly pool data mimicking WETH/USDC Uniswap v3 pools.

    Price follows driftless geometric Brownian motion (annualized vol = sigma_annualized).
    Volume is lognormal around V_mean.
    Liquidity follows a mean-reverting process around L_mean.

    Parameters
    ----------
    n_hours          : Number of hourly observations
    seed             : Random seed for reproducibility
    f                : Fee rate
    h                : Log tick spacing (defaults to H_03)
    sigma_annualized : Annualised price volatility
    q0               : Initial price
    L_mean           : Mean active liquidity
    V_mean           : Mean traded volume
    """
    rng = np.random.default_rng(seed)
    if h is None:
        h = H_03

    # Price: GBM dln(q) = sigma/sqrt(8760) * N(0,1)
    sigma_hourly = sigma_annualized / np.sqrt(8760.0)
    log_returns  = rng.normal(0.0, sigma_hourly, size=n_hours)
    log_prices   = np.log(q0) + np.cumsum(log_returns)
    q_t          = np.exp(log_prices)

    # Volume: lognormal with mean V_mean
    log_V_mean = np.log(V_mean) - 0.5 * 0.8 ** 2   # lognormal parameterisation
    V_t = np.exp(rng.normal(log_V_mean, 0.8, size=n_hours))
    V_t = np.maximum(V_t, 0.0)

    # Liquidity: mean-reverting (Ornstein-Uhlenbeck discretised)
    theta_L = 0.05     # mean-reversion speed
    sigma_L = L_mean * 0.1
    L_t     = np.empty(n_hours)
    L_t[0]  = L_mean
    for t in range(1, n_hours):
        L_t[t] = (
            L_t[t - 1]
            + theta_L * (L_mean - L_t[t - 1])
            + sigma_L * rng.standard_normal()
        )
        L_t[t] = max(L_t[t], L_mean * 0.1)  # prevent non-positive liquidity

    timestamps = pd.date_range("2024-01-01", periods=n_hours, freq="h", tz="UTC")

    df = pd.DataFrame({
        "timestamp":  timestamps,
        "q_t":        q_t,
        "V_t":        V_t,
        "L_t":        L_t,
        "f":          f,
        "h":          h,
        "log_return": log_returns,
    })
    return df


# ── Realized volatility ───────────────────────────────────────────────────────

def compute_rolling_realized_vol(
    log_returns: pd.Series,
    window: int = 720,
    annualization: int = 8760,
) -> pd.Series:
    """
    Rolling 30-day (720-hour) realized volatility from hourly log returns.

    Convention: RV = sqrt(annualization × mean(r_t²)) over the window.
    First (window-1) values are NaN (requires full window).

    Parameters
    ----------
    log_returns   : Series of hourly log returns
    window        : Rolling window length in hours (default 720 = 30 days)
    annualization : Annualisation factor (default 8760 = hours per year)
    """
    rv_sq = log_returns ** 2
    rv    = rv_sq.rolling(window=window, min_periods=window).mean()
    return np.sqrt(annualization * rv)


# ── Pathwise B_1 estimator ────────────────────────────────────────────────────

def compute_pathwise_B1(
    q_t: pd.Series,
    V_t: pd.Series,
    L_t: pd.Series,
    f: float,
    h: float,
    window: int = 720,
) -> pd.Series:
    """
    Pathwise rolling B_1 estimator EXACTLY as specified by the paper.

    Paper formula:
        B̂_1 = (1/H) Σ_{t=1}^{H} [ f · V_t / (√q_t · h · L_t) ]

    Critical: this is the AVERAGE OF RATIOS, NOT the RATIO OF AVERAGES.
    Hours with L_t <= 0 are excluded (replaced by NaN before averaging).

    Parameters
    ----------
    q_t, V_t, L_t : Hourly price, volume, liquidity series
    f             : Fee rate
    h             : Log tick spacing
    window        : Rolling window (default 720 hours)
    """
    # Paper formula: pathwise B1 estimator (Section 5)
    ratio = pd.Series(np.where(L_t > 0, f * V_t / (np.sqrt(q_t) * h * L_t), np.nan),
                      index=q_t.index)
    # Rolling mean of ratios (pathwise average)
    return ratio.rolling(window=window, min_periods=window).mean()


# ── Rolling margin computation ────────────────────────────────────────────────

def compute_rolling_margins(
    df: pd.DataFrame,
    window: int = 720,
    annualization: int = 8760,
) -> pd.DataFrame:
    """
    Compute rolling sigma_hat, B1_hat, sigma_c_hat, s_hat for a pool DataFrame.

    Input DataFrame must contain columns: timestamp, q_t, V_t, L_t, f, h, log_return.

    Adds columns: sigma_hat, B1_hat, sigma_c_hat, s_hat.
    B1_hat      = per-hour pathwise average of f·V/(√q·h·L)
    sigma_c_hat = 2·sqrt(h · B1_hat · annualization)   [annualised units]
    s_hat       = sigma_hat / sigma_c_hat
    """
    result = df.copy()
    f_val  = float(df["f"].iloc[0])
    h_val  = float(df["h"].iloc[0])

    result["sigma_hat"]   = compute_rolling_realized_vol(
        df["log_return"], window=window, annualization=annualization
    )
    result["B1_hat"]      = compute_pathwise_B1(
        df["q_t"], df["V_t"], df["L_t"], f_val, h_val, window=window
    )
    # Annualise B1_hat (per-hour) before computing sigma_c (per-year units)
    result["sigma_c_hat"] = 2.0 * np.sqrt(h_val * result["B1_hat"] * annualization)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result["s_hat"] = result["sigma_hat"] / result["sigma_c_hat"]

    return result


# ── Full-sample estimates ─────────────────────────────────────────────────────

def compute_full_sample_margins(df: pd.DataFrame, annualization: int = 8760) -> dict:
    """
    Full-sample estimates of sigma, B1, sigma_c, s for a pool DataFrame.

    Uses the same pathwise (average-of-ratios) B1 estimator.
    """
    f_val = float(df["f"].iloc[0])
    h_val = float(df["h"].iloc[0])

    # Annualized realized vol (mean of squared returns × annualization)
    rv_sq    = df["log_return"] ** 2
    sigma_fs = float(np.sqrt(annualization * rv_sq.mean()))

    # Pathwise B1 (full-sample average of per-hour ratios)
    mask   = df["L_t"] > 0
    ratios = f_val * df.loc[mask, "V_t"] / (np.sqrt(df.loc[mask, "q_t"]) * h_val * df.loc[mask, "L_t"])
    B1_fs  = float(ratios.mean())

    # Annualise per-hour B1 to match annualised sigma_fs
    sc_fs  = float(2.0 * np.sqrt(h_val * B1_fs * annualization))
    s_fs   = float(sigma_fs / sc_fs) if sc_fs > 0 else np.inf

    return {"sigma": sigma_fs, "B1": B1_fs, "sigma_c": sc_fs, "s": s_fs}


# ── Block bootstrap CI ────────────────────────────────────────────────────────

def block_bootstrap_ci(
    df: pd.DataFrame,
    n_boot: int = 2000,
    block_size: int = 168,
    ci_level: float = 0.95,
    seed: int = 42,
    annualization: int = 8760,
) -> dict:
    """
    Circular block bootstrap 95% CI for full-sample participation margin s.

    Resamples hourly tuples (q_t, V_t, L_t, log_return) jointly to preserve
    cross-variable dependence. Block size = 168 hours (one week).

    # Using arch.bootstrap.CircularBlockBootstrap — Context7 confirmed
    """
    f_val = float(df["f"].iloc[0])
    h_val = float(df["h"].iloc[0])

    data = df[["q_t", "V_t", "L_t", "log_return"]].values.copy()
    n    = len(data)

    bs = CircularBlockBootstrap(block_size, data, seed=seed)

    s_boots: list[float] = []
    for pos_data, _ in bs.bootstrap(n_boot):
        sample = pos_data[0]  # shape (n, 4)
        q_s    = sample[:, 0]
        V_s    = sample[:, 1]
        L_s    = sample[:, 2]
        lr_s   = sample[:, 3]

        # Full-sample sigma from boot sample
        sigma_b = float(np.sqrt(annualization * np.mean(lr_s ** 2)))

        # Pathwise B1 from boot sample
        mask_b = L_s > 0
        if mask_b.sum() == 0:
            continue
        ratio_b = f_val * V_s[mask_b] / (np.sqrt(q_s[mask_b]) * h_val * L_s[mask_b])
        B1_b    = float(np.mean(ratio_b))
        sc_b    = 2.0 * np.sqrt(h_val * B1_b * annualization)  # annualise per-hour B1
        s_b     = sigma_b / sc_b if sc_b > 0 else np.inf
        s_boots.append(s_b)

    s_boots_arr = np.array(s_boots)
    alpha       = (1.0 - ci_level) / 2.0
    s_hat_fs    = compute_full_sample_margins(df, annualization)["s"]

    return {
        "s_hat":       s_hat_fs,
        "ci_lower":    float(np.percentile(s_boots_arr, 100 * alpha)),
        "ci_upper":    float(np.percentile(s_boots_arr, 100 * (1.0 - alpha))),
        "boot_samples": s_boots_arr,
    }


# ── Pool panel runner ─────────────────────────────────────────────────────────

def run_pool_panel(pool_configs: list[dict], n_boot: int = 2000, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic data for each pool config and compute full-sample margins + CI.

    Each config dict must contain: pool_id, f, h, sigma, n_hours, L_mean, V_mean, q0.

    Returns summary DataFrame with columns:
        pool_id, f, h, sigma_hat, sigma_c_hat, s_hat, ci_lower, ci_upper.
    """
    rows = []
    for cfg in pool_configs:
        pool_seed = seed + hash(cfg["pool_id"]) % 10000
        df = generate_synthetic_pool_data(
            n_hours          = cfg["n_hours"],
            seed             = pool_seed,
            f                = cfg["f"],
            h                = cfg["h"],
            sigma_annualized = cfg["sigma"],
            q0               = cfg["q0"],
            L_mean           = cfg["L_mean"],
            V_mean           = cfg["V_mean"],
        )
        margins = compute_full_sample_margins(df)
        try:
            ci = block_bootstrap_ci(df, n_boot=n_boot, seed=pool_seed)
            ci_lo, ci_hi = ci["ci_lower"], ci["ci_upper"]
        except Exception:
            ci_lo, ci_hi = np.nan, np.nan

        rows.append({
            "pool_id":    cfg["pool_id"],
            "chain":      cfg.get("chain", ""),
            "protocol":   cfg.get("protocol", ""),
            "f":          cfg["f"],
            "h":          cfg["h"],
            "sigma_hat":  margins["sigma"],
            "sigma_c_hat": margins["sigma_c"],
            "s_hat":      margins["s"],
            "ci_lower":   ci_lo,
            "ci_upper":   ci_hi,
        })
    return pd.DataFrame(rows)
