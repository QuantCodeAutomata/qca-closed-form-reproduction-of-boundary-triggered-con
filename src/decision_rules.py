"""
Economic decision rules for concentrated-liquidity LP participation.

Implements exclusion thresholds, capital viability, pool classification,
and the implied-volatility relation from the paper.

All classifications are model-based and valid only under boundary-triggered
provision and fast execution assumptions.

Custom — Context7 found no library equivalent for DEX LP exclusion thresholds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

from src.approximations import s_star as _s_star, g_max as _g_max


def compute_exclusion_threshold(kappa: float) -> float:
    """
    Hard exclusion threshold from proportional costs only.

    Paper formula: s_excl = (1 + (7/4)·(12·κ/7)^(2/3))^(-1/2)

    If observed s >= s_excl, no capital level can restore profitability
    under boundary-triggered provision.

    Parameters
    ----------
    kappa : Proportional transaction cost
    """
    # Paper formula: exclusion threshold (same as approximations.s_excl)
    inner = (12.0 * kappa / 7.0) ** (2.0 / 3.0)
    return (1.0 + (7.0 / 4.0) * inner) ** (-0.5)


def classify_pool(
    s: float,
    kappa: float,
    g: Optional[float] = None,
    g0: Optional[float] = None,
    c: Optional[float] = None,
) -> str:
    """
    Classify a pool under boundary-triggered provision rules.

    Classification labels:
        'excluded_all_capital'    : s >= s_excl(kappa)
        'above_break_even'        : s > 1 and s < s_excl
        'near_free_entry'         : s_star(g) < s <= 1 (requires g)
        'open_if_capital_suff'    : s <= 1 and s < s_excl

    NOTE: All labels are model-based, valid only under boundary-triggered
    provision and fast rebalancing (χ→0).
    """
    # Paper formula: pool classification logic
    se = compute_exclusion_threshold(kappa)

    if s >= se:
        return "excluded_all_capital"
    if s > 1.0:
        return "above_break_even"
    if g is not None and s > _s_star(g):
        return "near_free_entry"
    return "open_if_capital_suff"


def compute_c_min(s: float, g0: float, kappa: float) -> float:
    """
    Minimum viable capital c_min(s).

    Paper formula:
        g_max = (7/12)·[(2/√7)·√(s^(-2) - 1)]³
        c_min = g0 / (g_max - κ)  if g_max > κ, else ∞

    Parameters
    ----------
    s     : Participation margin σ/σ_c
    g0    : Fixed cost (same units as model proportional costs)
    kappa : Proportional cost parameter
    """
    # Paper formula: capital viability
    gm = _g_max(s)
    if gm <= kappa:
        return np.inf
    return g0 / (gm - kappa)


def compute_sigma_IV(
    V_t: np.ndarray,
    q_t: np.ndarray,
    L_bar: float,
    f: float,
    h: float,
) -> float:
    """
    On-chain implied volatility from active-tick TVL and volume.

    Paper formula:
        TVL_tick ≈ √q̄ · h · L̄ / 2
        σ_IV     = 2·f·√(V̄ / TVL_tick)

    Parameters
    ----------
    V_t   : Array of hourly volumes
    q_t   : Array of hourly prices
    L_bar : Active-tick background liquidity
    f     : Fee rate
    h     : Log tick spacing
    """
    # Paper formula: implied volatility from on-chain data
    V_mean  = float(np.mean(V_t))
    q_bar   = float(np.mean(q_t))
    TVL_tick = np.sqrt(q_bar) * h * L_bar / 2.0
    return 2.0 * f * np.sqrt(V_mean / TVL_tick)


def analyze_implied_vol_relation(
    sigma_c_val: float,
    sigma_IV: float,
    h: float,
    f: float,
) -> dict:
    """
    Test the relation σ_c ≈ sqrt(h/(2f)) · σ_IV.

    Under standard volatile-tier grids where h ≈ 2f, this ratio ≈ 1.

    Returns
    -------
    dict with keys:
        sigma_c, sigma_IV, predicted_ratio, observed_ratio,
        close_for_h_approx_2f.
    """
    # Paper formula: implied vol relation
    predicted_ratio = np.sqrt(h / (2.0 * f))
    observed_ratio  = sigma_c_val / sigma_IV if sigma_IV > 0 else np.nan
    close_for_h_2f  = abs(h - 2.0 * f) < 0.001

    return {
        "sigma_c":             sigma_c_val,
        "sigma_IV":            sigma_IV,
        "predicted_ratio":     predicted_ratio,
        "observed_ratio":      observed_ratio,
        "close_for_h_approx_2f": close_for_h_2f,
    }


def build_exclusion_table(pools: list[dict]) -> pd.DataFrame:
    """
    Build a pool-level exclusion classification table.

    Each pool dict must contain: pool_id, kappa, s, f, h.
    Optional: g (for free-entry classification).

    Returns DataFrame with columns:
        pool_id, kappa, s_excl, s_observed, verdict.
    """
    rows = []
    for p in pools:
        se      = compute_exclusion_threshold(p["kappa"])
        s_obs   = p["s"]
        g_val   = p.get("g", None)
        label   = classify_pool(s_obs, p["kappa"], g=g_val)

        # Human-readable verdict
        if label == "excluded_all_capital":
            verdict = "excluded (no capital helps)"
        elif label == "above_break_even":
            verdict = "above break-even (fee-only loss)"
        elif label == "near_free_entry":
            verdict = "near free-entry region"
        else:
            verdict = "open (capital-sufficient)"

        rows.append({
            "pool_id":    p["pool_id"],
            "kappa":      p["kappa"],
            "s_excl":     se,
            "s_observed": s_obs,
            "verdict":    verdict,
            "label":      label,
        })

    return pd.DataFrame(rows)
