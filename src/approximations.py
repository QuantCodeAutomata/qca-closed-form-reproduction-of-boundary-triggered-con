"""
Leading-order closed-form approximations for optimal width and participation thresholds.

Implements the asymptotic formulas from the paper for exp_2 validation.

Custom — Context7 found no library equivalent for these DEX LP-specific
leading-order threshold approximations.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

from src.models import (
    B1_coeff, sigma_c as compute_sigma_c,
    optimal_width, Pi, H_03,
)


def w0_star(g: float, sigma: float, h: float, sc: float) -> float:
    """
    Leading-order approximation for optimal half-width w0*.

    Paper formula (α=1 empirical case):
        w0* = 8·g·σ² / [h·(σ_c² - σ²)]

    Valid only when σ < σ_c.

    Parameters
    ----------
    g     : Total transaction cost
    sigma : Realised volatility
    h     : Log tick spacing
    sc    : Critical volatility σ_c
    """
    # Paper formula: leading-order width approximation
    if sigma >= sc:
        return np.inf
    return 8.0 * g * sigma ** 2 / (h * (sc ** 2 - sigma ** 2))


def s_star(g: float) -> float:
    """
    Leading-order free-entry participation margin approximation s*.

    Paper formula:
        x* = (12·g / 7)^(1/3)
        s* = (1 + 7·x*²/4)^(-1/2)

    At g = 5 bp = 0.0005: x* ≈ 0.095, s* ≈ 0.992.
    """
    # Paper formula: leading-order free-entry approximation
    x_star = (12.0 * g / 7.0) ** (1.0 / 3.0)
    return (1.0 + 7.0 * x_star ** 2 / 4.0) ** (-0.5)


def s_excl(kappa: float) -> float:
    """
    Hard exclusion threshold from proportional costs alone.

    Paper formula: s_excl = (1 + (7/4)·(12·κ/7)^(2/3))^(-1/2)

    If observed s >= s_excl, no capital level can restore profitability
    under the boundary-triggered model.
    """
    # Paper formula: exclusion threshold
    inner = (12.0 * kappa / 7.0) ** (2.0 / 3.0)
    return (1.0 + (7.0 / 4.0) * inner) ** (-0.5)


def g_max(s: float) -> float:
    """
    Maximum admissible total transaction cost given participation margin s.

    Paper formula:
        g_max(s) = (7/12) · [(2/√7)·√(s^(-2) - 1)]³

    Returns 0.0 when s >= 1 (no room for costs at or above the critical ratio).
    """
    # Paper formula: maximum admissible cost
    if s >= 1.0:
        return 0.0
    inner = (2.0 / np.sqrt(7.0)) * np.sqrt(s ** (-2) - 1.0)
    return (7.0 / 12.0) * inner ** 3


def c_min(s: float, g0: float, kappa: float) -> float:
    """
    Minimum viable capital c_min(s).

    Paper formula:
        g_max = g_max(s)
        c_min = g0 / (g_max - κ)  if g_max > κ, else ∞

    Parameters
    ----------
    s     : Observed participation margin σ/σ_c
    g0    : Fixed cost (same units as proportional costs)
    kappa : Proportional cost parameter
    """
    # Paper formula: capital viability
    gm = g_max(s)
    if gm <= kappa:
        return np.inf
    return g0 / (gm - kappa)


def compute_exact_breakeven_s(
    B1: float, h: float, g: float,
    f: float, V_bar: float, L_bar: float, q_bar: float,
    W_max: int = 500,
    tol: float = 1e-6,
) -> float:
    """
    Compute the exact break-even participation margin s_exact via binary search.

    Finds s ∈ (0, 1) such that max_w Π(w) = 0 by varying σ = s · σ_c.

    Parameters
    ----------
    B1, h, g, f, V_bar, L_bar, q_bar : Model parameters
    W_max : Maximum integer width for grid search
    tol   : Binary search tolerance
    """
    # Custom — Context7 found no library equivalent (paper-specific break-even search)
    sc = compute_sigma_c(B1, h)
    t_j = q_bar

    def max_pi(s_val: float) -> float:
        sigma_val = s_val * sc
        result = optimal_width(f, V_bar, L_bar, t_j, h, sigma_val, g, W_max)
        return result["Pi_opt"]

    # Check feasibility: need Pi > 0 for some s
    if max_pi(0.01) <= 0:
        return 0.0   # never profitable

    lo, hi = 0.01, 0.9999
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if max_pi(mid) > 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return (lo + hi) / 2.0


def approximation_errors(
    B1: float,
    h: float,
    f: float,
    V_bar: float,
    L_bar: float,
    q_bar: float,
    g_grid: np.ndarray,
    kappa_grid: np.ndarray,
    sigma_sigma_c_grid: np.ndarray,
    W_max: int = 200,
) -> pd.DataFrame:
    """
    Compute a table of exact vs. approximate quantities over parameter grids.

    Returns a DataFrame with columns:
        g, kappa, s, w_star_exact, w0_star_approx,
        s_star_approx, error_w, error_s_star.
    """
    # Custom — Context7 found no library equivalent
    sc   = compute_sigma_c(B1, h)
    t_j  = q_bar
    rows = []

    for g_val in g_grid:
        for kappa_val in kappa_grid:
            for s_val in sigma_sigma_c_grid:
                sigma_val = s_val * sc
                if sigma_val >= sc:
                    continue

                # Exact optimal width
                result  = optimal_width(f, V_bar, L_bar, t_j, h, sigma_val, g_val, W_max)
                w_exact = result["w_star"]
                pi_opt  = result["Pi_opt"]

                # Leading-order width approximation
                w0 = w0_star(g_val, sigma_val, h, sc)

                # Leading-order free-entry approximation
                ss = s_star(g_val)

                rows.append({
                    "g":              g_val,
                    "kappa":          kappa_val,
                    "s":              s_val,
                    "w_star_exact":   w_exact,
                    "w0_star_approx": w0,
                    "s_star_approx":  ss,
                    "pi_opt":         pi_opt,
                    "error_w":        abs(w_exact - w0) if np.isfinite(w0) else np.nan,
                    "error_s_star":   abs(s_val - ss),
                })

    return pd.DataFrame(rows)
