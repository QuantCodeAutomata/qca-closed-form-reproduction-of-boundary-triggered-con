"""
Core analytical model for boundary-triggered concentrated-liquidity LP profitability.

Implements exact closed-form results from the paper under driftless log-price diffusion
and fast rebalancing (chi -> 0). All formulas are valid under boundary-triggered
provision and the local-flatness exact-dynamics simplification.

Custom — Context7 found no library equivalent for concentrated-liquidity DEX
boundary-triggered LP model (paper core formulas).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

# ── Protocol tick-spacing constants ─────────────────────────────────────────
H_03  = 60.0 * np.log(1.0001)   # Uniswap v3 0.3% tier  ≈ 0.005999700
H_005 = 10.0 * np.log(1.0001)   # Uniswap v3 0.05% tier ≈ 0.000999950


@dataclass
class CLModel:
    """Parameter container for a concentrated-liquidity pool scenario."""
    f: float       # Fee rate (e.g. 0.003 for 0.3%)
    h: float       # Log tick spacing (H_03 or H_005)
    V_bar: float   # Background traded volume (same numeraire as q_bar)
    L_bar: float   # Background active-tick liquidity
    q_bar: float   # Reference/center price level
    sigma: float   # Annualised log-price volatility
    g: float       # Total proportional + fixed transaction cost per rebalancing
    kappa: float   # Proportional cost component
    g0: float = 0.0   # Fixed cost (USD)
    c: float  = 1.0   # Capital (USD)


# ── Cost-per-unit-liquidity ──────────────────────────────────────────────────

def epsilon_ab(q: float, a: float, b: float) -> float:
    """
    Exact cost-per-unit-liquidity for a position [a, b] evaluated at price q.

    Paper formula: ε_{a,b}(q) = 2√q - √a - q/√b
    """
    # Paper formula: Section on position cost
    return 2.0 * np.sqrt(q) - np.sqrt(a) - q / np.sqrt(b)


def epsilon_w_centered(t_j: float, w: int, h: float) -> float:
    """
    Centered cost-per-unit-liquidity for half-width w and center price t_j.

    Paper formula (centered simplification): ε_w = 2√t_j (1 - exp(-wh/2))
    """
    # Paper formula: centered position, a = t_{j-w}, b = t_{j+w}
    return 2.0 * np.sqrt(t_j) * (1.0 - np.exp(-w * h / 2.0))


def ell_w(t_j: float, w: int, h: float) -> float:
    """
    Supplied liquidity per unit capital: ℓ_w = 1 / ε_w.
    """
    eps = epsilon_w_centered(t_j, w, h)
    assert eps > 0, f"epsilon_w must be positive for w={w}, got {eps}"
    return 1.0 / eps


# ── Impermanent loss kernel ──────────────────────────────────────────────────

def il_kernel(q_prime: float, q: float, a: float, b: float,
              eps: Optional[float] = None) -> float:
    """
    Clipped impermanent-loss kernel for a position [a, b].

    Paper formula:
        q̄  = clip(q,  [a, b])
        q̄' = clip(q', [a, b])
        Θ̂(q', q) = √q̄ + q'/√q̄ - √q̄' - q'/√q̄'
        IL = Θ̂ / ε_{a,b}(q)   (if eps is None, uses epsilon_ab)
    """
    # Paper formula: clipped IL kernel
    q_bar       = float(np.clip(q,       a, b))
    q_bar_prime = float(np.clip(q_prime, a, b))
    theta_hat = (
        np.sqrt(q_bar) + q_prime / np.sqrt(q_bar)
        - np.sqrt(q_bar_prime) - q_prime / np.sqrt(q_bar_prime)
    )
    denom = eps if eps is not None else epsilon_ab(q, a, b)
    if denom == 0.0:
        return 0.0
    return theta_hat / denom


def il_centered_inrange(w: int, n: int, h: float) -> float:
    """
    Simplified in-range impermanent loss for centered position at offset n.

    Valid for |n| <= w.

    Paper formula: IL(w, n) = (1 - exp(n·h/2))² / (2(1 - exp(-w·h/2)))
    """
    # Paper formula: Eq. IL centered in-range
    assert abs(n) <= w, f"Offset |n|={abs(n)} must be <= w={w}"
    numerator   = (1.0 - np.exp(n * h / 2.0)) ** 2
    denominator = 2.0 * (1.0 - np.exp(-w * h / 2.0))
    return numerator / denominator


# ── Boundary decomposition functions ────────────────────────────────────────

def varphi(w: int, h: float) -> float:
    """
    Boundary IL decomposition function φ(w).

    Paper formula: φ(w) = (exp(wh/2) - 1) · cosh(wh/2)
    """
    # Paper formula: boundary decomposition, varphi
    x = w * h / 2.0
    return (np.exp(x) - 1.0) * np.cosh(x)


def eta(w: int, h: float) -> float:
    """
    Boundary IL asymmetry function η(w).

    Paper formula: η(w) = (exp(wh/2) - 1) · sinh(wh/2)
    """
    # Paper formula: boundary decomposition, eta
    x = w * h / 2.0
    return (np.exp(x) - 1.0) * np.sinh(x)


# ── Effective rebalancing rate ───────────────────────────────────────────────

def sigma_h_param(sigma: float, h: float) -> float:
    """σ_h = σ / h  (volatility in units of tick spacing)."""
    return sigma / h


def lambda_eff(sigma: float, h: float, w: int) -> float:
    """
    Effective cycle/rebalancing rate under driftless diffusion.

    Paper formula: λ_eff = σ_h² / w²,  σ_h = σ/h
    """
    # Paper formula: fast-rebalancing limit
    sh = sigma_h_param(sigma, h)
    return sh ** 2 / (w ** 2)


# ── Cycle occupation law ─────────────────────────────────────────────────────

def rho_w(n: int, w: int) -> float:
    """
    Triangular cycle occupation law.

    Paper formula: ρ_w(n) = (w - |n|) / w²   for n ∈ {-w, …, w}
    """
    # Paper formula: stationary cycle occupation law
    assert abs(n) <= w, f"|n|={abs(n)} must be <= w={w}"
    return (w - abs(n)) / (w ** 2)


def rho_w_array(w: int) -> np.ndarray:
    """Return the full ρ_w(n) array for n in {-w,...,w}."""
    ns = np.arange(-w, w + 1)
    return (w - np.abs(ns)) / (w ** 2)


# ── Income and cost rates ────────────────────────────────────────────────────

def R_cyc(w: int, f: float, V_bar: float, L_bar: float,
          t_j: float, h: float) -> float:
    """
    Cycle income rate under local-flatness exact-dynamics simplification.

    Paper formula: R_cyc(w) = f·V̄ / (L̄ · ε_w)
    """
    # Paper formula: local-flatness exact-dynamics income
    eps = epsilon_w_centered(t_j, w, h)
    return f * V_bar / (L_bar * eps)


def R_0(w: int, sigma: float, h: float, g: float, t_j: float) -> float:
    """
    Long-run cost rate under boundary-triggered re-centering.

    Paper formula: R_0(w) = λ_eff · (0.5·φ(w) + g)
    """
    # Paper formula: cost rate
    lam = lambda_eff(sigma, h, w)
    phi = varphi(w, h)
    return lam * (0.5 * phi + g)


def Pi(w: int, f: float, V_bar: float, L_bar: float, t_j: float,
       h: float, sigma: float, g: float) -> float:
    """
    Long-run profit rate: Π(w) = R_cyc(w) - R_0(w).
    """
    # Paper formula: profit rate
    return R_cyc(w, f, V_bar, L_bar, t_j, h) - R_0(w, sigma, h, g, t_j)


# ── Optimal width ────────────────────────────────────────────────────────────

def optimal_width(f: float, V_bar: float, L_bar: float, t_j: float,
                  h: float, sigma: float, g: float,
                  W_max: int = 500) -> dict:
    """
    Numerically compute the optimal width w* by discrete grid search.

    Returns a dict with keys: w_star, Pi_opt, Pi_grid, w_grid.
    Grid search is the correct method since w is integer-valued.
    """
    # Custom — Context7 found no library equivalent (discrete integer grid search)
    ws = np.arange(1, W_max + 1, dtype=int)
    pi_vals = np.array([Pi(int(w), f, V_bar, L_bar, t_j, h, sigma, g) for w in ws])
    idx = int(np.argmax(pi_vals))
    return {
        "w_star":  int(ws[idx]),
        "Pi_opt":  float(pi_vals[idx]),
        "Pi_grid": pi_vals,
        "w_grid":  ws,
    }


# ── Value function ───────────────────────────────────────────────────────────

def value_function_V_star(n_arr: np.ndarray, w_star: int,
                          h: float, g: float) -> np.ndarray:
    """
    Stationary value function V*(n) in the boundary-triggered regime.

    Paper formula:
        V*(n) = -[(φ(w*) + 2g) / (2 w*²)] n² - [η(w*)/(2 w*)] n + C
    where C ∈ [0, g]; we set C = g/2 (midpoint).
    """
    # Paper formula: stationary value function
    phi   = varphi(w_star, h)
    et    = eta(w_star, h)
    C     = g / 2.0
    coeff_n2 = -(phi + 2.0 * g) / (2.0 * w_star ** 2)
    coeff_n1 = -et / (2.0 * w_star)
    return coeff_n2 * n_arr ** 2 + coeff_n1 * n_arr + C


# ── Critical volatility ──────────────────────────────────────────────────────

def B1_coeff(f: float, V_bar: float, q_bar: float, h: float, L_bar: float) -> float:
    """
    Pathwise B_1 coefficient from pool fundamentals.

    Paper formula: B_1 = f·V̄ / (√q̄ · h · L̄)
    """
    # Paper formula: critical volatility derivation
    return f * V_bar / (np.sqrt(q_bar) * h * L_bar)


def sigma_c(B1: float, h: float) -> float:
    """
    Critical volatility above which provision is unprofitable.

    Paper formula: σ_c = 2√(B_1·h)
    """
    # Paper formula: sigma_c definition
    return 2.0 * np.sqrt(B1 * h)


# ── Full scenario runner ─────────────────────────────────────────────────────

def run_exp1_scenario(params: CLModel, W_max: int = 500) -> dict:
    """
    Run the full exp_1 analytical scenario for a CLModel parameter set.

    Returns a dict with all key analytical objects:
        w_star, Pi_opt, sigma_c, s (=sigma/sigma_c),
        V_star (value function array on offset grid),
        rho (cycle occupation law on offset grid),
        Pi_grid, w_grid, R_cyc_grid, R_0_grid,
        epsilon_grid, varphi_grid, eta_grid, lambda_eff_grid.

    Assertion: valid only under boundary-triggered provision and fast rebalancing (χ→0).
    """
    assert W_max >= 1, "W_max must be at least 1"

    t_j = params.q_bar   # centre price = q_bar

    # Compute B1 and sigma_c
    B1   = B1_coeff(params.f, params.V_bar, params.q_bar, params.h, params.L_bar)
    sc   = sigma_c(B1, params.h)
    s    = params.sigma / sc if sc > 0 else np.inf

    # Optimal width
    opt = optimal_width(
        params.f, params.V_bar, params.L_bar, t_j,
        params.h, params.sigma, params.g, W_max
    )
    w_star  = opt["w_star"]
    Pi_opt  = opt["Pi_opt"]
    Pi_grid = opt["Pi_grid"]
    w_grid  = opt["w_grid"]

    # Component grids
    R_cyc_grid    = np.array([R_cyc(int(w), params.f, params.V_bar,
                                    params.L_bar, t_j, params.h) for w in w_grid])
    R_0_grid      = np.array([R_0(int(w), params.sigma, params.h,
                                   params.g, t_j) for w in w_grid])
    epsilon_grid  = np.array([epsilon_w_centered(t_j, int(w), params.h) for w in w_grid])
    varphi_grid   = np.array([varphi(int(w), params.h) for w in w_grid])
    eta_grid      = np.array([eta(int(w), params.h) for w in w_grid])
    lam_grid      = np.array([lambda_eff(params.sigma, params.h, int(w)) for w in w_grid])

    # Value function on offset grid for w_star
    n_arr  = np.arange(-w_star, w_star + 1, dtype=float)
    V_star = value_function_V_star(n_arr, w_star, params.h, params.g)

    # Cycle occupation law
    rho = rho_w_array(w_star)

    # Verification assertions
    rho_sum = float(np.sum(rho))
    assert abs(rho_sum - 1.0) < 1e-12, f"rho_w must sum to 1, got {rho_sum}"
    assert all(epsilon_grid > 0), "All epsilon_w must be positive"

    return {
        "w_star":          w_star,
        "Pi_opt":          Pi_opt,
        "sigma_c":         sc,
        "B1":              B1,
        "s":               s,
        "V_star":          V_star,
        "n_arr":           n_arr,
        "rho":             rho,
        "Pi_grid":         Pi_grid,
        "w_grid":          w_grid,
        "R_cyc_grid":      R_cyc_grid,
        "R_0_grid":        R_0_grid,
        "epsilon_grid":    epsilon_grid,
        "varphi_grid":     varphi_grid,
        "eta_grid":        eta_grid,
        "lambda_eff_grid": lam_grid,
        "params":          params,
    }
