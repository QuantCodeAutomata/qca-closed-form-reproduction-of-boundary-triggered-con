"""
Tests for Experiment 1: Closed-form LP profitability formulas.

Validates exact paper formulas for:
- Cost-per-unit-liquidity ε
- Impermanent loss kernel
- In-range IL simplification
- Boundary decomposition functions φ(w) and η(w)
- Cycle occupation law ρ_w(n)
- Profit rate Π(w)
- Optimal width w*
- Value function V*(n)
- Critical volatility σ_c
"""
import pytest
import numpy as np

from src.models import (
    H_03, H_005,
    CLModel,
    epsilon_ab, epsilon_w_centered, ell_w,
    il_kernel, il_centered_inrange,
    varphi, eta,
    lambda_eff, sigma_h_param,
    rho_w, rho_w_array,
    R_cyc, R_0, Pi,
    optimal_width, value_function_V_star,
    B1_coeff, sigma_c,
    run_exp1_scenario,
)


# ── Constants for tests ───────────────────────────────────────────────────────
T_J = 2000.0          # reference price
H   = H_03
W   = 10              # representative width
G   = 0.0005
F   = 0.003
V_BAR  = 2e7
L_BAR  = 8e8
SIGMA  = 0.85
KAPPA  = 0.0005


# ── epsilon_ab tests ─────────────────────────────────────────────────────────

def test_epsilon_ab_positive():
    """ε_{a,b}(q) > 0 for q within range [a,b]."""
    a = T_J * np.exp(-W * H)
    b = T_J * np.exp( W * H)
    assert epsilon_ab(T_J, a, b) > 0


def test_epsilon_ab_centered_equivalence():
    """ε_centered = ε_ab(t_j, t_{j-w}, t_{j+w})."""
    a   = T_J * np.exp(-W * H)
    b   = T_J * np.exp( W * H)
    eps_general  = epsilon_ab(T_J, a, b)
    eps_centered = epsilon_w_centered(T_J, W, H)
    assert abs(eps_general - eps_centered) < 1e-10


def test_epsilon_w_formula():
    """ε_w = 2√t_j (1 - exp(-wh/2)) matches explicit computation."""
    expected = 2.0 * np.sqrt(T_J) * (1.0 - np.exp(-W * H / 2.0))
    computed = epsilon_w_centered(T_J, W, H)
    assert abs(expected - computed) < 1e-12


def test_epsilon_w_increases_with_w():
    """ε_w should be increasing in w (wider range → more cost)."""
    eps_vals = [epsilon_w_centered(T_J, w, H) for w in range(1, 20)]
    assert all(b > a for a, b in zip(eps_vals, eps_vals[1:]))


def test_epsilon_w_edge_case_w1():
    """ε_w for w=1 is positive and small relative to w=10."""
    eps1  = epsilon_w_centered(T_J, 1, H)
    eps10 = epsilon_w_centered(T_J, 10, H)
    assert eps1 > 0
    assert eps10 > eps1


def test_epsilon_w_extreme_w():
    """ε_w must be positive even for very large w."""
    eps = epsilon_w_centered(T_J, 200, H)
    assert eps > 0


def test_ell_w_positive():
    """Liquidity per unit capital ℓ_w = 1/ε_w must be positive."""
    ell = ell_w(T_J, W, H)
    assert ell > 0


# ── IL kernel tests ───────────────────────────────────────────────────────────

def test_il_kernel_zero_at_center():
    """IL is 0 when q' = q (no price move)."""
    a   = T_J * np.exp(-W * H)
    b   = T_J * np.exp( W * H)
    eps = epsilon_w_centered(T_J, W, H)
    il  = il_kernel(T_J, T_J, a, b, eps=eps)
    assert abs(il) < 1e-12


def test_il_kernel_non_negative():
    """Impermanent loss should be non-negative for boundary price moves."""
    a = T_J * np.exp(-W * H)
    b = T_J * np.exp( W * H)
    eps = epsilon_w_centered(T_J, W, H)
    # Move to upper boundary
    il_upper = il_kernel(b, T_J, a, b, eps=eps)
    # Move to lower boundary
    il_lower = il_kernel(a, T_J, a, b, eps=eps)
    assert il_upper >= 0
    assert il_lower >= 0


def test_il_centered_inrange_formula():
    """IL(w, n) = (1 - exp(nh/2))² / (2(1 - exp(-wh/2))) for |n| <= w."""
    for n in range(-W, W + 1):
        il = il_centered_inrange(W, n, H)
        expected = (1.0 - np.exp(n * H / 2.0)) ** 2 / (2.0 * (1.0 - np.exp(-W * H / 2.0)))
        assert abs(il - expected) < 1e-12, f"Failed at n={n}"


def test_il_centered_inrange_n0():
    """IL at offset n=0 (centered position) should be 0."""
    il = il_centered_inrange(W, 0, H)
    assert abs(il) < 1e-12


def test_il_centered_inrange_positive_offsets():
    """IL(w, n) should be positive for non-zero offsets."""
    for n in list(range(-W, 0)) + list(range(1, W + 1)):
        il = il_centered_inrange(W, n, H)
        assert il > 0, f"IL should be positive for n={n}"


# ── Boundary decomposition ─────────────────────────────────────────────────────

def test_varphi_positive():
    """φ(w) must be positive for w >= 1."""
    for w in [1, 2, 5, 10, 20]:
        assert varphi(w, H) > 0


def test_eta_positive():
    """η(w) must be positive for w >= 1."""
    for w in [1, 2, 5, 10, 20]:
        assert eta(w, H) > 0


def test_varphi_formula():
    """φ(w) = (exp(wh/2) - 1)·cosh(wh/2)."""
    x      = W * H / 2.0
    expect = (np.exp(x) - 1.0) * np.cosh(x)
    assert abs(varphi(W, H) - expect) < 1e-12


def test_eta_formula():
    """η(w) = (exp(wh/2) - 1)·sinh(wh/2)."""
    x      = W * H / 2.0
    expect = (np.exp(x) - 1.0) * np.sinh(x)
    assert abs(eta(W, H) - expect) < 1e-12


def test_varphi_monotone_in_w():
    """φ(w) should be increasing in w."""
    phi_vals = [varphi(w, H) for w in range(1, 15)]
    assert all(b > a for a, b in zip(phi_vals, phi_vals[1:]))


# ── Cycle occupation law ───────────────────────────────────────────────────────

def test_rho_w_sums_to_one():
    """ρ_w(n) must sum exactly to 1 over {-w,...,w}."""
    for w in [1, 2, 5, 10, 20, 50, 100]:
        rho = rho_w_array(w)
        total = float(np.sum(rho))
        assert abs(total - 1.0) < 1e-12, f"ρ_w({w}) sum = {total}"


def test_rho_w_non_negative():
    """ρ_w(n) must be non-negative."""
    rho = rho_w_array(W)
    assert np.all(rho >= 0)


def test_rho_w_triangular_shape():
    """ρ_w(n) = (w - |n|)/w² has triangular shape peaking at n=0."""
    rho = rho_w_array(W)
    ns  = np.arange(-W, W + 1)
    peak_idx = np.argmax(rho)
    assert ns[peak_idx] == 0


def test_rho_w_symmetric():
    """ρ_w(n) must be symmetric around n=0."""
    rho = rho_w_array(W)
    assert np.allclose(rho, rho[::-1])


def test_rho_w_formula():
    """ρ_w(n) = (w - |n|)/w² directly matches rho_w() function."""
    for n in range(-W, W + 1):
        rho_n    = rho_w(n, W)
        expected = (W - abs(n)) / (W ** 2)
        assert abs(rho_n - expected) < 1e-15


def test_rho_w_zero_at_boundary():
    """ρ_w(±w) = 0."""
    assert rho_w(-W, W) == 0.0
    assert rho_w( W, W) == 0.0


# ── Profit rate ────────────────────────────────────────────────────────────────

def test_R_cyc_positive():
    """Cycle income R_cyc must be positive for valid parameters."""
    rc = R_cyc(W, F, V_BAR, L_BAR, T_J, H)
    assert rc > 0


def test_R_0_positive():
    """Cost rate R_0 must be positive."""
    r0 = R_0(W, SIGMA, H, G, T_J)
    assert r0 > 0


def test_Pi_formula():
    """Π(w) = R_cyc(w) - R_0(w)."""
    pi_val   = Pi(W, F, V_BAR, L_BAR, T_J, H, SIGMA, G)
    expected = R_cyc(W, F, V_BAR, L_BAR, T_J, H) - R_0(W, SIGMA, H, G, T_J)
    assert abs(pi_val - expected) < 1e-15


def test_Pi_decreasing_for_large_w():
    """For w >> w*, R_0 dominates and Π is less than at the optimum."""
    opt    = optimal_width(F, V_BAR, L_BAR, T_J, H, SIGMA, G, W_max=500)
    w_star = opt["w_star"]
    pi_opt = opt["Pi_opt"]
    # w much larger than w* must have lower Pi (R_0 grows super-exponentially)
    w_large   = min(w_star + 100, 500)
    pi_large  = Pi(w_large, F, V_BAR, L_BAR, T_J, H, SIGMA, G)
    assert pi_large < pi_opt, (
        f"Pi({w_large})={pi_large:.4e} should be < Pi(w*={w_star})={pi_opt:.4e}"
    )


# ── Optimal width ─────────────────────────────────────────────────────────────

def test_optimal_width_interior():
    """w* should be an interior maximum for σ < σ_c."""
    B1 = B1_coeff(F, V_BAR, T_J, H, L_BAR)
    sc = sigma_c(B1, H)
    if SIGMA < sc:
        opt = optimal_width(F, V_BAR, L_BAR, T_J, H, SIGMA, G, W_max=200)
        assert opt["w_star"] >= 1
        assert opt["Pi_opt"] > 0


def test_optimal_width_at_high_sigma():
    """For σ >> σ_c, max profit should be negative or zero."""
    B1 = B1_coeff(F, V_BAR, T_J, H, L_BAR)
    sc = sigma_c(B1, H)
    opt = optimal_width(F, V_BAR, L_BAR, T_J, H, 3.0 * sc, G, W_max=50)
    assert opt["Pi_opt"] <= 0.0


def test_optimal_width_is_integer():
    """w* must be a positive integer."""
    opt = optimal_width(F, V_BAR, L_BAR, T_J, H, SIGMA, G, W_max=50)
    assert isinstance(opt["w_star"], int)
    assert opt["w_star"] >= 1


# ── Value function ─────────────────────────────────────────────────────────────

def test_value_function_quadratic_shape():
    """V*(n) is quadratic in n (downward-opening due to costs)."""
    n_arr = np.arange(-W, W + 1, dtype=float)
    V     = value_function_V_star(n_arr, W, H, G)
    # Quadratic: finite differences of second differences should be ~constant
    d2 = np.diff(V, n=2)
    assert np.std(d2) < 1e-10


def test_value_function_peak_at_center():
    """V*(n) peaks at n=0 (center of range)."""
    n_arr = np.arange(-W, W + 1, dtype=float)
    V     = value_function_V_star(n_arr, W, H, G)
    peak  = np.argmax(V)
    center_idx = W   # index of n=0
    assert peak == center_idx


def test_value_function_C_in_range():
    """V*(0) = C = g/2 ∈ [0, g]."""
    n_arr = np.arange(-W, W + 1, dtype=float)
    V     = value_function_V_star(n_arr, W, H, G)
    C     = V[W]   # n=0 is at index W
    assert 0 <= C <= G + 1e-14


# ── Critical volatility ───────────────────────────────────────────────────────

def test_sigma_c_formula():
    """σ_c = 2√(B_1·h)."""
    B1 = B1_coeff(F, V_BAR, T_J, H, L_BAR)
    sc = sigma_c(B1, H)
    expected = 2.0 * np.sqrt(B1 * H)
    assert abs(sc - expected) < 1e-15


def test_sigma_c_positive():
    """σ_c must be positive for valid pool parameters."""
    B1 = B1_coeff(F, V_BAR, T_J, H, L_BAR)
    assert sigma_c(B1, H) > 0


def test_above_sigma_c_unprofitable():
    """No width should yield positive profit when σ > σ_c."""
    B1 = B1_coeff(F, V_BAR, T_J, H, L_BAR)
    sc = sigma_c(B1, H)
    opt = optimal_width(F, V_BAR, L_BAR, T_J, H, 2.0 * sc, G, W_max=200)
    assert opt["Pi_opt"] <= 0.0


def test_sigma_c_increases_with_f():
    """σ_c increases with fee rate f (higher fees support higher vol)."""
    B1_low  = B1_coeff(0.0005, V_BAR, T_J, H, L_BAR)
    B1_high = B1_coeff(0.003,  V_BAR, T_J, H, L_BAR)
    sc_low  = sigma_c(B1_low,  H)
    sc_high = sigma_c(B1_high, H)
    assert sc_high > sc_low


# ── Full scenario runner ──────────────────────────────────────────────────────

def test_run_exp1_scenario_keys():
    """run_exp1_scenario returns all required keys."""
    params = CLModel(f=F, h=H, V_bar=V_BAR, L_bar=L_BAR, q_bar=T_J,
                     sigma=SIGMA, g=G, kappa=KAPPA)
    result = run_exp1_scenario(params, W_max=50)
    required_keys = [
        "w_star", "Pi_opt", "sigma_c", "s", "V_star", "n_arr",
        "rho", "Pi_grid", "w_grid", "R_cyc_grid", "R_0_grid",
    ]
    for key in required_keys:
        assert key in result, f"Missing key: {key}"


def test_run_exp1_scenario_rho_sums_to_one():
    """Cycle occupation law in full scenario sums to 1."""
    params = CLModel(f=F, h=H, V_bar=V_BAR, L_bar=L_BAR, q_bar=T_J,
                     sigma=SIGMA, g=G, kappa=KAPPA)
    result = run_exp1_scenario(params, W_max=50)
    assert abs(float(np.sum(result["rho"])) - 1.0) < 1e-12


def test_run_exp1_high_sigma():
    """Full scenario with σ > σ_c should still run without error."""
    params = CLModel(f=F, h=H, V_bar=V_BAR, L_bar=L_BAR, q_bar=T_J,
                     sigma=5.0, g=G, kappa=KAPPA)
    result = run_exp1_scenario(params, W_max=50)
    assert result["Pi_opt"] <= 0.0


# ── Protocol constants ─────────────────────────────────────────────────────────

def test_H_03_value():
    """H_03 should be 60·ln(1.0001) ≈ 0.005999700."""
    expected = 60.0 * np.log(1.0001)
    assert abs(H_03 - expected) < 1e-12


def test_H_005_value():
    """H_005 should be 10·ln(1.0001) ≈ 0.000999950."""
    expected = 10.0 * np.log(1.0001)
    assert abs(H_005 - expected) < 1e-12
