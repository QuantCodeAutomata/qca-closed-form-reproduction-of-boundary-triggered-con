"""
Tests for Experiment 2: Leading-order approximations.

Validates:
- w0* leading-order width approximation
- s_star free-entry margin
- s_excl exclusion threshold (reference values from the paper)
- g_max and c_min functions
- Paper example: g=5bp, x*≈0.095, s*≈0.992
"""
import pytest
import numpy as np

from src.approximations import w0_star, s_star, s_excl, g_max, c_min
from src.models import B1_coeff, sigma_c, H_03, H_005


H    = H_03
F    = 0.003
V    = 2e7
Q    = 2000.0
L    = 8e8
B1   = B1_coeff(F, V, Q, H, L)
SC   = sigma_c(B1, H)


# ── s_star tests ──────────────────────────────────────────────────────────────

def test_s_star_paper_example_g5bp():
    """At g=5bp, s*≈0.992 and x*≈0.095 per the paper."""
    g  = 0.0005   # 5 bp
    x_star = (12.0 * g / 7.0) ** (1.0 / 3.0)
    ss = s_star(g)
    assert abs(x_star - 0.095) < 0.015, f"x* = {x_star:.4f}"
    assert abs(ss    - 0.992) < 0.015, f"s* = {ss:.4f}"


def test_s_star_in_unit_interval():
    """s* must lie in (0, 1) for positive g."""
    for g in [0.0001, 0.0005, 0.001, 0.005]:
        ss = s_star(g)
        assert 0 < ss < 1, f"s_star({g}) = {ss}"


def test_s_star_decreases_with_g():
    """Higher transaction costs → lower free-entry threshold."""
    ss_low  = s_star(0.0001)
    ss_high = s_star(0.005)
    assert ss_low > ss_high


def test_s_star_formula_direct():
    """s_star = (1 + 7·x*²/4)^(-1/2) matches formula directly."""
    g      = 0.0005
    x_star = (12.0 * g / 7.0) ** (1.0 / 3.0)
    expect = (1.0 + 7.0 * x_star ** 2 / 4.0) ** (-0.5)
    assert abs(s_star(g) - expect) < 1e-15


# ── s_excl tests ──────────────────────────────────────────────────────────────

def test_s_excl_paper_reference_values():
    """
    Reference values from the paper for specified kappa:
        kappa=17.5bp → s_excl≈0.9823
        kappa=5bp    → s_excl≈0.9922
        kappa=4.76bp → s_excl≈0.9924
        kappa=3bp    → s_excl≈0.9944
    """
    ref = {0.00175: 0.9823, 0.0005: 0.9922, 0.000476: 0.9924, 0.0003: 0.9944}
    for kp, se_ref in ref.items():
        se = s_excl(kp)
        assert abs(se - se_ref) < 0.002, f"kappa={kp}: s_excl={se:.4f}, ref={se_ref}"


def test_s_excl_in_unit_interval():
    """s_excl must lie in (0, 1)."""
    for kp in [0.0001, 0.0005, 0.001, 0.005]:
        se = s_excl(kp)
        assert 0 < se < 1


def test_s_excl_less_than_1():
    """Exclusion threshold must be strictly less than 1."""
    assert s_excl(0.0001) < 1.0


def test_s_excl_decreases_with_kappa():
    """Higher kappa → lower exclusion threshold (harder to be profitable)."""
    se_low  = s_excl(0.0001)
    se_high = s_excl(0.005)
    assert se_low > se_high


def test_s_excl_formula():
    """s_excl = (1 + (7/4)·(12κ/7)^(2/3))^(-1/2)."""
    kp = 0.0005
    inner  = (12.0 * kp / 7.0) ** (2.0 / 3.0)
    expect = (1.0 + (7.0 / 4.0) * inner) ** (-0.5)
    assert abs(s_excl(kp) - expect) < 1e-15


# ── g_max tests ───────────────────────────────────────────────────────────────

def test_g_max_zero_at_s1():
    """g_max(s=1) = 0 (no room for costs at break-even)."""
    assert g_max(1.0) == 0.0


def test_g_max_positive_below_1():
    """g_max(s) > 0 for s < 1."""
    for s in [0.7, 0.8, 0.9, 0.99]:
        assert g_max(s) > 0, f"g_max({s}) should be positive"


def test_g_max_decreasing_in_s():
    """g_max decreases as s increases toward 1."""
    g1 = g_max(0.8)
    g2 = g_max(0.9)
    assert g1 > g2


def test_g_max_formula():
    """g_max = (7/12)·[(2/√7)·√(s^(-2) - 1)]³."""
    s = 0.9
    inner  = (2.0 / np.sqrt(7.0)) * np.sqrt(s ** (-2) - 1.0)
    expect = (7.0 / 12.0) * inner ** 3
    assert abs(g_max(s) - expect) < 1e-15


# ── c_min tests ───────────────────────────────────────────────────────────────

def test_c_min_inf_when_g_max_le_kappa():
    """c_min = ∞ when g_max(s) <= kappa."""
    # s very close to 1 → g_max ≈ 0 < any kappa
    cm = c_min(0.9999, 0.0001, 0.0005)
    assert np.isinf(cm)


def test_c_min_finite_when_g_max_gt_kappa():
    """c_min is finite when g_max(s) > kappa."""
    cm = c_min(0.85, 0.0001, 0.0001)
    assert np.isfinite(cm)
    assert cm > 0


def test_c_min_decreasing_in_s():
    """Larger margin (lower s) allows lower minimum capital."""
    cm_low_s  = c_min(0.80, 0.0001, 0.0001)
    cm_high_s = c_min(0.90, 0.0001, 0.0001)
    # Both should be finite here
    if np.isfinite(cm_low_s) and np.isfinite(cm_high_s):
        assert cm_low_s < cm_high_s


def test_c_min_zero_g0():
    """c_min = 0 when g0 = 0 (no fixed cost)."""
    cm = c_min(0.9, 0.0, 0.0001)
    assert cm == 0.0


# ── w0_star tests ─────────────────────────────────────────────────────────────

def test_w0_star_inf_at_sigma_c():
    """w0* = ∞ exactly when σ = σ_c."""
    result = w0_star(0.0005, SC, H, SC)
    assert np.isinf(result)


def test_w0_star_inf_above_sigma_c():
    """w0* = ∞ when σ > σ_c."""
    result = w0_star(0.0005, 2.0 * SC, H, SC)
    assert np.isinf(result)


def test_w0_star_positive():
    """w0* > 0 for σ < σ_c and g > 0."""
    sigma = 0.5 * SC
    result = w0_star(0.0005, sigma, H, SC)
    assert result > 0


def test_w0_star_increases_with_g():
    """Larger cost g → larger optimal width (wider hedge)."""
    s = 0.7 * SC
    w0_low  = w0_star(0.0001, s, H, SC)
    w0_high = w0_star(0.005,  s, H, SC)
    assert w0_high > w0_low


def test_w0_star_formula():
    """w0* = 8·g·σ² / [h·(σ_c² - σ²)]."""
    g = 0.0005
    sigma = 0.7 * SC
    expect = 8.0 * g * sigma ** 2 / (H * (SC ** 2 - sigma ** 2))
    assert abs(w0_star(g, sigma, H, SC) - expect) < 1e-12
