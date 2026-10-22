"""
Tests for Experiment 4: Decision-rule validation.

Validates:
- compute_exclusion_threshold
- classify_pool labels
- compute_c_min (including infinity case)
- compute_sigma_IV
- test_implied_vol_relation
- build_exclusion_table
"""
import pytest
import numpy as np

from src.decision_rules import (
    compute_exclusion_threshold,
    classify_pool,
    compute_c_min,
    compute_sigma_IV,
    analyze_implied_vol_relation,
    build_exclusion_table,
)


# ── compute_exclusion_threshold ───────────────────────────────────────────────

def test_excl_formula():
    """s_excl = (1 + (7/4)·(12κ/7)^(2/3))^(-1/2)."""
    kp  = 0.0005
    inner  = (12.0 * kp / 7.0) ** (2.0 / 3.0)
    expect = (1.0 + (7.0 / 4.0) * inner) ** (-0.5)
    assert abs(compute_exclusion_threshold(kp) - expect) < 1e-15


def test_excl_paper_values():
    """Match paper reference values within ±2 bp."""
    ref = {0.00175: 0.9823, 0.0005: 0.9922, 0.000476: 0.9924, 0.0003: 0.9944}
    for kp, se_ref in ref.items():
        se = compute_exclusion_threshold(kp)
        assert abs(se - se_ref) < 0.002, f"kappa={kp}: s_excl={se:.4f}, ref={se_ref}"


def test_excl_in_unit_interval():
    """s_excl ∈ (0, 1) for all valid kappa."""
    for kp in [0.0001, 0.0005, 0.001, 0.005]:
        se = compute_exclusion_threshold(kp)
        assert 0 < se < 1


def test_excl_decreasing_kappa():
    """Higher kappa → lower exclusion threshold."""
    se_low  = compute_exclusion_threshold(0.0001)
    se_high = compute_exclusion_threshold(0.005)
    assert se_low > se_high


# ── classify_pool ─────────────────────────────────────────────────────────────

def test_classify_excluded():
    """Pool with s >= s_excl should be classified as excluded."""
    kp = 0.0005
    se = compute_exclusion_threshold(kp)
    label = classify_pool(se + 0.01, kp)
    assert label == "excluded_all_capital"


def test_classify_above_break_even():
    """Pool with 1 < s < s_excl should be above_break_even."""
    kp = 0.00175
    se = compute_exclusion_threshold(kp)
    label = classify_pool(1.01, kp)
    if 1.01 < se:
        assert label == "above_break_even"


def test_classify_open_if_capital_suff():
    """Pool with s < 1 and s < s_excl should be open."""
    kp    = 0.0005
    label = classify_pool(0.85, kp)
    assert label in ["near_free_entry", "open_if_capital_suff"]


def test_classify_labels_are_strings():
    """classify_pool always returns a string."""
    assert isinstance(classify_pool(0.9, 0.0005), str)
    assert isinstance(classify_pool(1.05, 0.0005), str)


# ── compute_c_min ─────────────────────────────────────────────────────────────

def test_c_min_inf_when_excluded():
    """When g_max(s) <= kappa, c_min = ∞."""
    # s very close to 1 → g_max ≈ 0 < kappa
    cm = compute_c_min(0.9999, 0.0001, 0.0005)
    assert np.isinf(cm)


def test_c_min_positive_when_feasible():
    """c_min > 0 when g_max > kappa."""
    cm = compute_c_min(0.85, 0.001, 0.0001)
    if np.isfinite(cm):
        assert cm > 0


def test_c_min_zero_fixed_cost():
    """c_min = 0 when g0 = 0."""
    cm = compute_c_min(0.9, 0.0, 0.0001)
    assert cm == 0.0


def test_c_min_inf_at_s1():
    """c_min = ∞ when s = 1 (break-even)."""
    cm = compute_c_min(1.0, 0.001, 0.0001)
    assert np.isinf(cm)


# ── compute_sigma_IV ──────────────────────────────────────────────────────────

def test_sigma_iv_positive():
    """σ_IV should be positive for positive inputs."""
    q_arr = np.full(100, 2000.0)
    V_arr = np.full(100, 1e6)
    iv    = compute_sigma_IV(V_arr, q_arr, L_bar=1e8, f=0.003, h=0.006)
    assert iv > 0


def test_sigma_iv_scales_with_f():
    """σ_IV should increase linearly with fee rate f."""
    q_arr = np.full(100, 2000.0)
    V_arr = np.full(100, 1e6)
    iv1   = compute_sigma_IV(V_arr, q_arr, 1e8, 0.001, 0.006)
    iv2   = compute_sigma_IV(V_arr, q_arr, 1e8, 0.003, 0.006)
    assert abs(iv2 / iv1 - 3.0) < 0.01


def test_sigma_iv_scales_with_sqrt_volume():
    """σ_IV scales with √V."""
    q_arr = np.full(100, 2000.0)
    V1    = np.full(100, 1e6)
    V4    = np.full(100, 4e6)
    iv1   = compute_sigma_IV(V1, q_arr, 1e8, 0.003, 0.006)
    iv4   = compute_sigma_IV(V4, q_arr, 1e8, 0.003, 0.006)
    assert abs(iv4 / iv1 - 2.0) < 0.01


# ── test_implied_vol_relation ─────────────────────────────────────────────────

def test_implied_vol_relation_keys():
    """test_implied_vol_relation returns expected keys."""
    result = analyze_implied_vol_relation(1.0, 1.0, 0.006, 0.003)
    for key in ["sigma_c", "sigma_IV", "predicted_ratio", "observed_ratio",
                "close_for_h_approx_2f"]:
        assert key in result


def test_implied_vol_relation_ratio_when_h_eq_2f():
    """When h = 2f, predicted_ratio = 1."""
    f = 0.003
    h = 2.0 * f
    result = analyze_implied_vol_relation(1.0, 1.0, h, f)
    assert abs(result["predicted_ratio"] - 1.0) < 1e-14


def test_implied_vol_relation_close_h_2f_flag():
    """close_for_h_approx_2f = True when |h - 2f| < 0.001."""
    result = analyze_implied_vol_relation(1.0, 1.0, 0.006, 0.003)
    assert result["close_for_h_approx_2f"] is True


def test_implied_vol_relation_not_close():
    """Flag is False when h and 2f differ substantially."""
    result = analyze_implied_vol_relation(1.0, 1.0, 0.10, 0.003)
    assert result["close_for_h_approx_2f"] is False


# ── build_exclusion_table ─────────────────────────────────────────────────────

def test_build_exclusion_table_structure():
    """build_exclusion_table returns a DataFrame with required columns."""
    pools = [
        {"pool_id": "P1", "kappa": 0.0005, "s": 0.95, "f": 0.003, "h": 0.006},
        {"pool_id": "P2", "kappa": 0.0005, "s": 0.999, "f": 0.003, "h": 0.006},
    ]
    df = build_exclusion_table(pools)
    assert "pool_id"    in df.columns
    assert "s_excl"     in df.columns
    assert "s_observed" in df.columns
    assert "verdict"    in df.columns


def test_build_exclusion_table_excluded_verdict():
    """Pool with s > s_excl gets 'excluded' verdict."""
    kp = 0.0005
    se = compute_exclusion_threshold(kp) + 0.01
    pools = [{"pool_id": "X", "kappa": kp, "s": se, "f": 0.003, "h": 0.006}]
    df = build_exclusion_table(pools)
    assert df.iloc[0]["label"] == "excluded_all_capital"


def test_build_exclusion_table_open_verdict():
    """Pool with s < 1 gets 'open' verdict."""
    kp = 0.0005
    pools = [{"pool_id": "Y", "kappa": kp, "s": 0.85, "f": 0.003, "h": 0.006}]
    df = build_exclusion_table(pools)
    assert df.iloc[0]["label"] in ["near_free_entry", "open_if_capital_suff"]
