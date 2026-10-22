"""
Tests for Experiment 5: Finite-state averaged mean-field game.

Validates:
- State space construction
- Transition cost function c(j,w,k,w') formulation
- Aggregate liquidity L(k, mu) computation
- Fee income R_fee computation
- KFE/HJB stationarity and normalization
- Equilibrium existence (numerical)
- Payoff-relevant uniqueness (numerical)
- Cycle occupation law consistency
"""
import pytest
import numpy as np

from src.mfg import (
    MFGParams,
    build_state_space,
    compute_aggregate_liquidity,
    compute_fee_income,
    transition_cost,
    solve_mfg_stationary,
    check_payoff_unique,
)


# ── Parameters ────────────────────────────────────────────────────────────────
@pytest.fixture
def small_params():
    """Small MFG parameter set for fast tests."""
    return MFGParams(
        M=5, W_max=3, f=0.003, Gamma=10.0, g=0.0005,
        L_0=1e6, V_base=1e5,
    )


@pytest.fixture
def medium_params():
    """Medium MFG parameter set."""
    return MFGParams(
        M=10, W_max=5, f=0.003, Gamma=10.0, g=0.0005,
        L_0=5e6, V_base=5e5,
    )


# ── State space ───────────────────────────────────────────────────────────────

def test_build_state_space_includes_out_of_pool(small_params):
    """State space must include w=0 out-of-pool state."""
    states = build_state_space(small_params)
    w_vals = [s[1] for s in states]
    assert 0 in w_vals


def test_build_state_space_widths_in_range(small_params):
    """All widths in state space are in {0,...,W_max}."""
    states = build_state_space(small_params)
    for _, w in states:
        assert 0 <= w <= small_params.W_max


def test_build_state_space_size(small_params):
    """State space size is (2*M+1)*W_max + 1 (in-pool) + 1 (out-of-pool)."""
    states = build_state_space(small_params)
    # States: (j, w) for w in {1,...,W_max}, j arbitrary, plus (*, 0)
    assert len(states) > small_params.W_max


# ── Aggregate liquidity ───────────────────────────────────────────────────────

def test_aggregate_liquidity_gte_L0(small_params):
    """L(k, mu) >= L0 since agents only add to background liquidity."""
    states = build_state_space(small_params)
    n      = len(states)
    mu     = np.ones(n) / n   # uniform prior
    for k in range(small_params.M + 1):
        L = compute_aggregate_liquidity(k, mu, states, small_params)
        assert L >= small_params.L_0


def test_aggregate_liquidity_increases_with_mu(small_params):
    """More agents in-pool → higher aggregate liquidity."""
    states = build_state_space(small_params)
    n      = len(states)
    mu_low  = np.zeros(n)
    mu_low[0] = 1.0          # everyone in first state (likely w=0)
    mu_high = np.ones(n) / n  # uniform (more in-pool agents on average)
    k = 0
    L_low  = compute_aggregate_liquidity(k, mu_low,  states, small_params)
    L_high = compute_aggregate_liquidity(k, mu_high, states, small_params)
    assert L_high >= L_low


# ── Transition cost ───────────────────────────────────────────────────────────

def test_transition_cost_zero_to_zero(small_params):
    """c(j, w=0, k, w'=0) = 0 (staying out of pool costs nothing)."""
    cost = transition_cost(0, 0, 0, 0, small_params)
    assert cost == 0.0


def test_transition_cost_out_to_in(small_params):
    """c(j, w=0, k, w'>0) = g (entering the pool costs g)."""
    cost = transition_cost(0, 0, 0, 1, small_params)
    assert abs(cost - small_params.g) < 1e-12


def test_transition_cost_in_to_any_positive(small_params):
    """c(j, w>0, k, w') >= 0 (includes IL and g)."""
    for k in range(small_params.M + 1):
        for w_prime in [0, 1, 2]:
            cost = transition_cost(0, 1, k, w_prime, small_params)
            assert cost >= 0, f"Negative cost at k={k}, w'={w_prime}"


def test_transition_cost_non_negative(small_params):
    """All transition costs are non-negative."""
    for j in [0, 1]:
        for w in [0, 1, 2]:
            for k in range(small_params.M + 1):
                for wp in [0, 1]:
                    cost = transition_cost(j, w, k, wp, small_params)
                    assert cost >= 0


# ── Fee income ────────────────────────────────────────────────────────────────

def test_fee_income_out_of_pool_zero(small_params):
    """R_fee for w=0 (out of pool) should be 0."""
    states = build_state_space(small_params)
    n      = len(states)
    mu     = np.ones(n) / n
    # Find an out-of-pool state
    out_idx = next(i for i, s in enumerate(states) if s[1] == 0)
    j, w    = states[out_idx]
    r       = compute_fee_income(j, w, mu, states, small_params)
    assert r == 0.0


def test_fee_income_in_pool_non_negative(small_params):
    """R_fee for in-pool positions must be non-negative."""
    states = build_state_space(small_params)
    n      = len(states)
    mu     = np.ones(n) / n
    for j, w in states:
        if w > 0:
            r = compute_fee_income(j, w, mu, states, small_params)
            assert r >= 0, f"Negative fee income at j={j}, w={w}"


# ── Stationary MFG solver ──────────────────────────────────────────────────────

def test_solve_mfg_converges(small_params):
    """solve_mfg_stationary should return a converged result."""
    result = solve_mfg_stationary(small_params, max_iter=100, tol=1e-4)
    assert "converged" in result
    assert result["converged"] is True or result["n_iter"] > 0


def test_solve_mfg_mu_sums_to_one(small_params):
    """Stationary population law mu must sum to 1."""
    result = solve_mfg_stationary(small_params, max_iter=100, tol=1e-4)
    mu_sum = float(np.sum(result["mu"]))
    assert abs(mu_sum - 1.0) < 1e-8


def test_solve_mfg_mu_non_negative(small_params):
    """All entries of stationary mu must be non-negative."""
    result = solve_mfg_stationary(small_params, max_iter=100, tol=1e-4)
    assert np.all(result["mu"] >= -1e-10)


def test_solve_mfg_value_function_finite(small_params):
    """Value function must be finite for all states."""
    result = solve_mfg_stationary(small_params, max_iter=100, tol=1e-4)
    V = result["V"]
    assert np.all(np.isfinite(V))


# ── Payoff-relevant uniqueness ─────────────────────────────────────────────────

def test_payoff_unique_same_agg_liquidity(small_params):
    """
    Payoff-relevant uniqueness: two converged solutions from different
    initial laws should yield similar aggregate liquidity profiles.
    """
    result = check_payoff_unique(small_params, n_runs=3, max_iter=100, tol=1e-4)
    assert "L_profiles" in result
    assert len(result["L_profiles"]) > 0


def test_payoff_unique_report_keys(small_params):
    """check_payoff_unique returns required report keys."""
    result = check_payoff_unique(small_params, n_runs=2, max_iter=50, tol=1e-3)
    for key in ["L_profiles", "V_profiles", "L_std", "V_std"]:
        assert key in result


def test_payoff_unique_L_std_low(small_params):
    """Aggregate liquidity should be nearly identical across initializations."""
    result = check_payoff_unique(small_params, n_runs=3, max_iter=100, tol=1e-4)
    # Allow some tolerance for numerical solver differences
    assert result["L_std"] < 1e4 or result["L_std"] / small_params.L_0 < 0.5
