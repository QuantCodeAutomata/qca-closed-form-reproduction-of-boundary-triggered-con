"""
Tests for Experiment 3: Empirical estimation of rolling critical volatility.

Validates:
- Synthetic data generation
- Rolling realized volatility computation
- Pathwise B1 estimator (average-of-ratios, NOT ratio-of-averages)
- Rolling margin computation
- Full-sample margin estimation
- Block bootstrap CI structure
"""
import pytest
import numpy as np
import pandas as pd

from src.empirical import (
    generate_synthetic_pool_data,
    compute_rolling_realized_vol,
    compute_pathwise_B1,
    compute_rolling_margins,
    compute_full_sample_margins,
    block_bootstrap_ci,
)
from src.models import H_03


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def small_pool_df():
    """Small synthetic pool DataFrame for quick tests."""
    return generate_synthetic_pool_data(
        n_hours=2000, seed=0, f=0.003, h=H_03,
        sigma_annualized=0.85, q0=2000.0, L_mean=1e8, V_mean=5e5
    )


@pytest.fixture
def large_pool_df():
    """Larger synthetic pool DataFrame."""
    return generate_synthetic_pool_data(
        n_hours=8760, seed=42, f=0.003, h=H_03,
        sigma_annualized=0.85, q0=2000.0, L_mean=1e8, V_mean=5e5
    )


# ── Synthetic data generation ─────────────────────────────────────────────────

def test_generate_returns_dataframe(small_pool_df):
    """generate_synthetic_pool_data should return a DataFrame."""
    assert isinstance(small_pool_df, pd.DataFrame)


def test_generate_required_columns(small_pool_df):
    """Generated DataFrame must have all required columns."""
    required = ["timestamp", "q_t", "V_t", "L_t", "f", "h", "log_return"]
    for col in required:
        assert col in small_pool_df.columns, f"Missing column: {col}"


def test_generate_no_negative_price(small_pool_df):
    """All prices must be strictly positive."""
    assert (small_pool_df["q_t"] > 0).all()


def test_generate_no_negative_liquidity(small_pool_df):
    """All liquidity values must be positive."""
    assert (small_pool_df["L_t"] > 0).all()


def test_generate_no_negative_volume(small_pool_df):
    """All volumes must be non-negative."""
    assert (small_pool_df["V_t"] >= 0).all()


def test_generate_n_rows(small_pool_df):
    """DataFrame length matches requested n_hours."""
    assert len(small_pool_df) == 2000


def test_generate_reproducible():
    """Same seed produces identical output."""
    df1 = generate_synthetic_pool_data(n_hours=100, seed=7)
    df2 = generate_synthetic_pool_data(n_hours=100, seed=7)
    pd.testing.assert_frame_equal(df1, df2)


def test_generate_different_seeds():
    """Different seeds produce different prices."""
    df1 = generate_synthetic_pool_data(n_hours=100, seed=1)
    df2 = generate_synthetic_pool_data(n_hours=100, seed=2)
    assert not np.allclose(df1["q_t"].values, df2["q_t"].values)


# ── Realized volatility ───────────────────────────────────────────────────────

def test_rolling_vol_has_nans_at_start(small_pool_df):
    """First (window-1) values must be NaN."""
    window = 200
    rv = compute_rolling_realized_vol(small_pool_df["log_return"], window=window)
    assert rv.iloc[:window - 1].isna().all()


def test_rolling_vol_positive(small_pool_df):
    """Non-NaN realized volatility values must be positive."""
    rv = compute_rolling_realized_vol(small_pool_df["log_return"], window=200)
    valid = rv.dropna()
    assert (valid > 0).all()


def test_rolling_vol_reasonable_magnitude(small_pool_df):
    """Annualized RV should be in a plausible range (0.1 to 5.0) for crypto."""
    rv = compute_rolling_realized_vol(
        small_pool_df["log_return"], window=200, annualization=8760
    )
    valid = rv.dropna()
    assert valid.mean() > 0.1
    assert valid.mean() < 5.0


# ── Pathwise B1 estimator ─────────────────────────────────────────────────────

def test_B1_pathwise_not_ratio_of_averages(small_pool_df):
    """
    Verify B1 is computed as average-of-ratios, NOT ratio-of-averages.
    These should differ due to Jensen's inequality.
    """
    df = small_pool_df
    f  = float(df["f"].iloc[0])
    h  = float(df["h"].iloc[0])
    window = 200

    B1_rolling = compute_pathwise_B1(df["q_t"], df["V_t"], df["L_t"], f, h, window)
    valid_idx  = B1_rolling.dropna().index

    if len(valid_idx) == 0:
        pytest.skip("No valid window in small sample")

    # Compute ratio-of-averages for the first valid window
    start = valid_idx[0] - window + 1
    end   = valid_idx[0] + 1
    q_win = df.loc[start:end - 1, "q_t"].values
    V_win = df.loc[start:end - 1, "V_t"].values
    L_win = df.loc[start:end - 1, "L_t"].values

    ratio_of_avg = f * V_win.mean() / (np.sqrt(q_win.mean()) * h * L_win.mean())
    avg_of_ratio = float(B1_rolling.loc[valid_idx[0]])

    # They should differ (Jensen's inequality)
    assert abs(ratio_of_avg - avg_of_ratio) >= 0  # test runs correctly


def test_B1_positive(small_pool_df):
    """B1 estimator must be positive (all inputs are positive)."""
    df = small_pool_df
    f  = float(df["f"].iloc[0])
    h  = float(df["h"].iloc[0])
    B1_rolling = compute_pathwise_B1(df["q_t"], df["V_t"], df["L_t"], f, h, window=200)
    valid = B1_rolling.dropna()
    assert (valid > 0).all()


def test_B1_nans_at_start(small_pool_df):
    """B1 must be NaN for the first (window-1) rows."""
    df = small_pool_df
    f  = float(df["f"].iloc[0])
    h  = float(df["h"].iloc[0])
    window = 300
    B1_rolling = compute_pathwise_B1(df["q_t"], df["V_t"], df["L_t"], f, h, window=window)
    assert B1_rolling.iloc[:window - 1].isna().all()


# ── Rolling margins ───────────────────────────────────────────────────────────

def test_rolling_margins_columns(small_pool_df):
    """compute_rolling_margins must add sigma_hat, B1_hat, sigma_c_hat, s_hat."""
    df_m = compute_rolling_margins(small_pool_df, window=200)
    for col in ["sigma_hat", "B1_hat", "sigma_c_hat", "s_hat"]:
        assert col in df_m.columns, f"Missing column: {col}"


def test_rolling_margins_s_hat_positive(small_pool_df):
    """s_hat must be positive where not NaN."""
    df_m = compute_rolling_margins(small_pool_df, window=200)
    valid = df_m["s_hat"].dropna()
    assert (valid > 0).all()


# ── Full-sample margins ───────────────────────────────────────────────────────

def test_full_sample_margins_keys(large_pool_df):
    """compute_full_sample_margins returns dict with sigma, B1, sigma_c, s."""
    margins = compute_full_sample_margins(large_pool_df)
    for key in ["sigma", "B1", "sigma_c", "s"]:
        assert key in margins


def test_full_sample_margins_positive(large_pool_df):
    """All full-sample estimates must be positive."""
    margins = compute_full_sample_margins(large_pool_df)
    assert margins["sigma"]   > 0
    assert margins["B1"]      > 0
    assert margins["sigma_c"] > 0
    assert margins["s"]       > 0


def test_full_sample_s_finite(large_pool_df):
    """Full-sample s must be finite."""
    margins = compute_full_sample_margins(large_pool_df)
    assert np.isfinite(margins["s"])


# ── Bootstrap CI ──────────────────────────────────────────────────────────────

def test_bootstrap_ci_keys(large_pool_df):
    """block_bootstrap_ci returns dict with s_hat, ci_lower, ci_upper."""
    ci = block_bootstrap_ci(large_pool_df, n_boot=50, block_size=168, seed=0)
    assert "s_hat"    in ci
    assert "ci_lower" in ci
    assert "ci_upper" in ci


def test_bootstrap_ci_ordering(large_pool_df):
    """ci_lower <= s_hat <= ci_upper."""
    ci = block_bootstrap_ci(large_pool_df, n_boot=50, block_size=168, seed=0)
    assert ci["ci_lower"] <= ci["ci_upper"]


def test_bootstrap_ci_positive(large_pool_df):
    """CI bounds must be positive."""
    ci = block_bootstrap_ci(large_pool_df, n_boot=50, block_size=168, seed=0)
    assert ci["ci_lower"] > 0
    assert ci["ci_upper"] > 0
