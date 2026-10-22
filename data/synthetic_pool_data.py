"""
Pool configuration presets mimicking the paper's target WETH/USDC pools.

All data generated here is SYNTHETIC and intended for methodology validation only.
Do not interpret synthetic outputs as empirical findings about real pools.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.models import H_03, H_005
from src.empirical import generate_synthetic_pool_data

# ── Pool configurations ───────────────────────────────────────────────────────
# V_mean and L_mean are calibrated so that the reconstructed participation margin
# s = sigma_hat / sigma_c_hat matches the paper's reported values per pool.
# sigma_c_hat = 2·sqrt(h · B1_annual) where B1_annual = 8760 · mean_t(f·V_t/(√q_t·h·L_t)).
# L_mean represents active-tick liquidity (not total TVL).
# Calibration derivation: V_mean/L_mean = sigma_c_target² · sqrt(q0) / (4·8760·f)
# kappa values follow the paper's protocol-specific proportional cost assumptions.

POOL_CONFIGS: list[dict] = [
    {
        # Target: s ≈ 0.98 (paper: ~0.98, near free-entry)
        # V_mean numerically calibrated: seed 2878, scale k=1.3590 applied to 3.2e6
        "pool_id":  "ETH_UNIV3_WETH_USDC_030",
        "chain":    "ethereum",
        "protocol": "uniswap_v3",
        "f":        0.003,
        "h":        H_03,
        "sigma":    0.85,
        "n_hours":  22_000,
        "L_mean":   1e7,
        "V_mean":   4.3489e6,
        "q0":       2000.0,
        "kappa":    0.00175,          # paper: 17.5 bp
    },
    {
        # Target: s ≈ 1.06 (paper: ~1.06, above break-even)
        # V_mean numerically calibrated: seed 4808, scale k=0.9190 applied to 8.21e5
        "pool_id":  "ETH_UNIV3_WETH_USDC_005",
        "chain":    "ethereum",
        "protocol": "uniswap_v3",
        "f":        0.0005,
        "h":        H_005,
        "sigma":    0.85,
        "n_hours":  22_000,
        "L_mean":   5e5,
        "V_mean":   7.5452e5,
        "q0":       2000.0,
        "kappa":    0.0005,           # paper: 5 bp
    },
    {
        # Target: s ≈ 1.05 (paper: ~1.05)
        # V_mean numerically calibrated: seed 7317, scale k=0.9763 applied to 8.36e5
        "pool_id":  "BASE_UNIV3_WETH_USDC_005",
        "chain":    "base",
        "protocol": "uniswap_v3",
        "f":        0.0005,
        "h":        H_005,
        "sigma":    0.85,
        "n_hours":  22_000,
        "L_mean":   5e5,
        "V_mean":   8.1622e5,
        "q0":       2000.0,
        "kappa":    0.0005,
    },
    {
        # Target: s ≈ 1.15 (paper: ~1.15; fee-only, no token emissions)
        # V_mean numerically calibrated: seed 6438, scale k=0.9557 applied to 3.09e5
        "pool_id":  "BASE_AERO_WETH_USDC",
        "chain":    "base",
        "protocol": "aerodrome",
        "f":        0.000452,
        "h":        H_005,
        "sigma":    0.85,
        "n_hours":  18_000,
        "L_mean":   2e5,
        "V_mean":   2.9530e5,
        "q0":       2000.0,
        "kappa":    0.000476,         # paper: 4.76 bp
    },
    {
        # Target: s ≈ 1.18 (posted fee); LP-effective s ≈ 1.44 (paper)
        # V_mean numerically calibrated: seed 6531, scale k=0.7407 applied to 3.31e5
        "pool_id":  "BASE_PANCAKE_WETH_USDC_001",
        "chain":    "base",
        "protocol": "pancakeswap_v3",
        "f":        0.0001,
        "h":        H_005,
        "sigma":    0.85,
        "n_hours":  20_000,
        "L_mean":   5e4,
        "V_mean":   2.4516e5,
        "q0":       2000.0,
        "kappa":    0.0003,           # paper: 3 bp (LP-effective)
        "f_lp":     0.0001 * 0.67,   # 67% of swap fee goes to LPs
    },
]


def _pool_seed(pool_id: str, base_seed: int) -> int:
    """Deterministic per-pool seed offset using SHA-256 (not Python's hash())."""
    import hashlib
    digest = int(hashlib.sha256(pool_id.encode()).hexdigest(), 16)
    return base_seed + digest % 10_000


def load_pool_data(pool_id: str, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic hourly data for a specified pool.

    Parameters
    ----------
    pool_id : One of the IDs in POOL_CONFIGS
    seed    : Random seed for reproducibility

    Returns
    -------
    DataFrame with columns: timestamp, q_t, V_t, L_t, f, h, log_return.
    Annotated with pool metadata in attrs.
    """
    cfg = next((c for c in POOL_CONFIGS if c["pool_id"] == pool_id), None)
    if cfg is None:
        available = [c["pool_id"] for c in POOL_CONFIGS]
        raise ValueError(f"Unknown pool_id '{pool_id}'. Available: {available}")

    pool_seed = _pool_seed(pool_id, seed)
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
    # Attach metadata
    df.attrs.update({k: v for k, v in cfg.items() if k != "pool_id"})
    df.attrs["pool_id"] = pool_id
    return df
