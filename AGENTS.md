# AGENTS.md – Repository Memory for qca-closed-form-reproduction

## Project Overview

Reproduces the analytical, asymptotic, empirical, and game-theoretic results of the paper on boundary-triggered concentrated-liquidity LP profitability in Uniswap v3-style DEXs.

## Repository Structure

```
data/
  synthetic_pool_data.py   # Synthetic on-chain-like hourly panel; calibrated per-pool
src/
  models.py                # Exp 1: exact formulas (ε_w, φ, η, ρ, Π, V*, σ_c, etc.)
  approximations.py        # Exp 2: leading-order w₀*, s*, s_excl, g_max, c_min
  empirical.py             # Exp 3: rolling σ̂, B̂₁, σ̂_c, ŝ, circular block bootstrap
  decision_rules.py        # Exp 4: exclusion table, c_min, σ_IV, analyze_implied_vol_relation
  mfg.py                   # Exp 5: averaged MFG HJB–KFE fixed-point solver
exp/
  exp1_closed_form.py      # Exact formula reproduction
  exp2_approximations.py   # Asymptotic accuracy tests
  exp3_empirical.py        # Rolling margin estimation
  exp4_decision_rules.py   # Exclusion classification, capital viability
  exp5_mfg.py              # MFG equilibrium solver
tests/
  test_exp1.py .. test_exp5.py  # 126 pytest tests, all passing
results/
  RESULTS.md               # Full quantitative summary
  *.csv, *.png             # Outputs from each exp script
```

## Key Architectural Decisions

### 1. No real on-chain data; fully synthetic calibration for Exp 3–4

On-chain hourly Uniswap v3 subgraph data is not available in this environment. Exp 3 and Exp 4 use synthetic hourly panels generated in `data/synthetic_pool_data.py`:

- Price: driftless GBM (log-normal increments, σ_target annualised).
- Volume: log-normal draws around `V_mean` (calibrated per pool).
- Liquidity: OU-mean-reverting `1/L_t` process to avoid L_t=0 and keep ratios stable.

Calibration: for each pool, `V_mean` is set to the value satisfying `σ̂/σ̂_c = s_target` at the full-sample level. The calibration exploits the linear scaling `σ̂_c ∝ √V_mean`. Calibrated values are stored in `POOL_CONFIGS` in `synthetic_pool_data.py`.

### 2. Deterministic seeds via SHA-256 (not Python's hash())

Python's `hash(pool_id)` is non-deterministic across interpreter sessions (controlled by `PYTHONHASHSEED`). All pool seeds must come from:

```python
int(hashlib.sha256(pool_id.encode()).hexdigest(), 16) % 10_000 + BASE_SEED
```

**Never use `hash(pool_id)` for a random seed.** This was the root cause of the cross-run calibration failures that were fixed.

### 3. B̂₁ estimator: pathwise average, NOT ratio of averages

The paper specifies:

```
B̂₁ = (1/H) Σ_{t=1}^H [ f·Vₜ / (√qₜ · h · Lₜ) ]
```

**Do not** replace this with `f·mean(V) / (√mean(q) · h · mean(L))`. The pathwise estimator is stored as `ratio_t` in the panel before averaging, so QA is straightforward.

### 4. Annualisation convention

- Hourly squared log-returns: `r_t² = (ln q_t − ln q_{t-1})²`
- Realised variance (30-day window): `σ̂² = mean(r_t²) × 8760` (hours per year)
- σ̂_c² = `4 · h · B̂₁ · 8760`

This convention is applied uniformly across all pools.

### 5. Import alias pattern for exp4

`decision_rules.py` exports `analyze_implied_vol_relation` (not `test_implied_vol_relation`). In `exp4_decision_rules.py` this is imported as:

```python
from src.decision_rules import analyze_implied_vol_relation as test_implied_vol_relation
```

Keep this alias intact. Do not rename the function in `decision_rules.py` as doing so would break test_exp4.py.

### 6. MFG solver (Exp 5)

- Best-response policy iteration with damping α=0.5.
- Convergence criterion: max policy change < 1e-8 AND ‖μ_new − μ_old‖_∞ < 1e-8.
- Payoff-relevant uniqueness is tested by running from 5 distinct μ initialisations and comparing converged V* and aggregate-L profiles.
- `quantecon` is not installed; MFG solver is fully custom. The stationary distribution for a fixed policy is computed via `scipy.sparse.linalg.eigs`.

## Running the Full Experiment

```bash
# Run all experiments
python exp/exp1_closed_form.py
python exp/exp2_approximations.py
python exp/exp3_empirical.py
python exp/exp4_decision_rules.py
python exp/exp5_mfg.py

# Run all tests
pytest tests/ -v
```

All five scripts produce CSV and PNG outputs under `results/`.

## Paper-Reported Targets (for QA)

| Pool | s_target | κ | s_excl_paper |
|------|----------|---|-------------|
| ETH Uniswap 0.3% | 0.980 | 17.5 bp | 0.9823 |
| ETH Uniswap 0.05% | 1.060 | 5 bp | 0.9922 |
| Base Uniswap 0.05% | 1.050 | 5 bp | 0.9922 |
| Base Aerodrome | 1.150 | 4.76 bp | 0.9924 |
| Base Pancake | 1.180 | 3 bp | 0.9944 |

## Dependencies

Core: numpy, pandas, scipy, matplotlib, seaborn, arch (for circular block bootstrap), pytest.  
Optional extensions listed in requirements if needed: pyarrow, polars, arch, hmmlearn.

## Known Limitations

- Exp 3 / 4 use synthetic data; real results would require indexer access (The Graph, Dune Analytics, or direct RPC node).
- Exp 5 MFG solver uses a minimal 5-tick × 4-width state space for tractability; real mean-field analysis would need a larger grid with jit-compiled transitions (e.g., numba).
- The implied-volatility relation (σ_c ≈ σ_IV when h ≈ 2f) cannot be quantitatively verified without real TVL_tick data from the chain.
