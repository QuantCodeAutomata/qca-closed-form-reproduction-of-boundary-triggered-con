# Boundary-Triggered Concentrated-Liquidity LP Profitability

Closed-form reproduction and numerical validation of a boundary-triggered liquidity-provider profitability model for Uniswap v3-style concentrated-liquidity DEXs under driftless log-price diffusion and fast rebalancing.

## Overview

This repository implements five experiments:

| Exp | Title | Key output |
|-----|-------|-----------|
| 1 | Exact closed-form formulas | Π(w), w\*, V\*(n), σ_c |
| 2 | Asymptotic approximation accuracy | s\*, s_excl, c_min vs exact |
| 3 | Rolling critical volatility reconstruction | ŝ = σ̂/σ̂_c per pool |
| 4 | Exclusion thresholds & capital viability | Pool verdicts, c_min table |
| 5 | Mean-field game: existence & uniqueness | Equilibrium V\*, μ\*, comparative statics |

## Repository Structure

```
data/                   # Synthetic hourly pool panel generation
src/
  models.py             # Exact formulas (ε_w, φ, η, ρ, Π, σ_c, V*)
  approximations.py     # Leading-order w₀*, s*, s_excl, g_max, c_min
  empirical.py          # Rolling B̂₁, σ̂_c, circular block bootstrap
  decision_rules.py     # Exclusion table, capital viability, σ_IV
  mfg.py                # HJB–KFE fixed-point solver
exp/                    # Experiment runner scripts
tests/                  # 126 pytest tests
results/                # Plots, CSVs, RESULTS.md
AGENTS.md               # Persistent session memory
```

## Installation

```bash
pip install numpy pandas scipy matplotlib seaborn arch pytest
```

## Usage

```bash
# Run individual experiments
python exp/exp1_closed_form.py
python exp/exp2_approximations.py
python exp/exp3_empirical.py
python exp/exp4_decision_rules.py
python exp/exp5_mfg.py

# Run test suite
pytest tests/ -v
```

All outputs (CSV tables and PNG plots) are saved to `results/`.

## Key Results

- **σ_c threshold**: σ_c = 2√(B₁h), where B₁ = f V̄ / (√q̄ · h · L̄). Replicated to paper precision.
- **ETH 0.3% pool**: ŝ ≈ 0.980, near the free-entry margin — potentially profitable under boundary-triggered provision.
- **0.05% pools**: ŝ > 1.05, above the break-even critical threshold on a fee-only basis.
- **Exclusion thresholds**: s_excl(κ=17.5bp) = 0.9823, s_excl(κ=5bp) = 0.9922 — matching paper to 4+ decimal places.
- **MFG**: Payoff-relevant uniqueness confirmed numerically across 5 initializations; congestion (fee ∝ 1/L₀) verified.

See [`results/RESULTS.md`](results/RESULTS.md) for the full quantitative summary.

## Methodology

### Core formulas (Exp 1)

Position cost per unit liquidity (centered at tick j, half-width w):

```
ε_w = 2√q_j · (1 − exp(−wh/2))
```

Cycle occupation law (triangular):

```
ρ_w(n) = (w − |n|) / w²,  n ∈ {−w, …, w}
```

Long-run profit rate:

```
Π(w) = fV̄/(L̄ε_w) − (σ/h)²/w² · [½φ(w) + g]
```

Critical volatility:

```
σ_c = 2√(B₁h),  B₁ = fV̄/(√q̄ · h · L̄)
```

### Data (Exp 3–4)

Synthetic hourly panels are generated with deterministic SHA-256 seeds and calibrated V_mean values so that the full-sample participation margin ŝ matches each pool's paper-reported target.

## License

MIT
