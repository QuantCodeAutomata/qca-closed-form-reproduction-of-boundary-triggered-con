"""
Finite-state averaged mean-field game for strategic liquidity provision in
concentrated-liquidity DEXs.

Implements the coupled HJB-KFE system for the averaged re-centering game.
Numerical solutions provide evidence consistent with the paper's theoretical
existence and payoff-relevant uniqueness claims (this is numerical replication
of theory, not a mathematical proof).

Custom — Context7 found no library equivalent for this specific MFG formulation.
Uses scipy.sparse for linear system assembly.
"""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Any, Optional

import scipy.sparse as sp
import scipy.sparse.linalg as spla

from src.models import (
    epsilon_w_centered,
    il_kernel,
    H_03,
)

logger = logging.getLogger(__name__)


@dataclass
class MFGParams:
    """Parameters for the finite-state averaged mean-field game."""
    M:        int   = 10             # Number of center-tick states
    W_max:    int   = 5              # Maximum half-width
    f:        float = 0.003          # Fee rate
    V_base:   float = 1_000_000.0   # Base volume (uniform across ticks)
    L_0:      float = 1e8            # Background (exogenous) active-tick liquidity
    Gamma:    float = 1.0            # Execution rate
    g:        float = 0.0005         # Transaction cost
    h:        float = field(default_factory=lambda: 60.0 * np.log(1.0001))
    kappa:    float = 0.0005         # Proportional cost
    max_iter: int   = 200            # Maximum policy iterations
    tol:      float = 1e-8           # Fixed-point convergence tolerance
    damping:  float = 0.5            # Damping factor for fixed-point iteration
    q_ref:    float = 2000.0         # Reference price for tick 0


class MFGSolver:
    """
    Solver for the averaged re-centering mean-field game on a finite state space.

    State space: (j, w) where j ∈ {0,...,M-1} (center tick) and w ∈ {0,...,W_max}
    (half-width; w=0 denotes the out-of-pool state).

    The averaged game uses the stationary active-tick law π_k = 1/(M+1)
    for k ∈ {0,...,M}.
    """

    def __init__(self, params: MFGParams) -> None:
        self.p   = params
        self._build_state_space()

    # ── State space ───────────────────────────────────────────────────────────

    def _build_state_space(self) -> None:
        """Build (j, w) state index mapping."""
        self.states: list[tuple[int, int]] = []
        for j in range(self.p.M):
            for w in range(self.p.W_max + 1):
                self.states.append((j, w))
        self.n_states  = len(self.states)
        self._idx      = {s: i for i, s in enumerate(self.states)}
        # Active-tick grid {0,...,M}; law is uniform
        self.K         = self.p.M + 1
        self.pi        = np.ones(self.K) / self.K   # π_k = 1/(M+1)

    def _state_index(self, j: int, w: int) -> int:
        """Map (j, w) to integer state index."""
        return self._idx[(j % self.p.M, w)]

    def _all_states(self) -> list[tuple[int, int]]:
        return self.states

    def _tick_price(self, j: int) -> float:
        """t_j = q_ref · exp(j·h)."""
        return self.p.q_ref * np.exp(j * self.p.h)

    def _volume(self, k: int) -> float:
        """Uniform volume across ticks."""
        return self.p.V_base

    def _epsilon_w_jw(self, j: int, w: int) -> float:
        """Centered ε_w for state (j, w); returns 0 for w=0."""
        if w == 0:
            return 0.0
        return epsilon_w_centered(self._tick_price(j), w, self.p.h)

    def _ell(self, j: int, w: int) -> float:
        """Liquidity per unit capital; 0 for out-of-pool."""
        if w == 0:
            return 0.0
        eps = self._epsilon_w_jw(j, w)
        return 1.0 / eps if eps > 0 else 0.0

    def _aggregate_liquidity(self, k: int, mu: np.ndarray) -> float:
        """
        L(k, μ) = L_0 + Σ_{(j',w'): w'>0} μ(j',w') · ℓ(j',w')
        """
        total = self.p.L_0
        for i, (j, w) in enumerate(self.states):
            if w > 0:
                total += mu[i] * self._ell(j, w)
        return total

    def _transition_cost(self, j: int, w: int, k: int, w_prime: int) -> float:
        """
        Transition cost c(j, w, k, w').
            - w > 0 → rebalancing: IL + g
            - w = 0, w' > 0 → entering pool: g
            - w = 0, w' = 0 → stay out: 0
        """
        # Paper formula: transition cost function
        if w == 0 and w_prime == 0:
            return 0.0
        if w == 0 and w_prime > 0:
            return self.p.g
        # w > 0: compute IL cost at tick k
        t_j  = self._tick_price(j)
        t_k  = self._tick_price(k % self.p.M)   # active price
        a    = self._tick_price(j) * np.exp(-w * self.p.h)
        b    = self._tick_price(j) * np.exp( w * self.p.h)
        eps  = self._epsilon_w_jw(j, w)
        il   = il_kernel(t_k, t_j, a, b, eps=eps) if eps > 0 else 0.0
        return il + self.p.g

    def _fee_income(self, j: int, w: int, mu: np.ndarray) -> float:
        """
        R_fee(j, w, μ) = Σ_k π_k · f·V(k)·ℓ(j,w) / L(k,μ)
        Only non-zero for w > 0.
        """
        if w == 0:
            return 0.0
        ell_jw = self._ell(j, w)
        income = 0.0
        for k in range(self.K):
            Lk     = self._aggregate_liquidity(k, mu)
            income += self.pi[k] * self.p.f * self._volume(k) * ell_jw / Lk
        return income

    # ── HJB solver ────────────────────────────────────────────────────────────

    def solve_hjb(self, mu: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Solve the stationary HJB using discounted Bellman value iteration.

        V(j,w) = R_fee(j,w,μ) + γ · max_{w'} Σ_k π_k [V(j,w') − c(j,w,k,w')]

        where γ = Gamma/(Gamma+1) is the per-event discount factor ensuring
        contraction (avoids singular linear systems caused by self-loop policies).
        In the averaged re-centering game the centre tick j is unchanged on
        re-centering, so the destination state is always (j, w').

        Returns
        -------
        V      : Value function array (n_states,)
        policy : Policy array (n_states,) — optimal w' for each state
        """
        # Discount factor: Gamma/(Gamma+1) in [0,1), guarantees contraction.
        gamma = self.p.Gamma / (self.p.Gamma + 1.0)
        n     = self.n_states

        # Pre-compute average transition costs c_avg[i, w'] = Σ_k π_k c(j,w,k,w')
        c_avg = np.zeros((n, self.p.W_max + 1), dtype=np.float64)
        for i, (j, w) in enumerate(self.states):
            for wp in range(self.p.W_max + 1):
                c_avg[i, wp] = float(sum(
                    self.pi[k] * self._transition_cost(j, w, k, wp)
                    for k in range(self.K)
                ))

        # Destination state index map: (j, w) → (j, w') stays at same j
        i_dest = np.zeros((n, self.p.W_max + 1), dtype=int)
        for i, (j, _) in enumerate(self.states):
            for wp in range(self.p.W_max + 1):
                i_dest[i, wp] = self._state_index(j, wp)

        # Fee income for each state (income while positioned at current (j,w))
        R_fee = np.array([self._fee_income(j, w, mu) for j, w in self.states])

        V      = np.zeros(n, dtype=np.float64)
        best_w = np.zeros(n, dtype=int)

        # Bellman iterations (guaranteed to converge because γ < 1)
        for _ in range(500):
            V_new = np.empty(n, dtype=np.float64)
            for i in range(n):
                best_val = -np.inf
                best_wp  = 0
                for wp in range(self.p.W_max + 1):
                    # Expected net gain from re-centering to w'
                    val = V[i_dest[i, wp]] - c_avg[i, wp]
                    if val > best_val:
                        best_val = val
                        best_wp  = wp
                best_w[i]  = best_wp
                V_new[i]   = R_fee[i] + gamma * best_val

            diff = float(np.max(np.abs(V_new - V)))
            V    = V_new
            if diff < self.p.tol:
                break

        return V, best_w

    # ── KFE solver ────────────────────────────────────────────────────────────

    def solve_kfe(self, policy: np.ndarray) -> np.ndarray:
        """
        Compute the stationary population law μ given a deterministic policy.

        Under boundary-triggered re-centering each agent at (j,w) moves to
        (j, policy[i]) at the next boundary event. The Markov chain is
        deterministic, so each state leads transitively to an absorbing state
        (where policy[i] == w). The stationary mass at each absorbing state
        equals the fraction of initial states that flow into it.

        Cycles in the policy graph are broken by treating the first node
        encountered as absorbing.

        Returns
        -------
        mu : Stationary distribution array (n_states,), non-negative, sums to 1.
        """
        n = self.n_states

        # Deterministic successor: (j, w) → (j, policy[j,w])
        successor = np.array([
            self._state_index(self.states[i][0], int(policy[i]))
            for i in range(n)
        ], dtype=int)

        # Identify absorbing states (self-loop in the deterministic policy chain)
        is_absorbing = successor == np.arange(n)
        absorbed_by  = np.where(is_absorbing, np.arange(n), -1)

        # Propagate absorption: follow successor links until we reach an absorber
        for _ in range(n + 1):
            changed = False
            for i in range(n):
                if absorbed_by[i] == -1:
                    s = successor[i]
                    if absorbed_by[s] != -1:
                        absorbed_by[i] = absorbed_by[s]
                        changed = True
            if not changed:
                break

        # Break remaining cycles (absorbed_by still -1) by designating the
        # lowest-index node in each cycle as absorbing.
        for i in range(n):
            if absorbed_by[i] != -1:
                continue
            # Trace cycle from i; mark cycle's minimum-index node as absorber
            visited: list[int] = []
            cur = i
            while cur not in visited and absorbed_by[cur] == -1:
                visited.append(cur)
                cur = successor[cur]
            if absorbed_by[cur] != -1:
                cycle_absorber = absorbed_by[cur]
            else:
                # cur is in the cycle already visited
                idx = visited.index(cur)
                cycle_absorber = min(visited[idx:])
            for v in visited:
                absorbed_by[v] = cycle_absorber

        # Distribute unit mass (1/n each state) into absorbing states
        mu = np.zeros(n, dtype=np.float64)
        for i in range(n):
            target = absorbed_by[i]
            if target >= 0:
                mu[target] += 1.0

        total = mu.sum()
        if total > 0.0:
            mu /= total
        else:
            mu = np.ones(n, dtype=np.float64) / n

        return mu

    # ── Main solver ───────────────────────────────────────────────────────────

    def _run_once(self, mu_init: np.ndarray) -> dict[str, Any]:
        """Single fixed-point iteration starting from mu_init."""
        p         = self.p
        mu        = mu_init.copy()
        residuals = []
        converged = False

        for iteration in range(p.max_iter):
            V, policy  = self.solve_hjb(mu)
            mu_new     = self.solve_kfe(policy)

            # Damped update
            mu_proposed = mu_new
            mu_next     = p.damping * mu_proposed + (1.0 - p.damping) * mu
            residual    = float(np.max(np.abs(mu_next - mu)))
            residuals.append(residual)

            logger.debug("iter %d  residual=%.3e", iteration, residual)

            mu = mu_next
            if residual < p.tol:
                converged = True
                break

        return {
            "V_final":     V,
            "mu_final":    mu,
            "policy":      policy,
            "converged":   converged,
            "iterations":  len(residuals),
            "residuals":   residuals,
        }

    def solve(self, n_init: int = 3, seed: int = 42) -> dict[str, Any]:
        """
        Run fixed-point iteration from multiple initializations.

        Tests payoff-relevant uniqueness by comparing aggregate liquidity and
        value functions across converged solutions.

        Returns
        -------
        dict with: V_final, mu_final, policy_final, converged, iterations,
                   residuals, all_mu_finals, all_V_finals.
        """
        rng          = np.random.default_rng(seed)
        all_results  = []

        for init_idx in range(n_init):
            if init_idx == 0:
                # Uniform initialization
                mu_init = np.ones(self.n_states) / self.n_states
            else:
                # Random initialization from Dirichlet
                mu_init = rng.dirichlet(np.ones(self.n_states))
            result = self._run_once(mu_init)
            all_results.append(result)
            logger.info(
                "Init %d: converged=%s iterations=%d",
                init_idx, result["converged"], result["iterations"]
            )

        # Use the result from the first (uniform) initialization as primary
        primary = all_results[0]
        return {
            "V_final":      primary["V_final"],
            "mu_final":     primary["mu_final"],
            "policy_final": primary["policy"],
            "converged":    primary["converged"],
            "iterations":   primary["iterations"],
            "residuals":    primary["residuals"],
            "all_mu_finals": [r["mu_final"] for r in all_results],
            "all_V_finals":  [r["V_final"]  for r in all_results],
            "all_converged": [r["converged"] for r in all_results],
        }

    def get_aggregate_liquidity_profile(self, mu: np.ndarray) -> np.ndarray:
        """Aggregate liquidity L(k, μ) for each active-tick state k."""
        return np.array([self._aggregate_liquidity(k, mu) for k in range(self.K)])

    def comparative_statics(
        self, param_name: str, param_values: list[float]
    ) -> pd.DataFrame:
        """
        Run solver for each parameter value and record equilibrium statistics.

        Illustrates congestion: higher L_0 reduces fee income through dilution.

        Parameters
        ----------
        param_name   : One of 'f', 'L_0', 'g', 'V_base'
        param_values : List of values to iterate over
        """
        original = getattr(self.p, param_name)
        rows     = []

        for val in param_values:
            setattr(self.p, param_name, val)
            self._build_state_space()
            result = self._run_once(np.ones(self.n_states) / self.n_states)
            mu = result["mu_final"]
            V  = result["V_final"]

            # Aggregate liquidity at k=0
            agg_L = self._aggregate_liquidity(0, mu)

            # Mean fee income across in-pool states
            fee_incomes = [
                self._fee_income(j, w, mu)
                for j, w in self.states if w > 0
            ]
            mean_fee = float(np.mean(fee_incomes)) if fee_incomes else 0.0

            # Policy entropy
            policy = result["policy"]
            p_counts = np.bincount(policy, minlength=self.p.W_max + 1).astype(float)
            p_dist   = p_counts / p_counts.sum()
            entropy  = float(-np.sum(p_dist * np.log(p_dist + 1e-15)))

            rows.append({
                param_name:   val,
                "agg_L":      agg_L,
                "mean_fee":   mean_fee,
                "mean_V":     float(np.mean(V)),
                "entropy":    entropy,
                "converged":  result["converged"],
                "iterations": result["iterations"],
            })

        # Restore original parameter
        setattr(self.p, param_name, original)
        self._build_state_space()
        return pd.DataFrame(rows)


# ── Functional API wrappers ──────────────────────────────────────────────────
# These thin wrappers expose the MFGSolver methods as module-level functions
# for ergonomic use in experiments and tests.


def build_state_space(params: MFGParams) -> list[tuple[int, int]]:
    """Return the (j, w) state list for the given MFGParams."""
    solver = MFGSolver(params)
    return solver.states


def compute_aggregate_liquidity(
    k: int, mu: np.ndarray, states: list[tuple[int, int]], params: MFGParams
) -> float:
    """Compute aggregate liquidity L(k, μ)."""
    solver = MFGSolver(params)
    return solver._aggregate_liquidity(k, mu)


def compute_fee_income(
    j: int, w: int, mu: np.ndarray,
    states: list[tuple[int, int]], params: MFGParams
) -> float:
    """Compute expected fee income R_fee(j, w, μ)."""
    solver = MFGSolver(params)
    return solver._fee_income(j, w, mu)


def transition_cost(
    j: int, w: int, k: int, w_prime: int, params: MFGParams
) -> float:
    """Compute transition cost c(j, w, k, w')."""
    solver = MFGSolver(params)
    return solver._transition_cost(j, w, k, w_prime)


def solve_mfg_stationary(
    params: MFGParams,
    max_iter: int = 200,
    tol: float = 1e-8,
) -> dict:
    """
    Solve the stationary MFG from a uniform initialisation.

    Returns dict with: V, mu, policy, converged, n_iter, residuals.
    """
    p = MFGParams(
        M=params.M, W_max=params.W_max, f=params.f,
        V_base=params.V_base, L_0=params.L_0, Gamma=params.Gamma,
        g=params.g, h=params.h, kappa=params.kappa,
        max_iter=max_iter, tol=tol, damping=params.damping,
        q_ref=params.q_ref,
    )
    solver = MFGSolver(p)
    result = solver._run_once(np.ones(solver.n_states) / solver.n_states)
    return {
        "V":        result["V_final"],
        "mu":       result["mu_final"],
        "policy":   result["policy"],
        "converged": result["converged"],
        "n_iter":   result["iterations"],
        "residuals": result["residuals"],
    }


def check_payoff_unique(
    params: MFGParams,
    n_runs: int = 3,
    max_iter: int = 200,
    tol: float = 1e-8,
) -> dict:
    """
    Test payoff-relevant uniqueness across multiple random initialisations.

    Returns dict with: L_profiles, V_profiles, L_std (max over k), V_std (max).
    """
    p = MFGParams(
        M=params.M, W_max=params.W_max, f=params.f,
        V_base=params.V_base, L_0=params.L_0, Gamma=params.Gamma,
        g=params.g, h=params.h, kappa=params.kappa,
        max_iter=max_iter, tol=tol, damping=params.damping,
        q_ref=params.q_ref,
    )
    solver  = MFGSolver(p)
    rng     = np.random.default_rng(42)
    L_profiles: list[np.ndarray] = []
    V_profiles: list[np.ndarray] = []

    for run_i in range(n_runs):
        if run_i == 0:
            mu_init = np.ones(solver.n_states) / solver.n_states
        else:
            mu_init = rng.dirichlet(np.ones(solver.n_states))
        result  = solver._run_once(mu_init)
        mu_sol  = result["mu_final"]
        V_sol   = result["V_final"]
        L_prof  = solver.get_aggregate_liquidity_profile(mu_sol)
        L_profiles.append(L_prof)
        V_profiles.append(V_sol)

    L_arr = np.stack(L_profiles, axis=0)  # (n_runs, K)
    V_arr = np.stack(V_profiles, axis=0)  # (n_runs, n_states)
    return {
        "L_profiles": L_profiles,
        "V_profiles": V_profiles,
        "L_std":      float(np.max(np.std(L_arr, axis=0))),
        "V_std":      float(np.max(np.std(V_arr, axis=0))),
    }
