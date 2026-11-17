"""Markov chain Monte Carlo for a small correlated-default model, classical and quantum-assisted.

Plan section 13.2 (R1). The target is a Boltzmann distribution over default indicators,

    pi(z) proportional to exp(-beta * E(z)),   E = -sum_i h_i s_i - sum_{i<j} J_ij s_i s_j,

with ``s_i = +1`` for "no default" (bit 0) and ``s_i = -1`` for "default" (bit 1). Positive
couplings make names in a sector default together, which produces separated modes that
single-flip samplers cross slowly.

Samplers:

* Metropolis with a local (single flip) or uniform proposal, and random-scan Gibbs;
* parallel tempering, as the stronger classical baseline;
* Metropolis with a **quantum proposal** (Layden et al., "Quantum-enhanced Markov chain Monte
  Carlo", Nature 619, 282 (2023)): evolve the current state under a mix of the model
  Hamiltonian and a transverse field for a random time and measure. The evolution operator of
  a real symmetric Hamiltonian is a symmetric matrix, so the proposal is symmetric and the
  plain Metropolis rule keeps the target exact. The Trotter steps used here are palindromic,
  which preserves that symmetry; the tests check it.

The quantum circuit only proposes. Acceptance is classical, and the output is classical
samples. This file imports no quantum SDK; the proposal takes a circuit runner.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from qflab.circuit import Circuit


@dataclass(frozen=True)
class DefaultModel:
    fields: np.ndarray  # h_i
    couplings: np.ndarray  # symmetric J_ij, zero diagonal
    beta: float
    exposures: np.ndarray  # loss if name i defaults

    @property
    def n(self) -> int:
        return len(self.fields)

    def spins(self) -> np.ndarray:
        """Row ``index`` holds s_i = 1 - 2 * bit_i of that state index."""
        index = np.arange(2**self.n)
        return 1 - 2 * ((index[:, None] >> np.arange(self.n)) & 1)

    def energies(self) -> np.ndarray:
        s = self.spins()
        return -(s @ self.fields) - 0.5 * np.einsum("ki,ij,kj->k", s, self.couplings, s)

    def distribution(self) -> np.ndarray:
        weights = np.exp(-self.beta * (self.energies() - self.energies().min()))
        return weights / weights.sum()

    def defaults(self) -> np.ndarray:
        return (1 - self.spins()).sum(axis=1) // 2

    def losses(self) -> np.ndarray:
        return ((1 - self.spins()) // 2) @ self.exposures

    def with_beta(self, beta: float) -> DefaultModel:
        return DefaultModel(self.fields, self.couplings, beta, self.exposures)


def sector_model(
    sectors: list[int],
    bias: float,
    within: float,
    across: float,
    beta: float,
    seed: int,
    jitter: float = 0.15,
) -> DefaultModel:
    """A specified (not fitted) credit testbed: ``sectors[i]`` is the sector of name ``i``.

    ``bias`` > 0 favours survival; ``within`` couples names of one sector; ``across`` links
    consecutive sectors through one pair of names. Small seeded jitter breaks exact symmetry.
    """
    n = len(sectors)
    rng = np.random.default_rng(seed)
    fields = bias * (1 + jitter * rng.uniform(-1, 1, n))
    couplings = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if sectors[i] == sectors[j]:
                couplings[i, j] = couplings[j, i] = within * (1 + jitter * rng.uniform(-1, 1))
    firsts = {s: sectors.index(s) for s in sorted(set(sectors))}
    order = sorted(firsts)
    for a, b in zip(order, order[1:], strict=False):
        i, j = firsts[a], firsts[b]
        couplings[i, j] = couplings[j, i] = across
    return DefaultModel(fields, couplings, beta, 1 + 0.5 * rng.uniform(0, 1, n))


# ------------------------------------------------------------------ transition matrices
def metropolis_matrix(model: DefaultModel, proposal: np.ndarray) -> np.ndarray:
    """Transition matrix of Metropolis with a symmetric proposal matrix."""
    pi = model.distribution()
    ratio = np.minimum(1.0, pi[None, :] / pi[:, None])
    matrix = proposal * ratio
    np.fill_diagonal(matrix, 0.0)
    np.fill_diagonal(matrix, 1 - matrix.sum(axis=1))
    return matrix


def local_proposal(n: int) -> np.ndarray:
    size = 2**n
    matrix = np.zeros((size, size))
    for index in range(size):
        for bit in range(n):
            matrix[index, index ^ (1 << bit)] = 1 / n
    return matrix


def uniform_proposal(n: int) -> np.ndarray:
    return np.full((2**n, 2**n), 1 / 2**n)


def gibbs_matrix(model: DefaultModel) -> np.ndarray:
    pi = model.distribution()
    size, n = len(pi), model.n
    matrix = np.zeros((size, size))
    for index in range(size):
        for bit in range(n):
            other = index ^ (1 << bit)
            matrix[index, other] = pi[other] / (pi[index] + pi[other]) / n
    np.fill_diagonal(matrix, 1 - matrix.sum(axis=1))
    return matrix


def spectral_gap(transition: np.ndarray, pi: np.ndarray) -> float:
    """Absolute spectral gap 1 - max |lambda| over the non-stationary eigenvalues."""
    root = np.sqrt(pi)
    symmetric = root[:, None] * transition / root[None, :]
    values = np.sort(np.abs(np.linalg.eigvalsh((symmetric + symmetric.T) / 2)))
    return float(1 - values[-2])


# -------------------------------------------------------------------- quantum proposal
class QuantumProposal:
    """Proposal by measuring ``exp(-i H t)|s>`` with H = (1-g) a H_model + g sum_i X_i."""

    def __init__(self, model: DefaultModel, runner, seed: int, step: float = 0.8) -> None:
        self.model, self.runner, self.step = model, runner, step
        self.rng = np.random.default_rng(seed)
        norm = math.sqrt(float((model.fields**2).sum() + (np.triu(model.couplings, 1) ** 2).sum()))
        self.alpha = math.sqrt(model.n) / norm
        self.pairs = [
            (i, j, float(model.couplings[i, j]))
            for i in range(model.n)
            for j in range(i + 1, model.n)
            if model.couplings[i, j] != 0
        ]
        self.executions = 0
        self.gates = 0

    def draw_settings(self) -> tuple[float, float]:
        return float(self.rng.uniform(0.25, 0.6)), float(self.rng.uniform(2.0, 20.0))

    def circuit(self, state: int, gamma: float, time: float) -> Circuit:
        n = self.model.n
        steps = max(1, round(time / self.step))
        dt = time / steps
        weight = (1 - gamma) * self.alpha
        circuit = Circuit(n)
        for bit in range(n):
            if state >> bit & 1:
                circuit.x(bit)

        def model_part(fraction: float) -> None:
            # exp(-i * fraction * dt * weight * (-h Z - J ZZ))
            for i in range(n):
                circuit.rz(-2 * fraction * dt * weight * float(self.model.fields[i]), i)
            for i, j, strength in self.pairs:
                circuit.cx(i, j).rz(-2 * fraction * dt * weight * strength, j).cx(i, j)

        model_part(0.5)
        for index in range(steps):
            for i in range(n):
                circuit.rx(2 * dt * gamma, i)
            model_part(1.0 if index < steps - 1 else 0.5)
        return circuit

    def __call__(self, state: int) -> int:
        gamma, time = self.draw_settings()
        circuit = self.circuit(state, gamma, time)
        self.executions += 1
        self.gates += len(circuit.ops)
        bits = next(iter(self.runner.sample(circuit, 1)))
        return sum(int(b) << i for i, b in enumerate(bits))

    def matrix(self, settings: list[tuple[float, float]]) -> np.ndarray:
        """Exact proposal matrix averaged over the given (gamma, time) pairs.

        Uses the simulator statevector for every starting state: privileged access.
        """
        size = 2**self.model.n
        matrix = np.zeros((size, size))
        for gamma, time in settings:
            for state in range(size):
                matrix[state] += np.abs(self.runner.state(self.circuit(state, gamma, time))) ** 2
        return matrix / len(settings)


# ------------------------------------------------------------------------------ chains
def metropolis_chain(model: DefaultModel, propose, steps: int, start: int, rng) -> np.ndarray:
    energies = model.energies()
    chain = np.empty(steps, dtype=int)
    state = start
    for index in range(steps):
        candidate = propose(state)
        if candidate != state and math.log(rng.random()) < -model.beta * (
            energies[candidate] - energies[state]
        ):
            state = candidate
        chain[index] = state
    return chain


def local_move(n: int, rng):
    return lambda state: state ^ (1 << int(rng.integers(n)))


def uniform_move(n: int, rng):
    return lambda state: int(rng.integers(2**n))


def gibbs_chain(model: DefaultModel, steps: int, start: int, rng) -> np.ndarray:
    energies = model.energies()
    chain = np.empty(steps, dtype=int)
    state = start
    for index in range(steps):
        other = state ^ (1 << int(rng.integers(model.n)))
        delta = model.beta * (energies[other] - energies[state])
        if rng.random() < 1 / (1 + math.exp(min(delta, 700.0))):
            state = other
        chain[index] = state
    return chain


def parallel_tempering(
    model: DefaultModel, betas: list[float], steps: int, start: int, rng, swap_every: int = 5
) -> tuple[np.ndarray, int]:
    """Replica exchange with local moves. Returns the target-temperature chain and the number
    of single-replica updates spent (replicas x steps)."""
    if not math.isclose(max(betas), model.beta):
        raise ValueError("The coldest replica must be at the target beta")
    energies = model.energies()
    ladder = sorted(betas)
    states = [start] * len(ladder)
    chain = np.empty(steps, dtype=int)
    for index in range(steps):
        for r, beta in enumerate(ladder):
            candidate = states[r] ^ (1 << int(rng.integers(model.n)))
            if math.log(rng.random()) < -beta * (energies[candidate] - energies[states[r]]):
                states[r] = candidate
        if index % swap_every == 0:
            for r in range(len(ladder) - 1):
                delta = (ladder[r + 1] - ladder[r]) * (
                    energies[states[r + 1]] - energies[states[r]]
                )
                if math.log(rng.random()) < delta:
                    states[r], states[r + 1] = states[r + 1], states[r]
        chain[index] = states[-1]
    return chain, len(ladder) * steps


# -------------------------------------------------------------------------- diagnostics
def autocorrelation_time(series: np.ndarray) -> float:
    """Integrated autocorrelation time with Sokal's automatic window (c = 5)."""
    series = np.asarray(series, dtype=float)
    centered = series - series.mean()
    variance = float(centered @ centered)
    if variance == 0:
        return float("inf")  # the chain never moved in this observable
    size = len(series)
    spectrum = np.fft.rfft(centered, 2 * size)
    correlation = np.fft.irfft(spectrum * np.conj(spectrum))[:size] / variance
    tau = 1.0
    for lag in range(1, size):
        tau += 2 * correlation[lag]
        if lag >= 5 * tau:
            break
    return max(tau, 1.0)


def total_variation(chain: np.ndarray, pi: np.ndarray) -> float:
    empirical = np.bincount(chain, minlength=len(pi)) / len(chain)
    return float(0.5 * np.abs(empirical - pi).sum())
