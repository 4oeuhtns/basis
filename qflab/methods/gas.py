"""Grover adaptive search (GAS) for the K-fund selection, independent of any quantum SDK.

References: Durr and Hoyer, "A quantum algorithm for finding the minimum" (1996); Gilliam,
Woerner and Gonciulea, "Grover adaptive search for constrained polynomial binary
optimization", Quantum 5, 428 (2021).

How it works here:

* The search space is the K-hot bitstrings only. The initial state is a Dicke state and the
  Grover diffusion reflects about it, so the cardinality constraint needs no penalty.
* The objective is computed coherently into a ``value_bits``-qubit two's-complement register
  using phase arithmetic and an inverse Fourier transform. The oracle is a Z on that register's
  sign bit after subtracting the current threshold: it marks portfolios that beat the
  threshold.
* The financial coefficients are integers around 1e10, far too wide for a small register, so
  they are **rounded** to fit. The search is run on the rounded objective; every returned
  portfolio is then scored on the original one and the difference is reported separately.

Counting: one *oracle call* is one compute-compare-uncompute of the objective inside a Grover
iteration. Each circuit execution also costs one classical evaluation of the measured
portfolio.
"""

from __future__ import annotations

import itertools
import math
import time

import numpy as np

from qflab.circuit import Circuit, dicke_state, qft
from qflab.optimizer_worker import energy


def quantize(model: dict, value_bits: int) -> dict:
    """Round the financial objective so that every difference fits the value register.

    The range is bounded from the coefficients alone (largest K linear terms and largest
    K-choose-2 pair terms), without enumerating portfolios.
    """
    if not 3 <= value_bits <= 12:
        raise ValueError("value_bits must be between 3 and 12")
    symbols, k = model["symbols"], model["select_count"]
    linear = [int(model["linear"][s]) for s in symbols]
    pairs = {(p["left"], p["right"]): int(p["coefficient"]) for p in model["pairs"]}
    pair_values = sorted(pairs.values())
    pair_count = k * (k - 1) // 2
    upper = sum(sorted(linear)[-k:]) + sum(v for v in pair_values[-pair_count:] if v > 0)
    lower = sum(sorted(linear)[:k]) + sum(v for v in pair_values[:pair_count] if v < 0)
    # Rounding moves each of the K + K(K-1)/2 terms by at most half a level.
    slack = k + pair_count
    levels = 2 ** (value_bits - 1) - 1 - slack
    if levels < 1:
        raise ValueError("value_bits is too small for this selection size")
    unit = max(1, math.ceil((upper - lower) / levels))
    position = {s: i for i, s in enumerate(symbols)}
    return {
        "value_bits": value_bits,
        "unit_integer": unit,
        "levels": levels,
        "linear": [round(v / unit) for v in linear],
        "pairs": [
            (position[a], position[b], round(v / unit))
            for (a, b), v in pairs.items()
            if round(v / unit)
        ],
        "range_bound_integer": upper - lower,
    }


def quantized_value(rounded: dict, bits) -> int:
    return sum(v for v, b in zip(rounded["linear"], bits, strict=True) if b) + sum(
        v for a, b, v in rounded["pairs"] if bits[a] and bits[b]
    )


def value_circuit(rounded: dict, n: int, offset: int) -> Circuit:
    """Write ``quantized_value(x) + offset`` (mod 2^m, two's complement) into qubits n..n+m-1."""
    m = rounded["value_bits"]
    circuit = Circuit(n + m)
    turn = 2 * math.pi / 2**m

    def phase(value: int, j: int) -> float:
        return (value * 2**j % 2**m) * turn

    for j in range(m):
        target = n + j
        circuit.h(target)
        if phase(offset, j):
            circuit.r1(phase(offset, j), target)
        for i, value in enumerate(rounded["linear"]):
            if phase(value, j):
                circuit.cr1(phase(value, j), i, target)
        for a, b, value in rounded["pairs"]:
            if phase(value, j):
                circuit.ccr1(phase(value, j), a, b, target)
    return qft(circuit, list(range(n, n + m)), inverse=True)


def grover_circuit(rounded: dict, n: int, k: int, threshold: int, rotations: int) -> Circuit:
    """Dicke state, then ``rotations`` Grover iterations marking values below ``threshold``."""
    m = rounded["value_bits"]
    circuit = Circuit(n + m)
    prepare = dicke_state(Circuit(n + m), list(range(n)), k)
    compute = value_circuit(rounded, n, -threshold)
    circuit.extend(prepare)
    for _ in range(rotations):
        circuit.extend(compute)
        circuit.z(n + m - 1)  # sign bit: value - threshold < 0
        circuit.extend(compute.inverse())
        circuit.extend(prepare.inverse())
        for q in range(n):
            circuit.x(q)
        circuit.mcz(0, n - 1, n - 1)
        for q in range(n):
            circuit.x(q)
        circuit.extend(prepare)
    return circuit


def grover_adaptive_search(
    model: dict,
    runner,
    value_bits: int,
    seed: int,
    patience: int = 8,
    max_oracle_calls: int = 400,
    growth: float = 8 / 7,
) -> dict:
    """Minimize the rounded objective over K-hot bitstrings; score the result on the original.

    Stops after ``patience`` consecutive circuit executions without improvement once the
    rotation range has reached its cap, or when the oracle budget is spent.
    """
    started = time.perf_counter()
    symbols, k = model["symbols"], model["select_count"]
    n = len(symbols)
    rounded = quantize(model, value_bits)
    rng = np.random.default_rng(seed)
    space = math.comb(n, k)
    cap = math.sqrt(space)

    def measure(threshold: int, rotations: int):
        counts = runner.sample(grover_circuit(rounded, n, k, threshold, rotations), 1)
        bits = [int(c) for c in next(iter(counts))[:n]]
        return bits, quantized_value(rounded, bits)

    bits, best = measure(0, 0)  # zero rotations: a uniform draw from the Dicke state
    best_bits = bits
    history = [{"rotations": 0, "value": best, "improved": True, "feasible": sum(bits) == k}]
    oracle_calls, executions, limit, misses, deepest = 0, 1, 1.0, 0, 0
    status = "patience_exhausted"
    while True:
        rotations = int(rng.integers(0, math.ceil(limit)))
        if oracle_calls + rotations > max_oracle_calls:
            status = "budget_exhausted"
            break
        bits, value = measure(best, rotations)
        oracle_calls += rotations
        executions += 1
        deepest = max(deepest, rotations)
        improved = sum(bits) == k and value < best
        history.append(
            {
                "rotations": rotations,
                "value": value,
                "improved": improved,
                "feasible": sum(bits) == k,
            }
        )
        if improved:
            best, best_bits, limit, misses = value, bits, 1.0, 0
        else:
            limit = min(limit * growth, cap)
            misses = misses + 1 if limit >= cap else 0
            if misses >= patience:
                break
    selected = [s for s, b in zip(symbols, best_bits, strict=True) if b]
    return {
        "status": status,
        "selected": selected,
        "objective_integer": str(energy(model, selected)),
        "quantized_value": best,
        "value_bits": value_bits,
        "unit_integer": str(rounded["unit_integer"]),
        "oracle_calls": oracle_calls,
        "circuit_executions": executions,
        "classical_evaluations": executions,
        "max_rotations": deepest,
        "search_space": space,
        "history": history,
        "deepest_circuit": runner.resources(grover_circuit(rounded, n, k, best, max(deepest, 1))),
        "per_rotation_gates": len(grover_circuit(rounded, n, k, best, 1).ops)
        - len(grover_circuit(rounded, n, k, best, 0).ops),
        "seconds": time.perf_counter() - started,
    }


def quantization_audit(model: dict, bit_range: range) -> list[dict]:
    """Classical check of what rounding costs, per register width (no circuits involved)."""
    symbols, k = model["symbols"], model["select_count"]
    combos = list(itertools.combinations(range(len(symbols)), k))
    true = [energy(model, [symbols[i] for i in combo]) for combo in combos]
    optimum = min(true)
    rows = []
    for bits in bit_range:
        try:
            rounded = quantize(model, bits)
        except ValueError:
            continue
        values = [
            quantized_value(rounded, [int(i in combo) for i in range(len(symbols))])
            for combo in combos
        ]
        low = min(values)
        ties = [true[i] for i, v in enumerate(values) if v == low]
        rows.append(
            {
                "value_bits": bits,
                "distinct_levels": len(set(values)),
                "tied_at_minimum": len(ties),
                "true_optimum_is_a_minimizer": optimum in ties,
                "worst_gap_among_minimizers_integer": str(max(ties) - optimum),
                "fits_register": max(values) - low <= 2 ** (bits - 1) - 1,
            }
        )
    return rows
