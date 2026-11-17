"""A transparent classical heuristic for the same selection problem QAOA searches."""

from __future__ import annotations

import numpy as np

from qflab.optimizer_worker import energy


def swap_search(model: dict, seed: int, max_evaluations: int) -> dict:
    """First-improvement swap search from a seeded random selection.

    One evaluation scores one portfolio with the canonical financial objective. The search
    stops at a selection no single swap improves (``local_optimum``) or when the declared
    budget runs out (``budget_exhausted``).
    """
    if max_evaluations < 1:
        raise ValueError("The search needs a budget of at least one evaluation")
    symbols, k = model["symbols"], model["select_count"]
    rng = np.random.default_rng(seed)
    current = sorted(rng.choice(symbols, size=k, replace=False).tolist())
    best = energy(model, current)
    evaluations, status, improved = 1, "local_optimum", True
    while improved:
        improved = False
        moves = [(out, new) for out in current for new in symbols if new not in current]
        for index in rng.permutation(len(moves)):
            if evaluations >= max_evaluations:
                status = "budget_exhausted"
                break
            out, new = moves[index]
            candidate = sorted([s for s in current if s != out] + [new])
            score = energy(model, candidate)
            evaluations += 1
            if score < best:
                current, best, improved = candidate, score, True
                break
        if status == "budget_exhausted":
            break
    return {
        "method": "swap_search",
        "status": status,
        "selected": current,
        "objective_integer": str(best),
        "evaluations": evaluations,
        "seed": seed,
    }
