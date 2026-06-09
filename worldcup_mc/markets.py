"""
Per-match markets derived from the Dixon-Coles score matrix.

`model.outcome_probs()` already gives 1X2 for a single fixture; this module
turns the *same* score-probability matrix into the rest of the per-match
card you need to predict and price an individual game as the group stage
plays out: totals (over/under), both-teams-to-score, and the most likely
correct scores, plus the model's expected goals.

Everything is read off one matrix, so the markets are mutually consistent
(the 1X2, totals and BTTS all come from the identical scoreline
distribution). Pair the probabilities here with `odds.compare_market()` to
get edge / EV against a book's per-match prices.

NOTE on de-vigging per-match value (same rule as the outright market): to
get a fair book probability you must pass the *complete* market to
`compare_market` -- for 1X2 that's all three of home/draw/away, for a
totals market both over and under. EV_per_unit is correct from the raw
quoted odds regardless.
"""

from __future__ import annotations

import numpy as np

from .model import MatchModel, Surface


def score_matrix(
    model: MatchModel, home: str, away: str, neutral: bool = True,
    surface: "Surface | None" = None,
) -> np.ndarray:
    """The (max_goals+1) x (max_goals+1) matrix P(home=i, away=j) for a
    fixture, including the Dixon-Coles low-score correction and any surface
    effect. Reuses the model's own machinery so it stays consistent with the
    simulator and outcome_probs()."""
    lam_h, lam_a = model.lambdas(home, away, neutral, surface)
    return model._score_matrix(lam_h, lam_a)


def match_1x2(matrix: np.ndarray) -> dict[str, float]:
    """Home / Draw / Away probabilities from a score matrix."""
    return {
        "home": float(np.tril(matrix, -1).sum()),
        "draw": float(np.trace(matrix)),
        "away": float(np.triu(matrix, 1).sum()),
    }


def totals(matrix: np.ndarray, lines=(0.5, 1.5, 2.5, 3.5)) -> dict[float, dict]:
    """
    Over/Under probabilities for each goal line. For half lines there is no
    push; for integer lines the exact-total mass is reported as `push`
    (stake returned on Asian/whole-number totals).
    Returns {line: {"over": p, "under": p, "push": p}}.
    """
    n = matrix.shape[0]
    gi, gj = np.indices(matrix.shape)
    total_goals = gi + gj
    out: dict[float, dict] = {}
    for line in lines:
        over = float(matrix[total_goals > line].sum())
        under = float(matrix[total_goals < line].sum())
        push = float(matrix[total_goals == line].sum()) if float(line).is_integer() else 0.0
        out[float(line)] = {"over": over, "under": under, "push": push}
    return out


def btts(matrix: np.ndarray) -> dict[str, float]:
    """Both-teams-to-score yes/no."""
    yes = float(matrix[1:, 1:].sum())  # home>=1 and away>=1
    return {"yes": yes, "no": float(1.0 - yes)}


def correct_score(matrix: np.ndarray, top: int = 6) -> list[tuple[tuple[int, int], float]]:
    """The `top` most likely exact scorelines as ((home, away), prob)."""
    flat = matrix.ravel()
    ncols = matrix.shape[1]
    idx = np.argsort(flat)[::-1][:top]
    return [((int(i // ncols), int(i % ncols)), float(flat[i])) for i in idx]


def expected_goals(matrix: np.ndarray) -> dict[str, float]:
    """Expected goals for each side and the total, read off the matrix."""
    gi, gj = np.indices(matrix.shape)
    eh = float((matrix * gi).sum())
    ea = float((matrix * gj).sum())
    return {"home": eh, "away": ea, "total": eh + ea}


def predict_fixture(
    model: MatchModel, home: str, away: str, neutral: bool = True,
    surface: "Surface | None" = None, totals_lines=(1.5, 2.5, 3.5),
    correct_score_top: int = 6,
) -> dict:
    """
    Full per-match card for one fixture, all derived from a single score
    matrix so the markets are internally consistent. Returns a dict with
    keys: home, away, neutral, lambdas, expected_goals, '1x2', 'totals',
    'btts', 'correct_score'. Feed the relevant sub-dict to
    odds.compare_market() with the book's prices to get edge/EV.
    """
    lam_h, lam_a = model.lambdas(home, away, neutral, surface)
    mat = model._score_matrix(lam_h, lam_a)
    return {
        "home": home,
        "away": away,
        "neutral": neutral,
        "lambdas": {"home": float(lam_h), "away": float(lam_a)},
        "expected_goals": expected_goals(mat),
        "1x2": match_1x2(mat),
        "totals": totals(mat, lines=totals_lines),
        "btts": btts(mat),
        "correct_score": correct_score(mat, top=correct_score_top),
    }
