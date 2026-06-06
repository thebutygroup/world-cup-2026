"""
Compare model probabilities to bookmaker prices.

Bookmaker odds bake in a margin (the "overround" / vig): the implied
probabilities across a market sum to more than 1. To compare like with
like you must strip that margin to get the book's *fair* probability, then
set it against the model.

Two de-vig methods are provided:
  * proportional ("multiplicative") -- divide each implied prob by the
    overround. Simple, standard baseline.
  * shin -- Shin (1992) method, which attributes the margin to informed
    ("insider") money and tends to shade favourites less than the
    proportional method. Usually a better fit for outright markets.

Value is then expressed two ways:
  * edge  = model_prob - fair_book_prob
  * EV    = expected profit per 1 unit staked at the *quoted* odds
            = model_prob * (odds - 1) - (1 - model_prob)
A positive EV is a value bet under the model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def implied_probs(decimal_odds: np.ndarray) -> np.ndarray:
    return 1.0 / np.asarray(decimal_odds, dtype=float)


def overround(decimal_odds: np.ndarray) -> float:
    return float(implied_probs(decimal_odds).sum())


def devig_proportional(decimal_odds: np.ndarray) -> np.ndarray:
    p = implied_probs(decimal_odds)
    return p / p.sum()


def devig_shin(decimal_odds: np.ndarray, tol: float = 1e-10, iters: int = 200) -> np.ndarray:
    """
    Shin's method. Solves for the insider-trading proportion z such that the
    recovered fair probabilities sum to 1, then returns them.
    """
    booksum = implied_probs(decimal_odds).sum()
    pi = implied_probs(decimal_odds) / booksum  # normalised quoted probs

    z = 0.0
    for _ in range(iters):
        root = np.sqrt(z**2 + 4.0 * (1.0 - z) * pi**2 / booksum)
        p = (root - z) / (2.0 * (1.0 - z))
        s = p.sum()
        # adjust z toward making s == 1
        new_z = z + (s - 1.0)
        new_z = min(max(new_z, 0.0), 0.2)
        if abs(new_z - z) < tol:
            z = new_z
            break
        z = new_z
    root = np.sqrt(z**2 + 4.0 * (1.0 - z) * pi**2 / booksum)
    p = (root - z) / (2.0 * (1.0 - z))
    return p / p.sum()


def compare_market(
    odds: dict[str, float],
    model_probs: dict[str, float],
    method: str = "shin",
) -> pd.DataFrame:
    """
    odds        : selection -> best available decimal odds
    model_probs : selection -> model probability (e.g. P(win) from the sim)

    Returns a DataFrame sorted by EV with the quoted odds, fair book prob,
    model prob, edge, EV per unit staked, and the model's own fair odds.
    """
    sels = list(odds.keys())
    dec = np.array([odds[s] for s in sels], dtype=float)

    if method == "proportional":
        fair = devig_proportional(dec)
    elif method == "shin":
        fair = devig_shin(dec)
    else:
        raise ValueError("method must be 'proportional' or 'shin'")

    rows = []
    for s, o, fb in zip(sels, dec, fair):
        mp = float(model_probs.get(s, 0.0))
        ev = mp * (o - 1.0) - (1.0 - mp)
        rows.append(
            {
                "selection": s,
                "odds": o,
                "book_implied": 1.0 / o,
                "book_fair": fb,
                "model_prob": mp,
                "model_fair_odds": (1.0 / mp) if mp > 0 else np.inf,
                "edge": mp - fb,
                "EV_per_unit": ev,
            }
        )
    df = pd.DataFrame(rows).set_index("selection")
    df.attrs["overround"] = overround(dec)
    return df.sort_values("EV_per_unit", ascending=False)
