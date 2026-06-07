"""
Pitch-surface effects: turn venue metadata into a per-match `Surface`, and
calibrate a surface's magnitude from observed matches.

Two ideas, both deliberately conservative:

  * surface_from_risk / load_venue_surfaces map the preliminary 1-5
    `surface_risk` in venues_2026.csv to a Surface, scaled by a global
    `effect_strength`. effect_strength defaults to 0.0 -> every surface is
    identity (no effect). The dimension stays OFF until data justifies
    turning it up.

  * calibrate_surface estimates (pace, compression) from a set of matches
    played on a surface, comparing observed goals to a baseline model's
    expectation, then shrinks hard toward "no effect" because early samples
    (a handful of friendlies) are tiny and noisy. This is the hook for
    gleaning an early read off warm-up games and, later, the first group
    matches.

Mechanism recap (see model.Surface): pace < 1 lowers total goals;
compression in [0,1] pulls the two teams' expectations together, raising the
draw/upset probability. A "bad" pitch is modelled as a variance compressor.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .model import MatchModel, Surface

# Magnitudes at risk=5 and effect_strength=1.0. PLACEHOLDERS to be replaced
# by calibration -- they only set the shape, not a validated size.
_MAX_PACE_DROP = 0.15      # worst pitch scores ~15% fewer goals
_MAX_COMPRESSION = 0.20    # worst pitch shrinks supremacy by ~20%


def surface_from_risk(risk: float, effect_strength: float = 0.0,
                      name: str | None = None) -> Surface:
    """Map a 1-5 risk score to a Surface. effect_strength=0 -> identity."""
    frac = max(0.0, (float(risk) - 1.0) / 4.0) * effect_strength
    return Surface(pace=1.0 - _MAX_PACE_DROP * frac,
                   compression=_MAX_COMPRESSION * frac, name=name)


def load_venue_surfaces(venues_csv: str, effect_strength: float = 0.0
                        ) -> dict[str, Surface]:
    """venue name -> Surface, scaled by effect_strength (default 0 = off)."""
    df = pd.read_csv(venues_csv)
    col = "surface_risk_1to5"
    out: dict[str, Surface] = {}
    for row in df.itertuples(index=False):
        risk = getattr(row, col, 1.0)
        out[row.venue] = surface_from_risk(risk, effect_strength, name=row.venue)
    return out


def calibrate_surface(
    matches: pd.DataFrame,
    model: MatchModel,
    pace_pseudocount: float = 10.0,
    comp_pseudocount: float = 20.0,
    name: str | None = None,
) -> tuple[Surface, dict]:
    """
    Estimate a Surface from observed matches on that pitch.

    `matches` columns: home_team, away_team, home_score, away_score,
    and optional `neutral` (default True). `model` supplies the baseline
    (surface-free) expectation each match would have had.

    Shrinkage: with n matches, the estimate is pulled toward identity by
    w = n / (n + pseudocount). A few friendlies => heavy shrinkage, so the
    early read barely moves the model -- by design.

    Returns (Surface, diagnostics).
    """
    exp_total = []
    exp_diff = []
    obs_total = []
    obs_diff = []
    for r in matches.itertuples(index=False):
        neutral = bool(getattr(r, "neutral", True))
        lam_h, lam_a = model.lambdas(r.home_team, r.away_team, neutral=neutral)
        exp_total.append(lam_h + lam_a)
        exp_diff.append(lam_h - lam_a)
        obs_total.append(r.home_score + r.away_score)
        obs_diff.append(r.home_score - r.away_score)

    n = len(obs_total)
    if n == 0:
        return Surface(name=name), {"n": 0, "note": "no matches"}

    exp_total = np.array(exp_total, float)
    exp_diff = np.array(exp_diff, float)
    obs_total = np.array(obs_total, float)
    obs_diff = np.array(obs_diff, float)

    # --- pace: ratio of observed to expected total goals, shrunk to 1 ---
    pace_hat = obs_total.sum() / max(exp_total.sum(), 1e-9)
    w_pace = n / (n + pace_pseudocount)
    pace = (1.0 - w_pace) * 1.0 + w_pace * pace_hat
    pace = float(np.clip(pace, 0.5, 1.2))

    # --- compression: how much the supremacy spread shrank ---
    # best scale k mapping expected diffs to observed diffs (through origin)
    denom = float(np.dot(exp_diff, exp_diff))
    k_hat = float(np.dot(obs_diff, exp_diff) / denom) if denom > 1e-9 else 1.0
    comp_hat = max(0.0, 1.0 - k_hat)          # k<1 => supremacy compressed
    w_comp = n / (n + comp_pseudocount)
    compression = float(np.clip(w_comp * comp_hat, 0.0, 0.5))

    surf = Surface(pace=pace, compression=compression, name=name)
    info = {
        "n": n,
        "pace_hat_raw": round(pace_hat, 3),
        "pace_shrunk": round(pace, 3),
        "k_hat_raw": round(k_hat, 3),
        "compression_shrunk": round(compression, 3),
        "w_pace": round(w_pace, 3),
        "w_comp": round(w_comp, 3),
        "warning": "tiny-sample estimate; treat as a wide-error-bar prior, not truth",
    }
    return surf, info
