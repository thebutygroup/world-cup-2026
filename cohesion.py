"""
Team "cohesion" signal: shots on target per unit of weekly wage.

Idea (user): a side with a huge aggregate wage bill but few shots on target
may be a collection of individuals not yet gelled as a team. Early on
(friendlies) that disparity is a weak, noisy PRIOR -- an initial nudge, not
a verdict -- which later matches override. We expect poorly-gelled but
talented teams to "take advantage" of their talent as they gel, so the
signal should be small, shrunk hard at low sample, and decay-weighted toward
recent form.

Pipeline:
  1. wages: per-player weekly wage. Missing players -> imputed at the bottom
     `impute_pct` percentile of known wages (user's rule). See SOURCES.csv
     for where to get wages (EA FC/SofIFA on Kaggle is the broad-coverage
     base; Capology for accurate top-end wages).
  2. per match: STWR = (team shots on target) / (sum of on-field weekly wages
     in millions). Higher = more chances created per pound of talent.
  3. roll STWR per team with exponential decay (recent friendlies/games
     weighted more), then shrink toward the cross-team mean by sample size.
  4. convert each team's relative STWR into an attacking-lambda multiplier
     via MatchModel.apply_cohesion. Default sensitivity 0.0 -> no effect.

Also provides fit_sot_to_goals(): the empirical shots-on-target -> goals
relationship, so the strength of this whole signal is measured, not assumed.
Prior from public data: ~0.30 goals per shot on target (~31% conversion).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

GOALS_PER_SOT_PRIOR = 0.30  # ~31% of shots on target are scored (public data)
LN2 = math.log(2.0)


# --------------------------------------------------------------------------
# wages
# --------------------------------------------------------------------------
def load_wages(path: str, wage_col: str = "weekly_wage_gbp",
               name_col: str = "player_name") -> dict[str, float]:
    """player name -> weekly wage. CSV cols default: player_name, weekly_wage_gbp."""
    df = pd.read_csv(path)
    df = df.dropna(subset=[name_col, wage_col])
    return dict(zip(df[name_col], df[wage_col].astype(float)))


def impute_floor(wages: dict[str, float], pct: float = 10.0) -> float:
    """Bottom-`pct` percentile of known wages, used for unknown players."""
    if not wages:
        return 0.0
    return float(np.percentile(list(wages.values()), pct))


def team_wage_bill(lineup: list[str], wages: dict[str, float],
                   floor: float) -> float:
    """Sum weekly wages for an on-field lineup, imputing unknowns at `floor`."""
    return float(sum(wages.get(p, floor) for p in lineup))


# --------------------------------------------------------------------------
# shots on target -> goals (signal-strength check)
# --------------------------------------------------------------------------
def fit_sot_to_goals(matches: pd.DataFrame, sot_col: str = "shots_on_target",
                     goals_col: str = "goals") -> dict:
    """
    Estimate goals as a function of shots on target. Returns the conversion
    slope (goals per SoT), the correlation, and R^2 -- i.e. how much of goal
    variation SoT actually explains. Expect a modest R^2: SoT is correlated
    with goals but far from deterministic, so this is a weak-ish signal.
    """
    s = matches[sot_col].to_numpy(float)
    g = matches[goals_col].to_numpy(float)
    n = len(s)
    if n < 3 or s.std() == 0:
        return {"n": n, "note": "insufficient/degenerate data"}
    # slope through origin (goals ~ c * SoT) and ordinary correlation
    c = float(np.dot(s, g) / np.dot(s, s))
    r = float(np.corrcoef(s, g)[0, 1])
    # R^2 of the through-origin model
    ss_res = float(np.sum((g - c * s) ** 2))
    ss_tot = float(np.sum((g - g.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {
        "n": n,
        "goals_per_sot": round(c, 3),
        "correlation": round(r, 3),
        "r_squared": round(r2, 3),
        "prior_goals_per_sot": GOALS_PER_SOT_PRIOR,
    }


# --------------------------------------------------------------------------
# the cohesion metric + lambda multiplier
# --------------------------------------------------------------------------
def rolling_stwr(
    records: pd.DataFrame,
    asof: str | pd.Timestamp,
    half_life_days: float = 120.0,
    wage_scale: float = 1e6,
    sot_col: str = "shots_on_target",
) -> dict[str, dict]:
    """
    Per-team decay-weighted components for the cohesion signal.

    `records` columns: date, team, wage_bill, and the shots column named by
    `sot_col` (default 'shots_on_target' = chances created; pass
    'shots_on_target_against' for the defensive version). Returns
    team -> {sot, wage, stwr, eff_n}:
      sot   - decay-weighted mean of `sot_col` per match
      wage  - decay-weighted mean on-field wage bill (in `wage_scale` units)
      stwr  - sot / wage (reference only; the multiplier uses the
              wealth-normalised residual, not this ratio)
      eff_n - effective (decayed) match count, used for shrinkage
    """
    asof = pd.Timestamp(asof)
    out: dict[str, dict] = {}
    for team, g in records.groupby("team"):
        age = (asof - pd.to_datetime(g["date"])).dt.days.to_numpy(float)
        w = np.exp(-LN2 * np.clip(age, 0, None) / half_life_days)
        wsum = float(w.sum())
        sot = float(np.sum(w * g[sot_col].to_numpy(float)) / wsum)
        wage = float(np.sum(w * g["wage_bill"].to_numpy(float)) / wsum) / wage_scale
        out[team] = {"sot": sot, "wage": wage,
                     "stwr": (sot / wage if wage > 0 else np.nan), "eff_n": wsum}
    return out


def defensive_stwr(records: pd.DataFrame, asof, **kw) -> dict[str, dict]:
    """Defensive counterpart: shots on target CONCEDED per wage. Expects a
    `shots_on_target_against` column. Feed the result to cohesion_multipliers
    and apply via MatchModel.apply_cohesion(defence_mults=...). Because a
    strong defence concedes FEW shots, it gets a residual < 0 and a multiplier
    < 1, which (applied to the opponent) correctly suppresses their goal rate."""
    return rolling_stwr(records, asof, sot_col="shots_on_target_against", **kw)


def cohesion_multipliers(
    stwr: dict[str, dict],
    sensitivity: float = 0.0,
    shrink_pseudocount: float = 6.0,
    clip: float = 0.15,
) -> dict[str, float]:
    """
    WEALTH-NORMALISED cohesion multipliers.

    Fits log(SoT) = a + b*log(wage) across teams (the shots a wage bill
    *predicts*), then each team's cohesion signal is its residual: did it
    create more or fewer shots on target than its talent implies? This is
    orthogonal to wealth, so a poor team and a rich team that both meet their
    wage-implied baseline get a neutral multiplier -- no blanket penalty on
    expensive squads.

    Residuals are shrunk by eff_n/(eff_n+k), scaled by `sensitivity`
    (0.0 = OFF), exponentiated and clipped. Returns team -> multiplier.
    Needs >=4 teams with positive wage/SoT and wage variation to fit; else
    returns all 1.0.
    """
    if sensitivity == 0.0:
        return {t: 1.0 for t in stwr}

    usable = {t: d for t, d in stwr.items()
              if d["sot"] and d["wage"] and d["sot"] > 0 and d["wage"] > 0}
    if len(usable) < 4:
        return {t: 1.0 for t in stwr}

    teams = list(usable)
    lw = np.log(np.array([usable[t]["wage"] for t in teams]))
    ls = np.log(np.array([usable[t]["sot"] for t in teams]))
    if lw.std() < 1e-6:
        return {t: 1.0 for t in stwr}

    b, a = np.polyfit(lw, ls, 1)            # log(SoT) ~ a + b*log(wage)
    resid = {t: ls[i] - (a + b * lw[i]) for i, t in enumerate(teams)}

    mults: dict[str, float] = {}
    for t, d in stwr.items():
        r = resid.get(t)
        if r is None:
            mults[t] = 1.0
            continue
        shrink = d["eff_n"] / (d["eff_n"] + shrink_pseudocount)
        adj = float(np.clip(sensitivity * shrink * r, -clip, clip))
        mults[t] = math.exp(adj)
    return mults
