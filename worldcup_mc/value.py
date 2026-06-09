"""
Live value card: turn the model's per-fixture probabilities into a ranked,
stake-tiered bet list against the bookmaker prices you actually see, with a
confidence interval on every edge.

WHAT THIS IS FOR. You bet EARLY into the soft opening/midweek line (that is
where the edge is, before the market sharpens toward kickoff). This module
takes (a) the model's probability for each market and (b) the complete book
market you can see right now, de-vigs the book, and reports edge + EV per
selection -- ranked, so you can put money on the biggest gaps first. The
closing line is NOT used here; it is the after-the-fact scorecard (see the
backtest) for whether you got positive CLV. Two different jobs.

THE CONFIDENCE INTERVAL. A single point-estimate probability hides how much
the ratings themselves are uncertain. So instead of one fit we bootstrap:
resample the training history with replacement, refit Dixon-Coles B times,
and evaluate every fixture under all B models. That gives a *distribution* of
the model's probability for each selection -> a CI on the probability, hence a
CI on edge and EV. Stake tiers are gated on the CI, not the point estimate:
the higher stake only fires when even the pessimistic end of the interval
still clears the bar. Early on, with placeholder ratings, expect wide CIs and
mostly "skip"/"small" -- that is the tool being honest, not broken.

DE-VIG NEEDS THE COMPLETE MARKET. For 1X2 you must pass all three prices; for
a totals line, both over and under. EV is computed at the *quoted* odds and is
correct regardless, but `edge` (model prob - fair book prob) and the overround
are only meaningful with the full market.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .fit import compute_weights, fit_dixon_coles
from . import markets as mkt
from .odds import devig_shin, devig_proportional, overround


# --------------------------------------------------------------------------
# bootstrap an ensemble of fitted models for one as-of date
# --------------------------------------------------------------------------
def bootstrap_models(
    history: pd.DataFrame,
    asof: str | pd.Timestamp,
    n_boot: int = 200,
    half_life_days: float = 1460.0,
    friendly_weight: float = 0.3,
    ridge: float = 0.05,
    min_history_date: str | None = "2010-01-01",
    seed: int = 0,
):
    """
    Resample the pre-`asof` history with replacement and refit Dixon-Coles
    `n_boot` times. Returns a list of MatchModel. One refit per resample, all
    sharing the same as-of cut, so a whole slate of fixtures is priced from
    the SAME ensemble (B refits total, not B per fixture).
    """
    asof = pd.Timestamp(asof)
    h = history.copy()
    h["date"] = pd.to_datetime(h["date"])
    train = h[h["date"] < asof]
    if min_history_date is not None:
        train = train[train["date"] >= pd.Timestamp(min_history_date)]
    if len(train) < 100:
        raise ValueError(f"only {len(train)} training matches before {asof.date()}")

    rng = np.random.default_rng(seed)
    n = len(train)
    models = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        samp = train.iloc[idx].reset_index(drop=True)
        w = compute_weights(samp, asof=asof, half_life_days=half_life_days,
                            friendly_weight=friendly_weight)
        fit = fit_dixon_coles(samp, w, ridge=ridge)
        models.append(fit.to_match_model())
    return models


def _selection_samples(models, home, away, neutral, total_line=2.5):
    """For each bootstrap model, read the fixture's selection probabilities.
    Returns {selection: np.array of B probs}. Models missing either team are
    skipped (their team never appeared in that resample)."""
    keys = ["home", "draw", "away", "over", "under"]
    acc = {k: [] for k in keys}
    for m in models:
        if home not in m.teams or away not in m.teams:
            continue
        p_h, p_d, p_a = m.outcome_probs(home, away, neutral=neutral)
        matrix = mkt.score_matrix(m, home, away, neutral=neutral)
        tot = mkt.totals(matrix, lines=(total_line,))[float(total_line)]
        for k, v in zip(keys, (p_h, p_d, p_a, tot["over"], tot["under"])):
            acc[k].append(v)
    return {k: np.asarray(v, dtype=float) for k, v in acc.items()}


# --------------------------------------------------------------------------
# value of one selection given its prob samples and the book's complete market
# --------------------------------------------------------------------------
@dataclass
class Selection:
    market: str          # "1X2" or "OU2.5"
    selection: str       # "home"/"draw"/"away"/"over"/"under"
    odds: float          # the quoted decimal odds you can take
    p_model: float       # model probability (mean over bootstrap)
    p_lo: float          # CI lower / upper on the model probability
    p_hi: float
    p_fair_book: float   # de-vigged book probability for this selection
    overround: float     # book margin on this market (sum of implied - 1)
    edge: float          # p_model - p_fair_book
    ev: float            # EV per unit at quoted odds, point estimate
    ev_lo: float         # EV at the pessimistic end of the prob CI
    stake: float         # suggested stake from the tiering rule


def _devig(odds_vec, method):
    return devig_shin(odds_vec) if method == "shin" else devig_proportional(odds_vec)


def _ev(p, odds):
    return p * (odds - 1.0) - (1.0 - p)


def evaluate_market(
    samples: dict[str, np.ndarray],
    market: str,
    selections: list[str],
    book_odds: dict[str, float],
    devig_method: str = "shin",
    ci: float = 0.90,
    stake_tiers=(10.0, 50.0),
    ev_small: float = 0.0,
    ev_big: float = 0.03,
    margin_mult: float = 0.5,
    min_edge: float = 0.02,
) -> list[Selection]:
    """
    Score one complete market (all its selections). `book_odds` must cover the
    whole market (e.g. home/draw/away). Staking:
      - margin-scaled bar: a selection must clear ev_threshold, where
        ev_threshold = ev_small + margin_mult * overround  (fatter vig => need
        more cushion, since model error costs more there).
      - big stake (tiers[1]) only if the *pessimistic* end of the prob CI
        still gives EV >= ev_big AND edge >= min_edge -- i.e. the edge
        survives the model's own uncertainty.
      - small stake (tiers[0]) if the point EV clears the bar but the CI does
        not. Otherwise skip (stake 0).
    """
    odds_vec = np.array([book_odds[s] for s in selections], dtype=float)
    fair = _devig(odds_vec, devig_method)
    ov = overround(odds_vec) - 1.0
    lo_q, hi_q = (1 - ci) / 2, 1 - (1 - ci) / 2

    out = []
    for s, f in zip(selections, fair):
        ps = samples[s]
        if ps.size == 0:
            continue
        p = float(ps.mean())
        p_lo, p_hi = (float(x) for x in np.quantile(ps, [lo_q, hi_q]))
        o = float(book_odds[s])
        ev_point = _ev(p, o)
        ev_pessimistic = _ev(p_lo, o)          # low prob end = worst case
        edge = p - float(f)

        ev_threshold = ev_small + margin_mult * ov
        stake = 0.0
        if ev_pessimistic >= max(ev_big, ev_threshold) and edge >= min_edge:
            stake = stake_tiers[1]
        elif ev_point >= ev_threshold and edge >= min_edge:
            stake = stake_tiers[0]

        out.append(Selection(
            market=market, selection=s, odds=o, p_model=p, p_lo=p_lo, p_hi=p_hi,
            p_fair_book=float(f), overround=ov, edge=edge, ev=ev_point,
            ev_lo=ev_pessimistic, stake=stake))
    return out


# --------------------------------------------------------------------------
# whole-fixture convenience
# --------------------------------------------------------------------------
def price_fixture(
    models, home, away, neutral, book_row: dict, **kwargs
) -> list[Selection]:
    """
    Price every market we have complete book odds for in `book_row`. Expects
    keys like odds_home/odds_draw/odds_away and optionally
    odds_over25/odds_under25. Returns a flat list of Selection.
    """
    samples = _selection_samples(models, home, away, neutral)
    rows: list[Selection] = []
    if all(k in book_row and pd.notna(book_row[k])
           for k in ("odds_home", "odds_draw", "odds_away")):
        rows += evaluate_market(
            samples, "1X2", ["home", "draw", "away"],
            {"home": book_row["odds_home"], "draw": book_row["odds_draw"],
             "away": book_row["odds_away"]}, **kwargs)
    if all(k in book_row and pd.notna(book_row[k])
           for k in ("odds_over25", "odds_under25")):
        rows += evaluate_market(
            samples, "OU2.5", ["over", "under"],
            {"over": book_row["odds_over25"], "under": book_row["odds_under25"]},
            **kwargs)
    return rows
