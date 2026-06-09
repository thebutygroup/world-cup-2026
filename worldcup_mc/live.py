"""
Live in-tournament loop: ingest results as the group stage plays out, refit,
and predict the next fixture using the most up-to-date information.

The workflow you described -- "after each game, re-run so the next game for
either team uses the latest data" -- is three steps:

  1. APPEND the finished result to a live results file (martj42 schema), as
     each group game completes. `append_live_result` does this; the file is
     separate from the historical download so you never mutate the base data.
  2. REFIT on history + everything recorded live so far, as of today. Because
     the fitter is time-decay weighted, yesterday's group games already get
     near-full weight; `live_boost` lets you up-weight current-tournament
     matches further if a backtest says in-tournament form is worth extra.
     `refit` returns a ready-to-use MatchModel.
  3. PREDICT the specific upcoming fixture with `predict_next`, which returns
     the full per-match card (1X2, totals, BTTS, correct score) from
     markets.predict_fixture -- pair it with the book's per-match prices via
     odds.compare_market to read the edge.

Refitting is a full MLE over every national team, so it takes a little time;
run it once between matches. Per the project's standing rule, `live_boost`
defaults to 1.0 (no extra weight) -- turn it up only if the backtest earns it.

A note on host advantage: 2026 has three hosts (USA, Mexico, Canada). For
their home matches pass neutral=False so the fitted home_adv applies; all
other 2026 matches are neutral. host_advantage_2026.csv lists them.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .fit import load_results, compute_weights, fit_dixon_coles, LN2
from .markets import predict_fixture

LIVE_COLUMNS = ["date", "home_team", "away_team", "home_score", "away_score",
                "tournament", "city", "country", "neutral"]


def append_live_result(
    live_path: str,
    date: str,
    home_team: str,
    away_team: str,
    home_score: int,
    away_score: int,
    tournament: str = "FIFA World Cup 2026",
    city: str = "",
    country: str = "",
    neutral: bool = True,
) -> None:
    """Append one finished match to the live results CSV (created if absent),
    in the same schema as the historical results file so they concatenate
    cleanly. Set neutral=False only for a host nation's home game."""
    row = {
        "date": date, "home_team": home_team, "away_team": away_team,
        "home_score": int(home_score), "away_score": int(away_score),
        "tournament": tournament, "city": city, "country": country,
        "neutral": bool(neutral),
    }
    if os.path.exists(live_path):
        df = pd.read_csv(live_path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row], columns=LIVE_COLUMNS)
    df.to_csv(live_path, index=False)


def build_results_with_live(
    historical_path: str,
    live_path: str | None = None,
    min_date: str | None = "2014-01-01",
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Load historical results and concatenate any live results recorded so far.
    Returns (combined_df, is_live_mask) where is_live_mask[i] is True for rows
    that came from the live file -- used by refit() to optionally up-weight
    current-tournament matches.
    """
    hist = load_results(historical_path, min_date=min_date)
    hist_live = np.zeros(len(hist), dtype=bool)
    if live_path and os.path.exists(live_path):
        live = load_results(live_path, min_date=None)
        combined = pd.concat([hist, live], ignore_index=True)
        mask = np.concatenate([hist_live, np.ones(len(live), dtype=bool)])
        return combined.reset_index(drop=True), mask
    return hist, hist_live


def refit(
    historical_path: str,
    live_path: str | None = None,
    asof: str = "2026-06-11",
    half_life_days: float = 730.0,
    friendly_weight: float = 0.3,
    ridge: float = 0.05,
    min_date: str | None = "2014-01-01",
    live_boost: float = 1.0,
    **model_kwargs,
):
    """
    Refit ratings on history + live results as of `asof`, and return
    (MatchModel, FitResult). `live_boost` multiplies the decay weight of
    current-tournament (live-file) matches; 1.0 = rely on recency decay only.
    Pass model_kwargs (e.g. max_goals) through to the MatchModel.
    """
    df, is_live = build_results_with_live(historical_path, live_path, min_date)
    w = compute_weights(df, asof=asof, half_life_days=half_life_days,
                        friendly_weight=friendly_weight)
    if live_boost != 1.0 and is_live.any():
        w = w.copy()
        w[is_live] *= live_boost
    fit = fit_dixon_coles(df, w, ridge=ridge)
    return fit.to_match_model(**model_kwargs), fit


def predict_next(
    model, home: str, away: str, neutral: bool = True, **kwargs,
) -> dict:
    """Per-match card for the next fixture (1X2, totals, BTTS, correct score).
    Thin wrapper over markets.predict_fixture; pass surface=... or
    totals_lines=... through kwargs. Use neutral=False for host home games."""
    return predict_fixture(model, home, away, neutral=neutral, **kwargs)
