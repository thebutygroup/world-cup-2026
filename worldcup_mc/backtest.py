"""
Walk-forward backtest for the match model.

WHY THIS SHAPE. The success metric for this project is per-match prediction
quality (and, downstream, beating the closing line) -- NOT whether a single
simulated tournament happens to crown the right champion. One tournament is
a single draw from a 64-match, knockout-heavy distribution; the title is
decided by a handful of near-coin-flips, so "did we pick the 2022 winner"
carries almost no signal. The right test scores the *probabilities* the
model assigned to many individual matches it had never seen, using only
information available before each kickoff. That is what this module does.

METHOD (strict walk-forward, no leakage):
  1. Choose a test set of matches (a date window, and/or a tournament such
     as the 2022 World Cup or Euro 2024 -- both are in the martj42 results
     file `fetch_data.py` downloads).
  2. Group the test matches by date. For each distinct test date, refit the
     Dixon-Coles ratings on *all* results strictly BEFORE that date (with
     the usual time-decay + friendly down-weight + ridge), then predict
     every test match on that date. So each prediction uses only the past.
  3. Score each match's 1X2 forecast against the actual result with:
       - multiclass Brier score  : sum_k (p_k - y_k)^2   (0 best, 2 worst)
       - log loss                : -log(p_actual)        (0 best)
     and report the mean over the test set, plus a naive base-rate baseline
     so the numbers have a reference point. Lower is better for both.

This validates the *modelling* (are the probabilities calibrated and sharp?)
on teams that mostly reappear in 2026 -- exactly the cross-check you want
before trusting the live per-match values. Tune half_life / friendly_weight
/ ridge to MINIMISE out-of-sample Brier here, not by eye.

Refitting once per test date keeps it tractable: a single tournament is
~10-15 distinct dates, i.e. 10-15 fits. A multi-year window is heavier --
each fit is a full MLE over every national team -- so widen the window only
when you're ready to wait.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .fit import compute_weights, fit_dixon_coles

_OUTCOME_INDEX = {"home": 0, "draw": 1, "away": 2}


def _result_class(hs: int, as_: int) -> int:
    if hs > as_:
        return 0
    if hs == as_:
        return 1
    return 2


def filter_matches(
    df: pd.DataFrame,
    start: str | None = None,
    end: str | None = None,
    tournament_contains: str | None = None,
    year: int | None = None,
) -> pd.DataFrame:
    """
    Select the test matches. Any combination of a date window, a tournament
    name substring (e.g. 'FIFA World Cup', 'UEFA Euro', 'Gold Cup',
    'Nations League', 'Copa') and a calendar year. Case-insensitive on the
    tournament name.
    """
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    if start is not None:
        out = out[out["date"] >= pd.Timestamp(start)]
    if end is not None:
        out = out[out["date"] <= pd.Timestamp(end)]
    if tournament_contains is not None:
        out = out[out["tournament"].astype(str).str.contains(
            tournament_contains, case=False, na=False)]
    if year is not None:
        out = out[out["date"].dt.year == year]
    return out.reset_index(drop=True)


@dataclass
class BacktestResult:
    n_scored: int
    n_skipped: int
    brier: float
    log_loss: float
    baseline_brier: float
    baseline_log_loss: float
    per_match: pd.DataFrame = field(repr=False)

    def summary(self) -> str:
        return (
            f"matches scored: {self.n_scored} (skipped {self.n_skipped})\n"
            f"  model    Brier {self.brier:.4f} | log-loss {self.log_loss:.4f}\n"
            f"  baseline Brier {self.baseline_brier:.4f} | log-loss {self.baseline_log_loss:.4f}\n"
            f"  skill (Brier): {(1 - self.brier / self.baseline_brier) * 100:+.1f}% vs base rates"
        )


def backtest(
    history: pd.DataFrame,
    test: pd.DataFrame,
    half_life_days: float = 730.0,
    friendly_weight: float = 0.3,
    ridge: float = 0.05,
    min_history_date: str | None = "2010-01-01",
    min_matches_per_team: int = 3,
    verbose: bool = False,
) -> BacktestResult:
    """
    Walk-forward 1X2 backtest.

    history : full results DataFrame (martj42 schema) -- the universe the
              model may learn from. For each test date we use the subset of
              `history` strictly before that date.
    test    : the matches to score (from filter_matches). Should itself be a
              subset of the same results source so actual scores are present.

    A test match is skipped (not scored) if either side has fewer than
    `min_matches_per_team` prior matches in the training window, since its
    rating would be pure prior. Skips are counted, not hidden.
    """
    history = history.copy()
    history["date"] = pd.to_datetime(history["date"])
    test = test.copy()
    test["date"] = pd.to_datetime(test["date"])

    # Baseline = empirical H/D/A rate among NEUTRAL historical matches before
    # the earliest test date (most WC/Euro games are neutral). Gives the
    # "know nothing but the base rates" reference.
    first_test_date = test["date"].min()
    base_pool = history[history["date"] < first_test_date]
    if "neutral" in base_pool.columns:
        neutral_pool = base_pool[base_pool["neutral"].astype(str).str.lower().isin(
            ["true", "1", "yes"])]
        base_pool = neutral_pool if len(neutral_pool) > 50 else base_pool
    if len(base_pool):
        bc = np.array([
            (base_pool["home_score"].astype(int) > base_pool["away_score"].astype(int)).mean(),
            (base_pool["home_score"].astype(int) == base_pool["away_score"].astype(int)).mean(),
            (base_pool["home_score"].astype(int) < base_pool["away_score"].astype(int)).mean(),
        ], dtype=float)
        bc = bc / bc.sum()
    else:
        bc = np.array([1 / 3, 1 / 3, 1 / 3])

    rows = []
    n_skipped = 0
    for date, day_matches in test.groupby("date"):
        train = history[history["date"] < date]
        if min_history_date is not None:
            train = train[train["date"] >= pd.Timestamp(min_history_date)]
        if len(train) < 100:
            n_skipped += len(day_matches)
            continue

        counts = pd.concat([train["home_team"], train["away_team"]]).value_counts()
        w = compute_weights(train, asof=date, half_life_days=half_life_days,
                            friendly_weight=friendly_weight)
        fit = fit_dixon_coles(train, w, ridge=ridge)
        model = fit.to_match_model()
        if verbose:
            print(f"{date.date()}: trained on {len(train):,} matches, "
                  f"{len(fit.teams)} teams, scoring {len(day_matches)}")

        for r in day_matches.itertuples(index=False):
            h, a = r.home_team, r.away_team
            if (h not in fit.attack or a not in fit.attack
                    or counts.get(h, 0) < min_matches_per_team
                    or counts.get(a, 0) < min_matches_per_team):
                n_skipped += 1
                continue
            neutral = bool(getattr(r, "neutral", True))
            p_h, p_d, p_a = model.outcome_probs(h, a, neutral=neutral)
            p = np.array([p_h, p_d, p_a], dtype=float)
            p = p / p.sum()
            y_idx = _result_class(int(r.home_score), int(r.away_score))
            y = np.zeros(3); y[y_idx] = 1.0
            brier = float(np.sum((p - y) ** 2))
            ll = float(-math.log(max(p[y_idx], 1e-12)))
            rows.append({
                "date": date, "home": h, "away": a,
                "score": f"{int(r.home_score)}-{int(r.away_score)}",
                "p_home": p[0], "p_draw": p[1], "p_away": p[2],
                "result": ["H", "D", "A"][y_idx],
                "brier": brier, "log_loss": ll,
                # --- for bootstrap + calibration post-processing ---
                # per-match baseline Brier/log-loss against the same base-rate
                # vector, so a paired skill bootstrap recomputes the baseline on
                # each resample instead of freezing it:
                "base_brier": float(np.sum((bc - y) ** 2)),
                "base_log_loss": float(-math.log(max(bc[y_idx], 1e-12))),
                # how much training data the worse-documented side had, for the
                # data-richness calibration split:
                "min_team_matches": int(min(counts.get(h, 0), counts.get(a, 0))),
            })

    if not rows:
        raise ValueError("no matches scored -- check the test filter and that "
                         "history covers dates before the test window")

    per_match = pd.DataFrame(rows)
    y_idx_all = per_match["result"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
    base_brier = float(np.mean([np.sum((bc - np.eye(3)[i]) ** 2) for i in y_idx_all]))
    base_ll = float(np.mean([-math.log(max(bc[i], 1e-12)) for i in y_idx_all]))

    return BacktestResult(
        n_scored=len(per_match),
        n_skipped=n_skipped,
        brier=float(per_match["brier"].mean()),
        log_loss=float(per_match["log_loss"].mean()),
        baseline_brier=base_brier,
        baseline_log_loss=base_ll,
        per_match=per_match,
    )


def half_life_sweep(
    history: pd.DataFrame,
    test: pd.DataFrame,
    half_lives=(365.0, 547.0, 730.0, 1095.0, 1460.0),
    **kwargs,
) -> pd.DataFrame:
    """
    Run the backtest across several decay half-lives and return a table of
    out-of-sample Brier / log-loss. Pick the half_life that minimises Brier
    -- this is the intended way to set the decay knob (and the same pattern
    works for friendly_weight or ridge: loop and minimise OOS Brier).
    """
    out = []
    for hl in half_lives:
        res = backtest(history, test, half_life_days=hl, **kwargs)
        out.append({"half_life_days": hl, "brier": res.brier,
                    "log_loss": res.log_loss, "n_scored": res.n_scored,
                    "baseline_brier": res.baseline_brier})
    return pd.DataFrame(out).sort_values("brier").reset_index(drop=True)
