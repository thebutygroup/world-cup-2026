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
    warm_start: bool = True,
    refit_interval_days: float = 0.0,
    use_confederation_prior: bool = False,
    verbose: bool = False,
) -> BacktestResult:
    """
    Walk-forward 1X2 backtest.

    history : full results DataFrame (martj42 schema) -- the universe the
              model may learn from. For each test date we use the subset of
              `history` strictly before that date.
    test    : the matches to score (from filter_matches). Should itself be a
              subset of the same results source so actual scores are present.

    warm_start : initialise each date's fit from the previous date's optimum
              (teams matched by name). Consecutive training sets differ by a
              handful of matches, so this cuts fit time ~5-10x with the same
              solution -- which is what makes multi-year test windows and
              joint parameter grids affordable.
    refit_interval_days : 0 refits at every distinct test date (strictest).
              K > 0 reuses the last fitted ratings for up to K days before
              refitting -- still strictly walk-forward (ratings only ever see
              the past, just slightly staler). Use 14-30 for long qualifier
              windows where per-date refits are prohibitive.

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
    prev_fit = None          # warm-start carrier
    fit = model = counts = None
    last_fit_date = None
    for date, day_matches in test.groupby("date"):
        need_refit = (
            model is None
            or refit_interval_days <= 0
            or (date - last_fit_date).days >= refit_interval_days
        )
        if need_refit:
            train = history[history["date"] < date]
            if min_history_date is not None:
                train = train[train["date"] >= pd.Timestamp(min_history_date)]
            if len(train) < 100:
                n_skipped += len(day_matches)
                continue

            counts = pd.concat([train["home_team"], train["away_team"]]).value_counts()
            w = compute_weights(train, asof=date, half_life_days=half_life_days,
                                friendly_weight=friendly_weight)
            fit = fit_dixon_coles(train, w, ridge=ridge,
                                  warm_start=prev_fit if warm_start else None,
                                  use_confederation_prior=use_confederation_prior)
            if warm_start:
                prev_fit = fit
            model = fit.to_match_model()
            last_fit_date = date
            if verbose:
                print(f"{date.date()}: trained on {len(train):,} matches, "
                      f"{len(fit.teams)} teams, scoring {len(day_matches)}")
        elif verbose:
            print(f"{date.date()}: reusing fit from {last_fit_date.date()}, "
                  f"scoring {len(day_matches)}")

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


# ==========================================================================
# MULTI-FOLD, BOOTSTRAP-GATED PARAMETER TUNING
#
# Replaces "run one sweep on one tournament and pick the lowest Brier" --
# which, on ~64 matches, mostly picks noise -- with:
#
#   1. Several independent test FOLDS (different tournaments / windows), so
#      a winning config has to win (or tie) in different eras and team
#      mixes, not once.
#   2. A JOINT grid over (half_life, friendly_weight, ridge), because the
#      knobs interact: a longer half-life raises the effective sample per
#      team, which lowers the shrinkage you need, and vice versa.
#   3. A paired-bootstrap GATE vs a baseline config on the pooled matches:
#      a candidate only "beats" the baseline if the 95% CI of the per-match
#      Brier difference excludes zero. If nothing clears the gate, keep the
#      baseline -- a flat loss surface is an answer, not a failure.
# ==========================================================================

_CONFIG_KEYS = ("half_life_days", "friendly_weight", "ridge")
_MATCH_KEY = ["fold", "date", "home", "away"]


def _config_label(cfg: dict) -> str:
    return (f"hl={cfg['half_life_days']:.0f} "
            f"fw={cfg['friendly_weight']:.2f} "
            f"ridge={cfg['ridge']:.3f}")


def run_folds(
    history: pd.DataFrame,
    folds: dict[str, pd.DataFrame],
    config: dict,
    refit_interval_by_fold: dict[str, float] | None = None,
    verbose: bool = False,
    **backtest_kwargs,
) -> pd.DataFrame:
    """
    Run the walk-forward backtest with one parameter config over several
    named test folds. Returns the pooled per-match frame with a `fold`
    column; per-fold mean Briers are attached in .attrs["fold_brier"].

    refit_interval_by_fold lets long windows (e.g. multi-year qualifiers)
    use interval refits while finals keep strict per-matchday refits, e.g.
    {"quals_21_23": 14.0}. Folds not listed use backtest's default.
    """
    cfg = {k: config[k] for k in _CONFIG_KEYS}
    intervals = refit_interval_by_fold or {}
    parts, fold_brier = [], {}
    for name, test in folds.items():
        kw = dict(backtest_kwargs)
        if name in intervals:
            kw["refit_interval_days"] = intervals[name]
        res = backtest(history, test, **cfg, **kw)
        pm = res.per_match.copy()
        pm["fold"] = name
        parts.append(pm)
        fold_brier[name] = res.brier
        if verbose:
            print(f"  [{_config_label(cfg)}] fold {name:<16} "
                  f"n={res.n_scored:>4} brier={res.brier:.4f}")
    pooled = pd.concat(parts, ignore_index=True)
    pooled.attrs["fold_brier"] = fold_brier
    pooled.attrs["config"] = cfg
    return pooled


def _paired_boot_diff(da: np.ndarray, db: np.ndarray, n_boot: int,
                      ci: float, seed: int) -> tuple[float, float, float]:
    """Percentile bootstrap of mean(da - db) over paired rows."""
    diff = da - db
    rng = np.random.default_rng(seed)
    n = len(diff)
    b = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        b[i] = diff[idx].mean()
    lo_q, hi_q = (1 - ci) / 2, 1 - (1 - ci) / 2
    lo, hi = np.quantile(b, [lo_q, hi_q])
    return float(diff.mean()), float(lo), float(hi)


def gated_joint_sweep(
    history: pd.DataFrame,
    folds: dict[str, pd.DataFrame],
    grid: list[dict],
    baseline: dict,
    n_boot: int = 4000,
    ci: float = 0.95,
    seed: int = 0,
    refit_interval_by_fold: dict[str, float] | None = None,
    verbose: bool = True,
    **backtest_kwargs,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    Joint parameter sweep across multiple folds with a paired-bootstrap gate.

    For every config in `grid`, runs all folds, pools the per-match scores,
    inner-joins them to the baseline's matches on (fold, date, home, away)
    -- configs can skip different matches, pairing keeps it apples-to-apples
    -- and bootstraps the mean per-match Brier difference (config - baseline;
    negative = config better).

    Returns (table, per_match_by_label):
      table columns:
        half_life_days, friendly_weight, ridge, n_paired, brier,
        diff_vs_base, ci_lo, ci_hi,
        beats_baseline  -> CI entirely below 0 (a REAL improvement)
        folds_won       -> in how many folds this config's mean Brier was
                           lower than the baseline's (stability check: a
                           config that "wins" pooled but loses most folds is
                           leaning on one era)
      per_match_by_label: pooled per-match frame for every config (baseline
        included under its label) for downstream calibration on the winner.
    """
    base_cfg = {k: baseline[k] for k in _CONFIG_KEYS}
    if verbose:
        print(f"baseline: {_config_label(base_cfg)}")
    base_pm = run_folds(history, folds, base_cfg,
                        refit_interval_by_fold=refit_interval_by_fold,
                        verbose=verbose, **backtest_kwargs)
    base_fold_brier = base_pm.attrs["fold_brier"]
    store = {_config_label(base_cfg): base_pm}

    rows = [{**base_cfg, "n_paired": len(base_pm),
             "brier": float(base_pm["brier"].mean()),
             "diff_vs_base": 0.0, "ci_lo": 0.0, "ci_hi": 0.0,
             "beats_baseline": False,
             "folds_won": 0, "n_folds": len(folds), "is_baseline": True}]

    for cfg_in in grid:
        cfg = {k: cfg_in[k] for k in _CONFIG_KEYS}
        if cfg == base_cfg:
            continue
        if verbose:
            print(f"config:   {_config_label(cfg)}")
        pm = run_folds(history, folds, cfg,
                       refit_interval_by_fold=refit_interval_by_fold,
                       verbose=verbose, **backtest_kwargs)
        store[_config_label(cfg)] = pm

        merged = pm.merge(base_pm, on=_MATCH_KEY, suffixes=("_cfg", "_base"))
        point, lo, hi = _paired_boot_diff(
            merged["brier_cfg"].to_numpy(float),
            merged["brier_base"].to_numpy(float),
            n_boot=n_boot, ci=ci, seed=seed)
        folds_won = sum(
            1 for f, b in pm.attrs["fold_brier"].items()
            if b < base_fold_brier.get(f, np.inf))
        rows.append({**cfg, "n_paired": len(merged),
                     "brier": float(pm["brier"].mean()),
                     "diff_vs_base": point, "ci_lo": lo, "ci_hi": hi,
                     "beats_baseline": bool(hi < 0.0),
                     "folds_won": folds_won, "n_folds": len(folds),
                     "is_baseline": False})

    table = (pd.DataFrame(rows)
             .sort_values(["beats_baseline", "diff_vs_base"],
                          ascending=[False, True])
             .reset_index(drop=True))
    return table, store


def choose_config(
    table: pd.DataFrame,
    baseline: dict,
    min_folds_won_frac: float = 0.5,
) -> tuple[dict, str]:
    """
    Selection rule: stick with the baseline unless a candidate BOTH
      (a) beats it with the bootstrap CI excluding zero, AND
      (b) won at least `min_folds_won_frac` of the folds individually
          (stability across eras, not one lucky tournament).
    Among qualifiers, take the largest (most negative) pooled improvement.
    Returns (config, human-readable reason).
    """
    base_cfg = {k: baseline[k] for k in _CONFIG_KEYS}
    need = int(np.ceil(min_folds_won_frac * table["n_folds"].iloc[0]))
    cands = table[table["beats_baseline"] & (table["folds_won"] >= need)]
    if not len(cands):
        return base_cfg, (
            "no candidate beat the baseline with the CI excluding 0 AND "
            f"won >= {need} folds -- the loss surface is flat at this sample "
            "size; keeping the baseline (tuning further would be fitting noise)")
    best = cands.sort_values("diff_vs_base").iloc[0]
    cfg = {k: float(best[k]) for k in _CONFIG_KEYS}
    return cfg, (
        f"{_config_label(cfg)} beat the baseline by {best['diff_vs_base']:+.4f} "
        f"Brier [{best['ci_lo']:+.4f}, {best['ci_hi']:+.4f}] and won "
        f"{int(best['folds_won'])}/{int(best['n_folds'])} folds")
