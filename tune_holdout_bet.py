"""
End-to-end upgraded workflow:

  STAGE 1 -- TUNE (pre-2024 folds only)
    Joint grid over (half_life_days, friendly_weight, ridge) across several
    independent tournament folds, with a paired-bootstrap gate vs the
    baseline config. The baseline survives unless a candidate beats it with
    the 95% CI excluding zero AND wins at least half the folds.

  STAGE 2 -- HOLDOUT (2024+ matches, touched exactly once)
    The frozen winner is evaluated one time on data no tuning decision ever
    saw: Euro 2024, Copa America 2024, and 2024-25 qualifiers/Nations
    League. Brier/log-loss with bootstrap CIs, ECE, reliability table and
    the Murphy decomposition. These are the numbers to trust.
    DO NOT iterate against this stage. If you change the grid or folds and
    re-run, the holdout is burnt -- you'd need a fresh window (e.g. hold out
    2025-26 instead).

  STAGE 3 -- BET (the normal output, on the upgraded config)
    Refit on ALL history as of today with the chosen parameters, write
    teams_fitted.csv / params_fitted.json, price the per-match fixtures
    slate (1X2 + OU2.5 vs odds_slate.csv where prices exist, fair-odds
    price targets otherwise), and -- if a group draw is available -- run the
    Monte Carlo for outright value vs your book prices.

    python fetch_data.py        # once, on a networked machine
    python tune_holdout_bet.py
"""

from __future__ import annotations

import json
import os
import sys
from datetime import date, datetime
from itertools import product

import pandas as pd

from worldcup_mc.fit import load_results, compute_weights, fit_dixon_coles
from worldcup_mc.backtest import (
    filter_matches, run_folds, gated_joint_sweep, choose_config)
from worldcup_mc.calibration import (
    bootstrap_metrics, reliability_table, expected_calibration_error,
    brier_decomposition, split_by_data_richness)
from worldcup_mc import markets as mkt
from worldcup_mc.odds import devig_shin, overround

RESULTS = "worldcup_mc/data/results.csv"
ODDS_SLATE = "worldcup_mc/data/odds_slate.csv"
FIXTURES_SLATE = "worldcup_mc/data/fixtures_slate.csv"
OUT_DIR = "reports"

ASOF = str(date.today())          # decay measured back from today for Stage 3

# --- baseline = the current defaults; candidates must BEAT this, gated ---
BASELINE = {"half_life_days": 730.0, "friendly_weight": 0.3, "ridge": 0.05}

# --- joint grid: knobs interact, so sweep them together, not one at a time
GRID = [
    {"half_life_days": hl, "friendly_weight": fw, "ridge": rg}
    for hl, fw, rg in product(
        (365.0, 730.0, 1095.0, 1460.0),
        (0.15, 0.3, 0.5),
        (0.02, 0.05, 0.15),
    )
]

# walk-forward economy: warm starts everywhere; long qualifier windows
# refit every 14 days instead of every matchday (still strictly past-only)
BT = dict(warm_start=True)

MIN_FOLDS_WON_FRAC = 0.5
RICHNESS_THRESHOLD = 15


# --------------------------------------------------------------------------
# fold construction -- tuning strictly pre-2024, holdout strictly 2024+
# --------------------------------------------------------------------------
def tuning_folds(history: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "wc2018": filter_matches(history, tournament_contains="FIFA World Cup",
                                 year=2018),
        "euro2020": filter_matches(history, tournament_contains="UEFA Euro",
                                   year=2021),          # played summer 2021
        "wc2022": filter_matches(history, tournament_contains="FIFA World Cup",
                                 year=2022),
        "quals_21_23": filter_matches(history, tournament_contains="qualification",
                                      start="2021-01-01", end="2023-12-31"),
    }


# --------------------------------------------------------------------------
# stages
# --------------------------------------------------------------------------
def stage1_tune(history):
    print("=" * 72)
    print("STAGE 1 -- gated joint sweep on pre-2024 folds")
    print("=" * 72)
    folds = tuning_folds(history)
    for name, f in folds.items():
        span = (f"{f['date'].min().date()}..{f['date'].max().date()}"
                if len(f) else "EMPTY")
        print(f"  fold {name:<14} {len(f):>5} matches  ({span})")
    folds = {k: v for k, v in folds.items() if len(v) >= 20}

    # per-date refits for finals-sized folds; 14-day interval refits for the
    # long qualifier window (still strictly walk-forward, just staler)
    intervals = {k: 14.0 for k in folds if k.startswith("quals")}
    table, store = gated_joint_sweep(
        history, folds, GRID, BASELINE,
        n_boot=4000, ci=0.95, verbose=True,
        refit_interval_by_fold=intervals, **BT)

    print("\n--- sweep table (negative diff = better than baseline) ---")
    cols = ["half_life_days", "friendly_weight", "ridge", "n_paired", "brier",
            "diff_vs_base", "ci_lo", "ci_hi", "beats_baseline", "folds_won"]
    print(table[cols].round(4).to_string(index=False))

    cfg, reason = choose_config(table, BASELINE,
                                min_folds_won_frac=MIN_FOLDS_WON_FRAC)
    print(f"\nCHOSEN CONFIG: {cfg}\n  reason: {reason}")
    return cfg, table


def holdout_folds(history: pd.DataFrame) -> dict[str, pd.DataFrame]:
    quals = filter_matches(history, tournament_contains="qualification",
                           start="2024-01-01", end="2025-12-31")
    nations = filter_matches(history, tournament_contains="Nations League",
                             start="2024-01-01", end="2025-12-31")
    long_window = pd.concat([quals, nations], ignore_index=True
                            ).drop_duplicates(
        subset=["date", "home_team", "away_team"]).reset_index(drop=True)
    folds = {
        "euro2024": filter_matches(history, tournament_contains="UEFA Euro",
                                   year=2024),
        "copa2024": filter_matches(history, tournament_contains="Copa",
                                   year=2024),
        "quals_nl_24_25": long_window,
    }
    return {k: v for k, v in folds.items() if len(v) >= 20}


def stage2_holdout(history, cfg):
    print("\n" + "=" * 72)
    print("STAGE 2 -- one-shot holdout evaluation (2024+, never tuned on)")
    print("=" * 72)
    folds = holdout_folds(history)
    for name, f in folds.items():
        print(f"  fold {name:<16} {len(f):>5} matches  "
              f"({f['date'].min().date()}..{f['date'].max().date()})")

    pm = run_folds(history, folds, cfg,
                   refit_interval_by_fold={"quals_nl_24_25": 14.0},
                   verbose=False, **BT)

    brier = pm["brier"].mean()
    base_brier = pm["base_brier"].mean()
    print(f"\nmatches scored: {len(pm)}")
    print(f"  model    Brier {brier:.4f} | log-loss {pm['log_loss'].mean():.4f}")
    print(f"  baseline Brier {base_brier:.4f} | "
          f"log-loss {pm['base_log_loss'].mean():.4f}")
    print(f"  skill (Brier): {(1 - brier / base_brier) * 100:+.1f}% vs base rates")
    print("  per fold:", {k: round(v, 4)
                          for k, v in pm.attrs["fold_brier"].items()})

    print("\nbootstrap CIs:")
    for stat in bootstrap_metrics(pm, n_boot=5000).values():
        print(" ", stat)

    print(f"\nECE (quantile bins): {expected_calibration_error(pm):.4f}")
    print(brier_decomposition(pm, bins=10))
    print("\nreliability (gap = pred - obs, want ~0):")
    print(reliability_table(pm, bins=10, scheme="fixed")
          .round(3).to_string(index=False))

    print("\ndata-richness split (2026 has many data-poor sides -- "
          "check we stay calibrated there):")
    for label, part in split_by_data_richness(
            pm, threshold=RICHNESS_THRESHOLD).items():
        if len(part) < 30:
            print(f"  ({label}: only {len(part)} matches -- too few to read)")
            continue
        print(f"  {label}: n={len(part)}  brier={part['brier'].mean():.4f}  "
              f"ECE={expected_calibration_error(part):.4f}")
    return pm


def stage3_bet(history, cfg):
    print("\n" + "=" * 72)
    print(f"STAGE 3 -- betting output (fit as of {ASOF}, frozen config)")
    print("=" * 72)
    w = compute_weights(history, asof=ASOF, **{
        "half_life_days": cfg["half_life_days"],
        "friendly_weight": cfg["friendly_weight"]})
    fit = fit_dixon_coles(history, w, ridge=cfg["ridge"])
    print(f"fit {len(fit.teams)} teams on {fit.n_matches:,} matches | "
          f"base={fit.base:.3f} home_adv={fit.home_adv:.3f} rho={fit.rho:.3f}")
    fit.write_ratings_csv("worldcup_mc/data/teams_fitted.csv")
    fit.write_params("worldcup_mc/data/params_fitted.json")
    with open("worldcup_mc/data/tuned_config.json", "w") as f:
        json.dump({"config": cfg, "asof": ASOF}, f, indent=2)
    model = fit.to_match_model()

    # --- per-match slate vs book prices (the normal value output) ---
    if os.path.exists(ODDS_SLATE):
        slate = pd.read_csv(ODDS_SLATE)
        print(f"\n--- per-match value vs {ODDS_SLATE} ---")
        rows = []
        for r in slate.itertuples(index=False):
            if r.home not in model.teams or r.away not in model.teams:
                print(f"  skip {r.home} v {r.away}: team missing from fit")
                continue
            card = mkt.predict_fixture(model, r.home, r.away,
                                       neutral=bool(r.neutral))
            p = card["1x2"]
            odds_vec = [r.odds_home, r.odds_draw, r.odds_away]
            fair = devig_shin(pd.Series(odds_vec).to_numpy(float))
            ov = overround(pd.Series(odds_vec).to_numpy(float)) - 1.0
            for sel, prob, o, fb in zip(("home", "draw", "away"),
                                        (p["home"], p["draw"], p["away"]),
                                        odds_vec, fair):
                rows.append({
                    "date": r.date, "match": f"{r.home} v {r.away}",
                    "market": "1X2", "sel": sel, "odds": o,
                    "p_model": prob, "p_fair_book": float(fb),
                    "edge": prob - float(fb),
                    "EV_per_unit": prob * (o - 1) - (1 - prob),
                    "overround": ov,
                })
            t = card["totals"].get(2.5)
            if t and pd.notna(getattr(r, "odds_over25", None)):
                for sel, prob, o in (("over", t["over"], r.odds_over25),
                                     ("under", t["under"], r.odds_under25)):
                    rows.append({
                        "date": r.date, "match": f"{r.home} v {r.away}",
                        "market": "OU2.5", "sel": sel, "odds": o,
                        "p_model": prob, "p_fair_book": float("nan"),
                        "edge": float("nan"),
                        "EV_per_unit": prob * (o - 1) - (1 - prob),
                        "overround": float("nan"),
                    })
        card_df = pd.DataFrame(rows)
        if len(card_df):
            print(card_df.round(4).to_string(index=False))
            value = card_df[card_df["EV_per_unit"] > 0]
            print("\npositive-EV selections:")
            print(value.round(4).to_string(index=False) if len(value)
                  else "  none at these prices")
            os.makedirs(OUT_DIR, exist_ok=True)
            out_csv = os.path.join(
                OUT_DIR, f"betlist_{datetime.now():%Y%m%d_%H%M%S}.csv")
            card_df.to_csv(out_csv, index=False)
            print(f"\nbet card written to {out_csv}")
    else:
        print(f"\n(no {ODDS_SLATE} -- skipping book comparison)")

    # --- fair-odds price targets for fixtures with no book prices yet ---
    if os.path.exists(FIXTURES_SLATE):
        fx = pd.read_csv(FIXTURES_SLATE)
        print(f"\n--- fair-odds card for {FIXTURES_SLATE} "
              f"(min price to take = fair odds * 1.02) ---")
        for r in fx.itertuples(index=False):
            if r.home not in model.teams or r.away not in model.teams:
                continue
            p = mkt.predict_fixture(model, r.home, r.away,
                                    neutral=bool(r.neutral))["1x2"]
            line = "  ".join(
                f"{s}: p={p[s]:.3f} fair={1/p[s]:.2f} take>={1.02/p[s]:.2f}"
                for s in ("home", "draw", "away"))
            print(f"  {r.date} {r.home} v {r.away}:  {line}")

    print("\nNOTE: the value card here is point-estimate. For stake tiering")
    print("gated on rating uncertainty, feed the same config into")
    print("worldcup_mc.value.bootstrap_models + price_fixture (B refits).")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    history = load_results(RESULTS, min_date="2006-01-01")
    print(f"loaded {len(history):,} matches "
          f"({history['date'].min().date()}..{history['date'].max().date()})\n")

    cfg, table = stage1_tune(history)
    table.to_csv(os.path.join(OUT_DIR, "sweep_table.csv"), index=False)
    stage2_holdout(history, cfg)
    stage3_bet(history, cfg)


if __name__ == "__main__":
    main()
