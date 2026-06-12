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
from worldcup_mc import value as val

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

# --- Stage 3 price-target card ---
N_BOOT = 200          # model refits for the probability CI (raise for tighter CIs, slower)
CI = 0.90             # confidence level for the uncertainty-adjusted price
EV_SMALL = 0.02       # EV cushion baked into the naive fair-odds threshold
EV_BIG = 0.05         # EV cushion for the uncertainty-adjusted (CI-low) threshold


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
    print(f"STAGE 3 -- price-target card (fit as of {ASOF}, frozen config)")
    print("=" * 72)

    # point fit -> write ratings/params (and a quick strength sanity check)
    w = compute_weights(history, asof=ASOF, **{
        "half_life_days": cfg["half_life_days"],
        "friendly_weight": cfg["friendly_weight"]})
    fit = fit_dixon_coles(history, w, ridge=cfg["ridge"])
    print(f"fit {len(fit.teams)} teams on {fit.n_matches:,} matches | "
          f"base={fit.base:.3f} home_adv={fit.home_adv:.3f} rho={fit.rho:.3f}")
    fit.write_ratings_csv("worldcup_mc/data/teams_fitted.csv")
    fit.write_params("worldcup_mc/data/params_fitted.json")

    if not os.path.exists(FIXTURES_SLATE):
        print(f"\n(no {FIXTURES_SLATE} -- nothing to price. Put your fixtures "
              "there: date,group,home,away,neutral)")
        return

    fx = pd.read_csv(FIXTURES_SLATE)

    # --- bootstrap an ensemble ONCE (B refits, shared across all fixtures) ---
    # This is what puts a confidence interval on every probability, so we can
    # show the uncertainty-adjusted price next to the naive fair odds.
    print(f"\nbootstrapping {N_BOOT} model refits for the CI "
          "(this is the slow part)...")
    models = val.bootstrap_models(
        history, asof=ASOF, n_boot=N_BOOT,
        half_life_days=cfg["half_life_days"],
        friendly_weight=cfg["friendly_weight"], ridge=cfg["ridge"])

    print(f"\n--- PRICE-TARGET CARD ({CI:.0%} CI, no bookmaker odds needed) ---")
    print("Take the bet when the price you can find is at or above the "
          "threshold.")
    print("  fair_odds      = 1 / model_prob              (break-even if the "
          "model is exactly right)")
    print(f"  min_take_fair  = fair_odds with a {EV_SMALL:.0%} EV cushion "
          "on the point estimate")
    print(f"  min_take_safe  = priced off the {CI:.0%}-CI LOW prob with a "
          f"{EV_BIG:.0%} cushion  (survives model uncertainty)")
    print("  gap            = how much extra price the uncertainty margin "
          "demands\n")

    rows = []
    for r in fx.itertuples(index=False):
        if r.home not in models[0].teams or r.away not in models[0].teams:
            print(f"  skip {r.home} v {r.away}: team missing from fit")
            continue
        targets = val.fixture_targets(
            models, r.home, r.away, neutral=bool(r.neutral),
            ci=CI, ev_small=EV_SMALL, ev_big=EV_BIG)
        for t in targets:
            rows.append({
                "date": r.date,
                "match": f"{r.home} v {r.away}",
                "market": t.market,
                "sel": t.selection,
                "p_model": round(t.p_model, 4),
                "p_lo": round(t.p_lo, 4),
                "p_hi": round(t.p_hi, 4),
                "fair_odds": t.fair_odds,
                "min_take_fair": t.min_odds_small,
                "min_take_safe": t.min_odds_big,
                "gap": round(t.min_odds_big - t.min_odds_small, 2),
            })

    if not rows:
        print("  no fixtures could be priced (teams missing from the fit?)")
        return

    card = pd.DataFrame(rows)
    print(card.to_string(index=False))

    os.makedirs(OUT_DIR, exist_ok=True)
    out_csv = os.path.join(
        OUT_DIR, f"price_targets_{datetime.now():%Y%m%d_%H%M%S}.csv")
    card.to_csv(out_csv, index=False)
    print(f"\nprice-target card written to {out_csv}")
    print("\nHow to use it: shop around. If any book offers a price >= "
          "min_take_safe,")
    print("that's a confident bet; between min_take_fair and min_take_safe is "
          "a thin/")
    print("small-stake bet; below min_take_fair, pass. Wide gap = the model "
          "isn't sure")
    print("about that team yet (few games), so it demands a bigger margin "
          "before betting.")


CONFIG_PATH = "worldcup_mc/data/tuned_config.json"


def save_config(cfg, table=None):
    """Persist the chosen config so pricing can run without re-tuning."""
    payload = {"config": cfg, "tuned_on": str(date.today())}
    if table is not None:
        # keep a tiny audit trail of what won and by how much
        best = table.iloc[0].to_dict()
        payload["winning_row"] = {k: (float(v) if isinstance(v, (int, float))
                                      else v) for k, v in best.items()}
    with open(CONFIG_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nconfig saved to {CONFIG_PATH} -- "
          "run `python price.py` to price fixtures without re-tuning.")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    history = load_results(RESULTS, min_date="2006-01-01")
    print(f"loaded {len(history):,} matches "
          f"({history['date'].min().date()}..{history['date'].max().date()})\n")

    cfg, table = stage1_tune(history)
    table.to_csv(os.path.join(OUT_DIR, "sweep_table.csv"), index=False)
    save_config(cfg, table)          # <-- written here, before the slow stages
    stage2_holdout(history, cfg)
    stage3_bet(history, cfg)


if __name__ == "__main__":
    main()
