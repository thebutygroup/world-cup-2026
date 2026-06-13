"""
Price fixtures WITHOUT re-tuning.

Tuning (tune_holdout_bet.py) is the slow, occasional job -- it searches the
parameter grid across many tournament folds and writes the winning settings to
worldcup_mc/data/tuned_config.json. This script just READS that saved config
and produces the price-target card. Run it as often as you like; it never
re-tunes.

    python tune_holdout_bet.py    # SLOW, run occasionally (re-tune when you
                                  # have materially more data, e.g. after a
                                  # fresh batch of qualifiers)
    python price.py               # FAST, run any time you want fresh prices

What "fast" means: the only real cost here is the bootstrap (N_BOOT model
refits for the confidence interval). That is far cheaper than the full grid x
folds tuning search. If you want it faster still, lower --n-boot; if you want
tighter intervals, raise it.

The card has no bookmaker odds in it. For each selection it prints:
  fair_odds      1 / model_prob (break-even if the model is exactly right)
  min_take_fair  fair odds + a small EV cushion on the point estimate
  min_take_safe  priced off the CI-LOW probability + a bigger cushion, so the
                 bet stays +EV even if the model's true probability is at the
                 pessimistic end of its uncertainty
  gap            extra price the uncertainty margin demands (wide = thin data)

Shop around: take the bet when a book's price >= the threshold you're using.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import date, datetime

import pandas as pd

from worldcup_mc.fit import load_results, compute_weights, fit_dixon_coles
from worldcup_mc import value as val

RESULTS = "worldcup_mc/data/results.csv"
FIXTURES_SLATE = "worldcup_mc/data/fixtures_slate.csv"
CONFIG_PATH = "worldcup_mc/data/tuned_config.json"
OUT_DIR = "reports"

# defaults; all overridable on the command line
DEFAULTS = dict(n_boot=200, ci=0.90, ev_small=0.02, ev_big=0.05,
                min_history="2006-01-01")


def load_config(path: str) -> dict:
    if not os.path.exists(path):
        raise SystemExit(
            f"No tuned config at {path}.\n"
            "Run `python tune_holdout_bet.py` once first -- it writes the "
            "config this script reads.")
    with open(path) as f:
        payload = json.load(f)
    cfg = payload.get("config", payload)   # tolerate a bare config too
    for k in ("half_life_days", "friendly_weight", "ridge"):
        if k not in cfg:
            raise SystemExit(f"{path} is missing '{k}'. Re-run tuning.")
    tuned_on = payload.get("tuned_on", "unknown date")
    print(f"using tuned config from {path} (tuned on {tuned_on}):")
    print(f"  half_life_days={cfg['half_life_days']:.0f}  "
          f"friendly_weight={cfg['friendly_weight']:.2f}  "
          f"ridge={cfg['ridge']:.3f}")
    return cfg


def price(cfg: dict, asof: str, n_boot: int, ci: float,
          ev_small: float, ev_big: float, min_history: str,
          use_confederation_prior: bool = False):
    history = load_results(RESULTS, min_date=min_history)
    print(f"loaded {len(history):,} matches "
          f"({history['date'].min().date()}..{history['date'].max().date()})")

    if not os.path.exists(FIXTURES_SLATE):
        raise SystemExit(
            f"No fixtures at {FIXTURES_SLATE}. Add rows: "
            "date,group,home,away,neutral")
    fx = pd.read_csv(FIXTURES_SLATE)

    # one point fit for the ratings file + a strength sanity print
    w = compute_weights(history, asof=asof,
                        half_life_days=cfg["half_life_days"],
                        friendly_weight=cfg["friendly_weight"])
    fit = fit_dixon_coles(history, w, ridge=cfg["ridge"],
                          use_confederation_prior=use_confederation_prior)
    if fit.conf_strength:
        print("confederation offsets:",
              {k: round(v, 3) for k, v in sorted(fit.conf_strength.items())})
    fit.write_ratings_csv("worldcup_mc/data/teams_fitted.csv")
    fit.write_params("worldcup_mc/data/params_fitted.json")
    print(f"fit {len(fit.teams)} teams as of {asof} | "
          f"home_adv={fit.home_adv:.3f} rho={fit.rho:.3f}")

    print(f"\nbootstrapping {n_boot} refits for the CI...")
    models = val.bootstrap_models(
        history, asof=asof, n_boot=n_boot,
        half_life_days=cfg["half_life_days"],
        friendly_weight=cfg["friendly_weight"], ridge=cfg["ridge"],
        use_confederation_prior=use_confederation_prior)

    print(f"\n--- PRICE-TARGET CARD ({ci:.0%} CI, no bookmaker odds needed) ---")
    print("Take the bet when the price you can find is at or above the "
          "threshold.\n")
    rows = []
    for r in fx.itertuples(index=False):
        if r.home not in models[0].teams or r.away not in models[0].teams:
            print(f"  skip {r.home} v {r.away}: team missing from fit")
            continue
        for t in val.fixture_targets(models, r.home, r.away,
                                     neutral=bool(r.neutral),
                                     ci=ci, ev_small=ev_small, ev_big=ev_big):
            rows.append({
                "date": r.date, "match": f"{r.home} v {r.away}",
                "market": t.market, "sel": t.selection,
                "p_model": round(t.p_model, 4),
                "p_lo": round(t.p_lo, 4), "p_hi": round(t.p_hi, 4),
                "fair_odds": t.fair_odds,
                "min_take_fair": t.min_odds_small,
                "min_take_safe": t.min_odds_big,
                "gap": round(t.min_odds_big - t.min_odds_small, 2),
            })

    if not rows:
        raise SystemExit("no fixtures could be priced (teams missing?)")

    card = pd.DataFrame(rows)
    print(card.to_string(index=False))

    os.makedirs(OUT_DIR, exist_ok=True)
    out_csv = os.path.join(
        OUT_DIR, f"price_targets_{datetime.now():%Y%m%d_%H%M%S}.csv")
    card.to_csv(out_csv, index=False)
    print(f"\nwritten to {out_csv}")
    print("\nShop around: price >= min_take_safe = confident bet; between the "
          "two = thin/small; below min_take_fair = pass.")


def main():
    ap = argparse.ArgumentParser(description="Price fixtures from a saved tuned config (no re-tuning).")
    ap.add_argument("--asof", default=str(date.today()),
                    help="as-of date for the fit (default: today)")
    ap.add_argument("--n-boot", type=int, default=DEFAULTS["n_boot"],
                    help="bootstrap refits for the CI (lower=faster, higher=tighter)")
    ap.add_argument("--ci", type=float, default=DEFAULTS["ci"])
    ap.add_argument("--ev-small", type=float, default=DEFAULTS["ev_small"])
    ap.add_argument("--ev-big", type=float, default=DEFAULTS["ev_big"])
    ap.add_argument("--min-history", default=DEFAULTS["min_history"])
    ap.add_argument("--no-conf-prior", action="store_true",
                    help="disable confederation strength offsets (on by default; "
                         "validated across folds to remove weak-region bias)")
    ap.add_argument("--config", default=CONFIG_PATH)
    args = ap.parse_args()

    cfg = load_config(args.config)
    price(cfg, asof=args.asof, n_boot=args.n_boot, ci=args.ci,
          ev_small=args.ev_small, ev_big=args.ev_big,
          min_history=args.min_history,
          use_confederation_prior=not args.no_conf_prior)


if __name__ == "__main__":
    main()
