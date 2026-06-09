"""
Build a ranked bet card from the model vs the bookmaker prices you can see now.

    python fetch_data.py            # once, on a networked machine
    python value_example.py         # reads worldcup_mc/data/odds_slate.csv

INPUT: a CSV you fill in by hand (or pull from The Odds API, regions=uk) with
one row per fixture and the COMPLETE market for each market you want priced:

    date,home,away,neutral,odds_home,odds_draw,odds_away,odds_over25,odds_under25
    2026-06-12,Mexico,Poland,0,2.10,3.30,3.60,1.95,1.90
    ...

`neutral` = 1 for a neutral venue, 0 if `home` is really at home (the model's
home edge then applies). Leave the over/under columns blank to skip that
market. De-vig needs the whole market, so don't part-fill 1X2.

OUTPUT: a ranked card (biggest, most-confident edges first) printed to screen
and written to reports/, plus a CSV of every priced selection. Stake column is
the suggested £-tier from the CI-gated rule in value.py -- the £50 tier only
fires when the edge survives the model's own uncertainty.
"""

import os
import sys
from datetime import datetime

import pandas as pd

from worldcup_mc.fit import load_results
from worldcup_mc.value import bootstrap_models, price_fixture

RESULTS = "worldcup_mc/data/results.csv"
SLATE = "worldcup_mc/data/odds_slate.csv"
OUT_DIR = "reports"

N_BOOT = 200                    # bootstrap refits per as-of date (raise for tighter CIs)
HALF_LIFE = 1460.0
FRIENDLY_WEIGHT = 0.3
RIDGE = 0.05
DEVIG = "shin"
STAKE_TIERS = (10.0, 50.0)      # (small, big) in £
ASOF = None                     # None -> price as of "today"; or set "2026-06-10"


def _tee(path):
    class T:
        def __init__(s): s.f = open(path, "w", encoding="utf-8"); s.o = sys.stdout
        def write(s, x): s.o.write(x); s.f.write(x)
        def flush(s): s.o.flush(); s.f.flush()
        def close(s): s.f.close(); sys.stdout = s.o
    return T()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(OUT_DIR, f"betcard_{stamp}.txt")
    csv_path = os.path.join(OUT_DIR, f"betcard_{stamp}.csv")

    tee = _tee(report_path)
    sys.stdout = tee
    try:
        history = load_results(RESULTS, min_date="2006-01-01")
        slate = pd.read_csv(SLATE)
        slate["date"] = pd.to_datetime(slate["date"])
        asof = pd.Timestamp(ASOF) if ASOF else pd.Timestamp(datetime.now().date())
        print(f"Pricing {len(slate)} fixtures as-of {asof.date()} "
              f"({N_BOOT} bootstrap refits, de-vig={DEVIG})")

        # One ensemble for the whole slate (same as-of cut).
        models = bootstrap_models(
            history, asof=asof, n_boot=N_BOOT, half_life_days=HALF_LIFE,
            friendly_weight=FRIENDLY_WEIGHT, ridge=RIDGE)
        teams_known = set(models[0].teams) if models else set()

        rows = []
        for r in slate.itertuples(index=False):
            neutral = bool(int(getattr(r, "neutral", 1)))
            if r.home not in teams_known or r.away not in teams_known:
                print(f"  ! skipping {r.home} v {r.away}: team not in ratings "
                      f"(needs prior matches in history)")
                continue
            sels = price_fixture(models, r.home, r.away, neutral,
                                 book_row=r._asdict(), devig_method=DEVIG,
                                 stake_tiers=STAKE_TIERS)
            for s in sels:
                rows.append({
                    "date": r.date.date(), "match": f"{r.home} v {r.away}",
                    "market": s.market, "sel": s.selection, "odds": s.odds,
                    "p_model": s.p_model, "p_lo": s.p_lo, "p_hi": s.p_hi,
                    "p_fair_book": s.p_fair_book, "overround": s.overround,
                    "edge": s.edge, "ev": s.ev, "ev_lo": s.ev_lo,
                    "stake": s.stake})

        if not rows:
            print("\nNo selections priced -- check the slate teams are in the "
                  "ratings and the odds columns are filled.")
            return

        card = pd.DataFrame(rows)
        # Rank: bets we'd actually place first (by stake then EV), the rest by EV.
        card = card.sort_values(["stake", "ev"], ascending=[False, False]
                                ).reset_index(drop=True)

        show = card[card["stake"] > 0]
        print(f"\n=== SUGGESTED BETS ({len(show)} of {len(card)} selections clear the bar) ===")
        if len(show):
            print(show[["date", "match", "market", "sel", "odds", "p_model",
                        "p_lo", "p_fair_book", "edge", "ev", "ev_lo", "stake"]]
                  .round(3).to_string(index=False))
            print(f"\nTotal suggested stake: £{show['stake'].sum():.0f} "
                  f"across {len(show)} selections")
        else:
            print("None. The model is not confident enough vs these prices "
                  "to bet -- expected while ratings are still placeholders.")

        print("\n=== FULL CARD (all selections, ranked) ===")
        print(card[["match", "market", "sel", "odds", "p_model", "edge", "ev",
                    "stake"]].round(3).to_string(index=False))

        card.to_csv(csv_path, index=False)
    finally:
        print(f"\n--- written to ---\n  {os.path.abspath(report_path)}\n  {os.path.abspath(csv_path)}")
        tee.close()


if __name__ == "__main__":
    main()
