"""
Live in-tournament loop: record group-stage results as they finish, refit on
the latest data, and predict the next fixture either team plays.

    python fetch_data.py        # once, gets historical results.csv
    python predict_next.py

Run this between matches. As each group game ends, add an append_live_result
call (or maintain worldcup_mc/data/live_results.csv by hand in the same
schema), then re-run: the refit picks up every result recorded so far.
"""

import worldcup_mc as wc

RESULTS = "worldcup_mc/data/results.csv"
LIVE = "worldcup_mc/data/live_results.csv"
ASOF = "2026-06-15"          # 'today' during the tournament; decay measured from here


def main():
    # ---- 1. record finished results as they come in -------------------
    # Example (delete/replace with the real scorelines as they happen):
    wc.append_live_result(LIVE, "2026-06-12", "Mexico", "Poland", 2, 0,
                          tournament="FIFA World Cup 2026",
                          country="Mexico", neutral=False)   # host at home
    wc.append_live_result(LIVE, "2026-06-13", "Argentina", "Nigeria", 1, 1,
                          tournament="FIFA World Cup 2026", neutral=True)

    # ---- 2. refit on history + everything recorded live ---------------
    model, fit = wc.refit(RESULTS, LIVE, asof=ASOF,
                          half_life_days=730.0, live_boost=1.0)
    print(f"refit on {fit.n_matches:,} matches | base={fit.base:.3f} "
          f"home_adv={fit.home_adv:.3f} rho={fit.rho:.3f}")

    # ---- 3. predict the next fixture ----------------------------------
    card = wc.predict_next(model, "Argentina", "Mexico", neutral=True)
    print(f"\n{card['home']} vs {card['away']} (neutral={card['neutral']})")
    print(f"  expected goals: {card['expected_goals']['home']:.2f} - "
          f"{card['expected_goals']['away']:.2f}")
    p = card["1x2"]
    print(f"  1X2: home {p['home']:.3f} | draw {p['draw']:.3f} | away {p['away']:.3f}")
    ou = card["totals"][2.5]
    print(f"  O/U 2.5: over {ou['over']:.3f} | under {ou['under']:.3f}")
    b = card["btts"]
    print(f"  BTTS: yes {b['yes']:.3f} | no {b['no']:.3f}")
    print("  most likely scores: " +
          ", ".join(f"{h}-{a} ({pr:.2f})" for (h, a), pr in card["correct_score"][:4]))

    # ---- 4. (optional) per-match value vs the book's prices -----------
    # Pass the COMPLETE market (all three 1X2 selections) to de-vig correctly.
    book_1x2 = {"home": 2.4, "draw": 3.3, "away": 3.1}   # example decimal odds
    model_1x2 = {"home": p["home"], "draw": p["draw"], "away": p["away"]}
    print("\n  1X2 value vs book:")
    print(wc.compare_market(book_1x2, model_1x2, method="shin").round(4).to_string())


if __name__ == "__main__":
    main()
