"""
End-to-end example: simulate the tournament, then compare the model's
outright-winner probabilities to bookmaker prices to find value.

    python run_example.py

Replace teams_sample.csv with real ratings + the official draw, and replace
the BOOKMAKER_ODDS dict with prices you've pulled from UK books.
"""

import pandas as pd

import worldcup_mc as wc

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", None)

N = 50_000  # bump for tighter estimates; ~0.5 ms/sim


def main():
    wc.seed(2026)
    teams = wc.load_teams("worldcup_mc/data/teams_sample.csv")
    groups = wc.groups_from_teams(teams)

    # Neutral venues -> home_adv = 0. Give hosts a small edge if you like by
    # raising home_adv and flagging their matches as non-neutral.
    model = wc.MatchModel(teams, base=0.10, home_adv=0.0, rho=-0.05)

    print(f"Simulating {N:,} tournaments...")
    probs = wc.monte_carlo(model, groups, n=N)
    print("\n=== Advancement & title probabilities (top 16) ===")
    print(probs.head(16).round(3))

    # ----- Outright winner: model vs bookmaker -----
    # ILLUSTRATIVE UK outright decimal odds. Replace with real prices.
    BOOKMAKER_ODDS = {
        "Spain": 5.5, "Argentina": 6.5, "France": 7.0, "Brazil": 8.0,
        "England": 8.5, "Portugal": 13.0, "Germany": 15.0, "Netherlands": 17.0,
        "Belgium": 26.0, "Italy": 34.0, "Croatia": 51.0, "Morocco": 67.0,
    }
    model_win = probs["P(win)"].to_dict()

    table = wc.compare_market(BOOKMAKER_ODDS, model_win, method="shin")
    print(f"\n=== Outright value (book overround = {table.attrs['overround']:.3f}) ===")
    print(table.round(4))

    value = table[table["EV_per_unit"] > 0]
    print("\n=== Positive-EV selections (model thinks these are value) ===")
    print(value.round(4) if len(value) else "  none at these prices")


if __name__ == "__main__":
    main()
