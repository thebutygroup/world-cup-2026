"""
Backtest the match model against a real tournament held with the same
national teams that play in 2026 -- e.g. the 2022 World Cup or Euro 2024.

    python fetch_data.py        # once, on a networked machine
    python backtest_example.py

This scores the model's per-match 1X2 forecasts (Brier + log loss) on every
match of the chosen tournament, refitting on only the data available before
each matchday so there is no leakage. It also sweeps the decay half-life so
you can pick the value that minimises out-of-sample Brier.

Swap TOURNAMENT/YEAR for other validation sets, all present in the martj42
file:
    "FIFA World Cup", 2022
    "UEFA Euro",      2024
    "Copa",           2024     # Copa America
    "Gold Cup",       2023     # CONCACAF (the 'North American Cup')
    "Nations League", None     # UEFA/CONCACAF Nations League (many matches)
"""

from worldcup_mc.fit import load_results
from worldcup_mc.backtest import filter_matches, backtest, half_life_sweep

RESULTS = "worldcup_mc/data/results.csv"
TOURNAMENT = "FIFA World Cup"
YEAR = 2022


def main():
    # Full universe the model may learn from (kept wide; decay handles recency).
    history = load_results(RESULTS, min_date="2006-01-01")

    # The matches we score are a subset of that same source.
    test = filter_matches(history, tournament_contains=TOURNAMENT, year=YEAR)
    # Exclude third-place/odd labels if you like; here we keep all of them.
    print(f"test set: {len(test)} matches "
          f"({test['date'].min().date()}..{test['date'].max().date()})")

    res = backtest(history, test, half_life_days=730.0,
                   friendly_weight=0.3, ridge=0.05, verbose=True)
    print("\n=== Backtest ===")
    print(res.summary())

    print("\nWorst-predicted matches (highest Brier = biggest surprises):")
    print(res.per_match.sort_values("brier", ascending=False).head(8).round(3)
          .to_string(index=False))

    print("\n=== Half-life sweep (pick the lowest Brier) ===")
    sweep = half_life_sweep(history, test,
                            half_lives=(365.0, 547.0, 730.0, 1095.0, 1460.0),
                            friendly_weight=0.3, ridge=0.05)
    print(sweep.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
