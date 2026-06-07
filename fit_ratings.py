"""
Fit real attack/defence ratings from historical results, then write them in
the format worldcup_mc expects.

    python fetch_data.py        # once, on a networked machine
    python fit_ratings.py

Outputs:
    worldcup_mc/data/teams_fitted.csv   (name, attack, defence)
    worldcup_mc/data/params_fitted.json (base, home_adv, rho)

Then point run_example.py at teams_fitted.csv (and merge in your group draw
via the `group` column) instead of the synthetic teams_sample.csv.
"""

from worldcup_mc.fit import load_results, compute_weights, fit_dixon_coles

RESULTS = "worldcup_mc/data/results.csv"
ASOF = "2026-06-11"          # tournament kickoff; decay is measured back from here
HALF_LIFE_DAYS = 730.0       # ~2 yrs; tune via backtest
FRIENDLY_WEIGHT = 0.3        # down-weight friendlies, don't discard them
RIDGE = 0.05                 # shrink sparse teams toward average; tune via backtest
MIN_DATE = "2014-01-01"      # bound compute; decay handles recency


def main():
    df = load_results(RESULTS, min_date=MIN_DATE)
    print(f"loaded {len(df):,} matches, {df['date'].min().date()}..{df['date'].max().date()}")

    w = compute_weights(df, asof=ASOF, half_life_days=HALF_LIFE_DAYS,
                        friendly_weight=FRIENDLY_WEIGHT)
    print(f"effective sample size (sum of weights): {w.sum():.0f}")

    fit = fit_dixon_coles(df, w, ridge=RIDGE, verbose=False)
    print(f"fit {len(fit.teams)} teams | base={fit.base:.3f} "
          f"home_adv={fit.home_adv:.3f} rho={fit.rho:.3f}")

    fit.write_ratings_csv("worldcup_mc/data/teams_fitted.csv")
    fit.write_params("worldcup_mc/data/params_fitted.json")

    # quick top-10 by overall strength (attack + defence)
    rank = sorted(fit.teams, key=lambda t: fit.attack[t] + fit.defence[t], reverse=True)
    print("\nTop 10 by attack+defence:")
    for t in rank[:10]:
        print(f"  {t:<20} atk {fit.attack[t]:+.2f}  def {fit.defence[t]:+.2f}")


if __name__ == "__main__":
    main()
