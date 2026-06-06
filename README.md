# World Cup 2026 — Monte Carlo Simulator

A small, modular Python package that simulates the 48-team 2026 World Cup
many thousands of times and compares the resulting probabilities to UK
bookmaker prices to surface value bets.

## What it does

1. **Match model** (`worldcup_mc/model.py`) — scorelines from a
   Dixon–Coles bivariate Poisson. Each team has an `attack` and `defence`
   rating; expected goals are
   `λ_home = exp(base + attack_home − defence_away + home_adv)` and
   `λ_away = exp(base + attack_away − defence_home)`. The Dixon–Coles `rho`
   term corrects the dependence in 0-0/1-0/0-1/1-1 results that an
   independent Poisson gets wrong.
1. **Tournament engine** (`worldcup_mc/tournament.py`) — the real 2026
   structure: 12 groups of 4 (round-robin), top two of each group plus the
   eight best third-placed teams into a Round of 32, then R16 → QF → SF →
   final. Group tiebreakers: points, goal difference, goals for,
   head-to-head, then random draw. Knockout ties resolve via extra time
   (rates scaled to 1/3) and a mildly strength-weighted shootout.
1. **Monte Carlo** — runs N tournaments and returns each team’s probability
   of reaching every stage plus the outright title probability.
1. **Odds comparison** (`worldcup_mc/odds.py`) — strips the bookmaker margin
   (proportional or Shin method) to get fair book probabilities, then
   reports `edge = model − book` and `EV per unit staked` at the quoted
   odds. Positive EV = value under the model.

## Quick start

```bash
pip install numpy scipy pandas
python make_sample_data.py     # writes worldcup_mc/data/teams_sample.csv
python run_example.py
```

```python
import worldcup_mc as wc

wc.seed(2026)
teams  = wc.load_teams("worldcup_mc/data/teams_sample.csv")
groups = wc.groups_from_teams(teams)
model  = wc.MatchModel(teams, base=0.10, home_adv=0.0, rho=-0.05)

probs  = wc.monte_carlo(model, groups, n=50_000)   # ~0.5 ms/sim
print(probs.sort_values("P(win)", ascending=False).head(12))

odds = {"Spain": 5.5, "Argentina": 6.5, "France": 7.0, ...}  # full market!
print(wc.compare_market(odds, probs["P(win)"].to_dict(), method="shin"))
```

## Fitting real ratings (the part that matters most)

The single biggest accuracy lever is replacing the synthetic ratings with
ones fitted to real results. `worldcup_mc/fit.py` does this by weighted
maximum likelihood:

```bash
python fetch_data.py     # downloads results.csv from martj42 (networked machine)
python fit_ratings.py    # -> worldcup_mc/data/teams_fitted.csv + params_fitted.json
```

```python
from worldcup_mc import load_results, compute_weights, fit_dixon_coles
df  = load_results("worldcup_mc/data/results.csv", min_date="2014-01-01")
w   = compute_weights(df, asof="2026-06-11", half_life_days=730, friendly_weight=0.3)
fit = fit_dixon_coles(df, w, ridge=0.05)
model = fit.to_match_model()      # base/home_adv/rho all estimated
```

Key knobs (all to be tuned against a backtest, not guessed):

- **`half_life_days`** — exponential time decay. ~2 years by default; smooth
  decay rather than a hard cutoff, so old friendlies still bridge the
  confederations a qualification-only dataset would sever.
- **`friendly_weight`** — friendlies down-weighted, not discarded.
- **`ridge`** — shrinks sparse/minnow teams toward the average (cheap
  Bayesian-prior-toward-mean); raise it when a team has few games.

## Data sources

`worldcup_mc/data/SOURCES.csv` is the full manifest with URLs, licences and
what each layer feeds. Verified available as of June 2026:

- **Match results** — `martj42/international_results` (GitHub raw CSV, ~49k
  men’s internationals). The one required input. `fetch_data.py` pulls it.
- **Club Elo** — `api.clubelo.com` (free CSV API) for league-strength
  calibration in the later player-based prior.
- **2026 hosts** — `host_advantage_2026.csv` (USA, Canada, Mexico) for
  applying a non-neutral home edge.

Secondary sources for the player layer / backtest (Transfermarkt squad
values, FBref/Understat xG, Football-Data.co.uk historical odds) are listed
with access caveats — note Transfermarkt scraping is against its ToS.

## Plugging in real data (what to replace)

- **Ratings.** `teams_sample.csv` is synthetic (attack == defence for every
  team). Fit real ones with `fit.py` as above, or convert a single power
  rating with `attack_defence_from_rating()`.
- **The R32 bracket.** `DEFAULT_R32_BRACKET` is structurally valid but not
  the official pairing, and `assign_third_slots()` routes the eight best
  thirds by rank rather than FIFA’s fixed combination table. Replace both
  for a true-to-draw bracket — these only affect *who plays whom* in the
  knockouts, not the match model.

## Caveats worth knowing

- **De-vig needs the full market.** `compare_market()` normalises the
  selections you give it. If you pass only the favourites, the implied
  overround is understated and the `edge`/`book_fair` columns are wrong.
  Pass *all* outright selections. (`EV_per_unit` uses the raw quoted odds,
  so it’s correct regardless.)
- **Independence across matches.** Each match is simulated independently;
  there’s no in-tournament form/fatigue/injury dynamics.
- **Shootouts** are near coin-flips with a tiny strength lean — deliberately
  conservative.
- This is a modelling tool, not betting advice.

## Layout

```
worldcup_mc/
  __init__.py        public API
  model.py           Dixon-Coles match model + sampling
  tournament.py      group/knockout structure + Monte Carlo loop
  odds.py            de-vig + value comparison
  fit.py             time-decay Dixon-Coles MLE ratings fitter
  data/
    teams_sample.csv         synthetic placeholder ratings/draw
    SOURCES.csv              data-source manifest (URLs, licences)
    host_advantage_2026.csv  host nations + suggested home edge
fetch_data.py        downloads real results.csv (run on networked machine)
fit_ratings.py       fetch -> fit -> write teams_fitted.csv
make_sample_data.py  regenerates the placeholder ratings/draw
run_example.py       end-to-end simulation + odds demo
```