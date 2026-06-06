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

## Plugging in real data (what to replace)

The package runs out of the box on **synthetic placeholder data**. For a
real model, swap in:

- **Ratings.** `teams_sample.csv` ratings are made up. Use fitted
  attack/defence parameters, or convert a single power rating (World
  Football Elo etc.) with `attack_defence_from_rating()`. Best practice is
  to fit `base`, `home_adv`, `rho` and per-team attack/defence to recent
  international results by maximum likelihood (with Dixon–Coles time
  weighting) rather than hand-setting them.
- **The draw.** The `group` column and `groups_from_teams()` use an
  illustrative draw, not the official December 5 groups.
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
  data/teams_sample.csv
make_sample_data.py  regenerates the placeholder ratings/draw
run_example.py       end-to-end demo
```