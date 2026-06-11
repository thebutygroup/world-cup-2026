# World Cup 2026 — Monte Carlo Simulator

**In one sentence:** this code plays out the 2026 World Cup tens of thousands
of times on a computer, works out how often each team wins, and then checks
those numbers against bookmaker prices to spot bets where the odds look too
generous.

The idea behind it is simple. We can't know what will happen in a single
tournament — one tournament is decided by a handful of near-coin-flips. But
if we simulate it thousands of times, the share of simulations a team wins is
a good estimate of its true chance. If our estimate of a team's chance is
higher than the chance implied by the bookmaker's price, that's a potential
**value bet**: a price that pays out more than the risk justifies, *on
average, over many bets*.

The rest of this README walks through the pieces. Each section starts plainly,
then goes deeper for readers who want the technical detail.

## What it does

**Plainly:** four moving parts — a model of a single match, an engine that
plays a whole tournament, a loop that runs that tournament thousands of times,
and a comparison against bookmaker odds.

**In detail:**

1. **Match model** (`worldcup_mc/model.py`) — predicts the scoreline of one
   game. Each team carries an `attack` and a `defence` number; a strong attack
   and a weak opposing defence mean more expected goals. The maths is a
   *Dixon–Coles bivariate Poisson*: goals are modelled as a Poisson process
   (the standard model for "how many of a rare-ish event happen"), with a
   small correction (`rho`) for the fact that low-scoring results like 0-0,
   1-0, 0-1 and 1-1 happen at slightly different rates than independent
   randomness would predict. Expected goals are
   `λ_home = exp(base + attack_home − defence_away + home_adv)` and
   `λ_away = exp(base + attack_away − defence_home)`.
2. **Tournament engine** (`worldcup_mc/tournament.py`) — the real 2026
   structure: 12 groups of 4 (round-robin), top two of each group plus the
   eight best third-placed teams into a Round of 32, then R16 → QF → SF →
   final. Group tiebreakers: points, goal difference, goals for,
   head-to-head, then random draw. Knockout ties resolve via extra time
   (scoring rates scaled to 1/3 for the shorter period) and a mildly
   strength-weighted penalty shootout.
3. **Monte Carlo** — "Monte Carlo" just means *estimate something by running
   many random trials and averaging*. Here it runs N whole tournaments and
   reports each team's probability of reaching every stage, plus the outright
   title probability.
4. **Odds comparison** (`worldcup_mc/odds.py`) — bookmaker prices include a
   built-in margin (the "overround" or "vig") that makes the implied
   probabilities add up to more than 100%. We strip that margin off
   (proportional or Shin method) to recover the bookmaker's *fair* implied
   probability, then report `edge = model − book` and `EV per unit staked`
   (expected profit per £1 bet) at the quoted odds. Positive EV = value under
   the model.

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

**Plainly:** the attack and defence numbers are where almost all the accuracy
lives. We don't guess them — we *learn* them from tens of thousands of real
historical international results, letting the data tell us how strong each team
is. Recent games count more than old ones, friendlies count less than
competitive games, and teams with very few matches get pulled toward the
average so one fluky result doesn't make them look world-class.

**In detail:** `worldcup_mc/fit.py` fits the ratings by *weighted maximum
likelihood* — it finds the set of attack/defence numbers that makes the actual
historical scorelines most probable, with each match weighted by how much we
trust it.

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

Three knobs control the weighting. None of them should be guessed — each is
set by the backtest described in the next section.

- **`half_life_days`** (how fast old games fade) — instead of a hard cutoff
  that throws away everything before a date, every match's weight decays
  smoothly with age. A half-life of 730 days means a game from two years ago
  counts half as much as a game today. *Why smooth decay?* A hard
  "qualifiers-only" cutoff would sever the links between confederations —
  there's no competitive match connecting a European team to a South American
  one, so old friendlies are the only thread tying their rating scales
  together. Decay keeps that thread while still favouring recency.
- **`friendly_weight`** (how much friendlies count) — friendlies have
  experimental line-ups and low stakes, so they're down-weighted (0.3 = a
  friendly counts ~a third of a competitive match). They're *not* discarded,
  because they're often the only cross-confederation games that exist.
- **`ridge`** (how hard to shrink thin-data teams) — a team with only a few
  recorded games could look absurdly strong or weak on a couple of lucky
  scorelines. Ridge pulls every team's rating gently toward the average, with
  the pull mattering most for teams with little data. It's the cheapest form
  of a Bayesian "prior toward the mean": start from "average until proven
  otherwise." Raise it when teams are data-poor.

## Validating and tuning the model (the part that keeps us honest)

This is the heart of the project's rigor. A model that looks good on the data
it was built from tells you nothing — the real question is how well it predicts
games it has *never seen*. Everything below exists to answer that question
honestly and to set the three knobs above without fooling ourselves.

Run the whole thing with:

```bash
python fetch_data.py          # once, on a networked machine
python tune_holdout_bet.py    # tune -> holdout -> bet, end to end
```

### Walk-forward testing: only ever predict the future

**Plainly:** to test the model fairly, we pretend we're back in time. For each
historical match we want to score, we rebuild the model using *only* the games
that had happened before that match's kickoff, then ask it to predict that
match. The model never gets to peek at the result it's being tested on, or at
anything that happened afterward. We do this for hundreds of past matches and
see how good the predictions were.

**Why it matters:** the cardinal sin in this kind of work is *leakage* —
accidentally letting information from the future sneak into a prediction.
A model that has secretly seen the answer will look brilliant in testing and
then lose money in the real world. Walk-forward testing structurally prevents
that: information only ever flows forward in time.

**In detail** (`worldcup_mc/backtest.py`): we pick a test set (say every match
of the 2022 World Cup), group those matches by date, and for each date refit
the entire Dixon–Coles model on all results strictly *before* that date, then
score that day's matches. Every prediction uses only the past.

We score each match's home/draw/away forecast two ways, because *accuracy
alone is the wrong target* — we care about whether the probabilities are
trustworthy, not just whether the favourite was tipped:

- **Brier score** — the squared distance between the forecast and what actually
  happened. Forecast a team at 70% and they win, you're penalised for the
  missing 30%; forecast them at 99% and they lose, you're punished hard. Lower
  is better. It rewards being both *right* and *appropriately confident*.
- **Log loss** — similar, but punishes confident wrong calls far more
  severely. Saying something was nearly impossible and then watching it happen
  is very costly here.

We always report a **baseline** alongside — the score you'd get by predicting
nothing but the historical base rates (roughly how often home/draw/away occur
in general). If the model can't beat that, it has learned nothing useful. The
"skill" percentage is how much better than the baseline we are.

Matches where either team has almost no history are *skipped and counted*, not
silently dropped — their rating would be pure guesswork, and hiding the skips
would flatter the model.

### Tuning without fooling ourselves: the bootstrap gate

**Plainly:** we want to pick the best settings for the three knobs. The naive
approach — try several settings, keep whichever scores best — is a trap,
because on a few hundred matches the "best" score is mostly luck. Tiny
differences between settings are noise. So before we declare one setting better
than another, we make it *prove* the difference is real and not a fluke.

**Why it matters:** if you tune to noise, you'll pick settings that happened to
work on your test matches and will fail on new ones. This is overfitting, and
it's how confident-looking models quietly go broke.

**In detail:** the proof is a *paired bootstrap*. We take the per-match scores,
resample them thousands of times with replacement, and build a confidence
interval for the difference between two settings. A candidate only counts as
genuinely better than the baseline if that entire interval sits on the better
side of zero. If the interval straddles zero, the two settings are
indistinguishable and we keep the simpler baseline. `choose_config` enforces
this, and when nothing clears the bar it says so explicitly — *a flat result
is an answer, not a failure.* It means "stop tuning, you're chasing noise."

### Three improvements that make the tuning trustworthy

1. **Joint grid, not one knob at a time.** The three knobs interact — a longer
   half-life means more effective data per team, which means *less* shrinkage
   is needed. Tuning them one at a time can land on the wrong combination, so
   `gated_joint_sweep` sweeps them together over a grid.

2. **Multiple tournaments (folds), not one.** A setting that wins on the 2022
   World Cup alone might just suit that one event. We run the sweep across
   several independent test sets — the 2018 and 2022 World Cups, Euro 2020,
   and a multi-year block of qualifiers — and require a winning setting to hold
   up across most of them. Winning everywhere is evidence; winning once is
   luck. The `folds_won` column tracks this stability directly.

3. **Warm-starting, so all of the above is affordable.** Refitting the full
   model from scratch for every test date is slow, and a joint grid across many
   folds multiplies that cost enormously. But consecutive test dates differ by
   only a handful of matches, so the previous day's fitted ratings are an
   excellent starting point for the next. Feeding them in as the starting guess
   (`warm_start`) reaches the identical answer in a fraction of the iterations —
   verified to give a *bit-for-bit identical* solution at meaningfully higher
   speed. This is purely a speed optimization with no effect on results; it's
   what turns an overnight job into a coffee-break one and makes the honest,
   expensive validation above practical to actually run. Long qualifier windows
   can additionally refit every N days (`refit_interval_days`) instead of every
   single matchday — still strictly past-only, just slightly staler ratings.

### The three-stage pipeline

`tune_holdout_bet.py` ties it together with one hard rule: **the data you tune
on and the data you judge yourself on must never be the same data.**

- **Stage 1 — Tune** on pre-2024 tournaments only. The gated joint sweep across
  folds picks the settings (or keeps the baseline if nothing clears the gate).
- **Stage 2 — Holdout.** The chosen settings are frozen and evaluated *exactly
  once* on 2024-onward matches the tuning never touched (Euro 2024, Copa
  América 2024, 2024–25 qualifiers and Nations League). These are the only
  numbers you should actually trust, because no decision was made by looking at
  them. **Re-running Stage 1 and then re-checking Stage 2 burns the holdout** —
  once you've peeked, it's no longer unseen data, and you'd need a fresh window.
  Alongside Brier and log loss, this stage reports **calibration**: when the
  model says 30%, does it happen about 30% of the time? (See
  `worldcup_mc/calibration.py` for the reliability table, expected calibration
  error, and the Murphy reliability/resolution/uncertainty decomposition.)
  Calibration is the binding constraint on thin-edge betting — if our 30%
  forecasts only land 22% of the time, every "value" bet priced off them loses.
- **Stage 3 — Bet.** With settings locked, refit on *all* history as of today,
  write the ratings, and produce the normal value output: per-match 1X2 and
  over/under edges against book prices where you have them, and "minimum price
  to take" fair-odds targets for fixtures where you don't.

## Data sources

`worldcup_mc/data/SOURCES.csv` is the full manifest with URLs, licences and
what each layer feeds. Verified available as of June 2026:

- **Match results** — `martj42/international_results` (GitHub raw CSV, ~49k
  men's internationals). The one required input. `fetch_data.py` pulls it.
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
  thirds by rank rather than FIFA's fixed combination table. Replace both
  for a true-to-draw bracket — these only affect *who plays whom* in the
  knockouts, not the match model.

## Pitch surface dimension

**Plainly:** a bad pitch makes football scrappier — fewer goals, more even
contests, more upsets. This optional feature lets a poor playing surface nudge
the model toward lower-scoring, closer games. It's **off by default** and only
worth turning on once there's real data to size the effect.

`worldcup_mc/data/venues_2026.csv` carries a preliminary per-venue
`surface_risk_1to5`, and the model can apply a per-match `Surface` that
models a poor pitch as a *variance compressor*: `pace` (<1) lowers total
goals and `compression` (0..1) pulls the two teams' expectations together,
raising the draw/upset probability.

**It is OFF by default.** `load_venue_surfaces(..., effect_strength=0.0)`
returns identity surfaces, and `MatchModel` with no surface is unchanged.
Turn it up only once data sets the size:

```python
from worldcup_mc import load_venue_surfaces, calibrate_surface, Surface
m.outcome_probs("Spain", "Morocco", surface=Surface(pace=0.85, compression=0.2))

# estimate a surface from matches played on it (warm-ups, then group games):
surf, info = calibrate_surface(matches_df, baseline_model)  # shrinks to identity
```

`calibrate_surface` shrinks toward "no effect" by `n/(n+pseudocount)`, so a
handful of friendlies barely move the model — by design, the estimate
tightens as real matches accumulate. Wiring surfaces into the full
tournament needs the match → venue schedule (next step); the magnitudes in
`surface_from_risk` are placeholders until calibrated.

## Cohesion signal: shots on target per wage (weak prior, off by default)

**Plainly:** an expensive squad that isn't creating many chances may be a bag
of star individuals who haven't gelled into a team yet — a gap that often
closes as the tournament goes on. This optional signal tries to spot that by
comparing a team's shot output to what its wage bill implies it *should*
produce. It's a gentle nudge, not a verdict, and **off by default**.

`worldcup_mc/cohesion.py` builds a team "are they gelling?" signal: shots on
target per £m of on-field wages. The thesis: a big wage bill generating few
shots on target may be individuals, not a team — an early, exploitable
disparity that we expect to correct as they gel. So it's treated as a weak
prior, not a verdict.

```python
from worldcup_mc import (load_wages, impute_floor, team_wage_bill,
                         rolling_stwr, cohesion_multipliers, fit_sot_to_goals)

wages = load_wages("worldcup_mc/data/wages.csv")     # player -> weekly £
floor = impute_floor(wages, pct=10)                   # unknown players -> bottom 10%
# build match records (date, team, shots_on_target, wage_bill) using team_wage_bill(...)
stwr  = rolling_stwr(records, asof="2026-06-11", half_life_days=120)
mults = cohesion_multipliers(stwr, sensitivity=0.0)   # 0.0 = OFF
model.apply_cohesion(mults)                            # scales attacking lambda
```

Design points: missing wages are imputed at the bottom 10th percentile of
known wages (per the spec); STWR is decay-weighted so recent games dominate;
and the multiplier is shrunk by effective sample size `eff_n/(eff_n+k)`, so a
couple of friendlies barely move it. `sensitivity` defaults to 0 — the signal
is wired in but contributes nothing until a backtest justifies turning it up.

**Wealth-normalised, not a wealth tax.** A raw SoT/wage ratio would penalise
expensive squads automatically (more wages = lower ratio), which would just
fade the favourites rather than predict matches. Instead `cohesion_multipliers`
fits `log(SoT) = a + b*log(wage)` across teams and uses each team's *residual*
from that curve — did it create more or fewer shots than its talent predicts?
Two teams that both meet their wage-implied baseline get a neutral multiplier
regardless of wealth, so a poor, well-drilled side isn't punished for being
cheap and the signal carries information orthogonal to strength.

**Defensive version.** `defensive_stwr` runs the identical machinery on
shots on target *conceded* per wage (records need a
`shots_on_target_against` column). A side conceding fewer shots than its wage
bill predicts gets a residual < 0 and a multiplier < 1; feed it via
`model.apply_cohesion(defence_mults=...)` and it scales the goal rate the
team concedes (i.e. it's applied to opponents), so a well-drilled defence
correctly suppresses opponent scoring while a leaky one inflates it. Same
wealth-neutrality and shrinkage; same off-by-default `sensitivity`.

How strong is the signal? `fit_sot_to_goals(matches)` measures the
shots-on-target → goals relationship on your data. Public data puts it near
0.30 goals per shot on target (~31% conversion), but the R² is modest
(~0.3 in testing): SoT tracks goals loosely, not tightly, so expect this to
be a minor adjustment at most. It is also not opponent-adjusted — fewer SoT
against better defences — which is a further reason to keep `sensitivity`
low and lean on the backtest.

## Caveats worth knowing

- **De-vig needs the full market.** `compare_market()` normalises the
  selections you give it. If you pass only the favourites, the implied
  overround is understated and the `edge`/`book_fair` columns are wrong.
  Pass *all* outright selections. (`EV_per_unit` uses the raw quoted odds,
  so it's correct regardless.)
- **Independence across matches.** Each match is simulated independently;
  there's no in-tournament form/fatigue/injury dynamics.
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
  markets.py         per-match 1X2 / totals / BTTS / correct-score
  fit.py             time-decay Dixon-Coles MLE ratings fitter (+ warm start)
  backtest.py        walk-forward backtest + gated multi-fold tuning
  calibration.py     bootstrap CIs, reliability/ECE, Murphy decomposition
  value.py           bootstrap-ensemble value card with CI-gated stakes
  data/
    teams_sample.csv         synthetic placeholder ratings/draw
    SOURCES.csv              data-source manifest (URLs, licences)
    host_advantage_2026.csv  host nations + suggested home edge
    fixtures_slate.csv       upcoming fixtures (no odds needed)
    odds_slate.csv           upcoming fixtures with book prices
fetch_data.py        downloads real results.csv (run on networked machine)
fit_ratings.py       fetch -> fit -> write teams_fitted.csv
tune_holdout_bet.py  full pipeline: tune (Stage 1) -> holdout (Stage 2) -> bet (Stage 3)
backtest_example.py  single-tournament walk-forward backtest demo
calibration_example.py  calibration + bootstrap demo on a group-stage-like set
make_sample_data.py  regenerates the placeholder ratings/draw
run_example.py       end-to-end simulation + odds demo
```