"""
Fit team attack/defence ratings (plus base rate, home advantage and the
Dixon-Coles rho) by weighted maximum likelihood on historical results.

Design choices, following the quant discussion:

  * Time decay, not a hard cutoff. Each match is weighted by
    exp(-ln2 * age_days / half_life_days). Old games fade smoothly but still
    provide the cross-confederation "connective tissue" a knife-edge
    qualification-only cutoff would destroy.
  * Friendlies down-weighted (experimental line-ups, low stakes) but NOT
    discarded -- they're often the only games linking UEFA to CONMEBOL etc.
  * Ridge shrinkage on attack/defence. With ~10 games/year per nation, this
    pulls sparse/minnow teams toward the average instead of overfitting a
    couple of fluky scorelines -- the Bayesian-prior-toward-mean idea, in
    its cheapest form. Tune `ridge` up for more shrinkage.
  * Home advantage applied only to non-neutral matches (the `neutral` flag),
    so it's estimated from real host effects and can be reused for the 2026
    hosts.

Input schema (martj42/international_results results.csv):
    date, home_team, away_team, home_score, away_score, tournament,
    city, country, neutral

Output: name, attack, defence  (ready for worldcup_mc.load_teams), plus a
params dict {base, home_adv, rho} for the MatchModel constructor.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

LN2 = math.log(2.0)


# --------------------------------------------------------------------------
# data loading / weighting
# --------------------------------------------------------------------------
# Teams that appear in the martj42 results set but are NOT FIFA members
# eligible for the World Cup. They play mostly against each other or in
# regional games (Island Games, etc.), often racking up lopsided scores
# against very weak opposition. Left in, the fit reads those goal-fests as
# world-class attack with a barely-tested defence -- e.g. Isle of Man and
# Jersey rating above Brazil and Germany. Excluded by default.
NON_FIFA_TEAMS = frozenset({
    "Isle of Man", "Jersey", "Guernsey", "Alderney", "Greenland",
    "Monaco", "Vatican City", "Vatican", "Kiribati", "Tuvalu",
    "Federated States of Micronesia", "Micronesia", "Palau",
    "Northern Cyprus", "Western Sahara", "Zanzibar", "Saare County",
    "Yorkshire", "Kernow", "Cornwall", "Shetland", "Orkney",
    "Frøya", "Hitra", "Falkland Islands", "Saint Helena",
    "Niue", "Tokelau", "Wallis and Futuna", "Tahiti reserves",
    "Kosovo U21", "Padania", "Occitania", "Provence", "Sápmi",
    "Chagos Islands", "Saint Pierre and Miquelon", "Bonaire",
    "Sint Maarten", "French Guiana", "Martinique", "Guadeloupe",
    "Saint Martin", "Réunion", "Mayotte",
    "Elba Island", "Parishes of Jersey", "Surrey", "Artsakh", "Sealand",
})


def load_results(
    path: str,
    min_date: str | None = "2014-01-01",
    exclude_non_fifa: bool = True,
    extra_exclude: set[str] | None = None,
) -> pd.DataFrame:
    """Load results.csv (martj42 schema). `min_date` bounds compute size;
    decay handles recency, so keep enough history to bridge confederations
    (a few World Cup cycles).

    exclude_non_fifa: drop matches involving non-FIFA teams (Isle of Man,
    Jersey, Island-Games sides, French overseas territories that aren't FIFA
    members, etc.). These distort the ratings badly -- they pile up goals
    against minnows and never face real defences -- so they're removed by
    default. Set False to keep them; add your own names via extra_exclude.
    """
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["home_score", "away_score"]).copy()
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    if "neutral" in df.columns:
        df["neutral"] = df["neutral"].astype(str).str.lower().isin(["true", "1", "yes"])
    else:
        df["neutral"] = True
    if min_date is not None:
        df = df[df["date"] >= pd.Timestamp(min_date)]
    if exclude_non_fifa or extra_exclude:
        drop = set(NON_FIFA_TEAMS) if exclude_non_fifa else set()
        if extra_exclude:
            drop |= set(extra_exclude)
        before = len(df)
        df = df[~df["home_team"].isin(drop) & ~df["away_team"].isin(drop)]
        removed = before - len(df)
        if removed:
            print(f"load_results: excluded {removed} matches involving "
                  f"{len(drop)} non-FIFA/regional teams")
    return df.reset_index(drop=True)


def compute_weights(
    df: pd.DataFrame,
    asof: str | pd.Timestamp,
    half_life_days: float = 730.0,
    friendly_weight: float = 0.3,
    competition_weights: dict[str, float] | None = None,
) -> np.ndarray:
    """Exponential time decay * competition importance."""
    asof = pd.Timestamp(asof)
    age = (asof - df["date"]).dt.days.to_numpy(dtype=float)
    age = np.clip(age, 0.0, None)
    w = np.exp(-LN2 * age / half_life_days)

    tour = df["tournament"].astype(str)
    if competition_weights:
        cw = tour.map(competition_weights).fillna(1.0).to_numpy()
    else:
        cw = np.where(tour.str.contains("Friendly", case=False), friendly_weight, 1.0)
    return w * cw


# --------------------------------------------------------------------------
# model fit
# --------------------------------------------------------------------------
@dataclass
class FitResult:
    teams: list[str]
    attack: dict[str, float]
    defence: dict[str, float]
    base: float
    home_adv: float
    rho: float
    n_matches: int
    loglik: float
    conf_strength: dict[str, float] | None = None

    def write_ratings_csv(self, path: str, groups: dict[str, str] | None = None) -> None:
        from .confederations import confederation_of
        rows = []
        for t in self.teams:
            row = {"name": t, "attack": round(self.attack[t], 4),
                   "defence": round(self.defence[t], 4)}
            if self.conf_strength is not None:
                row["conf_off"] = round(
                    self.conf_strength.get(confederation_of(t), 0.0), 4)
            if groups is not None:
                row["group"] = groups.get(t, "")
            rows.append(row)
        pd.DataFrame(rows).to_csv(path, index=False)

    def write_params(self, path: str) -> None:
        payload = {"base": self.base, "home_adv": self.home_adv,
                   "rho": self.rho, "n_matches": self.n_matches,
                   "loglik": self.loglik}
        if self.conf_strength is not None:
            payload["conf_strength"] = self.conf_strength
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    def to_match_model(self, **kwargs):
        from .model import Team, MatchModel
        from .confederations import confederation_of
        cs = self.conf_strength or {}
        teams = {
            t: Team(t, self.attack[t], self.defence[t],
                    conf_off=cs.get(confederation_of(t), 0.0))
            for t in self.teams
        }
        return MatchModel(teams, base=self.base, home_adv=self.home_adv,
                          rho=self.rho, **kwargs)


def _dc_tau_log(x, y, lam, mu, rho):
    """log of the Dixon-Coles low-score correction, vectorised."""
    tau = np.ones_like(lam)
    m00 = (x == 0) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m10 = (x == 1) & (y == 0)
    m11 = (x == 1) & (y == 1)
    tau[m00] = 1.0 - lam[m00] * mu[m00] * rho
    tau[m01] = 1.0 + lam[m01] * rho
    tau[m10] = 1.0 + mu[m10] * rho
    tau[m11] = 1.0 - rho
    return np.log(np.clip(tau, 1e-10, None))


def fit_dixon_coles(
    df: pd.DataFrame,
    weights: np.ndarray,
    ridge: float = 0.05,
    max_home_adv: float = 1.0,
    verbose: bool = False,
    warm_start: "FitResult | None" = None,
    maxiter: int = 500,
    use_confederation_prior: bool = False,
) -> FitResult:
    """
    Weighted-MLE fit. Returns centred attack/defence (mean 0), a base rate,
    home advantage and rho.

    warm_start: a previous FitResult to initialise theta from (teams matched
    by name; unseen teams start at 0). In a walk-forward loop, consecutive
    training sets differ by a handful of matches, so the previous optimum is
    an excellent starting point -- L-BFGS converges in a small fraction of
    the cold-start iterations. The centred parameters are likelihood-
    equivalent to the raw optimum (the centring shift cancels inside
    base + attack - defence), so they are a valid initialisation.

    use_confederation_prior: adds one explicit strength-offset parameter per
    confederation to the likelihood, entering every match as the DIFFERENCE
    between the two sides' offsets. Within-region games are untouched (the
    difference is zero), so each team's attack/defence stays identified by
    regional play; the offsets are identified purely by cross-region games
    and put the regions on one absolute scale. Without this, "best team in a
    weak region" and "strong in absolute terms" are confounded because the
    fit can't tell how strong the region's opposition was. The cross-confederation
    games (World Cups, playoffs, friendlies) set where each confederation mean
    sits, so a team in a weak confederation is pulled toward a weaker baseline.
    """
    teams = sorted(set(df["home_team"]) | set(df["away_team"]))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)

    # confederation strength offsets: one scalar per confederation, shared by
    # all its teams, identified by CROSS-confederation games. This puts the
    # regions on a common absolute scale -- within-region games identify each
    # team's attack/defence relative to its peers, cross-region games identify
    # how strong the whole region is. Without this, "best in a weak region"
    # and "strong in absolute terms" are indistinguishable.
    n_conf = 0
    conf_home_i = conf_away_i = None
    if use_confederation_prior:
        from .confederations import confederation_of
        team_conf = [confederation_of(t) for t in teams]
        uniq = sorted(set(team_conf))
        n_conf = len(uniq)
        ci = {c: j for j, c in enumerate(uniq)}
        team_conf_idx = np.array([ci[c] for c in team_conf])
        conf_home_i = team_conf_idx[df["home_team"].map(idx).to_numpy()]
        conf_away_i = team_conf_idx[df["away_team"].map(idx).to_numpy()]
        _conf_names = uniq

    hi = df["home_team"].map(idx).to_numpy()
    ai = df["away_team"].map(idx).to_numpy()
    x = df["home_score"].to_numpy()
    y = df["away_score"].to_numpy()
    home_flag = (~df["neutral"].to_numpy()).astype(float)
    w = np.asarray(weights, dtype=float)

    # theta = [attack(n), defence(n), base, home_adv, rho, conf_str(n_conf)]
    # conf_str entries are absolute regional strength offsets; they only enter
    # via the DIFFERENCE between the two teams' confederations, so they vanish
    # for within-region games and are identified purely by cross-region games.
    def unpack(theta):
        atk = theta[:n]
        dfc = theta[n:2 * n]
        base, gamma, rho = theta[2 * n], theta[2 * n + 1], theta[2 * n + 2]
        cstr = theta[2 * n + 3:2 * n + 3 + n_conf] if n_conf else None
        return atk, dfc, base, gamma, rho, cstr

    def nll(theta):
        atk, dfc, base, gamma, rho, cstr = unpack(theta)
        if cstr is not None:
            # relative confederation strength enters each side's scoring rate:
            # a stronger region scores more / concedes less in cross-region games
            ch = cstr[conf_home_i]
            ca = cstr[conf_away_i]
            log_lam = base + atk[hi] - dfc[ai] + gamma * home_flag + (ch - ca)
            log_mu = base + atk[ai] - dfc[hi] + (ca - ch)
        else:
            log_lam = base + atk[hi] - dfc[ai] + gamma * home_flag
            log_mu = base + atk[ai] - dfc[hi]
        lam = np.exp(log_lam)
        mu = np.exp(log_mu)
        ll = (_dc_tau_log(x, y, lam, mu, rho)
              + x * log_lam - lam
              + y * log_mu - mu)
        neg = -np.sum(w * ll)
        neg += ridge * (np.dot(atk, atk) + np.dot(dfc, dfc))  # shrink + identify
        if cstr is not None:
            # light anchor on offsets: keeps them 0 unless cross-region games
            # provide real evidence (and pins the overall mean for identifiability)
            neg += ridge * np.dot(cstr, cstr)
        return neg

    n_par = 2 * n + 3 + n_conf
    theta0 = np.zeros(n_par)
    theta0[2 * n] = math.log(max(df[["home_score", "away_score"]].to_numpy().mean(), 0.3))
    theta0[2 * n + 1] = 0.25   # home_adv start
    theta0[2 * n + 2] = -0.05  # rho start

    if warm_start is not None:
        for t, i in idx.items():
            prev = warm_start.attack.get(t)
            if prev is not None:
                theta0[i] = prev
                theta0[n + i] = warm_start.defence[t]
        theta0[2 * n] = warm_start.base
        theta0[2 * n + 1] = min(max(warm_start.home_adv, 0.0), max_home_adv)
        theta0[2 * n + 2] = min(max(warm_start.rho, -0.2), 0.2)
        if n_conf and getattr(warm_start, "conf_strength", None):
            for c, j in ((c, j) for j, c in enumerate(_conf_names)):
                theta0[2 * n + 3 + j] = warm_start.conf_strength.get(c, 0.0)

    bounds = ([(None, None)] * (2 * n)
              + [(None, None), (0.0, max_home_adv), (-0.2, 0.2)]
              + [(None, None)] * n_conf)
    res = minimize(nll, theta0, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": maxiter, "disp": verbose})

    atk, dfc, base, gamma, rho, cstr = unpack(res.x)
    # centre for interpretability, folding the means into base
    a_mean, d_mean = atk.mean(), dfc.mean()
    base = base + a_mean - d_mean
    atk = atk - a_mean
    dfc = dfc - d_mean

    conf_strength = None
    if cstr is not None:
        # centre offsets to mean 0 (identifiability: only differences matter).
        # NOTE: offsets are NOT folded into attack/defence -- the likelihood
        # applies them as a per-match difference between the two sides'
        # confederations, and no attack/defence adjustment reproduces that
        # (folding either cancels in net rating or double-counts; both were
        # bugs in earlier attempts). They stay a separate term that
        # MatchModel applies at prediction time via Team.conf_off.
        cstr = cstr - cstr.mean()
        conf_strength = {c: float(cstr[j]) for j, c in enumerate(_conf_names)}

    return FitResult(
        teams=teams,
        attack={t: float(atk[idx[t]]) for t in teams},
        defence={t: float(dfc[idx[t]]) for t in teams},
        base=float(base), home_adv=float(gamma), rho=float(rho),
        n_matches=int(len(df)), loglik=float(-res.fun),
        conf_strength=conf_strength,
    )
