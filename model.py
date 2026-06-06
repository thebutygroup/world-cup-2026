"""
Match-outcome model.

Goals are modelled with a bivariate Poisson via the Dixon & Coles (1997)
low-score correction. Each team has an `attack` and a `defence` rating.
Expected goals for a fixture are:

    log(lambda_home) = base + attack_home - defence_away + home_adv
    log(lambda_away) = base + attack_away - defence_home

Conventions:
  * higher `attack`  -> scores more goals
  * higher `defence` -> concedes fewer goals
  * `home_adv` is applied only to the designated home side; for neutral
    World Cup venues set it to 0 (host nations are the usual exception).

The Dixon-Coles tau term corrects the dependence between the two scores for
the 0-0/1-0/0-1/1-1 results that an independent Poisson misses.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Module-level RNG. Call seed() for reproducibility.
_rng = np.random.default_rng()


def seed(value: int | None) -> None:
    global _rng
    _rng = np.random.default_rng(value)


@dataclass
class Team:
    name: str
    attack: float
    defence: float
    group: str | None = None


@dataclass(frozen=True)
class Surface:
    """
    A per-match pitch effect. The default is identity (no effect), so the
    surface dimension is OFF until calibration data sets its magnitude.

    pace        : multiplier on total goal expectation. <1 = slower pitch /
                  fewer goals; 1.0 = neutral.
    compression : in [0, 1]; pulls the two teams' goal expectations toward
                  their shared mean, shrinking supremacy and so raising the
                  draw/upset probability. 0.0 = neutral.
    name        : optional label (venue or surface tag).

    Both act in log space so they compose cleanly and the pace effect is
    level-only while compression is gap-only:
        log(lambda') = m + (1 - compression) * (log(lambda) - m) + log(pace)
    where m is the mean of the two teams' log expectations.
    """
    pace: float = 1.0
    compression: float = 0.0
    name: str | None = None

    @property
    def is_identity(self) -> bool:
        return self.pace == 1.0 and self.compression == 0.0


def load_teams(path: str) -> dict[str, Team]:
    """Load a CSV with columns: name, attack, defence, [group]."""
    df = pd.read_csv(path)
    required = {"name", "attack", "defence"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"teams file missing columns: {missing}")
    teams: dict[str, Team] = {}
    for row in df.itertuples(index=False):
        teams[row.name] = Team(
            name=row.name,
            attack=float(row.attack),
            defence=float(row.defence),
            group=getattr(row, "group", None),
        )
    return teams


def attack_defence_from_rating(
    rating: float, ref: float = 1500.0, scale: float = 200.0, spread: float = 0.5
) -> tuple[float, float]:
    """
    Convenience converter for people who only have a single power/Elo-style
    rating per team rather than separate attack/defence parameters.

    A team `spread` log-units above average both scores more and concedes
    less, symmetrically. `scale` controls how many rating points equal one
    standard strength unit. Returns (attack, defence).
    """
    z = (rating - ref) / scale
    return spread * z, spread * z


@dataclass
class MatchModel:
    """
    Holds ratings + global parameters and produces scoreline samples.

    Parameters
    ----------
    teams : mapping name -> Team
    base : intercept; exp(base) is roughly the goals an average side scores
           against an average side on neutral ground.
    home_adv : additive log home advantage (0 for neutral venues).
    rho : Dixon-Coles dependence parameter (typically small & negative).
    max_goals : truncation for the score grid.
    """

    teams: dict[str, Team]
    base: float = 0.1
    home_adv: float = 0.25
    rho: float = -0.05
    max_goals: int = 12
    _cache: dict = field(default_factory=dict, repr=False)

    # ---- expected goals -------------------------------------------------
    def lambdas(self, home: str, away: str, neutral: bool = True,
                surface: "Surface | None" = None) -> tuple[float, float]:
        h, a = self.teams[home], self.teams[away]
        adv = 0.0 if neutral else self.home_adv
        lam_h = math.exp(self.base + h.attack - a.defence + adv)
        lam_a = math.exp(self.base + a.attack - h.defence)
        if surface is not None and not surface.is_identity:
            Lh, La = math.log(lam_h), math.log(lam_a)
            m = 0.5 * (Lh + La)
            c = surface.compression
            lp = math.log(surface.pace)
            lam_h = math.exp(m + (1.0 - c) * (Lh - m) + lp)
            lam_a = math.exp(m + (1.0 - c) * (La - m) + lp)
        return lam_h, lam_a

    # ---- scoreline distribution ----------------------------------------
    def _score_matrix(self, lam_h: float, lam_a: float) -> np.ndarray:
        n = self.max_goals + 1
        gh = np.arange(n)
        ph = np.exp(-lam_h) * lam_h**gh / np.array([math.factorial(k) for k in gh])
        pa = np.exp(-lam_a) * lam_a**gh / np.array([math.factorial(k) for k in gh])
        mat = np.outer(ph, pa)  # independent Poisson

        # Dixon-Coles low-score correction
        rho = self.rho
        mat[0, 0] *= 1.0 - lam_h * lam_a * rho
        mat[0, 1] *= 1.0 + lam_h * rho
        mat[1, 0] *= 1.0 + lam_a * rho
        mat[1, 1] *= 1.0 - rho
        mat = np.clip(mat, 0.0, None)
        mat /= mat.sum()
        return mat

    def _cumulative(self, home: str, away: str, neutral: bool,
                    surface: "Surface | None" = None) -> tuple[np.ndarray, int]:
        key = (home, away, neutral, surface)
        cached = self._cache.get(key)
        if cached is None:
            lam_h, lam_a = self.lambdas(home, away, neutral, surface)
            mat = self._score_matrix(lam_h, lam_a)
            cum = np.cumsum(mat.ravel())
            cum[-1] = 1.0  # guard against fp drift
            cached = (cum, mat.shape[1])
            self._cache[key] = cached
        return cached

    # ---- sampling -------------------------------------------------------
    def sample_score(self, home: str, away: str, neutral: bool = True,
                     surface: "Surface | None" = None) -> tuple[int, int]:
        cum, ncols = self._cumulative(home, away, neutral, surface)
        idx = int(np.searchsorted(cum, _rng.random()))
        return divmod(idx, ncols)

    def sample_scores(
        self, home: str, away: str, size: int, neutral: bool = True,
        surface: "Surface | None" = None
    ) -> np.ndarray:
        """Vectorised: returns an (size, 2) int array of [goals_h, goals_a]."""
        cum, ncols = self._cumulative(home, away, neutral, surface)
        idx = np.searchsorted(cum, _rng.random(size))
        return np.column_stack(divmod(idx, ncols))

    # ---- analytic outcome probabilities (handy for sanity checks) -------
    def outcome_probs(
        self, home: str, away: str, neutral: bool = True,
        surface: "Surface | None" = None
    ) -> tuple[float, float, float]:
        """Returns (P_home_win, P_draw, P_away_win) over 90 minutes."""
        lam_h, lam_a = self.lambdas(home, away, neutral, surface)
        mat = self._score_matrix(lam_h, lam_a)
        p_home = np.tril(mat, -1).sum()
        p_draw = np.trace(mat)
        p_away = np.triu(mat, 1).sum()
        return float(p_home), float(p_draw), float(p_away)


def knockout_winner(
    model: MatchModel, home: str, away: str, neutral: bool = True,
    surface: "Surface | None" = None
) -> str:
    """
    Resolve a knockout tie: 90', then 30' extra time (rates scaled 1/3),
    then a penalty shootout weighted very mildly by attacking strength.
    Returns the winning team name.
    """
    gh, ga = model.sample_score(home, away, neutral, surface)
    if gh != ga:
        return home if gh > ga else away

    # Extra time: independent Poisson at one-third of the 90' rates.
    lam_h, lam_a = model.lambdas(home, away, neutral, surface)
    eh = int(_rng.poisson(lam_h / 3.0))
    ea = int(_rng.poisson(lam_a / 3.0))
    if eh != ea:
        return home if eh > ea else away

    # Shootout: near coin-flip with a small lean toward the stronger attack.
    edge = 0.5 + 0.04 * math.tanh((lam_h - lam_a))
    return home if _rng.random() < edge else away
