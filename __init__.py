"""Monte Carlo World Cup 2026 simulator.

Quick start
-----------
    from worldcup_mc import MatchModel, load_teams, monte_carlo, groups_from_teams
    from worldcup_mc import seed

    seed(42)
    teams = load_teams("worldcup_mc/data/teams_sample.csv")
    groups = groups_from_teams(teams)
    model = MatchModel(teams, home_adv=0.0)   # neutral venues
    probs = monte_carlo(model, groups, n=20000)
    print(probs.head(12))
"""

from .model import (
    MatchModel,
    Team,
    load_teams,
    attack_defence_from_rating,
    knockout_winner,
    seed,
)
from .tournament import (
    simulate_tournament,
    monte_carlo,
    select_qualifiers,
    GROUP_LETTERS,
    DEFAULT_R32_BRACKET,
)
from .odds import compare_market, devig_proportional, devig_shin, overround


def groups_from_teams(teams: dict) -> dict[str, list[str]]:
    """Build the {group_letter: [team, ...]} mapping from a loaded teams
    dict whose Team objects carry a `group` attribute."""
    groups: dict[str, list[str]] = {}
    for t in teams.values():
        if t.group is None:
            raise ValueError(f"team {t.name!r} has no group assigned")
        groups.setdefault(str(t.group), []).append(t.name)
    return dict(sorted(groups.items()))


__all__ = [
    "MatchModel", "Team", "load_teams", "attack_defence_from_rating",
    "knockout_winner", "seed", "simulate_tournament", "monte_carlo",
    "select_qualifiers", "groups_from_teams", "GROUP_LETTERS",
    "DEFAULT_R32_BRACKET", "compare_market", "devig_proportional",
    "devig_shin", "overround",
]
