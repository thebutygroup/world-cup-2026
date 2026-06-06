"""
Tournament structure for the 48-team format:

  * 12 groups (A-L) of 4 teams, single round-robin (3 games each).
  * Top 2 of every group (24) + 8 best third-placed teams -> Round of 32.
  * R32 -> R16 -> QF -> SF -> Final, single elimination, ties resolved by
    extra time + penalties.

Group tiebreakers applied here: points, then goal difference, then goals
for, then head-to-head points between the level teams, then a random draw
(FIFA's fair-play and drawing-of-lots steps collapsed into the random draw).

The Round-of-32 bracket is defined by slot labels (e.g. "1A" = winner of
group A, "2B" = runner-up of B, "3W".."3Z"+ = the ranked best thirds). The
DEFAULT_R32_BRACKET below is structurally valid but illustrative -- replace
it with the official 2026 pairings (and the official third-place routing
table) for a true-to-draw model. See assign_third_slots().
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .model import MatchModel, knockout_winner, _rng

GROUP_LETTERS = list("ABCDEFGHIJKL")  # 12 groups

# Eight best-third slots, filled in ranked order by assign_third_slots().
THIRD_SLOTS = [f"3{c}" for c in "STUVWXYZ"]

# 16 Round-of-32 ties. Adjacent ties feed the same R16 match (tie 0 vs tie 1,
# tie 2 vs tie 3, ...). Illustrative layout -- swap in the official bracket.
DEFAULT_R32_BRACKET: list[tuple[str, str]] = [
    ("1A", "3S"), ("1C", "2D"),
    ("1E", "3T"), ("1G", "2H"),
    ("1I", "3U"), ("1K", "2L"),
    ("1B", "3V"), ("1D", "2C"),
    ("1F", "3W"), ("1H", "2G"),
    ("1J", "3X"), ("1L", "2K"),
    ("2A", "3Y"), ("2E", "2F"),
    ("2I", "3Z"), ("2B", "2J"),
]

STAGES = ["group", "R32", "R16", "QF", "SF", "final", "winner"]


@dataclass
class TeamRow:
    name: str
    pts: int = 0
    gf: int = 0
    ga: int = 0

    @property
    def gd(self) -> int:
        return self.gf - self.ga


def simulate_group(model: MatchModel, teams: list[str]):
    """Round-robin a group; return (ranked_team_names, stats_by_team)."""
    rows = {t: TeamRow(t) for t in teams}
    h2h_pts: dict[tuple[str, str], int] = defaultdict(int)
    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            a, b = teams[i], teams[j]
            ga, gb = model.sample_score(a, b, neutral=True)
            rows[a].gf += ga; rows[a].ga += gb
            rows[b].gf += gb; rows[b].ga += ga
            if ga > gb:
                rows[a].pts += 3; h2h_pts[(a, b)] += 3
            elif gb > ga:
                rows[b].pts += 3; h2h_pts[(b, a)] += 3
            else:
                rows[a].pts += 1; rows[b].pts += 1
                h2h_pts[(a, b)] += 1; h2h_pts[(b, a)] += 1

    ranked = sorted(teams, key=lambda t: (rows[t].pts, rows[t].gd, rows[t].gf, _rng.random()), reverse=True)
    for k in range(len(ranked) - 1):
        x, y = ranked[k], ranked[k + 1]
        if (rows[x].pts, rows[x].gd, rows[x].gf) == (rows[y].pts, rows[y].gd, rows[y].gf):
            if h2h_pts[(y, x)] > h2h_pts[(x, y)]:
                ranked[k], ranked[k + 1] = y, x
    return ranked, rows


def assign_third_slots(third_keys: list[tuple]) -> list[str]:
    """
    Map the eight best thirds to bracket slots 3S..3Z in ranked order.

    NOTE: FIFA uses a fixed lookup table keyed by *which* groups the eight
    qualifying thirds come from, so that no third meets a side from its own
    group too early. This function applies the simplified "ranked order"
    routing. Override it with the official combination table for fidelity.
    """
    return THIRD_SLOTS[: len(third_keys)]


def select_qualifiers(
    model: MatchModel, groups: dict[str, list[str]]
) -> dict[str, str]:
    """Run all 12 groups; return a bracket-slot -> team mapping for the 32
    qualifiers (1A..1L, 2A..2L, and the eight best thirds in 3S..3Z)."""
    slots: dict[str, str] = {}
    third_pool: list[tuple[tuple, str]] = []

    for letter in GROUP_LETTERS:
        ranked, rows = simulate_group(model, groups[letter])
        slots[f"1{letter}"] = ranked[0]
        slots[f"2{letter}"] = ranked[1]
        t = ranked[2]
        key = (rows[t].pts, rows[t].gd, rows[t].gf, _rng.random())
        third_pool.append((key, t))

    third_pool.sort(key=lambda x: x[0], reverse=True)
    best_thirds = third_pool[:8]
    slot_names = assign_third_slots([k for k, _ in best_thirds])
    for slot, (_, team) in zip(slot_names, best_thirds):
        slots[slot] = team
    return slots


def simulate_tournament(
    model: MatchModel,
    groups: dict[str, list[str]],
    bracket: list[tuple[str, str]] = DEFAULT_R32_BRACKET,
) -> dict[str, str]:
    """
    Simulate one tournament. Returns team -> furthest stage reached
    (one of STAGES). Teams eliminated in groups map to 'group'.
    """
    all_teams = [t for g in groups.values() for t in g]
    reached = {t: "group" for t in all_teams}

    slots = select_qualifiers(model, groups)

    # Resolve R32 ties into a flat list of advancing teams.
    def team_of(slot: str) -> str:
        return slots[slot]

    round_teams = []
    for a_slot, b_slot in bracket:
        a, b = team_of(a_slot), team_of(b_slot)
        for t in (a, b):
            reached[t] = "R32"
        round_teams.append((a, b))

    stage_order = ["R16", "QF", "SF", "final", "winner"]
    # Play R32
    winners = [knockout_winner(model, a, b, neutral=True) for a, b in round_teams]

    for stage in stage_order:
        for w in winners:
            reached[w] = stage if stage != "winner" else "winner"
        if stage == "winner":
            break
        # pair adjacent winners for the next round
        next_round = [(winners[i], winners[i + 1]) for i in range(0, len(winners), 2)]
        winners = [knockout_winner(model, a, b, neutral=True) for a, b in next_round]

    return reached


def monte_carlo(
    model: MatchModel,
    groups: dict[str, list[str]],
    n: int = 10000,
    bracket: list[tuple[str, str]] = DEFAULT_R32_BRACKET,
) -> pd.DataFrame:
    """
    Run `n` tournaments. Returns a DataFrame indexed by team with the
    probability of reaching each stage (cumulative: reaching the final
    counts toward 'reach SF' etc.) plus the outright win probability.
    """
    all_teams = [t for g in groups.values() for t in g]
    order = {s: i for i, s in enumerate(STAGES)}
    # cumulative counters: reached at least stage k
    counts = {t: np.zeros(len(STAGES)) for t in all_teams}

    for _ in range(n):
        reached = simulate_tournament(model, groups, bracket)
        for t, st in reached.items():
            counts[t][: order[st] + 1] += 1

    rows = []
    for t in all_teams:
        c = counts[t] / n
        rows.append(
            {
                "team": t,
                "P(reach R32)": c[order["R32"]],
                "P(reach R16)": c[order["R16"]],
                "P(reach QF)": c[order["QF"]],
                "P(reach SF)": c[order["SF"]],
                "P(reach final)": c[order["final"]],
                "P(win)": c[order["winner"]],
            }
        )
    df = pd.DataFrame(rows).set_index("team").sort_values("P(win)", ascending=False)
    return df
