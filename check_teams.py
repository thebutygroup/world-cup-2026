"""
Audit which teams are in results.csv and flag likely non-FIFA / minnow sides
that distort the ratings. Run this on your REAL data to catch anything the
default NON_FIFA_TEAMS list in fit.py misses.

    python check_teams.py

It prints, for every team:
  - games played (in the loaded window)
  - average goal difference (lopsided + few games = suspicious)
  - whether the default exclusion already drops them

Eyeball the top of the "SUSPICIOUS" list. Any real minnow or non-FIFA side you
see there that ISN'T already excluded, add to NON_FIFA_TEAMS in fit.py (or pass
via extra_exclude).
"""
from __future__ import annotations
import pandas as pd
from worldcup_mc.fit import NON_FIFA_TEAMS

RESULTS = "worldcup_mc/data/results.csv"

# load WITHOUT exclusion so we can see everything
df = pd.read_csv(RESULTS)
df["date"] = pd.to_datetime(df["date"])
df = df.dropna(subset=["home_score", "away_score"])
df = df[df["date"] >= "2014-01-01"]

rows = []
for team in sorted(set(df.home_team) | set(df.away_team)):
    h = df[df.home_team == team]
    a = df[df.away_team == team]
    n = len(h) + len(a)
    gd = (h.home_score - h.away_score).sum() + (a.away_score - a.home_score).sum()
    gf = h.home_score.sum() + a.away_score.sum()
    rows.append({
        "team": team, "games": n,
        "avg_gd": round(gd / max(n, 1), 2),
        "avg_gf": round(gf / max(n, 1), 2),
        "excluded": team in NON_FIFA_TEAMS,
    })

t = pd.DataFrame(rows)

# suspicious = high average goal difference AND relatively few games, and NOT
# already excluded. Those are the minnows inflating the ratings.
susp = t[(~t.excluded) & (t.avg_gd > 1.5) & (t.games < 60)].sort_values(
    "avg_gd", ascending=False)

print(f"{len(t)} teams total | {t.excluded.sum()} already excluded\n")
print("=== SUSPICIOUS (high goal diff, few games, NOT yet excluded) ===")
print("Check these by eye -- real minnows/non-FIFA sides should be added to")
print("NON_FIFA_TEAMS in fit.py:\n")
print(susp.to_string(index=False) if len(susp) else "  (none flagged)")

print("\n=== currently excluded by default ===")
exc = t[t.excluded].sort_values("avg_gd", ascending=False)
print(exc[["team", "games", "avg_gd"]].to_string(index=False) if len(exc)
      else "  (none of the excluded names appear in your data)")
