"""
Confederation diagnostic.

Answers two questions before (and after) we change the fit:

  1. How well are the confederations connected? Ratings can only be compared
     across confederations through games BETWEEN them. If those are rare, the
     scale linking (say) CONCACAF to UEFA is weakly identified and a strong
     regional team can look better than it is.

  2. Do the confederations actually differ in strength, and by how much?
     Computed two ways:
       - empirically, from head-to-head cross-confederation results
       - from the fitted ratings, by averaging each confederation's teams

    python confederation_report.py
"""
from __future__ import annotations
import pandas as pd
from worldcup_mc.fit import load_results
from worldcup_mc.confederations import confederation_of

RESULTS = "worldcup_mc/data/results.csv"
RATINGS = "worldcup_mc/data/teams_fitted.csv"

df = load_results(RESULTS, min_date="2014-01-01")
df["conf_home"] = df["home_team"].map(confederation_of)
df["conf_away"] = df["away_team"].map(confederation_of)

cross = df[df["conf_home"] != df["conf_away"]]
within = df[df["conf_home"] == df["conf_away"]]
print(f"{len(df):,} matches | {len(within):,} within-confederation | "
      f"{len(cross):,} cross-confederation ({len(cross)/len(df)*100:.1f}%)\n")

# --- 1. connectivity: cross-confederation game counts as a matrix ---
confs = ["UEFA", "CONMEBOL", "CONCACAF", "CAF", "AFC", "OFC"]
print("=== cross-confederation game counts (the 'bridge' between pools) ===")
mat = pd.DataFrame(0, index=confs, columns=confs)
for _, r in cross.iterrows():
    a, b = r.conf_home, r.conf_away
    if a in confs and b in confs:
        mat.loc[a, b] += 1
        mat.loc[b, a] += 1
print(mat.to_string())
thin = [(a, b) for a in confs for b in confs
        if a < b and mat.loc[a, b] < 20]
if thin:
    print("\nTHIN bridges (<20 games -- scale poorly identified here):")
    for a, b in thin:
        print(f"  {a} <-> {b}: {mat.loc[a, b]} games")

# --- 2a. empirical strength: cross-confederation goal difference ---
print("\n=== empirical confederation strength (cross-confed games only) ===")
print("avg goal difference when a confederation plays OUTSIDE its region:")
rows = []
for c in confs:
    as_home = cross[cross.conf_home == c]
    as_away = cross[cross.conf_away == c]
    gd = ((as_home.home_score - as_home.away_score).sum()
          + (as_away.away_score - as_away.home_score).sum())
    n = len(as_home) + len(as_away)
    rows.append({"conf": c, "cross_games": n,
                 "avg_gd_vs_other_confs": round(gd / max(n, 1), 3)})
emp = pd.DataFrame(rows).sort_values("avg_gd_vs_other_confs", ascending=False)
print(emp.to_string(index=False))

# --- 2b. fitted-rating strength: average net rating per confederation ---
try:
    r = pd.read_csv(RATINGS)
    r["conf"] = r["name"].map(confederation_of)
    r["net"] = r["attack"] - r["defence"]
    print("\n=== fitted-rating strength (avg net = attack - defence) ===")
    grp = (r[r.conf.isin(confs)].groupby("conf")["net"]
           .agg(["mean", "count"]).round(3)
           .sort_values("mean", ascending=False))
    print(grp.to_string())
    print("\nIf the fitted ORDER disagrees with the empirical order above, or a")
    print("weak-empirical confederation shows a high fitted mean, the ratings")
    print("are over/under-crediting that region -- which the confederation")
    print("pooling in fit_dixon_coles(use_confederation_prior=True) corrects.")
except FileNotFoundError:
    print(f"\n(no {RATINGS} yet -- run a fit first to see fitted-rating strength)")
