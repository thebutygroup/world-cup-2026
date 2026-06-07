"""
Download the historical results data the fitter needs.

Run this on your own machine (this fetches over the network):

    python fetch_data.py

It pulls the martj42 international results dataset straight from GitHub raw
(no Kaggle login needed) into worldcup_mc/data/.
"""

import os
import urllib.request

RAW = "https://raw.githubusercontent.com/martj42/international_results/master"
FILES = {
    "results.csv": f"{RAW}/results.csv",        # date,home_team,away_team,home_score,away_score,tournament,city,country,neutral
    "shootouts.csv": f"{RAW}/shootouts.csv",     # date,home_team,away_team,winner,first_shooter
    "goalscorers.csv": f"{RAW}/goalscorers.csv", # date,home_team,away_team,team,scorer,minute,own_goal,penalty
}

OUT_DIR = os.path.join(os.path.dirname(__file__), "worldcup_mc", "data")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for fname, url in FILES.items():
        dest = os.path.join(OUT_DIR, fname)
        print(f"downloading {fname} ...")
        urllib.request.urlretrieve(url, dest)
        size = os.path.getsize(dest)
        print(f"  -> {dest} ({size/1e6:.1f} MB)")
    print("done. The fitter only needs results.csv; the others are optional.")


if __name__ == "__main__":
    main()
