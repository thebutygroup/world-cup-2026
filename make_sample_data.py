"""Generate an ILLUSTRATIVE 48-team ratings file + group draw.

These ratings and groups are synthetic placeholders, NOT the official
2026 draw. Replace with real ratings (e.g. World Football Elo) and the
official group composition.
"""
import csv
import numpy as np

from worldcup_mc.model import attack_defence_from_rating

rng = np.random.default_rng(2026)

# notional power ratings (Elo-ish), purely illustrative
teams_by_pot = {
    1: [("Spain", 2080), ("Argentina", 2060), ("France", 2050), ("England", 2010),
        ("Brazil", 2000), ("Portugal", 1980), ("Netherlands", 1965), ("Germany", 1955),
        ("USA", 1820), ("Mexico", 1790), ("Canada", 1770), ("Belgium", 1945)],
    2: [("Croatia", 1900), ("Italy", 1905), ("Morocco", 1880), ("Colombia", 1860),
        ("Uruguay", 1855), ("Japan", 1840), ("Senegal", 1820), ("Switzerland", 1815),
        ("Denmark", 1810), ("Korea Republic", 1790), ("Ecuador", 1780), ("Austria", 1800)],
    3: [("Australia", 1740), ("Nigeria", 1750), ("Serbia", 1760), ("Egypt", 1730),
        ("Poland", 1745), ("Ivory Coast", 1735), ("Sweden", 1755), ("Peru", 1700),
        ("Tunisia", 1690), ("Paraguay", 1680), ("Norway", 1770), ("Turkey", 1765)],
    4: [("Ghana", 1660), ("Qatar", 1600), ("Saudi Arabia", 1620), ("Iran", 1700),
        ("Costa Rica", 1640), ("Jamaica", 1610), ("Cameroon", 1670), ("Algeria", 1690),
        ("New Zealand", 1560), ("Panama", 1630), ("South Africa", 1650), ("Uzbekistan", 1640)],
}

# Snake the four pots into 12 groups A-L, one team per pot per group.
letters = list("ABCDEFGHIJKL")
groups = {L: [] for L in letters}
for pot in (1, 2, 3, 4):
    pool = teams_by_pot[pot][:]
    rng.shuffle(pool)
    for L, (name, _) in zip(letters, pool):
        groups[L].append(name)

rating_of = {n: r for pot in teams_by_pot.values() for (n, r) in pot}

rows = []
for L in letters:
    for name in groups[L]:
        atk, dfc = attack_defence_from_rating(rating_of[name], ref=1800, scale=300, spread=0.55)
        rows.append((name, round(atk, 3), round(dfc, 3), L))

with open("worldcup_mc/data/teams_sample.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["name", "attack", "defence", "group"])
    w.writerows(rows)

print(f"wrote {len(rows)} teams across {len(letters)} groups")
