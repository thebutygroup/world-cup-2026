"""
Trustworthiness check for the per-match probabilities, focused on the group
stage (where ~69% of the 2026 matches and most of the betting volume will be:
72 group games vs 32 knockout games).

    python fetch_data.py        # once, on a networked machine
    python calibration_example.py

Pipeline:
  1. Build a "group-stage-like" test set. The martj42 file does not label the
     group phase, so we approximate it two ways (pick with TEST_MODE):
       "wc2022_groups" -> just the 2022 WC group stage (20 Nov - 2 Dec 2022),
                          48 matches. Clean but small for calibration bins.
       "groupish"      -> WC + continental qualifiers across a window. These
                          are the best analogue for the 2026 group stage:
                          uneven team strength, thin data on minnows. Bigger N
                          => meaningful calibration bins.
  2. Walk-forward backtest (no leakage), as before.
  3. Bootstrap CIs on Brier / log-loss / skill -- so you can see whether the
     half-life differences from the sweep are real or noise.
  4. Calibration: reliability table, ECE, and the Murphy decomposition
     (are the probabilities honest AND sharp?).
  5. Data-richness split: re-run calibration on data-poor vs data-rich
     fixtures to test the thesis that the soft, mispriceable games are the
     ones with little data behind the weaker side -- but only bankable if WE
     stay calibrated there too.

The base-rate baseline here is a punching bag, not the bookie. This step tells
you whether the model is internally trustworthy; whether there is MONEY in it
needs the closing line as the baseline (see the note printed at the end).
"""

import os
import sys
from datetime import datetime

from worldcup_mc.fit import load_results
from worldcup_mc.backtest import filter_matches, backtest
from worldcup_mc.calibration import (
    bootstrap_metrics, reliability_table, expected_calibration_error,
    brier_decomposition, split_by_data_richness)

RESULTS = "worldcup_mc/data/results.csv"
OUT_DIR = "reports"             # report text + per-match CSV land here

TEST_MODE = "groupish"          # "groupish" or "wc2022_groups"
HALF_LIFE = 1460.0              # carry over the sweep's best; revisit with the CI
FRIENDLY_WEIGHT = 0.3
RIDGE = 0.05
RICHNESS_THRESHOLD = 15         # < this many prior matches = "data-poor" side


class _Tee:
    """Write to both stdout and a file so the console stays live while a
    permanent copy accumulates on disk."""
    def __init__(self, path):
        self._file = open(path, "w", encoding="utf-8")
        self._stdout = sys.stdout
    def write(self, s):
        self._stdout.write(s)
        self._file.write(s)
    def flush(self):
        self._stdout.flush()
        self._file.flush()
    def close(self):
        self._file.close()
        sys.stdout = self._stdout


def build_test(history):
    if TEST_MODE == "wc2022_groups":
        wc = filter_matches(history, tournament_contains="FIFA World Cup",
                            year=2022)
        # group stage only: drop the knockout window
        return wc[(wc["date"] >= "2022-11-20") & (wc["date"] <= "2022-12-02")
                  ].reset_index(drop=True)
    # "groupish": qualifiers + finals group phases are the closest analogue to
    # the expanded 2026 group stage (lopsided fixtures, sparse minnow data).
    quals = filter_matches(history, tournament_contains="qualification",
                           start="2021-01-01", end="2024-12-31")
    euro_groups = filter_matches(history, tournament_contains="UEFA Euro",
                                 year=2024)
    euro_groups = euro_groups[euro_groups["date"] <= "2024-06-26"]
    import pandas as pd
    return pd.concat([quals, euro_groups], ignore_index=True)


def report_calibration(per_match, label):
    print(f"\n--- calibration: {label} (n={len(per_match)}) ---")
    print(f"ECE (quantile bins): {expected_calibration_error(per_match):.4f}")
    print(brier_decomposition(per_match, bins=10))
    print("reliability (equal-width; read 'gap' = pred - obs, want ~0):")
    print(reliability_table(per_match, bins=10, scheme="fixed")
          .round(3).to_string(index=False))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(OUT_DIR, f"calibration_{stamp}.txt")
    csv_path = os.path.join(OUT_DIR, f"per_match_{stamp}.csv")

    tee = _Tee(report_path)
    sys.stdout = tee
    try:
        _run(csv_path)
    finally:
        print(f"\n--- written to ---")
        print(f"  report : {os.path.abspath(report_path)}")
        print(f"  matches: {os.path.abspath(csv_path)}")
        tee.close()


def _run(csv_path):
    history = load_results(RESULTS, min_date="2006-01-01")
    test = build_test(history)
    print(f"TEST_MODE={TEST_MODE}: {len(test)} candidate matches "
          f"({test['date'].min().date()}..{test['date'].max().date()})")

    res = backtest(history, test, half_life_days=HALF_LIFE,
                   friendly_weight=FRIENDLY_WEIGHT, ridge=RIDGE, verbose=False)
    print("\n=== Backtest ===")
    print(res.summary())

    print("\n=== Bootstrap CIs (matches resampled with replacement) ===")
    for stat in bootstrap_metrics(res.per_match, n_boot=5000).values():
        print(" ", stat)

    report_calibration(res.per_match, "all scored matches")

    print("\n=== Data-richness split (thesis: edge lives where data is thin) ===")
    parts = split_by_data_richness(res.per_match, threshold=RICHNESS_THRESHOLD)
    for label, part in parts.items():
        if len(part) < 30:
            print(f"  ({label}: only {len(part)} matches -- too few to read)")
            continue
        report_calibration(part, label)

    print("\n" + "=" * 70)
    print("NEXT STEP FOR THE MONEY QUESTION: re-run with the bookmaker's")
    print("DE-VIGGED CLOSING line as the baseline, not base rates. Compare the")
    print("model's Brier to the closing line's Brier on the same matches, and")
    print("measure realised CLV/EV on the bets the value filter would fire.")
    print("Closing odds for internationals are the data bottleneck, not code.")

    # Export the full per-match frame so you can slice/pivot in a spreadsheet
    # or feed it to the closing-line backtest once you have historical odds.
    pm = res.per_match.copy()
    pm["date"] = pm["date"].dt.date
    pm.to_csv(csv_path, index=False)
    print(f"\nPer-match results ({len(pm)} rows) written to {csv_path}")


if __name__ == "__main__":
    main()