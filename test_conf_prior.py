"""ON vs OFF confederation prior across multiple folds, gated."""
from worldcup_mc.fit import load_results
from worldcup_mc.backtest import filter_matches, run_folds, _paired_boot_diff

h = load_results("worldcup_mc/data/results.csv")
folds = {
    "wc2018": filter_matches(h, tournament_contains="FIFA World Cup", year=2018),
    "wc2022": filter_matches(h, tournament_contains="FIFA World Cup", year=2022),
    "copa2024": filter_matches(h, tournament_contains="Copa", year=2024),
    "quals_24_25": filter_matches(h, tournament_contains="qualification",
                                  start="2024-01-01", end="2025-12-31"),
}
folds = {k: v for k, v in folds.items() if len(v) >= 20}
cfg = {"half_life_days": 730.0, "friendly_weight": 0.3, "ridge": 0.05}
intervals = {"quals_24_25": 14.0}

pm_off = run_folds(h, folds, cfg, refit_interval_by_fold=intervals)
pm_on = run_folds(h, folds, cfg, refit_interval_by_fold=intervals,
                  use_confederation_prior=True)

m = pm_on.merge(pm_off, on=["fold", "date", "home", "away"],
                suffixes=("_on", "_off"))
d, lo, hi = _paired_boot_diff(m["brier_on"].to_numpy(float),
                              m["brier_off"].to_numpy(float), 4000, 0.95, 0)
print(f"\npooled n={len(m)}")
print("per-fold OFF:", {k: round(v, 4) for k, v in pm_off.attrs['fold_brier'].items()})
print("per-fold ON: ", {k: round(v, 4) for k, v in pm_on.attrs['fold_brier'].items()})
print(f"ON-OFF brier diff {d:+.4f} [{lo:+.4f},{hi:+.4f}] -> "
      f"{'REAL improvement' if hi < 0 else 'not distinguishable'}")