"""
Trustworthiness of the per-match probabilities: bootstrap uncertainty on the
backtest scores, and calibration (are the probabilities honest, not just
sharp?).

WHY THIS EXISTS. A single Brier number on ~150 matches is one draw from a
noisy process; before tuning a knob on it -- or, more importantly, before
trusting a "slight edge" the value filter detects -- we need (a) how wide the
sampling error on that Brier is, and (b) whether a forecast of p actually
comes true p of the time. Calibration is the binding constraint on thin-edge
betting: if our 30% lands 22% of the time, every 30%-priced "value" bet is a
loser regardless of how the average Brier looks.

Everything here is post-processing on the `per_match` DataFrame that
`backtest()` returns, so nothing is refit and no network/data is needed beyond
the original backtest. It expects these columns:
    p_home, p_draw, p_away, result ("H"/"D"/"A"), brier, log_loss
and, for the skill bootstrap and the data-richness split, the columns added by
the backtest patch:
    base_brier, min_team_matches

SCALE NOTE on the decomposition. The reliability/resolution/uncertainty
breakdown is computed on the *pooled one-vs-rest binary* forecasts (every
(class, match) pair stacked), so its Brier is the multiclass Brier / 3
(~0.19 where the multiclass is ~0.58). That is the right scale for "when we
say X%, does X% happen?", which is the betting-relevant question -- a market
line is a single binary price.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

_CLASS_COLS = ("p_home", "p_draw", "p_away")
_RESULT_TO_IDX = {"H": 0, "D": 1, "A": 2}


# --------------------------------------------------------------------------
# pooling helpers
# --------------------------------------------------------------------------
def _stack_binary(per_match: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Stack every (forecast prob, hit 0/1) pair across the three outcome
    classes into one long binary-forecast array. 3*N pairs from N matches."""
    P = per_match[list(_CLASS_COLS)].to_numpy(dtype=float)          # (N,3)
    y_idx = per_match["result"].map(_RESULT_TO_IDX).to_numpy()       # (N,)
    Y = np.zeros_like(P)
    Y[np.arange(len(P)), y_idx] = 1.0
    return P.reshape(-1), Y.reshape(-1)


# --------------------------------------------------------------------------
# bootstrap on the backtest scores
# --------------------------------------------------------------------------
@dataclass
class BootstrapCI:
    metric: str
    point: float
    lo: float
    hi: float
    n: int
    n_boot: int

    def __str__(self) -> str:
        return (f"{self.metric:>14}: {self.point:.4f}  "
                f"[{self.lo:.4f}, {self.hi:.4f}]  "
                f"(n={self.n}, {self.n_boot} resamples)")


def bootstrap_metrics(
    per_match: pd.DataFrame,
    n_boot: int = 5000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict[str, BootstrapCI]:
    """
    Percentile-bootstrap CIs for mean Brier, mean log-loss, and -- if a
    `base_brier` column is present -- the *paired* skill score
    (1 - sum(brier)/sum(base_brier)) recomputed on each resample, so the
    baseline moves with the resampled outcome mix instead of being frozen.

    Resampling is over matches (rows), which is the unit of independence.
    """
    rng = np.random.default_rng(seed)
    n = len(per_match)
    brier = per_match["brier"].to_numpy(dtype=float)
    ll = per_match["log_loss"].to_numpy(dtype=float)
    has_base = "base_brier" in per_match.columns
    base = per_match["base_brier"].to_numpy(dtype=float) if has_base else None

    b_brier = np.empty(n_boot)
    b_ll = np.empty(n_boot)
    b_skill = np.empty(n_boot) if has_base else None
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        b_brier[i] = brier[idx].mean()
        b_ll[i] = ll[idx].mean()
        if has_base:
            b_skill[i] = 1.0 - brier[idx].sum() / base[idx].sum()

    lo_q, hi_q = (1 - ci) / 2, 1 - (1 - ci) / 2

    def _ci(name, samples, point):
        lo, hi = np.quantile(samples, [lo_q, hi_q])
        return BootstrapCI(name, float(point), float(lo), float(hi), n, n_boot)

    out = {
        "brier": _ci("brier", b_brier, brier.mean()),
        "log_loss": _ci("log_loss", b_ll, ll.mean()),
    }
    if has_base:
        point_skill = 1.0 - brier.sum() / base.sum()
        out["skill"] = _ci("skill", b_skill, point_skill)
    return out


def paired_bootstrap_diff(
    per_match_a: pd.DataFrame,
    per_match_b: pd.DataFrame,
    metric: str = "brier",
    n_boot: int = 5000,
    ci: float = 0.95,
    seed: int = 0,
    key=("date", "home", "away"),
) -> BootstrapCI:
    """
    Paired bootstrap of (mean metric_a - mean metric_b) over the matches both
    configs actually scored (inner join on `key`). Use this to ask whether two
    half-lives (or any two settings) are really distinguishable: if 0 is inside
    the interval, the difference is noise -- don't tune on it.

    A negative point estimate means config A has the lower (better) Brier.
    """
    key = list(key)
    merged = per_match_a.merge(
        per_match_b, on=key, suffixes=("_a", "_b"))
    if not len(merged):
        raise ValueError("no matches in common between the two backtests "
                         "(check the key / that both scored the same set)")
    da = merged[f"{metric}_a"].to_numpy(dtype=float)
    db = merged[f"{metric}_b"].to_numpy(dtype=float)
    diff = da - db
    rng = np.random.default_rng(seed)
    n = len(diff)
    b = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        b[i] = diff[idx].mean()
    lo_q, hi_q = (1 - ci) / 2, 1 - (1 - ci) / 2
    lo, hi = np.quantile(b, [lo_q, hi_q])
    return BootstrapCI(f"{metric} diff (A-B)", float(diff.mean()),
                       float(lo), float(hi), n, n_boot)


# --------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------
def reliability_table(
    per_match: pd.DataFrame,
    bins: int = 10,
    scheme: str = "fixed",
) -> pd.DataFrame:
    """
    Pooled one-vs-rest reliability table. Bin every forecast probability and
    compare the mean predicted prob in the bin to the observed hit frequency.
    A well-calibrated model has observed ~= predicted on every row.

    scheme="fixed"   -> equal-width bins on [0,1] (read calibration at a price
                        level: "our ~40% forecasts").  Best for betting.
    scheme="quantile"-> equal-count bins (stable counts; better for ECE).
    """
    p, y = _stack_binary(per_match)
    if scheme == "quantile":
        edges = np.quantile(p, np.linspace(0, 1, bins + 1))
        edges[0], edges[-1] = 0.0, 1.0
        edges = np.unique(edges)
    else:
        edges = np.linspace(0.0, 1.0, bins + 1)
    b = np.clip(np.digitize(p, edges[1:-1], right=False), 0, len(edges) - 2)

    rows = []
    for k in range(len(edges) - 1):
        m = b == k
        if not m.any():
            continue
        rows.append({
            "bin": f"[{edges[k]:.2f},{edges[k + 1]:.2f})",
            "n": int(m.sum()),
            "pred": float(p[m].mean()),
            "obs": float(y[m].mean()),
            "gap": float(p[m].mean() - y[m].mean()),
        })
    return pd.DataFrame(rows)


def expected_calibration_error(per_match: pd.DataFrame, bins: int = 10,
                               scheme: str = "quantile") -> float:
    """Count-weighted mean |predicted - observed| over the bins. 0 = perfect.
    A handy single number; quantile bins keep it from being dominated by an
    empty tail bin."""
    tab = reliability_table(per_match, bins=bins, scheme=scheme)
    w = tab["n"] / tab["n"].sum()
    return float((w * tab["gap"].abs()).sum())


@dataclass
class BrierDecomposition:
    brier_binary: float       # pooled one-vs-rest Brier (= multiclass/3)
    reliability: float        # calibration error; LOWER is better
    resolution: float         # sharpness/discrimination; HIGHER is better
    uncertainty: float        # irreducible (base-rate variance)
    recon: float              # REL - RES + UNC (should ~= brier_binary)
    n_pairs: int

    def __str__(self) -> str:
        return (
            f"  pooled binary Brier : {self.brier_binary:.4f}  "
            f"(= multiclass/3)\n"
            f"  reliability (REL)   : {self.reliability:.4f}   <- calibration, lower better\n"
            f"  resolution  (RES)   : {self.resolution:.4f}   <- sharpness, higher better\n"
            f"  uncertainty (UNC)   : {self.uncertainty:.4f}   <- irreducible\n"
            f"  REL - RES + UNC     : {self.recon:.4f}   (vs Brier {self.brier_binary:.4f}; "
            f"gap = within-bin spread)"
        )


def brier_decomposition(per_match: pd.DataFrame, bins: int = 10) -> BrierDecomposition:
    """
    Murphy (1973) reliability-resolution-uncertainty decomposition on the
    pooled one-vs-rest binary forecasts:  BS = REL - RES + UNC.

    The identity is exact only when every forecast inside a bin is identical;
    with continuous probabilities there is a small within-bin term, so
    REL - RES + UNC reconciles to the binary Brier up to that gap (reported).
    """
    p, y = _stack_binary(per_match)
    N = len(p)
    edges = np.linspace(0.0, 1.0, bins + 1)
    b = np.clip(np.digitize(p, edges[1:-1], right=False), 0, bins - 1)
    o_bar = y.mean()

    rel = res = 0.0
    for k in range(bins):
        m = b == k
        nk = int(m.sum())
        if not nk:
            continue
        f_k = p[m].mean()
        o_k = y[m].mean()
        rel += nk * (f_k - o_k) ** 2
        res += nk * (o_k - o_bar) ** 2
    rel /= N
    res /= N
    unc = o_bar * (1 - o_bar)
    brier_bin = float(np.mean((p - y) ** 2))
    return BrierDecomposition(brier_bin, rel, res, unc,
                              rel - res + unc, N)


# --------------------------------------------------------------------------
# the data-richness split (does sparse-team data hurt our calibration?)
# --------------------------------------------------------------------------
def split_by_data_richness(
    per_match: pd.DataFrame,
    threshold: int,
    col: str = "min_team_matches",
) -> dict[str, pd.DataFrame]:
    """
    Slice the scored matches by how much training data the *weaker-documented*
    side had (the min of the two teams' prior match counts). Returns
    {"data_poor": rows < threshold, "data_rich": rows >= threshold} so you can
    run reliability/decomposition on each and compare. Tests the thesis that
    edge lives where data is thin -- but only if WE stay calibrated there.
    """
    if col not in per_match.columns:
        raise KeyError(
            f"'{col}' not in per_match -- apply the backtest patch that records "
            "min_team_matches per fixture.")
    poor = per_match[per_match[col] < threshold].reset_index(drop=True)
    rich = per_match[per_match[col] >= threshold].reset_index(drop=True)
    return {"data_poor": poor, "data_rich": rich}
