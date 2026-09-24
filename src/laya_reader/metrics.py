"""How well do Laya's signals match the reader's ratings?"""

from __future__ import annotations

from dataclasses import dataclass

from .judge import PICK

N_BINS = 5


@dataclass
class CalibrationBin:
    lo: float
    hi: float
    n: int
    mean_p: float  # mean predicted P(relevant)
    frac_yes: float  # share the reader rated relevant


@dataclass
class Report:
    n_rated: int
    pick_precision: float | None  # of rated picks, share rated relevant
    n_rated_picks: int
    sim_auc: float | None  # does similarity rank wanted papers above unwanted ones?
    laya_auc: float | None  # same for Laya's P(relevant), on rated shortlisted papers
    n_laya: int
    laya_accuracy: float | None  # P >= 0.5 vs rating
    bins: list[CalibrationBin]
    ece: float | None


def auc(pairs: list[tuple[float, int]]) -> float | None:
    """P(a random wanted paper scores above a random unwanted one); 0.5 = chance."""
    pos = [s for s, y in pairs if y]
    neg = [s for s, y in pairs if not y]
    if not pos or not neg:
        return None
    return sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg))


def calibration(pairs: list[tuple[float, int]], n_bins: int = N_BINS) -> tuple[list[CalibrationBin], float | None]:
    bins = []
    ece = 0.0
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        inside = [(p, y) for p, y in pairs if lo <= p < hi or (i == n_bins - 1 and p == 1.0)]
        if not inside:
            bins.append(CalibrationBin(lo, hi, 0, 0.0, 0.0))
            continue
        mean_p = sum(p for p, _ in inside) / len(inside)
        frac = sum(y for _, y in inside) / len(inside)
        bins.append(CalibrationBin(lo, hi, len(inside), mean_p, frac))
        ece += len(inside) / len(pairs) * abs(mean_p - frac)
    return bins, (ece if pairs else None)


def report(rated_rows) -> Report:
    sims = [(r["sim"], r["rated"]) for r in rated_rows]
    laya = [(r["p_relevant"], r["rated"]) for r in rated_rows if r["p_relevant"] is not None]
    picks = [r["rated"] for r in rated_rows if r["bucket"] == PICK]
    bins, ece = calibration(laya)
    return Report(
        n_rated=len(sims),
        pick_precision=sum(picks) / len(picks) if picks else None,
        n_rated_picks=len(picks),
        sim_auc=auc(sims),
        laya_auc=auc(laya),
        n_laya=len(laya),
        laya_accuracy=sum((p >= 0.5) == bool(y) for p, y in laya) / len(laya) if laya else None,
        bins=bins,
        ece=ece,
    )
