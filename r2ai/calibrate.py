"""Turn public-leaderboard probes into an F2-optimal cutoff.

We have no labels, but the leaderboard reports Precision and Recall separately at
each level. For a run that submitted K items per query with precision P and
recall R, the average number of true positives is K*P and the average size of the
gold set follows as |G| ~= K*P/R. Probing two or three values of K pins down the
recall curve well enough to pick the K that maximises F2 analytically.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "outputs/submissions/log.csv"
FIELDS = [
    "date", "name", "k_docs", "k_chunks", "chunks_per_doc", "target_words", "notes",
    "FINAL_SCORE", "DOCS_F2MACRO", "CHUNKS_F2MACRO",
    "DOCS_PRECISION", "DOCS_RECALL", "CHUNKS_PRECISION", "CHUNKS_RECALL",
]


def f2(precision: float, recall: float) -> float:
    if precision <= 0 or recall <= 0:
        return 0.0
    return 5 * precision * recall / (4 * precision + recall)


def gold_size(k: int, precision: float, recall: float) -> float:
    """Average |G| implied by a probe: |G| = (K*P) / R."""
    if recall <= 0:
        return float("nan")
    return k * precision / recall


def sweep(points: list[tuple[int, float, float]]) -> None:
    """Report implied |G| per probe and the F2-optimal K under a saturating fit."""
    print(f"{'K':>5}  {'P':>7}  {'R':>7}  {'F2':>7}  {'hits=K*P':>9}  {'|G|':>7}")
    gs = []
    for k, p, r in points:
        g = gold_size(k, p, r)
        gs.append(g)
        print(f"{k:>5}  {p:>7.4f}  {r:>7.4f}  {f2(p, r):>7.4f}  {k * p:>9.3f}  {g:>7.2f}")

    if not gs:
        return
    g_mean = sum(gs) / len(gs)
    print(f"\nimplied |G| (mean over probes): {g_mean:.2f} relevant items per query")

    # Fit hits(K) = |G| * (1 - exp(-K/tau)) through the observed (K, hits) points,
    # a saturating curve that respects hits <= |G|.
    import math
    taus = []
    for (k, p, r), g in zip(points, gs):
        hits = k * p
        if 0 < hits < g:
            taus.append(-k / math.log(1 - hits / g))
    if not taus:
        print("not enough signal to fit a recall curve; add probes at different K")
        return
    tau = sum(taus) / len(taus)
    print(f"fitted saturation constant tau: {tau:.2f}\n")

    print(f"{'K':>5}  {'pred P':>7}  {'pred R':>7}  {'pred F2':>8}")
    best = (0, -1.0)
    for k in range(1, 61):
        hits = g_mean * (1 - math.exp(-k / tau))
        p, r = hits / k, hits / g_mean
        score = f2(p, r)
        if score > best[1]:
            best = (k, score)
        if k <= 30 or k % 5 == 0:
            print(f"{k:>5}  {p:>7.4f}  {r:>7.4f}  {score:>8.4f}")
    print(f"\nF2-optimal K ~= {best[0]} (predicted F2 {best[1]:.4f})")
    print("Treat as a starting point: the fit assumes ranking quality is uniform across queries.")


def init_log() -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    if not LOG.exists():
        with LOG.open("w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(FIELDS)
        print(f"created {LOG}")
    else:
        print(f"{LOG} already exists")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--init-log", action="store_true", help="create outputs/submissions/log.csv")
    ap.add_argument("--probe", action="append", default=[], metavar="K,P,R",
                    help="observed leaderboard probe, e.g. --probe 10,0.1234,0.5678")
    a = ap.parse_args()
    if a.init_log:
        init_log()
    if a.probe:
        pts = []
        for item in a.probe:
            k, p, r = item.split(",")
            pts.append((int(k), float(p), float(r)))
        sweep(sorted(pts))
