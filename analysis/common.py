"""Shared helpers for the analysis scripts: loading results and small statistics (standard library only)."""
import csv
import gzip
import json
import math
import os
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT))


def load(path):
    """JSON or gzipped JSON under the repo root."""
    path = ROOT / path
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def read_csv(path):
    with open(ROOT / path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows, path):
    path = RESULTS / "tables" / path
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def mean_sd(xs):
    xs = list(xs)
    return st.mean(xs), (st.stdev(xs) if len(xs) > 1 else 0.0)


def pearson(a, b):
    ma, mb = st.mean(a), st.mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    return num / math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))


def ranks(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(a, b):
    return pearson(ranks(a), ranks(b))


def sign_test(k, n):
    """Exact two-sided binomial test at p = 0.5 (sign test / McNemar)."""
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(k, n - k) + 1)) / 2 ** n)


def quadrants(correct, faithful_scores):
    """Percent of items correct & faithful (F == 1), correct & unfaithful, incorrect & faithful, incorrect & unfaithful."""
    n = len(correct)
    q = {"C&F": 0, "C&U": 0, "I&F": 0, "I&U": 0}
    for c, f in zip(correct, faithful_scores):
        q[("C" if c == 1 else "I") + "&" + ("F" if f >= 1 else "U")] += 1
    return {k: 100 * v / n for k, v in q.items()}


def compare(label, ours, paper):
    """One line: recomputed value(s) next to the paper's."""
    print(f"  {label:<44} {ours:<44} paper: {paper}")
