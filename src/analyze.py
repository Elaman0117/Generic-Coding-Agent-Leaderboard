#!/usr/bin/env python3
"""
Three-level hierarchy builder + pairwise-ratio solver for the
Artificial Analysis Coding Agents benchmark.

Pipeline
--------
1. Load `output/raw_data.json` (produced by `scrape.py`).

2. Build the 3-level hierarchy:

       Test (performance benchmark — Index / DeepSWE /
                            Terminal-Bench v2 / SWE-Atlas-QnA)
         └─ Model (display.model — includes thinking level, so
                                  "GPT-5.6 Sol (max)" and
                                  "GPT-5.6 Sol (xhigh)" are DIFFERENT models)
              └─ Frontend (agentName — Codex, Cursor CLI, Claude Code, ...
                          frontends have NO thinking-level variants)

   This gives 4 tests × ~38 models × 1..3 frontends per (test, model).

3. Filter — drop any (test, model) that has fewer than 2 frontends.
   Only models tested by 2+ frontends on a given test survive that test.

4. Pairwise ratios — for each surviving (test, model), enumerate every
   unordered frontend pair (A, B) with A < B alphabetically and record
   the ratio  reward(A) / reward(B).  All 4 benchmarks are pass@1 reward
   scores (higher = better), so a ratio > 1 always means "A is better
   than B" on that test.

5. Group ratios by frontend pair — all (A, B) ratios across all tests
   and models collapse into one bucket per pair.

6. Weighted log least squares — for each pair (A, B):
       M_AB = mean of all observed ratios
       w_AB = number of observations
   Solve the Laplacian system   L · x = b   (with one variable pinned to 0)
   where  L  is the weighted graph Laplacian of the frontend-pair graph and
          b  is the weighted log-ratio divergence. Then  s_i = exp(x_i).

7. Normalise — divide every  s_i  by max(s_i), so the best frontend = 1.

Output files
------------
- output/hierarchy.json        — the full 3-level tree (before filtering)
- output/filtered.json         — the tree after dropping 1-frontend models
- output/ratios.json           — every observed ratio, grouped by pair
- output/final_scores.json     — final per-frontend score (best = 1) +
                                 per-pair mean ratio, weight, and fit error

Method reference
----------------
The weighted log least squares approach is the standard way to combine
pairwise ratio observations into a global ranking. It is exactly the
"对数最小二乘法" described in the Qwen share
(https://chat.qwen.ai/s/t_513f5796-2d0e-48bb-bad8-1c95476b3281) and
the cross-ratio verification flow described in the DeepSeek share
(https://chat.deepseek.com/share/0lhgm7hnhlqburvoh7).
"""

import json
import math
import os
from collections import defaultdict
from fractions import Fraction
from itertools import combinations

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
RAW_FILE = os.path.join(OUTPUT_DIR, "raw_data.json")
HIERARCHY_FILE = os.path.join(OUTPUT_DIR, "hierarchy.json")
FILTERED_FILE = os.path.join(OUTPUT_DIR, "filtered.json")
RATIOS_FILE = os.path.join(OUTPUT_DIR, "ratios.json")
SCORES_FILE = os.path.join(OUTPUT_DIR, "final_scores.json")


# ──────────────────────────────────────────────────────────────────────
# The 4 performance benchmarks. All are pass@1 reward scores
# (higher = better), so no inversion is needed.
# ──────────────────────────────────────────────────────────────────────
METRIC_META = {
    "Index":            {"higher_is_better": True, "label": "Index"},
    "DeepSWE":          {"higher_is_better": True, "label": "DeepSWE"},
    "Terminal-Bench v2": {"higher_is_better": True, "label": "Terminal-Bench v2"},
    "SWE-Atlas-QnA":    {"higher_is_better": True, "label": "SWE-Atlas-QnA"},
}
TESTS = list(METRIC_META.keys())


# ──────────────────────────────────────────────────────────────────────
# 1. Load raw data
# ──────────────────────────────────────────────────────────────────────
def load_raw():
    if not os.path.exists(RAW_FILE):
        raise FileNotFoundError(
            f"{RAW_FILE} not found — run `python src/scrape.py` first."
        )
    with open(RAW_FILE, encoding="utf-8") as f:
        return json.load(f)


# ──────────────────────────────────────────────────────────────────────
# 2. Build 3-level hierarchy:  Test → Model → Frontend → value
# ──────────────────────────────────────────────────────────────────────
def build_hierarchy(raw):
    """
    Returns:
      hierarchy[test][model][frontend] = numeric_value
      (only entries where the value is a positive finite number)
    """
    hierarchy = {t: defaultdict(dict) for t in TESTS}
    skipped = 0
    for combo in raw:
        agent = combo.get("agent")
        model = combo.get("model")
        perf = combo.get("perf", {}) or {}
        if not agent or not model:
            skipped += 1
            continue
        for test in TESTS:
            v = perf.get(test)
            if v is None:
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            # Skip zero / negative / NaN / inf — they break ratios & logs.
            if not math.isfinite(fv) or fv <= 0:
                continue
            hierarchy[test][model][agent] = fv
    return hierarchy, skipped


def hierarchy_to_json(hierarchy):
    """Convert defaultdict nesting to plain dict for JSON output."""
    return {
        test: {
            model: dict(frontends)
            for model, frontends in models.items()
        }
        for test, models in hierarchy.items()
    }


# ──────────────────────────────────────────────────────────────────────
# 3. Filter — drop (test, model) entries with < 2 frontends
# ──────────────────────────────────────────────────────────────────────
def filter_hierarchy(hierarchy):
    """
    Returns a new hierarchy retaining only (test, model) groups that have
    at least 2 distinct frontends.
    """
    filtered = {}
    dropped = 0
    kept = 0
    for test, models in hierarchy.items():
        filtered[test] = {}
        for model, frontends in models.items():
            if len(frontends) >= 2:
                filtered[test][model] = dict(frontends)
                kept += 1
            else:
                dropped += 1
    return filtered, kept, dropped


# ──────────────────────────────────────────────────────────────────────
# 4. Pairwise ratios per (test, model) — one direction only
# ──────────────────────────────────────────────────────────────────────
def effective_value(test, value):
    """Return the 'higher = better' effective value for this test."""
    if METRIC_META[test]["higher_is_better"]:
        return value
    return 1.0 / value


def compute_pairwise_ratios(filtered):
    """
    For every (test, model) group, enumerate every unordered frontend pair
    (A, B) with A < B alphabetically. Record one observation:

        {pair: (A, B), test, model, ratio: eff(A)/eff(B)}

    Returns:
      observations: list of dicts (one per pair-per-group)
      by_pair: dict[(A, B)] -> list of ratio floats
    """
    observations = []
    by_pair = defaultdict(list)
    for test, models in filtered.items():
        for model, frontends in models.items():
            names = sorted(frontends.keys())
            for a, b in combinations(names, 2):
                va = effective_value(test, frontends[a])
                vb = effective_value(test, frontends[b])
                ratio = va / vb
                obs = {
                    "pair": [a, b],
                    "test": test,
                    "model": model,
                    "value_a": frontends[a],
                    "value_b": frontends[b],
                    "ratio": ratio,
                }
                observations.append(obs)
                by_pair[(a, b)].append(ratio)
    return observations, by_pair


# ──────────────────────────────────────────────────────────────────────
# 5. Weighted log least squares — solve for relative frontend scores
# ──────────────────────────────────────────────────────────────────────
#
# For each pair (A, B) with mean ratio M_AB and weight w_AB (#obs):
#     equation:  x_A - x_B = log(M_AB)     weight: w_AB
#
# Stack into matrix form  A x = d  with diagonal weight matrix W.
# Normal equations:  (Aᵀ W A) x = Aᵀ W d   ⇒   L x = b
# where L is the weighted graph Laplacian, b is the weighted log-ratio
# divergence.
#
# L is singular (kernel = all-ones vector). We pin the first frontend's
# log-score to 0 (i.e. score = 1), drop that row/column, solve the reduced
# (n-1)×(n-1) system, then exponentiate and renormalise so max(score)=1.
#

def weighted_log_least_squares(by_pair, all_frontends):
    """
    Args:
      by_pair: dict[(A, B)] -> list of ratio floats
      all_frontends: sorted list of every frontend that appears in any pair

    Returns:
      scores: dict[frontend] -> float (best frontend = 1.0)
      pairs_summary: dict[(A,B)] -> {mean_ratio, weight, log_mean, fit_ratio, abs_err}
    """
    n = len(all_frontends)
    if n < 2:
        return {all_frontends[0]: 1.0} if n == 1 else {}, {}

    idx = {name: i for i, name in enumerate(all_frontends)}

    # ── Build Laplacian L (n×n) and RHS b (n) ──
    L = [[0.0] * n for _ in range(n)]
    b = [0.0] * n
    pairs_summary = {}

    for (a, bb), ratios in by_pair.items():
        if not ratios:
            continue
        # Mean of ratios (arithmetic mean — consistent with the Qwen share
        # which averages the fractions then takes the value).
        # We also compute the log-mean (geometric mean) for cross-check.
        mean_ratio = sum(ratios) / len(ratios)
        w = len(ratios)
        log_mean = math.log(mean_ratio)

        i, j = idx[a], idx[bb]
        L[i][i] += w
        L[j][j] += w
        L[i][j] -= w
        L[j][i] -= w
        b[i] += w * log_mean
        b[j] -= w * log_mean

        pairs_summary[(a, bb)] = {
            "mean_ratio": mean_ratio,
            "weight": w,
            "log_mean": log_mean,
        }

    # ── Pin frontend[0] to x=0, solve the reduced (n-1)×(n-1) system ──
    # Reduced L_red = L[1:, 1:],  b_red = b[1:] - L[1:, 0] * 0 = b[1:]
    # (since x_0 = 0).
    L_red = [row[1:] for row in L[1:]]
    b_red = b[1:]

    x_red = _solve_linear(L_red, b_red)

    # Recover full x
    x = [0.0] + list(x_red)
    scores_raw = {all_frontends[i]: math.exp(x[i]) for i in range(n)}

    # ── Normalise so max score = 1 ──
    max_score = max(scores_raw.values())
    if max_score <= 0:
        # Degenerate — fall back to raw scores
        scores = scores_raw
    else:
        scores = {k: v / max_score for k, v in scores_raw.items()}

    # ── Compute fit ratios and absolute errors for verification ──
    for pair_key, info in pairs_summary.items():
        a, bb = pair_key
        fit_ratio = scores[a] / scores[bb]
        info["fit_ratio"] = fit_ratio
        info["abs_err"] = fit_ratio - info["mean_ratio"]
        info["rel_err_pct"] = (
            abs(info["abs_err"]) / info["mean_ratio"] * 100
            if info["mean_ratio"] != 0 else 0.0
        )

    return scores, pairs_summary


def _solve_linear(A, b):
    """Gaussian elimination with partial pivoting. Solves A x = b."""
    n = len(A)
    # Build augmented matrix as fractions for stability, then convert.
    # For our small systems (≤ 10 frontends) this is fast and exact.
    M = [[Fraction(A[i][j]).limit_denominator(10**12) for j in range(n)]
         + [Fraction(b[i]).limit_denominator(10**12)]
         for i in range(n)]

    for col in range(n):
        # Pivot
        pivot = max(range(col, n), key=lambda r: abs(float(M[r][col])))
        if float(M[pivot][col]) == 0:
            # Singular — shouldn't happen if the pair graph is connected
            # and we pinned one variable.
            continue
        M[col], M[pivot] = M[pivot], M[col]
        # Eliminate below
        for r in range(col + 1, n):
            if float(M[r][col]) == 0:
                continue
            factor = M[r][col] / M[col][col]
            for c in range(col, n + 1):
                M[r][c] -= factor * M[col][c]

    # Back-substitute
    x = [Fraction(0)] * n
    for r in range(n - 1, -1, -1):
        s = M[r][n]
        for c in range(r + 1, n):
            s -= M[r][c] * x[c]
        if float(M[r][r]) == 0:
            x[r] = Fraction(0)
        else:
            x[r] = s / M[r][r]
    return [float(v) for v in x]


# ──────────────────────────────────────────────────────────────────────
# 6. Main pipeline
# ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("[1/6] Loading raw data ...")
    raw = load_raw()
    print(f"  Loaded {len(raw)} combinations")

    print("\n[2/6] Building 3-level hierarchy (Test → Model → Frontend) ...")
    hierarchy, skipped = build_hierarchy(raw)
    total_entries = sum(
        len(frontends)
        for models in hierarchy.values()
        for frontends in models.values()
    )
    print(f"  Tests: {len(hierarchy)}")
    print(f"  Total (test, model, frontend) entries: {total_entries}")
    print(f"  Skipped (missing agent/model): {skipped}")
    with open(HIERARCHY_FILE, "w", encoding="utf-8") as f:
        json.dump(hierarchy_to_json(hierarchy), f, ensure_ascii=False, indent=2)
    print(f"  Saved → {HIERARCHY_FILE}")

    print("\n[3/6] Filtering: dropping (test, model) groups with < 2 frontends ...")
    filtered, kept, dropped = filter_hierarchy(hierarchy)
    print(f"  Kept {kept} (test, model) groups, dropped {dropped}")
    with open(FILTERED_FILE, "w", encoding="utf-8") as f:
        json.dump(hierarchy_to_json(filtered), f, ensure_ascii=False, indent=2)
    print(f"  Saved → {FILTERED_FILE}")

    # Summarise what survived
    surviving_frontends = set()
    surviving_models = set()
    for test, models in filtered.items():
        for model, frontends in models.items():
            surviving_frontends.update(frontends.keys())
            surviving_models.add(model)
    print(f"  Surviving frontends: {sorted(surviving_frontends)}")
    print(f"  Surviving models: {sorted(surviving_models)}")

    print("\n[4/6] Computing pairwise frontend ratios per (test, model) ...")
    observations, by_pair = compute_pairwise_ratios(filtered)
    print(f"  Total ratio observations: {len(observations)}")
    print(f"  Unique frontend pairs: {len(by_pair)}")
    for pair, ratios in sorted(by_pair.items()):
        print(f"    {pair[0]:15s} / {pair[1]:15s}  : {len(ratios):3d} obs")
    with open(RATIOS_FILE, "w", encoding="utf-8") as f:
        json.dump(
            {
                "observations": observations,
                "by_pair": {
                    f"{a}||{b}": ratios for (a, b), ratios in by_pair.items()
                },
            },
            f, ensure_ascii=False, indent=2,
        )
    print(f"  Saved → {RATIOS_FILE}")

    print("\n[5/6] Weighted log least squares fit (best frontend = 1) ...")
    all_frontends = sorted(surviving_frontends)
    scores, pairs_summary = weighted_log_least_squares(by_pair, all_frontends)
    print(f"  Frontends ranked (best = 1):")
    for name in sorted(scores, key=lambda k: -scores[k]):
        print(f"    {name:15s}  {scores[name]:.6f}")

    print(f"\n  Pair fit verification:")
    print(f"    {'pair':40s}  {'#obs':>4s}  {'mean':>8s}  {'fit':>8s}  {'rel%':>6s}")
    for (a, b), info in sorted(pairs_summary.items()):
        pair_str = f"{a} / {b}"
        print(
            f"    {pair_str:40s}  {info['weight']:4d}  "
            f"{info['mean_ratio']:8.4f}  {info['fit_ratio']:8.4f}  "
            f"{info['rel_err_pct']:6.2f}"
        )

    print("\n[6/6] Saving final scores ...")
    output = {
        "metadata": {
            "source": "https://artificialanalysis.ai/agents/coding-agents",
            "method": (
                "Weighted log least squares (Laplacian system) on pairwise "
                "frontend ratios. Each pair's mean ratio is weighted by its "
                "observation count. Final scores are normalised so the best "
                "frontend = 1. See README.md for the full derivation."
            ),
            "tests": TESTS,
            "metric_direction": {
                t: "higher_is_better" if METRIC_META[t]["higher_is_better"]
                else "lower_is_better"
                for t in TESTS
            },
            "total_combinations": len(raw),
            "surviving_models": sorted(surviving_models),
            "surviving_frontends": sorted(surviving_frontends),
            "total_observations": len(observations),
        },
        "scores": scores,
        "pairs": [
            {
                "frontend_a": a,
                "frontend_b": b,
                "mean_ratio": info["mean_ratio"],
                "weight": info["weight"],
                "fit_ratio": info["fit_ratio"],
                "abs_err": info["abs_err"],
                "rel_err_pct": info["rel_err_pct"],
            }
            for (a, b), info in sorted(pairs_summary.items())
        ],
    }
    with open(SCORES_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"  Saved → {SCORES_FILE}")


if __name__ == "__main__":
    main()
