#!/usr/bin/env python3
"""
Bar-chart visualiser for the final frontend scores.

Chart design (per the user's spec)
----------------------------------
Let N = number of frontends with at least one ratio observation.

* The plot frame is a SQUARE spanning (0, 0) to (N+1, N+1).
  With N=4 frontends that's a 5×5 box, divided into (N+1)² = 25 small squares.

* 9 HORIZONTAL light grid lines are drawn along the y-axis at
  y = (i/10) * (N+1) for i = 1..9 — i.e. a 10-row division of the box.
  This mirrors the reference repo's grid style (color #333333, alpha 0.5,
  linewidth 0.4) but keeps ONLY the horizontal lines and drops the vertical
  ones — the x-axis only carries frontend labels, so vertical grid lines
  would be noise.

* Frontends are sorted by score ascending and placed on the vertical
  lines x = 1, 2, 3, ..., N. So the best frontend sits at x = N
  (rightmost non-boundary line).

* Each bar starts at y = 1 (one unit above the bottom) and extends up to
  y = 1 + (N-1) * score_i. Since the best frontend has score = 1, its bar
  reaches y = 1 + (N-1) = N — i.e. it spans (x=N, y=1) to (x=N, y=N),
  exactly the user's "4,1 → 4,4" description for N=4.

* The bar at x = N (best frontend) is drawn with the cyan accent colour.

Style reference
---------------
Matches the dark-theme aesthetic of the reference repo
(https://github.com/Elaman0117/Generic-LLM-Leaderboard/) — black background,
white boundary (#FFFFFF, linewidth 1.5), grid #333333 alpha 0.5 linewidth
0.4, cyan accent #00E5FF for the best frontend, dark fill #4A4A4A for the
other bars (matching the reference repo's "other models" colour).

Output: output/frontend_comparison.png
"""

import json
import os

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
SCORES_FILE = os.path.join(OUTPUT_DIR, "final_scores.json")
OUTPUT_PNG = os.path.join(OUTPUT_DIR, "frontend_comparison.png")


# ──────────────────────────────────────────────────────────────────────
# Font setup — Noto Sans SC handles any CJK that might sneak into labels,
# DejaVu Sans catches the rest. English-only output here, but stay safe.
# ──────────────────────────────────────────────────────────────────────
def _register_fonts():
    for path in (
        "/usr/share/fonts/truetype/chinese/NotoSansSC-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        if os.path.exists(path):
            try:
                fm.fontManager.addfont(path)
            except Exception:
                pass
    plt.rcParams["font.sans-serif"] = ["Noto Sans SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


# ──────────────────────────────────────────────────────────────────────
# Colour palette — matches the reference repo exactly.
# ──────────────────────────────────────────────────────────────────────
BG = "#000000"        # figure + axes background
FG = "#FFFFFF"        # boundary frame + labels
GRID = "#333333"      # horizontal grid lines
ACCENT = "#00E5FF"    # best-frontend bar + highlight
BAR_OTHER = "#4A4A4A"  # other bars (matches reference repo's "other models")
LABEL_BG = "#1A1A1A"  # bbox facecolor for labels (matches reference)


# ──────────────────────────────────────────────────────────────────────
# Drawing
# ──────────────────────────────────────────────────────────────────────
def plot_scores(scores, pairs, metadata, out_path):
    """
    Args:
      scores: dict[frontend] -> float (best = 1.0)
      pairs:  list of dicts with frontend_a, frontend_b, mean_ratio, weight, ...
      metadata: dict with surviving_frontends, total_observations, etc.
    """
    _register_fonts()

    # Sort frontends by score ascending — best ends up at the rightmost slot.
    frontends = sorted(scores.keys(), key=lambda k: scores[k])
    N = len(frontends)
    if N < 2:
        raise ValueError(f"Need at least 2 frontends to plot, got {N}")

    box = N + 1  # box spans (0,0) to (box, box)

    # Same figure setup as the reference repo: square aspect, black bg.
    fig, ax = plt.subplots(figsize=(14, 14))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_aspect("equal", adjustable="box")

    # ── 9 HORIZONTAL light grid lines (10-row division of the box) ──
    # y = i * box / 10 for i = 1..9
    # Same colour / alpha / linewidth as the reference repo.
    # zorder=0 so they sit behind everything else.
    # NO vertical grid lines — the x-axis only carries frontend labels.
    for i in range(1, 10):
        y = i * box / 10
        ax.plot(
            [0, box], [y, y],
            color=GRID, alpha=0.5, linewidth=0.4, zorder=0,
        )

    # ── Boundary frame (matches reference repo) ──
    ax.plot([0, box], [0, 0], color=FG, linewidth=1.5, zorder=1)
    ax.plot([0, 0], [0, box], color=FG, linewidth=1.5, zorder=1)
    ax.plot([0, box], [box, box], color=FG, linewidth=1.5, zorder=1)
    ax.plot([box, box], [0, box], color=FG, linewidth=1.5, zorder=1)

    # ── Bars ──
    # Each frontend i (0-indexed from worst to best) sits on x = i+1.
    # Bar spans y = 1 .. y = 1 + (N-1) * score_i.
    # Bar width is intentionally narrow so the bars read as discrete
    # vertical lines on the integer x positions (1, 2, ..., N).
    bar_width = 0.35
    max_bar_len = N - 1  # length of the best frontend's bar

    for i, name in enumerate(frontends):
        x_center = i + 1
        score = scores[name]
        bar_len = max_bar_len * score
        y_bottom = 1.0
        y_top = y_bottom + bar_len

        is_best = (i == N - 1)
        # Best frontend: cyan accent + white edge (matches reference repo's
        # Pareto point style). Others: dark gray fill + subtle edge.
        color = ACCENT if is_best else BAR_OTHER
        edge = FG if is_best else "#888888"
        lw = 1.2 if is_best else 0.6
        alpha = 0.95 if is_best else 0.85

        rect = Rectangle(
            (x_center - bar_width / 2, y_bottom),
            bar_width,
            bar_len,
            facecolor=color,
            edgecolor=edge,
            linewidth=lw,
            alpha=alpha,
            zorder=2,
        )
        ax.add_patch(rect)

        # Score label on top of the bar — same bbox style as the reference
        # repo's Pareto labels (dark facecolor, cyan edge for best).
        ax.text(
            x_center, y_top + 0.15,
            f"{score:.4f}",
            ha="center", va="bottom",
            color=FG, fontsize=11, fontweight="bold", zorder=5,
            bbox=dict(
                boxstyle="round,pad=0.12",
                facecolor=LABEL_BG, alpha=0.85,
                edgecolor=ACCENT if is_best else "#666666",
                linewidth=0.6 if is_best else 0.4,
            ),
        )

        # Frontend name below the bar (below the x-axis frame)
        ax.text(
            x_center, -0.4,
            name,
            ha="center", va="top",
            color=FG, fontsize=13, fontweight="bold", zorder=5,
        )

        # Highlight the best frontend with a star marker above the frame
        if is_best:
            ax.text(
                x_center, box + 0.3,
                "★ best",
                ha="center", va="bottom",
                color=ACCENT, fontsize=13, fontweight="bold", zorder=5,
            )

    # ── Title (matches reference repo style) ──
    ax.set_title(
        "Coding-Agent Frontend Performance Comparison\n"
        f"({N} frontends · {metadata.get('total_observations', '?')} ratio observations · "
        f"weighted log least squares, best = 1)",
        fontsize=15, color=FG, fontweight="bold", pad=16,
    )

    # ── Axis limits & cleanup (matches reference repo) ──
    margin = 0.5
    ax.set_xlim(-margin - 0.7, box + margin)
    ax.set_ylim(-1.0, box + 0.9)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)
    ax.tick_params(axis="both", colors=FG, length=5, width=1.2)

    # ── Method footnote (bottom-right, same style as reference repo) ──
    method = (
        f"X轴: 前端 (按得分升序排列) | Y轴: 相对得分 (最佳=1)\n"
        f"得分来源: 4项性能基准 (Index / DeepSWE / Terminal-Bench v2 / SWE-Atlas-QnA)\n"
        f"★ 加权对数最小二乘法拟合 | 共{metadata.get('total_observations', '?')}个比值观测"
    )
    ax.text(
        0.98, 0.02, method, transform=ax.transAxes, fontsize=7,
        va="bottom", ha="right", color="#AAAAAA", style="italic",
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="#111111", alpha=0.85,
            edgecolor="#444444", linewidth=0.5,
        ),
    )

    # ── Pair fit verification (bottom-left, monospace) ──
    pair_lines = []
    for p in pairs:
        pair_lines.append(
            f"{p['frontend_a']:15s} / {p['frontend_b']:15s}  "
            f"mean={p['mean_ratio']:.4f}  fit={p['fit_ratio']:.4f}  "
            f"n={p['weight']:2d}  err={p['rel_err_pct']:.2f}%"
        )
    footer = "\n".join(pair_lines)
    ax.text(
        0.02, 0.02,
        "Pair fit verification:\n" + footer,
        transform=ax.transAxes,
        ha="left", va="bottom",
        color="#AAAAAA", fontsize=7, family="monospace",
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="#111111", alpha=0.85,
            edgecolor="#444444", linewidth=0.5,
        ),
    )

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor=BG)
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main():
    if not os.path.exists(SCORES_FILE):
        raise FileNotFoundError(
            f"{SCORES_FILE} not found — run `python src/analyze.py` first."
        )
    with open(SCORES_FILE, encoding="utf-8") as f:
        data = json.load(f)

    scores = data["scores"]
    pairs = data["pairs"]
    metadata = data.get("metadata", {})

    print(f"[plot] {len(scores)} frontends, {len(pairs)} pairs")
    plot_scores(scores, pairs, metadata, OUTPUT_PNG)


if __name__ == "__main__":
    main()
