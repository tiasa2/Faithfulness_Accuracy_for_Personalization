"""Redraw the paper's figures from results/ (writes results/figures/fig{2,3,4,5}.{png,pdf}).

Fig. 2  accuracy vs faithfulness over DPO iterations, one panel per preference-pair strategy
        (3-run means; ellipses = +/- 1 sd)
Fig. 3  correct-and-faithful share over DPO iterations, per strategy (3-run mean +/- sd)
Fig. 4  correct-and-faithful % per education and political-ideology group, before/after curriculum training
Fig. 5  the same for every demographic attribute

  python analysis/make_figures.py
"""
import statistics as st

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Ellipse

from common import RESULTS, read_csv
from faithful_accuracy import opinionqa as O
from reproduce_tables import DEMOGRAPHICS, cf_by_group

OUT = RESULTS / "figures"
STRATEGIES = [("Correctness only", "correctness_only"), ("Faithfulness only", "faithfulness_only"),
              ("Both correctness and faithfulness", "combined"), ("Curriculum", "curriculum")]
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # categorical slots 1-4 (CVD-validated)
MARKERS = ["o", "s", "^", "D"]
ITER = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]    # ordinal blue ramp for iterations 0-3
HEAT = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
GROUP_ORDER = {  # natural order where one exists; other attributes are listed alphabetically
    "AGE": ["18-29", "30-49", "50-64", "65+"],
    "EDUCATION": ["Less than high school", "High school graduate", "Some college, no degree", "Associate's degree",
                  "College graduate/some postgrad", "Postgraduate"],
    "INCOME": ["Less than $30,000", "$30,000-$50,000", "$50,000-$75,000", "$75,000-$100,000", "$100,000 or more"],
    "POLIDEOLOGY": ["Very liberal", "Liberal", "Moderate", "Conservative", "Very conservative"],
}
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF, "axes.facecolor": SURF})


def trajectories():
    """strategy -> 4 points (iterations 0-3) of {metric: (mean, sd)} over the 3 runs."""
    runs = read_csv("results/opinionqa/strategy_runs.csv")
    def point(strategy, it):
        rs = [r for r in runs if r["strategy"] == strategy and int(r["dpo_iteration"]) == it]
        vals = {"acc": [float(r["accuracy"]) for r in rs], "F": [float(r["faithfulness"]) for r in rs],
                "CF": [100 * int(r["correct_faithful"]) / int(r["n"]) for r in rs]}
        return {k: (st.mean(v), st.stdev(v)) for k, v in vals.items()}
    out = {}
    for label, key in STRATEGIES:
        # faithfulness-only and curriculum share iteration 1 (same objective); only the curriculum run of it exists
        first = "curriculum" if key == "faithfulness_only" else key
        out[label] = [point("untrained", 0), point(first, 1), point(key, 2), point(key, 3)]
    return out


def fig2(data):
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.2), sharex=True, sharey=True)
    for ax, (name, pts) in zip(axes.flat, data.items()):
        ax.plot([p["acc"][0] for p in pts], [p["F"][0] for p in pts], ls="--", lw=1, color=MUTED, zorder=1)
        for it, p in enumerate(pts):
            ax.add_patch(Ellipse((p["acc"][0], p["F"][0]), 2 * p["acc"][1], 2 * p["F"][1], color=ITER[it], alpha=0.25, lw=0, zorder=2))
            ax.scatter(p["acc"][0], p["F"][0], s=46, color=ITER[it], edgecolor=SURF, linewidth=1.5, zorder=3)
            ax.annotate("base" if it == 0 else str(it), (p["acc"][0], p["F"][0]), xytext=(5, 4), textcoords="offset points", fontsize=8, color=INK2)
        ax.set_title(name, fontsize=9.5, color=INK)
    for ax in axes[1]:
        ax.set_xlabel("Average correctness")
    for ax in axes[:, 0]:
        ax.set_ylabel("Average faithfulness")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, markersize=7) for c in ITER]
    fig.legend(handles, ["baseline", "iter 1", "iter 2", "iter 3"], loc="lower center", ncol=4, frameon=False, labelcolor=INK2)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


def fig3(data):
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for (name, pts), c, m in zip(data.items(), SERIES, MARKERS):
        ys = [p["CF"][0] for p in pts]
        ax.errorbar(range(4), ys, yerr=[p["CF"][1] for p in pts], color=c, marker=m, markersize=6, lw=2, capsize=3, markeredgecolor=SURF, label=name)
        ax.annotate(name, (3, ys[-1]), xytext=(8, 0), textcoords="offset points", va="center", fontsize=8, color=INK2)
    ax.set_xticks(range(4), ["baseline", "1", "2", "3"])
    ax.set_xlabel("DPO iteration")
    ax.set_ylabel("Correct & faithful samples (%)")
    ax.set_xlim(-0.2, 3.2)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper left")
    fig.tight_layout()
    return fig


def heatmaps(fields):
    before, after = cf_by_group("untrained"), cf_by_group("curriculum")
    def ordered(f):
        present = {g for (ff, g) in before if ff == f}
        return [g for g in GROUP_ORDER.get(f, []) if g in present] + sorted(present - set(GROUP_ORDER.get(f, [])))
    groups = {f: ordered(f) for f in fields}
    fig, axes = plt.subplots(len(fields), 1, figsize=(4.2, 0.32 * sum(len(g) for g in groups.values()) + 0.6 * len(fields)),
                             gridspec_kw={"height_ratios": [len(groups[f]) for f in fields]}, squeeze=False)
    for ax, f in zip(axes[:, 0], fields):
        vals = [[before[(f, g)], after[(f, g)]] for g in groups[f]]
        ax.imshow(vals, cmap=HEAT, vmin=0, vmax=100, aspect="auto")
        for i, row in enumerate(vals):
            for j, v in enumerate(row):
                ax.text(j, i, f"{v:.1f}%", ha="center", va="center", fontsize=7.5, color=INK if v < 55 else "#ffffff")
        ax.set_yticks(range(len(groups[f])), [g.replace("$", r"\$") for g in groups[f]], fontsize=7.5)
        ax.set_xticks([0, 1], ["Untrained", "Curriculum"] if f == fields[-1] else ["", ""], fontsize=8)
        ax.set_title(O.PERSONA_FIELDS[f][0].upper() + O.PERSONA_FIELDS[f][1:], fontsize=9, color=INK)
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.text(0.5, 0.005, "% of held-out answers that are correct and faithful", ha="center", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    return fig


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = trajectories()
    figs = [(fig2(data), "fig2_trajectories"), (fig3(data), "fig3_correct_faithful_by_iteration"),
            (heatmaps(["EDUCATION", "POLIDEOLOGY"]), "fig4_demographics_selected"), (heatmaps(DEMOGRAPHICS), "fig5_demographics_all")]
    for fig, stem in figs:
        for ext in ("png", "pdf"):
            fig.savefig(OUT / f"{stem}.{ext}", dpi=200, bbox_inches="tight", facecolor=SURF)
        print(f"wrote results/figures/{stem}.png/.pdf")


if __name__ == "__main__":
    main()
