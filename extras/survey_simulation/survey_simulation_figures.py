"""Figures for survey_simulation_analysis.py (reads survey_sim/*.csv, writes survey_sim/figures/)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

OUT = "extras/survey_simulation/results/survey_sim"
SURF, INK, INK2, GRID, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#898781"
COND = {"baseline": ("#2a78d6", "o"), "persona": ("#eb6834", "s"), "axioms": ("#1baf7a", "^"),
        "persona+axioms": ("#eda100", "D")}
LABEL = {"baseline": "No persona", "persona": "Demographic persona", "axioms": "Axioms",
         "persona+axioms": "Persona + axioms"}
BASE = {"BASELINE group modal": "Ref: group modal", "BASELINE LR argmax": "Ref: LR on other answers (argmax)",
        "BASELINE LR sampled": "Ref: LR on other answers (sampled)"}
MODELS = ["Qwen3-8B", "32B-thinking", "32B-instruct", "Llama-70B", "Distill-Llama-70B"]


def style(ax):
    ax.set_facecolor(SURF)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, length=0)


def save(fig, name):
    os.makedirs(f"{OUT}/figures", exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/figures/{name}.{ext}", dpi=200, facecolor=SURF, bbox_inches="tight")
    plt.close(fig)


def fig_distribution():
    s = pd.read_csv(f"{OUT}/a1_summary.csv")
    ceil = pd.read_csv(f"{OUT}/a1_splithalf_ceiling.csv")
    s = s[s["sample"] == "500 untrained"]; ceil = ceil[ceil["sample"] == "500 untrained"]
    panels = [("question", "all 15", "wd", "split", "Distribution match per question\n1 − EMD/(K−1), split-half (higher = better)"),
              ("group", "ALL attrs pooled", "var_ratio", "full", "Within-group variance ratio\npredicted / real (1 = no compression)"),
              ("group", "ALL attrs pooled", "modal_excess", "full", "Modal collapse within group\npredicted − real share of group mode")]
    rows = [("Qwen3-8B|" + c, LABEL[c], COND[c]) for c in COND] + [(k, v, (MUTED, "X")) for k, v in BASE.items()]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True, facecolor=SURF)
    for ax, (level, scope, metric, kind, title) in zip(axes, panels):
        style(ax)
        for y, (cond, lab, (col, mk)) in enumerate(rows):
            if kind == "split":
                cc = ceil[(ceil.level == level) & (ceil.scope == scope) & (ceil.metric == metric)]
                for r in (0, 1, 2) if cond.startswith("Qwen") else (None,):
                    nm = cond + f"|run{r}" if r is not None else cond
                    row = cc[cc.condition == nm]
                    if row.empty:
                        continue
                    row = row.iloc[0]
                    if r in (None, 0):
                        ax.plot([row.lo, row.hi], [y, y], color=col, lw=2, solid_capstyle="round")
                        ax.plot(row["mean"], y, mk, ms=8, color=col)
                    else:
                        ax.plot(row["mean"], y, mk, ms=6, mfc=SURF, color=col, mew=1.5)
                re = cc[cc.condition == "REAL split-half"].iloc[0]
                ax.axvspan(re.lo, re.hi, color=GRID, zorder=0)
                ax.axvline(re["mean"], color=INK2, lw=1)
            else:
                ss = s[(s.level == level) & (s.scope == scope) & (s.metric == metric)]
                for r in (0, 1, 2) if cond.startswith("Qwen") else (None,):
                    nm = cond + f"|run{r}" if r is not None else cond
                    row = ss[ss.condition == nm]
                    if row.empty:
                        continue
                    row = row.iloc[0]
                    if r in (None, 0):
                        ax.plot([row.lo, row.hi], [y, y], color=col, lw=2, solid_capstyle="round")
                        ax.plot(row.est, y, mk, ms=8, color=col)
                    else:
                        ax.plot(row.est, y, mk, ms=6, mfc=SURF, color=col, mew=1.5)
                ax.axvline(1 if metric == "var_ratio" else 0, color=INK2, lw=1)
        ax.set_title(title, color=INK, fontsize=10, loc="left")
    axes[0].set_yticks(range(len(rows))); axes[0].set_yticklabels([r[1] for r in rows], color=INK)
    axes[0].invert_yaxis()
    axes[0].text(0.02, -0.16, "Grey band = real respondents, half vs half (noise ceiling)", transform=axes[0].transAxes,
                 color=INK2, fontsize=8)
    fig.suptitle("Qwen3-8B (untrained), 500 respondents: filled = run 0 with 95% CI; hollow = runs 1–2",
                 color=INK, fontsize=11, x=0.01, ha="left", y=1.03)
    save(fig, "fig1_distribution_qwen8b")


def fig_models():
    s = pd.read_csv(f"{OUT}/a1_summary.csv")
    s = s[(s["sample"] == "500 untrained") & (s.level == "group") & (s.scope == "ALL attrs pooled")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True, facecolor=SURF)
    for ax, metric, title in zip(axes, ("wd", "var_ratio"),
                                 ("Group × question distribution match (1 − EMD/(K−1))",
                                  "Within-group variance ratio (predicted / real)")):
        style(ax)
        for y, model in enumerate(MODELS):
            for k, (c, (col, mk)) in enumerate(COND.items()):
                row = s[(s.condition == f"{model}|{c}|run0") & (s.metric == metric)]
                if row.empty:
                    continue
                row = row.iloc[0]; yy = y + (k - 1.5) * 0.17
                ax.plot([row.lo, row.hi], [yy, yy], color=col, lw=2, solid_capstyle="round")
                ax.plot(row.est, yy, mk, ms=7, color=col)
        for b, ls in (("BASELINE group modal", "--"), ("BASELINE LR argmax", ":")):
            v = s[(s.condition == b) & (s.metric == metric)].est.values[0]
            ax.axvline(v, color=MUTED, lw=1.2, ls=ls)
        if metric == "var_ratio":
            ax.axvline(1, color=INK2, lw=1)
        ax.set_title(title, color=INK, fontsize=10, loc="left")
    axes[0].set_yticks(range(len(MODELS)))
    axes[0].set_yticklabels(["Qwen3-8B", "32B thinking", "32B instruct*", "Llama-70B*", "Distill-Llama-70B*"], color=INK)
    axes[0].invert_yaxis()
    h = [plt.Line2D([], [], color=c, marker=m, lw=2, ms=7, label=LABEL[k]) for k, (c, m) in COND.items()]
    h += [plt.Line2D([], [], color=MUTED, ls="--", label="Ref: group modal"),
          plt.Line2D([], [], color=MUTED, ls=":", label="Ref: LR argmax")]
    fig.legend(handles=h, loc="lower center", ncol=6, frameon=False, fontsize=8.5, labelcolor=INK,
               bbox_to_anchor=(0.5, -0.1))
    fig.suptitle("All models, 500 respondents, pooled over demographic groups with ≥20 respondents "
                 "(*model ID unverified)", color=INK, fontsize=11, x=0.01, ha="left", y=1.02)
    save(fig, "fig2_distribution_all_models")


def fig_regression():
    c = pd.read_csv(f"{OUT}/a2_coefficients.csv")
    sm = pd.read_csv(f"{OUT}/a2_regression_summary.csv")
    c = c[c["sample"] == "500 untrained"]; sm = sm[sm["sample"] == "500 untrained"]
    conds = [("Qwen3-8B|" + k + "|run0", LABEL[k], COND[k][0]) for k in COND] + \
            [("BASELINE LR argmax", "Ref: LR argmax", MUTED)]
    fig, axes = plt.subplots(1, len(conds), figsize=(15, 3.4), sharex=True, sharey=True, facecolor=SURF)
    lim = np.nanpercentile(np.abs(c[["real", "synthetic"]].values), 99.5) * 1.05
    for ax, (cond, lab, col) in zip(axes, conds):
        style(ax); ax.grid(axis="y", color=GRID, lw=0.8)
        d = c[c.condition == cond]
        ax.scatter(d.real, d.synthetic, s=12, color=col, alpha=0.6, linewidths=0)
        ax.plot([-lim, lim], [-lim, lim], color=INK2, lw=1)
        r = sm[sm.condition == cond].iloc[0]
        nc = pd.read_csv(f"{OUT}/a2_noise_corrected.csv")
        nc = nc[(nc["sample"] == "500 untrained") & (nc.condition == cond)]
        extra = f"\nnoise-corrected slope ≈ {nc.slope_corrected.values[0]:.2f}" if len(nc) else ""
        ax.set_title(f"{lab}\nr = {r['corr']:.2f}, slope = {r.slope:.2f}{extra}", color=INK, fontsize=9.5, loc="left")
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_xlabel("real coefficient", color=INK2)
    axes[0].set_ylabel("synthetic coefficient", color=INK2)
    fig.suptitle("Bisbee-style replication: demographic coefficients, 8 ordinal questions, Qwen3-8B run 0 "
                 "(diagonal = perfect replication)", color=INK, fontsize=11, x=0.01, ha="left", y=1.16)
    save(fig, "fig3_regression_coefficients")


def fig_faithfulness():
    a = pd.read_csv(f"{OUT}/a3_faithfulness_vs_collapse.csv")
    a = a[(a.attr == "PARTYxIDEO") & (a.variant == "deviating & model wrong")]
    outs = [("F", "Faithfulness F"), ("grounding", "Grounding"), ("contradiction_rate", "Contradiction rate")]
    conds = [c for c in a.condition.unique()]
    fig, axes = plt.subplots(1, 3, figsize=(13, 0.32 * len(conds) + 1.6), sharey=True, facecolor=SURF)
    for ax, (o, title) in zip(axes, outs):
        style(ax)
        for y, cnd in enumerate(conds):
            r = a[(a.condition == cnd) & (a.outcome == o)]
            if r.empty:
                continue
            r = r.iloc[0]
            key = cnd.split("|")[1] if "|" in cnd else "persona+axioms"
            col = COND.get(key, (MUTED, "o"))[0]
            sig = r.lo > 0 or r.hi < 0
            ax.plot([r.lo, r.hi], [y, y], color=col, lw=2, solid_capstyle="round")
            ax.plot(r.coef, y, "o", ms=7, color=col if sig else SURF, mec=col, mew=2)
        ax.axvline(0, color=INK2, lw=1)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
        ax.set_title(f"{title}\npredicts group mode − does not", color=INK, fontsize=10, loc="left")
    axes[0].set_yticks(range(len(conds)))
    axes[0].set_yticklabels([f"{c.replace('|', ' · ')} [{s}]" for c, s in
                             a.drop_duplicates("condition")[["condition", "sample"]].values], color=INK, fontsize=8)
    axes[0].invert_yaxis()
    fig.suptitle("Analysis 3 — respondents who deviate from their party×ideology mode, model answer wrong:\n"
                 "is reasoning less faithful when the wrong answer IS the group mode? "
                 "(question FE, respondent-clustered 95% CI; filled = CI excludes 0)",
                 color=INK, fontsize=10.5, x=0.01, ha="left", y=1.05)
    save(fig, "fig4_faithfulness_vs_collapse")


def fig_training():
    c = pd.read_csv(f"{OUT}/a1_contrasts.csv"); c = c[c["sample"] == "test 100"]
    a3 = pd.read_csv(f"{OUT}/a3_training_contrast.csv")
    pairs = [("trained (best_model) - untrained (iter1_1)", "Trained (best) − untrained", "#2a78d6", "o"),
             ("trained (curriculum run 2) - untrained (iter1_1)", "Trained (curriculum run 2) − untrained", "#eb6834", "s"),
             ("untrained 4-setup P+A run1 - untrained 4-setup P+A run0", "Untrained run 1 − run 0 (noise reference)", MUTED, "X")]
    panels = [("question", "all 15", "wd", "Question-level distribution match"),
              ("group", "ALL attrs pooled", "var_ratio", "Within-group variance ratio"),
              ("group", "ALL attrs pooled", "modal_excess", "Modal collapse (excess modal share)"),
              (None, None, "F", "Analysis 3: F | predicts mode − not\n(deviating & wrong, party×ideology)")]
    fig, axes = plt.subplots(1, 4, figsize=(14, 2.6), sharey=True, facecolor=SURF)
    for ax, (level, scope, metric, title) in zip(axes, panels):
        style(ax)
        for y, (con, lab, col, mk) in enumerate(pairs):
            if level is None:
                r = a3[(a3.attr == "PARTYxIDEO") & (a3.variant == "deviating & model wrong") & (a3.outcome == "F")
                       & (a3.contrast == con.replace("untrained 4-setup P+A run1 - untrained 4-setup P+A run0",
                                                     "untrained 4-setup P+A run1 - untrained (iter1_1)"))]
            else:
                r = c[(c.level == level) & (c.scope == scope) & (c.metric == metric) & (c.contrast == con)]
            if r.empty:
                continue
            r = r.iloc[0]
            ax.plot([r.lo, r.hi], [y, y], color=col, lw=2, solid_capstyle="round")
            ax.plot(r["diff"], y, mk, ms=8, color=col)
        ax.axvline(0, color=INK2, lw=1)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
        ax.set_title(title, color=INK, fontsize=9.5, loc="left")
    axes[0].set_yticks(range(len(pairs))); axes[0].set_yticklabels([p[1] for p in pairs], color=INK, fontsize=9)
    axes[0].invert_yaxis()
    fig.suptitle("Analysis 4 — training effect on the 100 held-out respondents (persona + axioms), paired respondent "
                 "bootstrap 95% CI", color=INK, fontsize=11, x=0.01, ha="left", y=1.08)
    save(fig, "fig5_training_effect")


if __name__ == "__main__":
    fig_distribution(); fig_models(); fig_regression(); fig_faithfulness(); fig_training()
    print("figures written to", f"{OUT}/figures")
