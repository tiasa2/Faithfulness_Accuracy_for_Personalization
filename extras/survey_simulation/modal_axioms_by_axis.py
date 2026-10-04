"""Modal / axiom-typicality analysis of faithfulness, per demographic axis.

For each demographic attribute (and a no-grouping population baseline), an item is
"deviant" if the respondent departs from their demographic group's modal answer
(group modals from the 400 TRAIN respondents; groups with <MIN_GROUP train members
or tied modals are undefined). Three deviance definitions per axis:

  answer   : the respondent's TRUE answer to this question != group modal (binary)
  axioms14 : share of the respondent's other 14 axioms (= their other answers) that
             differ from the group modal (continuous, standardized)
  relevant : same share, restricted to the question's relevant axioms
             (hand-written map, relevant_axioms.json)

Outcomes (held-out 100 users x 15 Qs, BEFORE = iter1_1, AFTER = best_model):
F==1 (fully faithful), mean F, correctness. Each is regressed on deviance with
question fixed effects, separately for before, after and the paired gain
(after - before). A second model adds the population-baseline deviance as a
covariate, so the axis coefficient = deviance from the group BEYOND general
atypicality. CIs: user-clustered bootstrap. BH-FDR over the gain tests.

Mention analysis: share of responses whose `### Reasoning:` section (the part the
judge reads) or thinking block mentions each attribute, and the F==1 rate with vs
without a mention.

Outputs: modal_axioms_by_axis.csv, modal_axioms_by_axis_mentions.csv,
results/supplementary/modal_axioms_by_axis.png/.pdf
"""
import ast
import json
import re
from collections import Counter

import numpy as np
import pandas as pd

R = ""
BEFORE = "responses_reward_eval_full_Qwen_iter1_1.json"
AFTER = "best_responses_reward_for_best_model.json"
MIN_GROUP = 10
N_BOOT = 1000
SEED = 0

AXES = ["POPULATION", "SEX", "RELIG", "RELIGATTEND", "RACE", "AGE", "EDUCATION",
        "INCOME", "CREGION", "MARITAL", "POLPARTY", "POLIDEOLOGY", "PARTYxIDEO"]
DEFS = ["answer", "axioms14", "relevant"]

MENTION = {
    "SEX": r"\b(female|male|woman|women|man|men|gender)\b",
    "RELIG": r"\b(protestant|catholic|christian|atheist|agnostic|jewish|muslim|mormon|religio\w*)\b",
    "RELIGATTEND": r"\b(attend\w*|church\w*|services)\b",
    "RACE": r"\b(white|black|hispanic|latino|latina|asian)\b",
    "AGE": r"\b(18-29|30-49|50-64|65\+|age|aged|young|younger|older|elderly|retire\w*)\b",
    "EDUCATION": r"\b(high school|college|degree|postgrad\w*|graduate|educational background|education level)\b",
    "INCOME": r"\b(income|\$\d|wealth\w*|affluent|low-income|high-income|earn\w*)\b",
    "CREGION": r"\b(midwest\w*|south|southern|northeast\w*|west|western|region\w*)\b",
    "MARITAL": r"\b(married|divorced|widowed|separated|never been married|single|partner|marital)\b",
    "POLPARTY": r"\b(republican\w*|democrat\w*|independent|party|partisan)\b",
    "POLIDEOLOGY": r"\b(conservative\w*|liberal\w*|moderate|ideolog\w*)\b",
}


def P(s):
    return ast.literal_eval(s) if isinstance(s, str) else s


def group_key(p, axis):
    if axis == "POPULATION":
        return "ALL"
    if axis == "PARTYxIDEO":
        a, b = p.get("POLPARTY"), p.get("POLIDEOLOGY")
        return None if a is None or b is None else (a, b)
    return p.get(axis)


def load_rewards(path):
    out = []
    for x in json.load(open(R + path)):
        resp = P(x["responses"])[0]
        think = resp.split("</think>")[0] if "</think>" in resp else ""
        post = resp.split("</think>")[-1]
        m = re.search(r"###\s*Reasoning:\s*(.*)", post, re.S)
        out.append(dict(F=float(P(x["rewards"])[0]), C=float(P(x["correctness"])[0]),
                        think=think.lower(), reas=(m.group(1) if m else "").lower()))
    return pd.DataFrame(out)


def main():
    rng = np.random.default_rng(SEED)
    tr = pd.read_csv(R + "data/opinionqa/final_created_train.csv")
    te = pd.read_csv(R + "data/opinionqa/final_created_test.csv")
    qs = [c for c in te.columns if c not in ("Unnamed: 0", "Persona")]
    rel = {e["key"]: {s["source_key"] for s in e["relevant_axioms"]}
           for e in json.load(open(R + "data/opinionqa/relevant_axioms.json"))["keys"]}
    tr_p = [P(s) for s in tr.Persona]
    te_p = [P(s) for s in te.Persona]
    tr_a = [{q: P(r[q])["response"] for q in qs} for _, r in tr.iterrows()]
    te_a = [{q: P(r[q])["response"] for q in qs} for _, r in te.iterrows()]

    b, a = load_rewards(BEFORE), load_rewards(AFTER)
    assert len(b) == len(a) == 15 * len(te)
    users = np.repeat(np.arange(len(te)), 15)
    qcol = np.tile(qs, len(te))
    Q = pd.get_dummies(pd.Series(qcol)).values.astype(float)
    Y = np.column_stack([
        (b.F == 1), (a.F == 1), (a.F == 1).astype(float) - (b.F == 1),
        b.F, a.F, a.F - b.F,
        b.C, a.C, a.C - b.C]).astype(float)
    ynames = [f"{m}_{t}" for m in ("F1", "meanF", "acc") for t in ("before", "after", "gain")]

    # deviance per axis x definition
    X = {}
    for axis in AXES:
        size = Counter(group_key(p, axis) for p in tr_p)
        cnt = {}
        for p, ans in zip(tr_p, tr_a):
            g = group_key(p, axis)
            if g is None or size[g] < MIN_GROUP:
                continue
            for q in qs:
                cnt.setdefault((g, q), Counter())[ans[q]] += 1
        dev_user = []
        for p, ans in zip(te_p, te_a):
            g = group_key(p, axis)
            d = {}
            for q in qs:
                mc = cnt.get((g, q), Counter()).most_common() if g is not None else []
                d[q] = None if not mc or (len(mc) > 1 and mc[0][1] == mc[1][1]) else float(ans[q] != mc[0][0])
            dev_user.append(d)
        for de in DEFS:
            v = []
            for u, q in zip(users, qcol):
                d = dev_user[u]
                if de == "answer":
                    v.append(np.nan if d[q] is None else d[q])
                else:
                    pool = [x for x in qs if x != q and (de == "axioms14" or x in rel[q])]
                    vals = [d[x] for x in pool if d[x] is not None]
                    v.append(np.mean(vals) if vals else np.nan)
            X[(axis, de)] = np.array(v, dtype=float)

    ix = {u: np.where(users == u)[0] for u in np.unique(users)}
    boots = [np.concatenate([ix[u] for u in rng.choice(len(te), len(te))]) for _ in range(N_BOOT)]

    def fit(x, s, ctrl=None):
        cols = [x[s]] + ([ctrl[s]] if ctrl is not None else []) + [Q[s]]
        return np.linalg.lstsq(np.column_stack(cols), Y[s], rcond=None)[0][0]

    rows = []
    for (axis, de), x in X.items():
        ok = ~np.isnan(x)
        xs = x.copy()
        if de != "answer":  # per-SD for continuous deviance
            xs[ok] = (x[ok] - x[ok].mean()) / x[ok].std()
        variants = [("raw", None)]
        if axis != "POPULATION":
            pc = X[("POPULATION", de)].copy()
            okp = ~np.isnan(pc)
            if de != "answer":
                pc[okp] = (pc[okp] - pc[okp].mean()) / pc[okp].std()
            variants.append(("ctrl_population", pc))
        for vname, ctrl in variants:
            m = ok if ctrl is None else ok & ~np.isnan(ctrl)
            if m.sum() < 200 or np.nanstd(xs[m]) == 0:
                continue
            idx = np.where(m)[0]
            pt = fit(xs, idx, ctrl)
            bs = np.array([fit(xs, s[m[s]], ctrl) for s in boots])
            lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
            p2 = np.minimum(1, 2 * np.minimum((bs <= 0).mean(0), (bs >= 0).mean(0)))
            row = dict(axis=axis, definition=de, model=vname, n=int(m.sum()),
                       mean_deviance=float(x[m].mean()),
                       unit="nonmodal vs modal" if de == "answer" else "per SD of deviance")
            for k, yn in enumerate(ynames):
                row[yn] = pt[k]; row[yn + "_lo"] = lo[k]; row[yn + "_hi"] = hi[k]; row[yn + "_p"] = p2[k]
            rows.append(row)
    res = pd.DataFrame(rows)

    for yn in ("F1_gain", "meanF_gain", "acc_gain", "F1_before", "meanF_before"):
        for vname in res.model.unique():
            mk = (res.model == vname).values
            p = res.loc[mk, yn + "_p"].values
            order = np.argsort(p); q = np.empty_like(p); n = len(p)
            q[order] = np.minimum.accumulate((p[order] * n / np.arange(1, n + 1))[::-1])[::-1]
            res.loc[mk, yn + "_q"] = np.minimum(q, 1)
    res.to_csv("extras/survey_simulation/results/modal_axioms_by_axis.csv", index=False)

    # mentions
    mrows = []
    for axis, pat in MENTION.items():
        rx = re.compile(pat)
        for tag, d in (("before", b), ("after", a)):
            mt = d.think.map(lambda s: bool(rx.search(s))).values
            mr = d.reas.map(lambda s: bool(rx.search(s))).values
            f1 = (d.F == 1).values
            mrows.append(dict(axis=axis, model=tag, think_mention=mt.mean(), reasoning_mention=mr.mean(),
                              F1_if_reasoning_mentions=f1[mr].mean() if mr.any() else np.nan,
                              F1_if_not=f1[~mr].mean() if (~mr).any() else np.nan))
    pd.DataFrame(mrows).to_csv("extras/survey_simulation/results/modal_axioms_by_axis_mentions.csv", index=False)
    print(f"wrote modal_axioms_by_axis.csv ({len(res)} rows), modal_axioms_by_axis_mentions.csv")


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res = pd.read_csv("extras/survey_simulation/results/modal_axioms_by_axis.csv")
    style = {"answer": ("#2a78d6", "o", "answer ≠ group modal (nonmodal − modal)"),
             "axioms14": ("#eb6834", "s", "all 14 axioms (per SD of deviance)"),
             "relevant": ("#1baf7a", "^", "relevant axioms only (per SD)")}
    ink, ink2, grid = "#0b0b0b", "#52514e", "#e4e3df"
    labels = {"POPULATION": "No grouping (population)", "SEX": "Gender", "RELIG": "Religion",
              "RELIGATTEND": "Religious attendance", "RACE": "Race", "AGE": "Age",
              "EDUCATION": "Education", "INCOME": "Income", "CREGION": "Region",
              "MARITAL": "Marital status", "POLPARTY": "Party", "POLIDEOLOGY": "Ideology",
              "PARTYxIDEO": "Party × ideology"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 6.4), sharey=True, sharex=True, facecolor="#fcfcfb")
    titles = {"raw": "Deviance from the group's modal answers",
              "ctrl_population": "…controlling for population-level atypicality"}
    for ax, model in zip(axes, ["raw", "ctrl_population"]):
        ax.set_facecolor("#fcfcfb")
        for k, axis in enumerate(AXES):
            for j, de in enumerate(DEFS):
                r = res[(res.axis == axis) & (res.definition == de) & (res.model == model)]
                if r.empty:
                    continue
                r = r.iloc[0]
                y = k + (j - 1) * 0.22
                c, mk, _ = style[de]
                sig = r.F1_gain_lo > 0 or r.F1_gain_hi < 0
                ax.plot([100 * r.F1_gain_lo, 100 * r.F1_gain_hi], [y, y], color=c, lw=2, solid_capstyle="round")
                ax.plot(100 * r.F1_gain, y, mk, ms=8, color=c if sig else "#fcfcfb",
                        markeredgecolor=c, markeredgewidth=2, zorder=3)
        ax.axvline(0, color=ink2, lw=1)
        ax.set_yticks(range(len(AXES)))
        ax.set_yticklabels([labels[a] for a in AXES], color=ink)
        ax.grid(axis="x", color=grid, lw=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color(grid)
        ax.tick_params(colors=ink2, length=0)
        ax.set_title(titles[model], color=ink, fontsize=11, loc="left")
        ax.set_xlabel("Effect on training's F=1 gain (pp)", color=ink2)
    axes[0].invert_yaxis()  # shared y: invert once
    handles = [plt.Line2D([], [], color=c, marker=m, lw=2, ms=8, label=l) for c, m, l in style.values()]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9,
               labelcolor=ink, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Training's faithfulness gain vs. deviance from demographic group, by axis "
                 "(95% user-clustered CI; filled = CI excludes 0)", color=ink, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    import os
    os.makedirs("figures", exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"extras/survey_simulation/results/modal_axioms_by_axis.{ext}", dpi=200, facecolor=fig.get_facecolor())
    print("wrote results/supplementary/modal_axioms_by_axis.png/.pdf")


if __name__ == "__main__":
    import sys
    if "--plot-only" not in sys.argv:
        main()
    plot()
