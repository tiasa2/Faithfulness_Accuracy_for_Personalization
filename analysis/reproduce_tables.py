"""Recompute the paper's result tables from results/ and print each value next to the paper's.

  python analysis/reproduce_tables.py            # all tables
  python analysis/reproduce_tables.py 3 5 12     # selected tables

Writes results/tables/table<N>_*.csv. Tables 1, 2, 7, 8 and the human column of Table 9 are in
analysis/human_eval.py; Figures are in analysis/make_figures.py.
"""
import ast
import sys
from collections import defaultdict

from common import compare, load, mean_sd, pearson, quadrants, read_csv, sign_test, write_csv
from faithful_accuracy import opinionqa as O
from faithful_accuracy.parsing import selected_letter

SETUPS = ["Base", "Persona", "Axioms", "Persona_and_Axioms"]


def slot_means(path):
    """Mean correctness and F per prompting setup in one setups file."""
    acc, F = defaultdict(list), defaultdict(list)
    for r in load(path):
        for t, c, f in zip(r["response_types"], r["correctness"], r["rewards"]):
            acc[t].append(c)
            F[t].append(f)
    return {s: (sum(acc[s]) / len(acc[s]), sum(F[s]) / len(F[s])) for s in SETUPS}


def table3():
    print("\nTable 3: accuracy / faithfulness by prompting setup (500 respondents; mean +/- sd over runs)")
    paper = {
        "qwen3-8b_thinking": ".426/.049 .490/.160 .512/.944 .521/.919", "qwen3-8b_nothinking": ".410/.042 .470/.213 .495/.915 .503/.861",
        "qwen3-32b_thinking": ".441/.077 .509/.178 .524/.965 .529/.891", "qwen3-32b_nothinking": ".441/.087 .509/.251 .521/.933 .541/.866",
        "llama-3.3-70b-instruct": ".441/.115 .496/.158 .501/.866 .522/.786", "deepseek-r1-distill-llama-70b": ".440/.141 .491/.192 .510/.834 .526/.737"}
    rows = []
    for model, p in paper.items():
        runs = [f"results/opinionqa/setups/{model}_run{k}.json.gz" for k in range(3)] if "qwen" in model else [f"results/opinionqa/setups/{model}.json.gz"]
        per_run = [slot_means(r) for r in runs]
        cells = []
        for s in SETUPS:
            (a, a_sd), (f, f_sd) = mean_sd(x[s][0] for x in per_run), mean_sd(x[s][1] for x in per_run)
            rows.append({"model": model, "setup": s, "runs": len(runs), "accuracy": round(a, 4), "accuracy_sd": round(a_sd, 4),
                         "faithfulness": round(f, 4), "faithfulness_sd": round(f_sd, 4)})
            cells.append(f"{a:.3f}/{f:.3f}")
        print(f"  {model}\n    ours  {' '.join(cells)}\n    paper {p}")
    print("  Note: Qwen3-8B (thinking) persona+axioms accuracy is 0.512; the paper prints 0.521.")
    write_csv(rows, "table3_prompting_setups.csv")


def strategy(name, it):
    return [r for r in read_csv("results/opinionqa/strategy_runs.csv") if r["strategy"] == name and int(r["dpo_iteration"]) == it]


def quad_pct(runs):
    keys = [("C&F", "correct_faithful"), ("C&U", "correct_unfaithful"), ("I&F", "incorrect_faithful"), ("I&U", "incorrect_unfaithful")]
    return {k: mean_sd(100 * int(r[col]) / int(r["n"]) for r in runs) for k, col in keys}


def table4():
    print("\nTable 4: quadrant composition over curriculum iterations (OpinionQA held-out, 3 runs each)")
    paper = ["35.6 15.5 36.7 12.3", "38.3 14.1 35.8 11.8", "38.9 13.6 36.8 10.7", "39.9 11.9 37.1 11.0"]
    rows = []
    for it in range(4):
        q = quad_pct(strategy("untrained" if it == 0 else "curriculum", it))
        rows.append({"iteration": it, **{k: round(v[0], 2) for k, v in q.items()}})
        compare(f"{'baseline' if it == 0 else f'iteration {it}'}", " ".join(f"{q[k][0]:.1f}" for k in q), paper[it])
    write_csv(rows, "table4_curriculum_quadrants.csv")


def personamem_quadrants(model, k):
    recs = load(f"results/personamem/{model}_run{k}.json.gz")
    return quadrants([r["correctness"][0] for r in recs], [r["rewards"][0] for r in recs])


def table5():
    print("\nTable 5: quadrants before / after curriculum training (mean +/- sd over 3 runs)")
    print("  runs are repeated generations from the same checkpoint; PersonaMem curriculum run0 and run2 are identical files")
    paper = {"OpinionQA untrained": "35.56 36.67 15.51 12.27", "OpinionQA trained": "39.93 37.11 11.93 11.02",
             "PersonaMem untrained": "51.10 24.21 11.97 12.72", "PersonaMem trained": "52.92 24.86 10.73 11.49"}
    rows = []
    for label, q in [("OpinionQA untrained", quad_pct(strategy("untrained", 0))), ("OpinionQA trained", quad_pct(strategy("curriculum", 3)))]:
        rows.append({"setting": label, **{f"{k} %": f"{m:.2f} +/- {s:.2f}" for k, (m, s) in q.items()}})
        compare(label, " ".join(f"{q[k][0]:.2f}" for k in ("C&F", "I&F", "C&U", "I&U")), paper[label] + "  (C&F I&F C&U I&U)")
    for label, model in [("PersonaMem untrained", "untrained"), ("PersonaMem trained", "curriculum")]:
        runs = [personamem_quadrants(model, k) for k in range(3)]
        q = {k: mean_sd(r[k] for r in runs) for k in runs[0]}
        rows.append({"setting": label, **{f"{k} %": f"{m:.2f} +/- {s:.2f}" for k, (m, s) in q.items()}})
        compare(label, " ".join(f"{q[k][0]:.2f}" for k in ("C&F", "I&F", "C&U", "I&U")), paper[label])
    write_csv(rows, "table5_before_after.csv")


def table9():
    print("\nTable 9: mean faithfulness of the untrained and curriculum models under five judges (OpinionQA held-out)")
    paper = {"qwen3-32b": "0.922 0.942", "gpt-5": "0.458 0.473", "gpt-5-mini": "0.665 0.676", "gpt-4o-mini": "0.655 0.688", "glm-4.7-flash": "0.678 0.689"}
    rows = read_csv("results/opinionqa/judges.csv")
    out = []
    for judge, p in paper.items():
        u = sum(float(r[judge]) for r in rows if r["model"] == "untrained") / 1500
        t = sum(float(r[judge]) for r in rows if r["model"] == "curriculum") / 1500
        out.append({"judge": judge, "untrained": round(u, 3), "curriculum": round(t, 3), "delta": round(t - u, 3)})
        compare(judge, f"{u:.3f} {t:.3f} ({t - u:+.3f})", p)
    write_csv(out, "table9_judges.csv")


ORDINAL = {"PP5a", "PP6c", "FTRWORRYd", "SOURCESKEPc", "POP2", "RACESURV7", "LEGALIMG", "RACATTN"}
RACATTN_ORDER = {"Too little attention": 0, "About the right amount of attention": 1, "Too much attention": 2}


def table10():
    print("\nTable 10 / App. E: faithfulness vs accuracy on the OpinionQA held-out set")
    test = O.load_respondents("data/opinionqa/test.csv")
    rows = []
    for model, paper_r in [("untrained", "-0.08"), ("curriculum", "-0.04")]:
        recs = load(f"results/opinionqa/heldout_{model}.json.gz")
        dist = {True: [], False: []}
        for r in recs:
            q = r["question"]
            if r["correctness"][0] == 1 or q.split("_W")[0] not in ORDINAL:
                continue
            c = O.cell(test, q, r["respondent"])
            opts = O.options_dict(O.options_text(c["option_mapping"]))
            rank = {L: RACATTN_ORDER[t] for L, t in opts.items()} if q.startswith("RACATTN") else {L: n for n, L in enumerate(opts)}
            gold = next((L for L, t in opts.items() if t == c["response"]), None)  # None: respondent refused
            pred = selected_letter(r["responses"][0])  # same rule as scripts/judge.py
            if gold is not None and pred in rank:
                dist[r["rewards"][0] == 1].append(abs(rank[pred] - rank[gold]))
        F = [sum(r["rewards"][0] for r in recs if r["respondent"] == i) / 15 for i in range(100)]
        A = [sum(r["correctness"][0] for r in recs if r["respondent"] == i) / 15 for i in range(100)]
        r_ = pearson(F, A)
        mf, mu = sum(dist[True]) / len(dist[True]), sum(dist[False]) / len(dist[False])
        rows.append({"model": model, "faithful_mean_distance": round(mf, 3), "unfaithful_mean_distance": round(mu, 3),
                     "n_faithful": len(dist[True]), "n_unfaithful": len(dist[False]), "r_respondent_F_vs_accuracy": round(r_, 3)})
        compare(model, f"distance {mf:.2f} vs {mu:.2f}; r = {r_:+.3f}", f"~1.22 vs ~1.22; r = {paper_r}")
    print("  Note: the paper's Table 10 distances match the curriculum model.")
    write_csv(rows, "table10_faithfulness_vs_accuracy.csv")


def table11():
    print("\nTable 11 / Sec. 5.2: axiom-relevance ablation (untrained Qwen3-8B, persona + axioms)")
    paper = {"all": "0.5113", "minus_relevant_human": "0.4727", "minus_relevant_gpt5": "0.4927", "minus_random": "0.5147", "only_relevant_human": "-"}
    rows = read_csv("results/opinionqa/axiom_ablation.csv")
    out = []
    for cond, p in paper.items():
        acc = sum(int(r["correct"]) for r in rows if r["condition"] == cond) / 1500
        out.append({"condition": cond, "accuracy": round(acc, 4)})
        compare(cond, f"{acc:.4f}", p)
    write_csv(out, "table11_axiom_ablation.csv")


def prefeval_errors(judgment):
    """PrefEval's aggregation of its four checks into error types and accuracy."""
    ack = "yes" in judgment.get("acknow", {}).get("answer", "").lower()
    hal = ack and "yes" in judgment.get("hallucinate", {}).get("answer", "").lower()
    vio = "yes" in judgment.get("violate", {}).get("answer", "").lower()
    unh = "no" in judgment.get("helpful", {}).get("answer", "").lower()
    err = {"inconsistent": ack and not hal and vio and not unh, "hallucinated": ack and hal and vio and not unh,
           "unaware": (not ack) and vio and not unh, "unhelpful": unh}
    return not any(err.values()), err


def table12_13():
    cells = [("reminder_10turns", "Reminder, 10 turns", "0.536 0.558 +2.2 [-0.3, +4.7]"), ("reminder_0turns", "Reminder, 0 turns", "0.668 0.677 +0.9 [-1.4, +3.2]"),
             ("no_reminder_10turns", "No reminder, 10 turns", "0.079 0.081 +0.2 [-1.5, +1.9]"), ("no_reminder_0turns", "No reminder, 0 turns", "0.471 0.484 +1.3 [-1.1, +3.7]")]
    print("\nTable 12: PrefEval consensus accuracy (accurate only if GPT-5 and claude-sonnet-5 both say so)")
    net, topic_net, t12, t13 = defaultdict(int), defaultdict(int), [], []
    for cell, label, paper in cells:
        u, t = load(f"results/prefeval/untrained_{cell}.json.gz"), load(f"results/prefeval/curriculum_{cell}.json.gz")
        cu = [prefeval_errors(r["judge_gpt5"])[0] and prefeval_errors(r["judge_claude_sonnet_5"])[0] for r in u]
        ct = [prefeval_errors(r["judge_gpt5"])[0] and prefeval_errors(r["judge_claude_sonnet_5"])[0] for r in t]
        d = [int(b) - int(a) for a, b in zip(cu, ct)]
        m, sd = mean_sd(d)
        se = sd / len(d) ** 0.5
        fixed, broke = d.count(1), d.count(-1)
        for r, x in zip(u, d):
            net[r["index"]] += x
            topic_net[r["topic"]] += x
        t12.append({"condition": label, "untrained": round(sum(cu) / 1000, 3), "curriculum": round(sum(ct) / 1000, 3),
                    "delta_pp": round(100 * m, 1), "ci_low": round(100 * (m - 1.96 * se), 1), "ci_high": round(100 * (m + 1.96 * se), 1),
                    "fixed": fixed, "broken": broke, "mcnemar_p": round(sign_test(broke, fixed + broke), 3)})
        compare(label, f"{sum(cu) / 1000:.3f} {sum(ct) / 1000:.3f} {100 * m:+.1f} [{100 * (m - 1.96 * se):+.1f}, {100 * (m + 1.96 * se):+.1f}]", paper)
        # Table 13: under GPT-5, error type of items fixed (before) and broken (after) by training
        gu, gt = [prefeval_errors(r["judge_gpt5"]) for r in u], [prefeval_errors(r["judge_gpt5"]) for r in t]
        fx = [a[1] for a, b in zip(gu, gt) if not a[0] and b[0]]
        br = [b[1] for a, b in zip(gu, gt) if a[0] and not b[0]]
        count = lambda errs: {k: sum(e[k] for e in errs) for k in ("inconsistent", "hallucinated", "unaware", "unhelpful")}
        t13.append({"condition": label, "fixed": len(fx), **{f"fixed_{k}": v for k, v in count(fx).items()},
                    "broken": len(br), **{f"broken_{k}": v for k, v in count(br).items()}, "net": len(fx) - len(br)})
    pos, neg = sum(v > 0 for v in net.values()), sum(v < 0 for v in net.values())
    tp, tn = sum(v > 0 for v in topic_net.values()), sum(v < 0 for v in topic_net.values())
    compare("pooled per-item sign test", f"+{pos} / -{neg}, p = {sign_test(neg, pos + neg):.3f}", "p = 0.02")
    compare("topics with positive net change", f"{tp} of {len(topic_net)}", "15 of 20")
    print("  95% CI = paired normal approximation; three of the paper's CIs are not reproduced by it.")
    write_csv(t12, "table12_prefeval.csv")
    print("\nTable 13: PrefEval error changes under GPT-5")
    paper13 = ["87 69 +18", "95 85 +10", "31 25 +6", "102 92 +10"]
    for row, p in zip(t13, paper13):
        compare(row["condition"], f"fixed {row['fixed']} broken {row['broken']} net {row['net']:+d}", f"{p} (fixed broken net)")
    write_csv(t13, "table13_prefeval_errors.csv")


DEMOGRAPHICS = ["AGE", "CITIZEN", "EDUCATION", "INCOME", "MARITAL", "RACE", "CREGION", "RELIG", "POLIDEOLOGY", "SEX"]


def cf_by_group(model):
    """Correct-and-faithful % per demographic group on the held-out set."""
    test = O.load_respondents("data/opinionqa/test.csv")
    persona = [ast.literal_eval(p) for p in test["Persona"]]
    counts = defaultdict(lambda: [0, 0])
    for r in load(f"results/opinionqa/heldout_{model}.json.gz"):
        for field in DEMOGRAPHICS:
            if field not in persona[r["respondent"]]:
                continue
            g = counts[(field, persona[r["respondent"]][field])]
            g[0] += r["correctness"][0] == 1 and r["rewards"][0] == 1
            g[1] += 1
    return {k: 100 * c / n for k, (c, n) in counts.items()}


def table14():
    import statistics as st
    print("\nTable 14: spread of correct-and-faithful % across demographic groups (sample variance)")
    paper = {"AGE": "12.05 4.05", "CITIZEN": "490.26 202.70", "EDUCATION": "22.56 17.75", "INCOME": "3.96 17.66", "MARITAL": "20.60 29.56",
             "RACE": "109.83 52.31", "CREGION": "14.43 12.32", "RELIG": "39.27 21.20", "POLIDEOLOGY": "14.20 2.64", "SEX": "9.95 2.15"}
    before, after = cf_by_group("untrained"), cf_by_group("curriculum")
    rows = []
    for field in DEMOGRAPHICS:
        b = [v for (f, _), v in before.items() if f == field]
        a = [v for (f, _), v in after.items() if f == field]
        rows.append({"demographic": O.PERSONA_FIELDS[field], "groups": len(b), "variance_untrained": round(st.variance(b), 2),
                     "variance_curriculum": round(st.variance(a), 2), "sd_untrained": round(st.stdev(b), 2), "sd_curriculum": round(st.stdev(a), 2),
                     "range_untrained": round(max(b) - min(b), 2), "range_curriculum": round(max(a) - min(a), 2)})
        compare(O.PERSONA_FIELDS[field], f"{st.variance(b):.2f} {st.variance(a):.2f}", paper[field])
    write_csv(rows, "table14_demographic_spread.csv")


TABLES = {"3": table3, "4": table4, "5": table5, "9": table9, "10": table10, "11": table11, "12": table12_13, "14": table14}

if __name__ == "__main__":
    for key in sys.argv[1:] or TABLES:
        TABLES[key]()
