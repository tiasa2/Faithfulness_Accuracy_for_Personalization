"""Faithfulness-metric validation against human ratings (paper Sec. 4.3, App. C.3, App. D; Tables 1, 2, 7, 8, 9).

Input: data/human_eval/chains.csv, 250 reasoning chains (2 batches x 25 queries x 5 setups) with
their 1-5 human rating and the scores of our metric (Qwen3-32B judge), the NLI baseline and four
API judges. Writes results/tables/table{1,2,7,8,9_human}.csv.

  python analysis/human_eval.py
"""
import statistics as st

from common import compare, pearson, read_csv, spearman, write_csv

SETUPS = ["Base", "Persona", "Axioms", "Persona_and_Axioms"]
JUDGES = {"Qwen3-32B (ours)": "qwen_F", "GPT-5": "gpt5_F", "GPT-5-mini": "gpt5mini_F",
          "GPT-4o-mini": "gpt4omini_F", "GLM-4.7-Flash": "glm47flash_F"}


def main():
    rows = read_csv("data/human_eval/chains.csv")
    for r in rows:
        for k in ["human_rating", "qwen_F", "qwen_grounding", "qwen_contradiction_rate", "nli_F", *JUDGES.values()]:
            r[k] = float(r[k])
    batches = {b: [r for r in rows if r["batch"] == b] for b in ("1", "2")}

    print("\nTables 1 and 8: correlation with human ratings (pooled over the chains of a batch)")
    t8 = []
    for b, R in batches.items():
        h = [r["human_rating"] for r in R]
        for name, col in [("proposed metric", "qwen_F"), ("NLI faithfulness", "nli_F")]:
            x = [r[col] for r in R]
            t8.append({"batch": b, "metric": name, "pearson": round(pearson(h, x), 3), "spearman": round(spearman(h, x), 3)})
            compare(f"batch {b}, {name}", f"r={t8[-1]['pearson']:.3f} rho={t8[-1]['spearman']:.3f}",
                    {("1", "proposed metric"): "0.629 / 0.675", ("1", "NLI faithfulness"): "-0.116 / -0.083",
                     ("2", "proposed metric"): "0.779 / 0.835 (Table 1)", ("2", "NLI faithfulness"): "-0.150 / -0.155 (Table 1)"}[(b, name)])
    write_csv(t8, "table8_metric_vs_human.csv")
    write_csv([r for r in t8 if r["batch"] == "2"], "table1_metric_vs_human.csv")

    print("\nTable 2: mean score per prompting setup (mean of the two batches)")
    paper2 = {"Base": "0.724 / 0.314", "Persona": "0.778 / 0.358", "Axioms": "0.632 / 0.927", "Persona_and_Axioms": "0.689 / 0.894"}
    t2 = []
    for s in SETUPS:
        vals = {col: st.mean(st.mean(r[col] for r in R if r["setup"] == s) for R in batches.values()) for col in ("nli_F", "qwen_F")}
        t2.append({"setup": s, "nli_faithfulness": round(vals["nli_F"], 3), "proposed_metric": round(vals["qwen_F"], 3)})
        compare(s, f"NLI {vals['nli_F']:.3f} / proposed {vals['qwen_F']:.3f}", paper2[s])
    write_csv(t2, "table2_metric_by_setup.csv")

    print("\nTable 7 (App. C.3): multiplicative vs additive combination, by human rating")
    paper7 = {1: "85 0.31 0.99 0.304 0.649 +0.344", 2: "35 0.73 1.00 0.727 0.863 +0.137", 3: "38 0.90 1.00 0.899 0.950 +0.050",
              4: "51 0.94 1.00 0.938 0.969 +0.031", 5: "41 0.90 1.00 0.902 0.951 +0.049"}
    t7 = []
    for rating in range(1, 6):
        R = [r for r in rows if r["human_rating"] == rating]
        g = st.mean(r["qwen_grounding"] for r in R)
        nc = st.mean(1 - r["qwen_contradiction_rate"] for r in R)
        mult = st.mean(r["qwen_grounding"] * (1 - r["qwen_contradiction_rate"]) for r in R)
        add = st.mean((r["qwen_grounding"] + 1 - r["qwen_contradiction_rate"]) / 2 for r in R)
        t7.append({"human_rating": rating, "n": len(R), "mean_g": round(g, 3), "mean_1_minus_rho": round(nc, 3),
                   "multiplicative": round(mult, 3), "additive": round(add, 3), "gap": round(add - mult, 3)})
        compare(f"rating {rating}", f"{len(R)} {g:.2f} {nc:.2f} {mult:.3f} {add:.3f} {add - mult:+.3f}", paper7[rating])
    low = [r for r in rows if r["human_rating"] <= 2]
    print(f"  ratings 1-2 pooled additive over-credit: {st.mean((r['qwen_grounding'] + 1 - r['qwen_contradiction_rate']) / 2 - r['qwen_F'] for r in low):+.3f} (App. C.3 text: +0.28)")
    write_csv(t7, "table7_additive_vs_multiplicative.csv")

    print("\nTable 9, last column: per-query Spearman with human ratings, averaged over the 45 unique queries")
    print("  (5 queries were rated in both batches; their two ratings per chain are averaged)")
    paper9 = {"Qwen3-32B (ours)": "0.788", "GPT-5": "0.569", "GPT-5-mini": "0.567", "GPT-4o-mini": "0.224", "GLM-4.7-Flash": "0.615"}
    chains = {}
    for r in rows:
        chains.setdefault(r["query_id"], {}).setdefault(r["setup"], []).append(r)
    t9 = []
    for name, col in JUDGES.items():
        per_query = {}
        for q, by_setup in chains.items():
            h = [st.mean(r["human_rating"] for r in rs) for rs in by_setup.values()]
            x = [rs[0][col] for rs in by_setup.values()]
            if len(set(h)) > 1 and len(set(x)) > 1:  # Spearman undefined for constant scores
                per_query[q] = spearman(h, x)
        rho = st.mean(per_query.values())
        t9.append({"judge": name, "spearman_vs_human": round(rho, 3), "queries": len(per_query)})
        compare(name, f"{rho:.3f} ({len(per_query)} queries)", paper9[name])
    write_csv(t9, "table9_judges_vs_human.csv")


if __name__ == "__main__":
    main()
