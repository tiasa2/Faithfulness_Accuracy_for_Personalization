"""Does individual grounding preserve WHICH BELIEFS GO WITH WHICH ANSWERS?

Pre-specified design (selection uses only the 400 TRAIN respondents' human answers):
  For each target question Y (15), candidate beliefs B = Y's relevant questions in the hand-written
  map (relevant_axioms.json). Each candidate is binarized at the category cut closest to a 50/50
  split in the train data. The selected B is the candidate with the largest Cramer's V with Y in the
  train data. All 15 (Y, B) pairs are reported. B is one of the person's axioms (shown in the
  axioms conditions); Y is never in the prompt.

Metrics (on held-out respondents; humans vs model predictions for the SAME respondents, B = the
respondent's TRUE belief):
  cond_tv    mean over pairs and b in {0,1} of TV(P_model(Y|B=b), P_human(Y|B=b)); cells n>=MIN_CELL
  assoc_err  mean over pairs of 0.5*sum|d_model - d_human|, d = P(Y|B=1) - P(Y|B=0)   (relationship error)
  marg_tv    mean over pairs of TV(P_model(Y), P_human(Y))                            (reference)
  reweight   mean TV between weighted human and weighted model marginals of Y when the population is
             reweighted to P(B=1) in {.2,.35,.5,.65,.8}, within party strata (party mix preserved)
Identity-shuffling control: within party strata, permute each question's model predictions across
respondents (200x); keeps the stratum answer distribution, breaks alignment with individual beliefs.
Uncertainty: respondent bootstrap (B=1000), paired across conditions.

Conditions: test 100 - untrained persona-only / axioms-only / persona+axioms (3 runs each, 4-setup
files), untrained persona+axioms (iter1_1), trained persona+axioms (best_model; curriculum run 2).
Also all 500 respondents for the untrained context contrasts (no training involved).

Outputs: belief_cond/pairs.csv, metrics.csv, contrasts.csv, shuffle.csv, per_pair.csv
"""
import json
import os
from collections import Counter

import numpy as np
import pandas as pd

import survey_simulation_analysis as S

OUT = "extras/survey_simulation/results/belief_cond"
MIN_CELL = 10
TARGETS = [0.2, 0.35, 0.5, 0.65, 0.8]


def cramers_v(x, y):
    t = pd.crosstab(x, y).values.astype(float)
    if t.shape[0] < 2 or t.shape[1] < 2:
        return 0.0
    e = t.sum(1, keepdims=True) * t.sum(0, keepdims=True) / t.sum()
    chi2 = ((t - e) ** 2 / e).sum()
    return float(np.sqrt(chi2 / (t.sum() * (min(t.shape) - 1))))


def select_pairs(D, train_idx):
    rel = {e["key"]: [s["source_key"] for s in e["relevant_axioms"]]
           for e in json.load(open("data/opinionqa/relevant_axioms.json"))["keys"]}
    real, qs = D["real"], D["qs"]
    rows = []
    for jy, y in enumerate(qs):
        best = None
        for bq in rel[y]:
            jb = qs.index(bq)
            ok = train_idx[(real[train_idx, jy] >= 0) & (real[train_idx, jb] >= 0)]
            K = D["K"][jb]
            cuts = [(c, abs(np.mean(real[ok, jb] < c) - 0.5)) for c in range(1, K)]
            cut = min(cuts, key=lambda t: t[1])[0]
            bbin = (real[ok, jb] >= cut).astype(int)
            v = cramers_v(bbin, real[ok, jy])
            if best is None or v > best["train_V"]:
                lo = ", ".join(D["order"][bq][:cut]); hi = ", ".join(D["order"][bq][cut:])
                best = dict(Y=y, B=bq, cut=cut, B0=lo, B1=hi, train_V=v, train_share_B1=float(bbin.mean()),
                            n_train=len(ok))
        rows.append(best)
    return pd.DataFrame(rows)


def dist(vals, w, K):
    c = np.bincount(vals, weights=w, minlength=K)[:K]
    s = c.sum()
    return c / s if s > 0 else np.full(K, np.nan)


def metrics(D, pairs, pred, idx, w_resp, strata):
    """pred: (N x Q) categories. w_resp: bootstrap multiplicity per respondent in idx."""
    real, qs = D["real"], D["qs"]
    out = dict(cond_tv=[], assoc_err=[], marg_tv=[], reweight=[])
    per = []
    for _, p in pairs.iterrows():
        jy, jb, K = qs.index(p.Y), qs.index(p.B), D["K"][qs.index(p.Y)]
        t, b, m = real[idx, jy], real[idx, jb], pred[idx, jy]
        ok = (t >= 0) & (b >= 0) & (m >= 0) & (w_resp > 0)
        bb = (b >= p.cut).astype(int)
        w = w_resp.astype(float)
        cells, dh, dm = [], {}, {}
        for v in (0, 1):
            s = ok & (bb == v)
            if w[s].sum() < MIN_CELL:
                continue
            ph, pm = dist(t[s], w[s], K), dist(m[s], w[s], K)
            cells.append(0.5 * np.abs(ph - pm).sum()); dh[v], dm[v] = ph, pm
        if cells:
            out["cond_tv"].append(np.mean(cells))
        if len(dh) == 2:
            out["assoc_err"].append(0.5 * np.abs((dm[1] - dm[0]) - (dh[1] - dh[0])).sum())
        out["marg_tv"].append(0.5 * np.abs(dist(t[ok], w[ok], K) - dist(m[ok], w[ok], K)).sum())
        rw = []
        for pi in TARGETS:
            ww = np.zeros_like(w)
            for g in np.unique(strata):
                sg = ok & (strata == g)
                if w[sg].sum() == 0:
                    continue
                p1 = (w[sg] * bb[sg]).sum() / w[sg].sum()
                if p1 in (0, 1):
                    ww[sg] = w[sg]
                    continue
                ww[sg] = w[sg] * np.where(bb[sg] == 1, pi / p1, (1 - pi) / (1 - p1))
            rw.append(0.5 * np.abs(dist(t[ok], ww[ok], K) - dist(m[ok], ww[ok], K)).sum())
        out["reweight"].append(np.mean(rw))
        per.append(dict(Y=p.Y, B=p.B, cond_tv=np.mean(cells) if cells else np.nan,
                        assoc_err=out["assoc_err"][-1] if len(dh) == 2 else np.nan,
                        human_shift=0.5 * np.abs(dh[1] - dh[0]).sum() if len(dh) == 2 else np.nan,
                        model_shift=0.5 * np.abs(dm[1] - dm[0]).sum() if len(dh) == 2 else np.nan))
    return {k: float(np.mean(v)) if v else np.nan for k, v in out.items()}, per


def main():
    rng = np.random.default_rng(0)
    os.makedirs(OUT, exist_ok=True)
    D = S.load_real()
    N = len(D["pers"]); all_idx = np.arange(N); test_idx = D["test_idx"]
    train_idx = np.setdiff1d(all_idx, test_idx)
    pairs = select_pairs(D, train_idx)
    pairs.to_csv(f"{OUT}/pairs.csv", index=False)
    print(pairs[["Y", "B", "B0", "B1", "train_V", "train_share_B1"]].round(3).to_string(index=False))

    preds = {}
    for k in range(3):
        pr = S.from_reward_file(D, f"final_trained_results/responses_reward_0to500_{k}.json", 4)[0]
        for s, c in enumerate(S.COND4):
            preds[f"untrained {c} run{k}"] = pr[s]
    for lab, path in [("untrained persona+axioms iter1_1", "responses_reward_eval_full_Qwen_iter1_1.json"),
                      ("TRAINED persona+axioms best_model", "best_responses_reward_for_best_model.json"),
                      ("TRAINED persona+axioms curriculum run2",
                       "final_trained_results/responses_reward_eval_full_Qwen_iter4_curriculum (1).json")]:
        preds[lab] = S.from_eval_file(D, path)[0]
    party = np.array([str(p.get("POLPARTY")) for p in D["pers"]])

    rows, boots, perrows, shuf = [], {}, [], []
    for sample, idx in (("test100", test_idx), ("all500", all_idx)):
        n = len(idx)
        W = [np.ones(n, int)] + [np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(1000)]
        for name, pr in preds.items():
            if sample == "all500" and ("TRAINED" in name or "iter1_1" in name):
                continue
            res = [metrics(D, pairs, pr, idx, w, party[idx]) for w in W]
            pt = res[0][0]
            for mname in pt:
                bs = np.array([r[0][mname] for r in res[1:]])
                boots[(sample, name, mname)] = np.concatenate([[pt[mname]], bs])
                rows.append(dict(sample=sample, condition=name, metric=mname, est=pt[mname],
                                 lo=2 * pt[mname] - np.nanpercentile(bs, 97.5), hi=2 * pt[mname] - np.nanpercentile(bs, 2.5)))
            for r_ in res[0][1]:
                perrows.append(dict(sample=sample, condition=name, **r_))
            if sample == "test100" or name.endswith("run0"):
                sh = []
                for _ in range(200):
                    pp = pr.copy()
                    for g in np.unique(party[idx]):
                        members = idx[party[idx] == g]
                        for j in range(pp.shape[1]):
                            pp[members, j] = pr[rng.permutation(members), j]
                    sh.append(metrics(D, pairs, pp, idx, np.ones(n, int), party[idx])[0])
                for mname in ("cond_tv", "assoc_err"):
                    v = np.array([s_[mname] for s_ in sh])
                    shuf.append(dict(sample=sample, condition=name, metric=mname, observed=pt[mname],
                                     shuffled_mean=v.mean(), shuffled_lo=np.percentile(v, 2.5),
                                     shuffled_hi=np.percentile(v, 97.5), p_shuffled_le_observed=np.mean(v <= pt[mname])))
    pd.DataFrame(rows).to_csv(f"{OUT}/metrics.csv", index=False)
    pd.DataFrame(perrows).to_csv(f"{OUT}/per_pair.csv", index=False)
    pd.DataFrame(shuf).to_csv(f"{OUT}/shuffle.csv", index=False)

    con = []
    def contrast(sample, a, b):
        for m in ("cond_tv", "assoc_err", "marg_tv", "reweight"):
            if (sample, a, m) not in boots or (sample, b, m) not in boots:
                continue
            d = boots[(sample, a, m)] - boots[(sample, b, m)]
            db = d[1:][np.isfinite(d[1:])]
            piv = 2 * d[0] - db
            con.append(dict(sample=sample, contrast=f"{a} - {b}", metric=m, diff=d[0],
                            lo=np.percentile(piv, 2.5), hi=np.percentile(piv, 97.5),
                            p=min(1, 2 * min((piv <= 0).mean(), (piv >= 0).mean()))))
    for sample in ("test100", "all500"):
        for k in range(3):
            contrast(sample, f"untrained persona+axioms run{k}", f"untrained persona run{k}")
            contrast(sample, f"untrained axioms run{k}", f"untrained persona run{k}")
        contrast(sample, "untrained persona run1", "untrained persona run0")
        contrast(sample, "untrained persona+axioms run1", "untrained persona+axioms run0")
    for tr in ("TRAINED persona+axioms best_model", "TRAINED persona+axioms curriculum run2"):
        contrast("test100", tr, "untrained persona+axioms iter1_1")
        for k in range(3):
            contrast("test100", tr, f"untrained persona+axioms run{k}")
    pd.DataFrame(con).to_csv(f"{OUT}/contrasts.csv", index=False)
    print("done")


if __name__ == "__main__":
    main()
