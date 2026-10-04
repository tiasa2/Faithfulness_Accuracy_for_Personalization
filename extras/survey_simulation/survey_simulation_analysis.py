"""Survey-simulation analysis: do individual axioms (and curriculum DPO) preserve the
real OpinionQA response distribution better than demographic personas?

Framing: Bisbee et al. 2024 (Political Analysis) - LLM synthetic respondents match
means but compress variance and distort regression coefficients by collapsing
individuals to their group's modal answer.

Inputs (all local):
  final_created.csv                      500 real respondents (persona + 15 answers)
  final_created_train.csv / _test.csv    400 / 100 split (test = DPO held-out)
  final_trained_results/responses_reward_0to500_{0,1,2}.json
        Qwen3-8B UNTRAINED, 4 conditions x 500 respondents, 3 sampling runs, judged F
  responses_reward_0to500_32B_labeled.json
        32B thinking model, 5 conditions (incl. relevant-axioms) x 500, judged F
  final_created_llama8B_0to500_32B_instruct.json, final_combined_llama70B.json,
  final_combined_distilled_llama70B.json
        secondary models, 4-5 conditions x 500, answers only (model IDs unverified)
  responses_reward_eval_full_Qwen_iter1_1.json   untrained Qwen3-8B, P+A, test 100
  best_responses_reward_for_best_model.json      curriculum-trained, P+A, test 100
  final_trained_results/responses_reward_eval_full_Qwen_iter4_curriculum (1).json
        second trained run (robustness)

Analyses
  1  distribution match (question and group x question), dispersion, modal collapse,
     with a split-half noise ceiling and two reference baselines
  2  Bisbee-style regression replication (ordinal questions)
  3  is F lower when the model predicts the group mode on deviating items?
  4  untrained vs trained on the 100 test respondents (paired bootstrap)

Run:  python survey_simulation_analysis.py [--boot 500] [--splits 100]
Outputs go to survey_sim/ (CSVs, figures, summary.txt).
"""
import argparse
import ast
import json
import os
import re
from collections import Counter

import numpy as np
import pandas as pd

OUT = "extras/survey_simulation/results/survey_sim"
ORDINAL = ["PP5a_W49", "PP6c_W49", "FTRWORRYd_W41", "LEGALIMG_W41", "SOURCESKEPc_W45",
           "POP2_W42", "RACESURV7_W43", "RACATTN_W43"]
# ordinal order differs from option order only for RACATTN
REORDER = {"RACATTN_W43": ["Too little attention", "About the right amount of attention", "Too much attention"]}
ATTRS = ["POLPARTY", "POLIDEOLOGY", "PARTYxIDEO", "SEX", "AGE", "RACE", "RELIG", "RELIGATTEND",
         "EDUCATION", "INCOME", "CREGION", "MARITAL"]
COND4 = ["baseline", "persona", "axioms", "persona+axioms"]
MIN_GROUP = 20


def P(s):
    if not isinstance(s, str):
        return s
    try:
        return ast.literal_eval(s)
    except Exception:
        return s


# ----------------------------------------------------------------------------- data
def load_real():
    f = pd.read_csv("data/opinionqa/final_created.csv")
    qs = [c for c in f.columns if c not in ("Unnamed: 0", "Persona")]
    letters, order = {}, {}
    for q in qs:
        om = [o for o in P(P(f[q].iloc[0])["option_mapping"]) if o != "Refused"]
        letters[q] = om                      # prompt letter A.. -> option text
        order[q] = REORDER.get(q, om)        # ordinal order
    pers = [P(s) for s in f.Persona]
    for p in pers:
        p["PARTYxIDEO"] = (f"{p['POLPARTY']}|{p['POLIDEOLOGY']}"
                           if p.get("POLPARTY") and p.get("POLIDEOLOGY") else None)
    real = np.full((len(f), len(qs)), -1)
    for j, q in enumerate(qs):
        for i, s in enumerate(f[q]):
            a = P(s)["response"]
            real[i, j] = order[q].index(a) if a in order[q] else -1
    key = lambda r: (r["Persona"],) + tuple(P(r[q])["response"] for q in qs)
    fk = {key(r): i for i, r in f.iterrows()}
    te = pd.read_csv("data/opinionqa/final_created_test.csv")
    test_idx = np.array([fk[key(r)] for _, r in te.iterrows()])
    return dict(qs=qs, letters=letters, order=order, pers=pers, real=real, test_idx=test_idx,
                K=np.array([len(order[q]) for q in qs]))


def letter_to_cat(D, j, letter):
    q = D["qs"][j]
    if letter is None:
        return -1
    k = ord(letter) - 65
    if not 0 <= k < len(D["letters"][q]):
        return -1
    return D["order"][q].index(D["letters"][q][k])


SEL = re.compile(r"Selected Answer[\"'*:\s]*\**\s*[\"']?\s*\(?([A-E])\b")


def parse_letter(text):
    text = text if isinstance(text, str) else str(text)
    post = text.split("</think>")[-1]
    m = SEL.search(post)
    return m.group(1) if m else None


def from_reward_file(D, path, n_setups):
    """4/5-setup reward file -> pred[setup] (N x Q), F/G/CR[setup] (N x Q)."""
    d = json.load(open(path))
    N, Q = len(D["pers"]), len(D["qs"])
    pred = np.full((n_setups, N, Q), -1)
    F = np.full((n_setups, N, Q), np.nan); G = F.copy(); CR = F.copy(); C = F.copy()
    for t, x in enumerate(d):
        if "source_persona_position" in x:
            i, j = int(P(x["source_persona_position"])), D["qs"].index(P(x["question_key"]))
        else:
            i, j = divmod(t, Q)
        resp, lab, na = P(x["responses"]), P(x["contradiction_prompts"]), P(x["num_axioms"])
        g, r, c = P(x["grounding_prompts"]), P(x["rewards"]), P(x["correctness"])
        pos = 0
        for s in range(n_setups):
            pred[s, i, j] = letter_to_cat(D, j, parse_letter(resp[s]))
            ch = lab[pos:pos + na[s]]; pos += na[s]
            CR[s, i, j] = sum(l.strip() == "C" for l in ch) / na[s]
            G[s, i, j], F[s, i, j], C[s, i, j] = float(g[s]), float(r[s]), float(c[s])
    return pred, F, G, CR, C


def from_generation_file(D, path, setups):
    d = json.load(open(path))
    N, Q = len(D["pers"]), len(D["qs"])
    pred = np.full((len(setups), N, Q), -1)
    assert len(d) == N
    for i, x in enumerate(d):
        for j, q in enumerate(D["qs"]):
            for s, name in enumerate(setups):
                pred[s, i, j] = letter_to_cat(D, j, parse_letter(x.get(f"{name}_{q}", "")))
    return pred


def from_eval_file(D, path):
    """single-setup eval file on the 100 test respondents -> (N x Q) arrays, NaN/-1 elsewhere."""
    d = json.load(open(path))
    N, Q = len(D["pers"]), len(D["qs"])
    pred = np.full((N, Q), -1)
    F = np.full((N, Q), np.nan); G = F.copy(); CR = F.copy(); C = F.copy()
    for t, x in enumerate(d):
        u, j = divmod(t, Q)
        i = D["test_idx"][u]
        resp = P(x["responses"])[0]
        pred[i, j] = letter_to_cat(D, j, parse_letter(resp))
        F[i, j] = float(P(x["rewards"])[0]); G[i, j] = float(P(x["grounding_scores"])[0])
        CR[i, j] = float(P(x["contradiction_rate"])[0]); C[i, j] = float(P(x["correctness"])[0])
    return pred, F, G, CR, C


# ----------------------------------------------------------------------------- baselines
def onehot_answers(D, idx):
    return D["real"][idx]


def baseline_group_modal(D, train_idx, target_idx, attr="PARTYxIDEO", min_pool=5):
    """Group modal answer from train_idx, leaving the target out; fallback party -> population.
    Ties broken toward the lower category index."""
    real, pers = D["real"], D["pers"]
    Km = D["K"].max()
    out = np.full(real.shape, -1)
    tr_mask = np.zeros(len(pers), bool); tr_mask[train_idx] = True
    levels = [attr, "POLPARTY", None]
    lab = {a: np.array([str(p.get(a)) if (a and p.get(a) is not None) else ("ALL" if a is None else "NA")
                        for p in pers]) for a in levels}
    for j in range(real.shape[1]):
        tables = {}
        for a in levels:
            t = {}
            for i in np.where(tr_mask & (real[:, j] >= 0))[0]:
                t.setdefault(lab[a][i], np.zeros(Km))[real[i, j]] += 1
            tables[a] = t
        for i in target_idx:
            for a in levels:
                g = lab[a][i]
                if g == "NA" or g not in tables[a]:
                    continue
                c = tables[a][g].copy()
                if tr_mask[i] and real[i, j] >= 0:
                    c[real[i, j]] -= 1
                if c.sum() >= min_pool:
                    out[i, j] = int(c.argmax())
                    break
    return out


def baseline_lr(D, train_idx, target_idx, rng, folds=None, sample=False):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import OneHotEncoder
    real = D["real"]
    out = np.full(real.shape, -1)
    splits = [(train_idx, target_idx)] if folds is None else folds
    for tr, ta in splits:
        for j in range(real.shape[1]):
            other = [k for k in range(real.shape[1]) if k != j]
            trv = tr[real[tr, j] >= 0]
            enc = OneHotEncoder(handle_unknown="ignore").fit(real[trv][:, other])
            m = LogisticRegression(max_iter=3000).fit(enc.transform(real[trv][:, other]), real[trv, j])
            pr = m.predict_proba(enc.transform(real[ta][:, other]))
            if sample:
                cum = pr.cumsum(1)
                out[ta, j] = m.classes_[(rng.random((len(ta), 1)) > cum).sum(1).clip(max=len(m.classes_) - 1)]
            else:
                out[ta, j] = m.classes_[pr.argmax(1)]
    return out


# ----------------------------------------------------------------------------- metrics
def onehot(cat, Km):
    """(n x Q) categories, -1 missing -> (n x Q x Km) indicator array."""
    O = np.zeros(cat.shape + (Km,))
    ok = cat >= 0
    ii, jj = np.where(ok)
    O[ii, jj, cat[ok]] = 1
    return O


METRICS = ["wd", "js", "var_ratio", "collapsed_cell", "ent_pred", "ent_real", "modal_share_pred",
           "modal_share_real", "modal_excess"]


def dist_metrics(pc, rc, K):
    """pc, rc: (..., Q, Km) counts. Returns dict of (..., Q) arrays (NaN where undefined)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        npred, nreal = pc.sum(-1), rc.sum(-1)
        p = pc / npred[..., None]; r = rc / nreal[..., None]
        emd = np.abs(np.cumsum(p - r, -1)).sum(-1)
        wd = 1 - emd / (K - 1)
        m = (p + r) / 2
        def kl(a, b):
            return np.where(a > 0, a * np.log2(np.where(a > 0, a, 1) / np.where(b > 0, b, 1)), 0).sum(-1)
        js = 0.5 * kl(p, m) + 0.5 * kl(r, m)
        x = np.arange(p.shape[-1])
        def var(d):
            return (d * x ** 2).sum(-1) - (d * x).sum(-1) ** 2
        vp, vr_ = var(p), var(r)
        vr = np.where(vr_ > 1e-12, vp / vr_, np.nan)
        collapsed = np.where(np.isfinite(vp), (vp < 1e-12).astype(float), np.nan)
        def ent(d):
            return -np.where(d > 0, d * np.log(np.where(d > 0, d, 1)), 0).sum(-1) / np.log(K)
        rmode = np.nan_to_num(r).argmax(-1)
        pm = np.take_along_axis(p, rmode[..., None], -1)[..., 0]
        rm = np.take_along_axis(r, rmode[..., None], -1)[..., 0]
        bad = (npred == 0) | (nreal == 0)
    out = dict(wd=wd, js=js, var_ratio=vr, collapsed_cell=collapsed, ent_pred=ent(p), ent_real=ent(r),
               modal_share_pred=pm, modal_share_real=rm, modal_excess=pm - rm)
    return {k: np.where(bad, np.nan, v) for k, v in out.items()}


def group_ids(D, attr, idx, min_n):
    vals = [D["pers"][i].get(attr) for i in idx]
    c = Counter(v for v in vals if v is not None)
    keep = sorted(g for g, n in c.items() if n >= min_n)
    return keep, np.array([[v == g for g in keep] for v in vals], dtype=float).reshape(len(idx), len(keep))


# ----------------------------------------------------------------------------- analysis 1
def analysis1(D, preds, idx, B, S, rng, min_n, tag, keep_cell_boot=()):
    """Question- and group x question-level distribution metrics.
    Returns cells (point estimates), summ {(level, scope, condition, metric): (B+1,) mean over cells},
    cellboot {(condition, metric): (attr, group, q) -> (B+1,)}, split-half ceiling rows."""
    K = D["K"]; Km = K.max(); n = len(idx)
    W = np.vstack([np.ones(n), rng.multinomial(n, np.ones(n) / n, size=B)]).astype(float)
    Or = onehot(D["real"][idx], Km)
    Op = {k: onehot(v[idx], Km) for k, v in preds.items()}
    ordm = np.array([q in ORDINAL for q in D["qs"]])
    cells, summ, cellboot = [], {}, {}
    def nanmean_cells(v, mask=None):  # v: (B+1, ..., Q) -> (B+1,)
        v = v if mask is None else v[..., mask]
        return np.nanmean(v.reshape(v.shape[0], -1), 1)
    rq = np.einsum("bn,nqk->bqk", W, Or)
    for name, O in Op.items():
        m = dist_metrics(np.einsum("bn,nqk->bqk", W, O), rq, K)
        for key in METRICS:
            v = m[key]
            summ[("question", "all 15", name, key)] = nanmean_cells(v)
            summ[("question", "ordinal 8", name, key)] = nanmean_cells(v, ordm)
            for j, q in enumerate(D["qs"]):
                cells.append(dict(sample=tag, level="question", attr="-", group="-", question=q,
                                  ordinal=bool(ordm[j]), condition=name, metric=key, est=v[0, j]))
    pooled = {}
    for attr in ATTRS:
        keep, M = group_ids(D, attr, idx, min_n)
        if not keep:
            continue
        rg = np.einsum("bn,ng,nqk->gbqk", W, M, Or)
        for name, O in Op.items():
            m = dist_metrics(np.einsum("bn,ng,nqk->gbqk", W, M, O), rg, K)
            for key in METRICS:
                v = np.moveaxis(m[key], 0, 1)                      # (B+1, G, Q)
                summ[("group", attr, name, key)] = nanmean_cells(v)
                pooled.setdefault((name, key), []).append(v.reshape(v.shape[0], -1))
                for gi, g in enumerate(keep):
                    for j, q in enumerate(D["qs"]):
                        cells.append(dict(sample=tag, level="group", attr=attr, group=g, question=q,
                                          ordinal=bool(ordm[j]), condition=name, metric=key, est=v[0, gi, j],
                                          n_group=int(M[:, gi].sum())))
                        if (name, key) in keep_cell_boot:
                            cellboot.setdefault((name, key), {})[(attr, g, q)] = v[:, gi, j]
    for (name, key), vs in pooled.items():
        summ[("group", "ALL attrs pooled", name, key)] = np.nanmean(np.concatenate(vs, 1), 1)
    ceil = []
    for s in range(S):
        perm = rng.permutation(n)
        wa = np.zeros(n); wa[perm[: n // 2]] = 1
        wb = 1 - wa
        rB = np.einsum("n,nqk->qk", wb, Or)
        for name, O in [("REAL split-half", Or)] + list(Op.items()):
            mq = dist_metrics(np.einsum("n,nqk->qk", wa, O), rB, K)
            for key in ("wd", "js", "var_ratio", "modal_excess"):
                ceil.append(dict(sample=tag, level="question", scope="all 15", split=s, condition=name,
                                 metric=key, est=np.nanmean(mq[key])))
                ceil.append(dict(sample=tag, level="question", scope="ordinal 8", split=s, condition=name,
                                 metric=key, est=np.nanmean(mq[key][ordm])))
        for attr in ATTRS:
            keep, M = group_ids(D, attr, idx, min_n)
            if not keep:
                continue
            rBg = np.einsum("n,ng,nqk->gqk", wb, M, Or)
            for name, O in [("REAL split-half", Or)] + list(Op.items()):
                mg = dist_metrics(np.einsum("n,ng,nqk->gqk", wa, M, O), rBg, K)
                for key in ("wd", "js", "var_ratio", "modal_excess"):
                    ceil.append(dict(sample=tag, level="group", scope=attr, split=s, condition=name,
                                     metric=key, est=np.nanmean(mg[key])))
    return pd.DataFrame(cells), summ, cellboot, pd.DataFrame(ceil)


def summ_table(summ, tag):
    rows = []
    for (level, scope, cond, metric), v in summ.items():
        lo_, hi_ = np.nanpercentile(v[1:], [2.5, 97.5])
        rows.append(dict(sample=tag, level=level, scope=scope, condition=cond, metric=metric, est=v[0],
                         lo=2 * v[0] - hi_, hi=2 * v[0] - lo_))
    return pd.DataFrame(rows)


def contrast(summ, tag, a, b, extra):
    rows = []
    for (level, scope, cond, metric), v in summ.items():
        if cond != a or (level, scope, b, metric) not in summ:
            continue
        d = v - summ[(level, scope, b, metric)]
        db = d[1:][np.isfinite(d[1:])]
        if not len(db):
            continue
        piv = 2 * d[0] - db                      # basic bootstrap: reflect around the estimate
        rows.append(dict(sample=tag, level=level, scope=scope, metric=metric, contrast=f"{a} - {b}", diff=d[0],
                         lo=np.percentile(piv, 2.5), hi=np.percentile(piv, 97.5),
                         p=min(1, 2 * min((piv <= 0).mean(), (piv >= 0).mean())), **extra))
    return rows


def bh(p):
    p = np.asarray(p, float); n = len(p)
    if n == 0:
        return p
    o = np.argsort(p); q = np.empty(n)
    q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1)


# ----------------------------------------------------------------------------- analysis 2
def design(D, idx, min_cat):
    cols = []
    for attr in ["POLPARTY", "POLIDEOLOGY", "SEX", "AGE", "RACE", "RELIG", "RELIGATTEND", "EDUCATION",
                 "INCOME", "CREGION", "MARITAL"]:
        v = pd.Series([D["pers"][i].get(attr) or "NA" for i in idx])
        c = v.value_counts()
        v = v.where(v.map(c) >= min_cat, "Other/NA")
        ref = v.value_counts().index[0]
        for lvl in sorted(set(v) - {ref}):
            if (v == lvl).sum() >= min_cat:
                cols.append((f"{attr}={lvl}", (v == lvl).values.astype(float)))
    names = [c[0] for c in cols]
    X = np.column_stack([np.ones(len(idx))] + [c[1] for c in cols])
    return names, X


def ols(X, y, w=None):
    ok = y >= 0
    if w is not None:
        ok = ok & (w > 0)
    Xo, yo = X[ok], y[ok].astype(float)
    ww = np.ones(ok.sum()) if w is None else w[ok]
    sw = np.sqrt(ww)
    beta, *_ = np.linalg.lstsq(Xo * sw[:, None], yo * sw, rcond=None)
    return beta


def regression_ceiling(D, preds, idx, S, rng, min_cat, tag):
    """Split-half: coefficients from REAL half A vs REAL half B (noise ceiling), and
    synthetic half A vs REAL half B (like-for-like model comparison)."""
    names, X = design(D, idx, min_cat)
    qj = [D["qs"].index(q) for q in ORDINAL]
    n = len(idx); real = D["real"][idx]; rows = []
    def fit(cat, w):
        return np.stack([ols(X, cat[:, j], w)[1:] for j in qj]).ravel()
    for s in range(S):
        perm = rng.permutation(n); wa = np.zeros(n); wa[perm[: n // 2]] = 1; wb = 1 - wa
        rb = fit(real, wb)
        ra = fit(real, wa)
        r_real = np.corrcoef(ra, rb)[0, 1]
        for name, cat in [("REAL split-half", real)] + [(k, v[idx]) for k, v in preds.items()]:
            sa = fit(cat, wa); sb = fit(cat, wb)
            rows.append(dict(sample=tag, split=s, condition=name, corr=np.corrcoef(rb, sa)[0, 1],
                             slope=np.polyfit(rb, sa, 1)[0], sign_flip=np.mean(np.sign(rb) != np.sign(sa)),
                             real_split_r=r_real, self_split_r=np.corrcoef(sa, sb)[0, 1]))
    return pd.DataFrame(rows)


def analysis2(D, preds, idx, B, rng, min_cat, tag):
    names, X = design(D, idx, min_cat)
    n = len(idx)
    W = np.vstack([np.ones(n), rng.multinomial(n, np.ones(n) / n, size=B)]).astype(float)
    qj = [D["qs"].index(q) for q in ORDINAL]
    real = D["real"][idx]
    coef = {}
    def fit_all(cat):
        return np.stack([np.stack([ols(X, cat[:, j], w)[1:] for j in qj]) for w in W])  # (B+1, nQ, p)
    coef["REAL"] = fit_all(real)
    for name, pr in preds.items():
        coef[name] = fit_all(pr[idx])
    rows, crow = [], []
    R = coef["REAL"]
    # real-coefficient significance from bootstrap
    r_sig = (np.percentile(R[1:], 2.5, 0) > 0) | (np.percentile(R[1:], 97.5, 0) < 0)
    for name in preds:
        S = coef[name]
        def stats(r, s):
            r, s = r.ravel(), s.ravel()
            corr = np.corrcoef(r, s)[0, 1]
            slope = np.polyfit(r, s, 1)[0]
            flip = np.mean(np.sign(r) != np.sign(s))
            return corr, slope, flip
        pts = stats(R[0], S[0])
        bs = np.array([stats(R[b], S[b]) for b in range(1, len(W))])
        flip_sig = np.mean(np.sign(R[0][r_sig]) != np.sign(S[0][r_sig])) if r_sig.any() else np.nan
        D_ = S - R
        lo, hi = np.percentile(D_[1:], 2.5, 0), np.percentile(D_[1:], 97.5, 0)
        p = np.minimum(1, 2 * np.minimum((D_[1:] <= 0).mean(0), (D_[1:] >= 0).mean(0)))
        q = bh(p.ravel()).reshape(p.shape)
        sd_ratio = []
        for jj, j in enumerate(qj):
            ok = (real[:, j] >= 0) & (preds[name][idx][:, j] >= 0)
            sd_ratio.append(preds[name][idx][ok, j].std() / real[ok, j].std())
        rows.append(dict(sample=tag, condition=name, n_coef=R[0].size, corr=pts[0], slope=pts[1],
                         share_sig_diff=np.mean((lo > 0) | (hi < 0)), share_sig_diff_fdr=np.mean(q < 0.05),
                         sign_flip_all=pts[2], sign_flip_among_real_sig=flip_sig, n_real_sig=int(r_sig.sum()),
                         sd_ratio_mean=float(np.mean(sd_ratio))))
        for jj, qq in enumerate(ORDINAL):
            for k, nm in enumerate(names):
                crow.append(dict(sample=tag, condition=name, question=qq, coef=nm, real=R[0][jj, k],
                                 synthetic=S[0][jj, k], real_sig=bool(r_sig[jj, k]), diff_p=p[jj, k], diff_q=q[jj, k]))
    return pd.DataFrame(rows), pd.DataFrame(crow), coef


# ----------------------------------------------------------------------------- analysis 3
def analysis3(D, sets, idx, modal_from, B, rng, tag, attrs=("PARTYxIDEO", "POLPARTY", "POLIDEOLOGY")):
    """sets: {name: (pred, F, G, CR)}. Items whose TRUE answer != group modal (modal from `modal_from`,
    leave-one-out). Outcome ~ predicts_mode + question FE; respondent-cluster bootstrap with the SAME
    respondent draws for every condition (so differences between conditions are paired).
    Returns (rows, boots {(attr, cond, variant, outcome): (B,)})."""
    real, Q = D["real"], len(D["qs"])
    draws = [rng.choice(idx, len(idx)) for _ in range(B)]
    users_all = np.repeat(idx, Q); qcol = np.tile(np.arange(Q), len(idx))
    Qd = np.eye(Q)[qcol]
    rows, boots = [], {}
    for attr in attrs:
        gm = baseline_group_modal(D, modal_from, idx, attr)
        t = real[users_all, qcol]; m_ = gm[users_all, qcol]
        for name, (pr, F, G, CR) in sets.items():
            p_ = pr[users_all, qcol]; f = F[users_all, qcol]
            ys = {"F": f, "F==1": (f == 1).astype(float), "grounding": G[users_all, qcol],
                  "contradiction_rate": CR[users_all, qcol]}
            base_ok = (t >= 0) & (m_ >= 0) & (p_ >= 0) & np.isfinite(f) & (t != m_)
            x = (p_ == m_).astype(float)
            for variant, extra in (("deviating items", np.ones_like(base_ok)), ("deviating & model wrong", p_ != t)):
                ok = base_ok & extra
                rows_ok = np.where(ok)[0]
                by_user = {}
                for r_ in rows_ok:
                    by_user.setdefault(users_all[r_], []).append(r_)
                samp = [np.concatenate([by_user[u] for u in dr if u in by_user]) for dr in draws]
                Xo = np.column_stack([x, Qd])
                for yn, y in ys.items():
                    est = np.linalg.lstsq(Xo[rows_ok], y[rows_ok], rcond=None)[0][0]
                    bs = np.array([np.linalg.lstsq(Xo[sm], y[sm], rcond=None)[0][0] for sm in samp])
                    boots[(attr, name, variant, yn)] = bs
                    xo, yo = x[rows_ok], y[rows_ok]
                    ordq = np.isin(qcol[rows_ok], [D["qs"].index(q) for q in ORDINAL])
                    chance = np.mean(1 / (D["K"][qcol[rows_ok]][ordq] - (1 if variant != "deviating items" else 0)))
                    rows.append(dict(sample=tag, attr=attr, condition=name, variant=variant, outcome=yn,
                                     n_items=int(ok.sum()), share_predicting_mode=xo.mean(),
                                     share_mode_ordinal=xo[ordq].mean(), chance_mode_ordinal=chance,
                                     mean_if_mode=yo[xo == 1].mean(), mean_if_not=yo[xo == 0].mean(),
                                     coef=est, lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5),
                                     p=min(1, 2 * min((bs <= 0).mean(), (bs >= 0).mean()))))
    out = pd.DataFrame(rows)
    out["q_fdr"] = np.nan
    for key, g in out.groupby(["sample", "variant", "outcome"]):
        out.loc[g.index, "q_fdr"] = bh(g.p.values)
    return out, boots


def collapse_among_errors(D, preds, idx, modal_from, tag, attr="PARTYxIDEO"):
    """Ordinal questions, respondents whose TRUE answer != group mode, model wrong: share of the wrong
    answers that equal the group mode vs chance (uniform over the K-1 wrong options) and vs the
    population mode. Describes collapse for every condition (no faithfulness needed)."""
    gm = baseline_group_modal(D, modal_from, idx, attr)
    pm = baseline_group_modal(D, modal_from, idx, "__none__")      # falls back to population mode
    ordj = [D["qs"].index(q) for q in ORDINAL]
    rows = []
    for name, pr in preds.items():
        t, m, p, pop = (D["real"][idx][:, ordj], gm[idx][:, ordj], pr[idx][:, ordj], pm[idx][:, ordj])
        K = np.broadcast_to(D["K"][ordj], t.shape)
        ok = (t >= 0) & (m >= 0) & (p >= 0) & (t != m) & (p != t)
        same = ok & (m == pop)
        rows.append(dict(sample=tag, condition=name, n_wrong=int(ok.sum()), share_group_mode=(p == m)[ok].mean(),
                         chance=(1 / (K - 1))[ok].mean(), share_group_mode_when_group_mode_differs_from_pop=
                         (p == m)[ok & (m != pop)].mean() if (ok & (m != pop)).any() else np.nan,
                         share_pop_mode_when_differs=(p == pop)[ok & (m != pop)].mean() if (ok & (m != pop)).any() else np.nan,
                         n_group_ne_pop=int((ok & (m != pop)).sum())))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--splits", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    os.makedirs(OUT, exist_ok=True); os.makedirs(f"{OUT}/figures", exist_ok=True)
    D = load_real()
    N = len(D["pers"]); all_idx = np.arange(N); test_idx = D["test_idx"]
    train_idx = np.setdiff1d(all_idx, test_idx)

    # ---------------- load predictions
    inv, P500, F500 = [], {}, {}
    for k in range(3):
        pr, F, G, CR, C = from_reward_file(D, f"final_trained_results/responses_reward_0to500_{k}.json", 4)
        for s, c in enumerate(COND4):
            P500[f"Qwen3-8B|{c}|run{k}"] = pr[s]
            F500[f"Qwen3-8B|{c}|run{k}"] = (pr[s], F[s], G[s], CR[s])
            agree = np.mean(((pr[s] == D["real"]) == (C[s] == 1))[(pr[s] >= 0) & (D["real"] >= 0)])
            inv.append(dict(model="Qwen3-8B (untrained)", condition=c, run=k, file=f"responses_reward_0to500_{k}.json",
                            respondents=N, parse_rate=(pr[s] >= 0).mean(), correctness_agreement=agree, has_F=True))
    pr, F, G, CR, C = from_reward_file(D, "responses_reward_0to500_32B_labeled.json", 5)
    for s, c in enumerate(COND4 + ["relevant-axioms"]):
        P500[f"32B-thinking|{c}|run0"] = pr[s]
        F500[f"32B-thinking|{c}|run0"] = (pr[s], F[s], G[s], CR[s])
        agree = np.mean(((pr[s] == D["real"]) == (C[s] == 1))[(pr[s] >= 0) & (D["real"] >= 0)])
        inv.append(dict(model="32B thinking (Qwen3-32B?)", condition=c, run=0, file="responses_reward_0to500_32B_labeled.json",
                        respondents=N, parse_rate=(pr[s] >= 0).mean(), correctness_agreement=agree, has_F=True))
    for label, path, setups in [
            ("32B-instruct", "final_created_llama8B_0to500_32B_instruct.json",
             ["Base", "Persona", "Axioms", "Persona_and_Axioms", "Relevant_Axioms"]),
            ("Llama-70B", "final_combined_llama70B.json", ["Base", "Persona", "Axioms", "Persona_and_Axioms"]),
            ("Distill-Llama-70B", "final_combined_distilled_llama70B.json",
             ["Base", "Persona", "Axioms", "Persona_and_Axioms"])]:
        pr = from_generation_file(D, path, setups)
        for s, c in enumerate((COND4 + ["relevant-axioms"])[: len(setups)]):
            P500[f"{label}|{c}|run0"] = pr[s]
            inv.append(dict(model=f"{label} (ID unverified)", condition=c, run=0, file=path, respondents=N,
                            parse_rate=(pr[s] >= 0).mean(), correctness_agreement=np.nan, has_F=False))
    T = {}
    for label, path in [("untrained (iter1_1)", "responses_reward_eval_full_Qwen_iter1_1.json"),
                        ("trained (best_model)", "best_responses_reward_for_best_model.json"),
                        ("trained (curriculum run 2)", "final_trained_results/responses_reward_eval_full_Qwen_iter4_curriculum (1).json")]:
        pr, F, G, CR, C = from_eval_file(D, path)
        T[label] = (pr, F, G, CR)
        ok = test_idx
        agree = np.mean(((pr[ok] == D["real"][ok]) == (C[ok] == 1))[(pr[ok] >= 0) & (D["real"][ok] >= 0)])
        inv.append(dict(model=f"Qwen3-8B {label}", condition="persona+axioms", run=0, file=path,
                        respondents=len(test_idx), parse_rate=(pr[ok] >= 0).mean(), correctness_agreement=agree, has_F=True))
    pd.DataFrame(inv).to_csv(f"{OUT}/inventory.csv", index=False)
    print(pd.DataFrame(inv).round(4).to_string(index=False))

    folds = []
    perm = rng.permutation(N)
    for f_ in range(5):
        ta = np.sort(perm[f_::5]); folds.append((np.setdiff1d(all_idx, ta), ta))
    base500 = {"BASELINE group modal": baseline_group_modal(D, all_idx, all_idx),
               "BASELINE LR argmax": baseline_lr(D, None, None, rng, folds=folds),
               "BASELINE LR sampled": baseline_lr(D, None, None, rng, folds=folds, sample=True)}
    baseT = {"BASELINE group modal": baseline_group_modal(D, train_idx, test_idx),
             "BASELINE LR argmax": baseline_lr(D, train_idx, test_idx, rng),
             "BASELINE LR sampled": baseline_lr(D, train_idx, test_idx, rng, sample=True)}

    preds500 = {**P500, **base500}
    kcb = [(f"Qwen3-8B|{c}|run0", m) for c in ("axioms", "persona", "persona+axioms") for m in ("wd", "var_ratio", "modal_excess")]
    cells, summ500, cb500, ceil500 = analysis1(D, preds500, all_idx, args.boot, args.splits, rng, MIN_GROUP,
                                               "500 untrained", keep_cell_boot=kcb)
    predsT = {k: v[0] for k, v in T.items()}
    for k in range(3):
        predsT[f"untrained 4-setup P+A run{k}"] = P500[f"Qwen3-8B|persona+axioms|run{k}"]
    predsT.update(baseT)
    cellsT, summT, _, ceilT = analysis1(D, predsT, test_idx, args.boot, args.splits, rng, MIN_GROUP, "test 100")
    pd.concat([cells, cellsT]).to_csv(f"{OUT}/a1_cells.csv", index=False)
    summ_df = pd.concat([summ_table(summ500, "500 untrained"), summ_table(summT, "test 100")])
    summ_df.to_csv(f"{OUT}/a1_summary.csv", index=False)
    C_all = pd.concat([ceil500, ceilT])
    C_all.to_csv(f"{OUT}/a1_splithalf_raw.csv", index=False)
    C_all.groupby(["sample", "level", "scope", "condition", "metric"]).est.agg(
        mean="mean", lo=lambda v: np.nanpercentile(v, 2.5), hi=lambda v: np.nanpercentile(v, 97.5)
    ).reset_index().to_csv(f"{OUT}/a1_splithalf_ceiling.csv", index=False)

    con = []
    for model, runs in [("Qwen3-8B", [0, 1, 2]), ("32B-thinking", [0]), ("32B-instruct", [0]),
                        ("Llama-70B", [0]), ("Distill-Llama-70B", [0])]:
        for r in runs:
            for a, b in [("axioms", "persona"), ("persona+axioms", "persona"), ("persona", "baseline"),
                         ("axioms", "baseline"), ("persona+axioms", "axioms")]:
                con += contrast(summ500, "500 untrained", f"{model}|{a}|run{r}", f"{model}|{b}|run{r}",
                                dict(model=model, run=r))
    for a, b in [("trained (best_model)", "untrained (iter1_1)"), ("trained (curriculum run 2)", "untrained (iter1_1)"),
                 ("untrained 4-setup P+A run0", "untrained (iter1_1)"), ("untrained 4-setup P+A run1", "untrained 4-setup P+A run0")]:
        con += contrast(summT, "test 100", a, b, dict(model="Qwen3-8B", run=0))
    con = pd.DataFrame(con)
    con["q_fdr"] = np.nan
    for key, g in con.groupby(["sample", "level", "metric"]):
        con.loc[g.index, "q_fdr"] = bh(g.p.values)
    con.to_csv(f"{OUT}/a1_contrasts.csv", index=False)

    cell = []
    for a in ("axioms", "persona+axioms"):
        for metric in ("wd", "var_ratio", "modal_excess"):
            A, Bc = cb500[(f"Qwen3-8B|{a}|run0", metric)], cb500[(f"Qwen3-8B|persona|run0", metric)]
            for k in A:
                d = A[k] - Bc[k]
                db = d[1:][np.isfinite(d[1:])]
                if not np.isfinite(d[0]) or len(db) < 0.9 * (len(d) - 1):
                    continue
                piv = 2 * d[0] - db
                cell.append(dict(contrast=f"{a} - persona", metric=metric, attr=k[0], group=k[1], question=k[2],
                                 diff=d[0], p=min(1, 2 * min((piv <= 0).mean(), (piv >= 0).mean()))))
    cell = pd.DataFrame(cell)
    cell["q_fdr"] = np.nan
    for key, g in cell.groupby(["contrast", "metric"]):
        cell.loc[g.index, "q_fdr"] = bh(g.p.values)
    cell.to_csv(f"{OUT}/a1_cell_tests_vs_persona.csv", index=False)

    # ---------------- Analysis 2
    a2, a2c, coef500 = analysis2(D, preds500, all_idx, args.boot, rng, 20, "500 untrained")
    a2t, a2tc, coefT = analysis2(D, predsT, test_idx, args.boot, rng, 8, "test 100")
    pd.concat([a2, a2t]).to_csv(f"{OUT}/a2_regression_summary.csv", index=False)
    rc = pd.concat([regression_ceiling(D, preds500, all_idx, args.splits, rng, 20, "500 untrained"),
                    regression_ceiling(D, predsT, test_idx, args.splits, rng, 8, "test 100")])
    rc.to_csv(f"{OUT}/a2_splithalf_raw.csv", index=False)
    cc_rows = []
    pairs = [(f"{m}|{a}|run0", f"{m}|{b}|run0") for m in ("Qwen3-8B", "32B-thinking", "32B-instruct", "Llama-70B",
                                                          "Distill-Llama-70B")
             for a, b in (("persona", "axioms"), ("persona+axioms", "persona"), ("persona+axioms", "axioms"))]
    pairs += [("trained (best_model)", "untrained (iter1_1)"), ("trained (curriculum run 2)", "untrained (iter1_1)"),
              ("untrained 4-setup P+A run1", "untrained 4-setup P+A run0")]
    for smp, g in rc.groupby("sample"):
        piv = g.pivot_table(index="split", columns="condition", values=["corr", "slope"])
        for a_, b_ in pairs:
            if ("corr", a_) not in piv.columns or ("corr", b_) not in piv.columns:
                continue
            for st in ("corr", "slope"):
                d = (piv[(st, a_)] - piv[(st, b_)]).values
                cc_rows.append(dict(sample=smp, contrast=f"{a_} - {b_}", stat=st, mean_diff=d.mean(),
                                    lo=np.percentile(d, 2.5), hi=np.percentile(d, 97.5), share_positive=(d > 0).mean()))
    pd.DataFrame(cc_rows).to_csv(f"{OUT}/a2_condition_contrasts.csv", index=False)
    rc.groupby(["sample", "condition"])[["corr", "slope", "sign_flip", "real_split_r", "self_split_r"]].agg(
        ["mean", lambda v: np.percentile(v, 2.5), lambda v: np.percentile(v, 97.5)]).to_csv(f"{OUT}/a2_splithalf_ceiling.csv")
    # noise-corrected replication: Spearman-Brown full-sample reliabilities from the split-half correlations;
    # slope corrected for errors-in-variables in the REAL coefficients, corr for noise in both.
    # Interval reflects reliability uncertainty across splits only (point slope/corr held fixed).
    sb = lambda r: 2 * np.clip(r, 1e-3, None) / (1 + np.clip(r, 1e-3, None))
    dis = []
    full = pd.concat([a2, a2t]).set_index(["sample", "condition"])
    for (smp, cond), g in rc[rc.condition != "REAL split-half"].groupby(["sample", "condition"]):
        if (smp, cond) not in full.index:
            continue
        r0 = full.loc[(smp, cond)]
        rel_r, rel_s = sb(g.real_split_r.values), sb(g.self_split_r.values)
        cs = r0["slope"] / rel_r; cc = r0["corr"] / np.sqrt(rel_r * rel_s)
        dis.append(dict(sample=smp, condition=cond, rel_real=rel_r.mean(), rel_synthetic=rel_s.mean(),
                        slope_obs=r0["slope"], slope_corrected=np.median(cs), slope_corr_lo=np.percentile(cs, 2.5),
                        slope_corr_hi=np.percentile(cs, 97.5), corr_obs=r0["corr"], corr_disattenuated=np.median(cc),
                        corr_dis_lo=np.percentile(cc, 2.5), corr_dis_hi=np.percentile(cc, 97.5)))
    pd.DataFrame(dis).to_csv(f"{OUT}/a2_noise_corrected.csv", index=False)
    pd.concat([a2c, a2tc]).to_csv(f"{OUT}/a2_coefficients.csv", index=False)
    # ---------------- Analysis 3
    sets3 = {k: v for k, v in F500.items() if k.split("|")[1] in ("axioms", "persona+axioms", "relevant-axioms")}
    a3, _ = analysis3(D, sets3, all_idx, all_idx, args.boot, rng, "500 untrained")
    setsT = {**T, **{f"untrained 4-setup P+A run{k}": F500[f"Qwen3-8B|persona+axioms|run{k}"] for k in range(3)}}
    a3t, bT = analysis3(D, setsT, test_idx, train_idx, args.boot, rng, "test 100")
    pd.concat([a3, a3t]).to_csv(f"{OUT}/a3_faithfulness_vs_collapse.csv", index=False)
    rows = []
    for (attr, cond, variant, yn), bs in bT.items():
        ref = "untrained (iter1_1)"
        if cond == ref or (attr, ref, variant, yn) not in bT:
            continue
        e = a3t[(a3t.attr == attr) & (a3t.condition == cond) & (a3t.variant == variant) & (a3t.outcome == yn)].coef.values[0]
        e0 = a3t[(a3t.attr == attr) & (a3t.condition == ref) & (a3t.variant == variant) & (a3t.outcome == yn)].coef.values[0]
        d = bs - bT[(attr, ref, variant, yn)]
        rows.append(dict(attr=attr, contrast=f"{cond} - {ref}", variant=variant, outcome=yn, diff=e - e0,
                         lo=np.percentile(d, 2.5), hi=np.percentile(d, 97.5),
                         p=min(1, 2 * min((d <= 0).mean(), (d >= 0).mean()))))
    pd.DataFrame(rows).to_csv(f"{OUT}/a3_training_contrast.csv", index=False)
    pd.concat([collapse_among_errors(D, preds500, all_idx, all_idx, "500 untrained"),
               collapse_among_errors(D, predsT, test_idx, train_idx, "test 100")]
              ).to_csv(f"{OUT}/a3_collapse_among_errors.csv", index=False)
    print("done; outputs in", OUT)


if __name__ == "__main__":
    main()
