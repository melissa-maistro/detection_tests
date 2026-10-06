"""Keyless SynthID detector v3.

Improvements over v2:
1. Leave-one-pair-out counts: each pair is scored with counts from ALL other pairs
   (never itself or its partner), so training data goes from 80% to ~100%.
2. Two aggregations per context length: mean evidence and total evidence (sum / sqrt(n)).
3. Stacked logistic regression over all features (h=1..6), trained on pair differences
   with K-fold CV, so the weights are learned and tested on held-out pairs.

Usage:
  python keyless_v3.py --data synthid-text/data/human_eval.jsonl
"""

import argparse
import math
from collections import Counter

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

from keyless_v2 import keys_of, load_pairs, metrics


def features_for_h(pairs, h, min_count):
    kw = [keys_of(w, h) for _, w, _ in pairs]
    ku = [keys_of(u, h) for _, _, u in pairs]
    cw, cu = Counter(), Counter()
    for s in kw:
        cw.update(s)
    for s in ku:
        cu.update(s)

    def feats(keys, own_w, own_u):
        # counts from all other pairs: remove this pair's wm text and unwm text
        zs = []
        for k in keys:
            a = cw[k] - (k in own_w)
            b = cu[k] - (k in own_u)
            if a + b >= min_count:
                zs.append((a - b) / math.sqrt(a + b))
        n = len(zs)
        if n == 0:
            return [0.0, 0.0, 0.0]
        s = sum(zs)
        return [s / n, s / math.sqrt(n), n / max(len(keys), 1)]

    fw = np.array([feats(kw[i], kw[i], ku[i]) for i in range(len(pairs))])
    fu = np.array([feats(ku[i], kw[i], ku[i]) for i in range(len(pairs))])
    return fw, fu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--tokenizer", default="google/gemma-7b-it")
    ap.add_argument("--h", default="1,2,3,4,5,6")
    ap.add_argument("--min-count", type=int, default=2)
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--C", type=float, default=0.1, help="LR regularization (smaller = stronger)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pairs = load_pairs(args.data, None, None, args.tokenizer)
    hs = [int(x) for x in args.h.split(",")]
    print("\nleave-one-pair-out | chance = 0.500\n")

    FW, FU, names = [], [], []
    for h in hs:
        fw, fu = features_for_h(pairs, h, args.min_count)
        for j, kind in enumerate(["mean", "total"]):
            print(f"h={h} {kind:<5}  " + "  ".join(
                f"{k}={v:.3f}" for k, v in zip(["AUC", "acc", "lo", "hi"],
                                                metrics(fw[:, j], fu[:, j]))), flush=True)
        FW.append(fw)
        FU.append(fu)
        names += [f"h{h}_mean", f"h{h}_total", f"h{h}_cov"]
    FW, FU = np.hstack(FW), np.hstack(FU)

    # standardize with stats from all texts (label-free), then learn weights on pair differences
    allf = np.vstack([FW, FU])
    mu, sd = allf.mean(0), allf.std(0) + 1e-12
    FW, FU = (FW - mu) / sd, (FU - mu) / sd

    sw, su = np.zeros(len(pairs)), np.zeros(len(pairs))
    coefs = []
    for tr, te in KFold(args.folds, shuffle=True, random_state=args.seed).split(FW):
        d = FW[tr] - FU[tr]
        X = np.vstack([d, -d])
        y = np.r_[np.ones(len(d)), np.zeros(len(d))]
        lr = LogisticRegression(C=args.C, fit_intercept=False, max_iter=2000).fit(X, y)
        w = lr.coef_[0]
        coefs.append(w)
        sw[te], su[te] = FW[te] @ w, FU[te] @ w

    auc, acc, lo, hi = metrics(sw, su)
    print(f"\nSTACKED (all h, {args.folds}-fold)  AUC={auc:.3f}  paired acc={acc:.3f} "
          f"[95% CI {lo:.3f}-{hi:.3f}]")
    w = np.mean(coefs, 0)
    top = np.argsort(-np.abs(w))[:5]
    print("biggest weights: " + ", ".join(f"{names[i]}={w[i]:+.2f}" for i in top))


if __name__ == "__main__":
    main()