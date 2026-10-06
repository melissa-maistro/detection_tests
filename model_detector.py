"""Keyless detector with a model baseline.

For each (previous h tokens, next token) key, compare how often the token was chosen in
OTHER watermarked texts (observed) with how often Gemma expected it there (sum of its
no-watermark probabilities). Favored-by-key tokens show observed > expected. Same against
unwatermarked texts as a control. Each pair is scored without itself or its partner.
Temperature is fitted on unwatermarked texts only.

Usage:
  python model_detector.py --shards wm_gemma
"""

import argparse
import glob
import math
from collections import Counter, defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

from keyless_v2 import metrics

NAMES = ["zw_mean", "zw_total", "zu_mean", "zu_total", "zd_mean", "zd_total"]


def load(dirpath):
    texts = {}
    files = sorted(glob.glob(f"{dirpath}/shard_*.npz"))
    if not files:
        raise SystemExit(f"no shard_*.npz files in {dirpath}")
    for f in files:
        # read each array ONCE per shard; texts keep views into these arrays
        with np.load(f) as d:
            ids, ti, tl = d["ids"], d["topk_ids"], d["topk_logits"]
            lengths, pair, cls = d["lengths"], d["pair"], d["cls"]
        off = np.r_[0, np.cumsum(lengths)]
        for j in range(len(lengths)):
            a, b = off[j], off[j + 1]
            texts[(int(pair[j]), int(cls[j]))] = (ids[a:b].tolist(), ti[a:b], tl[a:b])
    pairs = sorted({p for p, _ in texts if (p, 0) in texts and (p, 1) in texts})
    print(f"{len(files)} shards, {len(pairs)} complete pairs", flush=True)
    return texts, pairs


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fit_temperature(texts, pairs):
    ids = np.concatenate([np.array(texts[(p, 1)][0], dtype=np.int32) for p in pairs])
    ti = np.concatenate([texts[(p, 1)][1] for p in pairs])
    tl = np.concatenate([texts[(p, 1)][2] for p in pairs]).astype(np.float32)
    match = ti == ids[:, None]
    hit = match.any(1)
    chosen = np.where(hit, (tl * match).sum(1), 0.0)
    best = None
    for T in np.arange(0.3, 1.55, 0.05):
        z = tl / T
        m = z.max(1)
        lse = m + np.log(np.exp(z - m[:, None]).sum(1))
        lp = np.where(hit, chosen / T - lse, math.log(1e-4))
        if best is None or lp.mean() > best[1]:
            best = (float(T), float(lp.mean()))
    print(f"temperature fitted on unwatermarked texts: T={best[0]:.2f} "
          f"(chosen token in top-K for {hit.mean():.1%} of positions)", flush=True)
    return best[0]


def first_positions(ids, h):
    """First occurrence of each context in a text, like SynthID's repetition mask."""
    seen, out = set(), []
    for j in range(h, len(ids)):
        ctx = tuple(ids[j - h:j])
        if ctx in seen:
            continue
        seen.add(ctx)
        out.append((ctx, ids[j], j))
    return out


def run_h(texts, pairs, h, T, min_n):
    pos = {k: first_positions(texts[k][0], h) for k in texts}
    tok_for = defaultdict(set)
    for lst in pos.values():
        for ctx, t, _ in lst:
            tok_for[ctx].add(t)

    def contrib(k):
        """ctx -> (chosen token, {candidate token: model prob}) for one text."""
        ti = texts[k][1]
        P = softmax(texts[k][2].astype(np.float32) / T)
        out = {}
        for ctx, t, j in pos[k]:
            row = dict(zip(ti[j].tolist(), P[j].tolist()))
            out[ctx] = (t, {u: row[u] for u in tok_for[ctx] if u in row})
        return out

    N = [Counter(), Counter()]
    O = [Counter(), Counter()]
    E = [defaultdict(float), defaultdict(float)]
    for p in pairs:
        for c in (0, 1):
            for ctx, (t, q) in contrib((p, c)).items():
                N[c][ctx] += 1
                O[c][(ctx, t)] += 1
                for u, v in q.items():
                    E[c][(ctx, u)] += v

    fw, fu = [], []
    for p in pairs:
        own = {0: contrib((p, 0)), 1: contrib((p, 1))}

        def score(cx):
            zw, zu, zd = [], [], []
            for ctx, (t, _) in cx.items():
                zc = {}
                for c in (0, 1):
                    o = own[c].get(ctx)
                    n = N[c][ctx] - (1 if o else 0)
                    if n < min_n:
                        continue
                    obs = O[c][(ctx, t)] - (1 if o and o[0] == t else 0)
                    e = E[c].get((ctx, t), 0.0) - (o[1].get(t, 0.0) if o else 0.0)
                    e = min(max(e, 0.0), n)
                    zc[c] = (obs - e) / math.sqrt(max(e * (1 - e / n), 0.05))
                if 0 in zc:
                    zw.append(zc[0])
                if 1 in zc:
                    zu.append(zc[1])
                if 0 in zc and 1 in zc:
                    zd.append(zc[0] - zc[1])

            def agg(z):
                return [float(np.mean(z)), sum(z) / math.sqrt(len(z))] if z else [0.0, 0.0]
            return agg(zw) + agg(zu) + agg(zd)

        fw.append(score(own[0]))
        fu.append(score(own[1]))
    return np.array(fw), np.array(fu)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True)
    ap.add_argument("--h", default="3,4,5")
    ap.add_argument("--min-n", type=int, default=1, help="min other occurrences of a context")
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--C", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    texts, pairs = load(args.shards)
    T = fit_temperature(texts, pairs)
    print("\nleave-one-pair-out | chance = 0.500 | zw = vs watermarked texts, "
          "zu = vs unwatermarked (control), zd = zw - zu\n", flush=True)

    FW, FU, names = [], [], []
    for h in [int(x) for x in args.h.split(",")]:
        fw, fu = run_h(texts, pairs, h, T, args.min_n)
        for j, nm in enumerate(NAMES):
            auc, acc, lo, hi = metrics(fw[:, j], fu[:, j])
            print(f"h={h} {nm:<9} AUC={auc:.3f}  acc={acc:.3f} [{lo:.3f}-{hi:.3f}]", flush=True)
        print(flush=True)
        FW.append(fw)
        FU.append(fu)
        names += [f"h{h}_{nm}" for nm in NAMES]
    FW, FU = np.hstack(FW), np.hstack(FU)

    allf = np.vstack([FW, FU])
    mu, sd = allf.mean(0), allf.std(0) + 1e-12
    FW, FU = (FW - mu) / sd, (FU - mu) / sd
    sw, su = np.zeros(len(FW)), np.zeros(len(FW))
    coefs = []
    for tr, te in KFold(args.folds, shuffle=True, random_state=args.seed).split(FW):
        d = FW[tr] - FU[tr]
        lr = LogisticRegression(C=args.C, fit_intercept=False, max_iter=2000).fit(
            np.vstack([d, -d]), np.r_[np.ones(len(d)), np.zeros(len(d))])
        coefs.append(lr.coef_[0])
        sw[te], su[te] = FW[te] @ lr.coef_[0], FU[te] @ lr.coef_[0]
    auc, acc, lo, hi = metrics(sw, su)
    print(f"STACKED ({args.folds}-fold)  AUC={auc:.3f}  paired acc={acc:.3f} [{lo:.3f}-{hi:.3f}]")
    w = np.mean(coefs, 0)
    print("biggest weights: " + ", ".join(f"{names[i]}={w[i]:+.2f}"
                                          for i in np.argsort(-np.abs(w))[:5]))


if __name__ == "__main__":
    main()