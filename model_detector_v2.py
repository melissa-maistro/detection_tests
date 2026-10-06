"""Model-baseline detector v2: ideas that don't need the real questions.

1. Skip near-certain positions (Gemma's top probability > max_p): the watermark can barely
   change a token the model was almost sure about, so those positions only add noise.
2. Optionally weight each key by the model's entropy at that position
   (more uncertain = more room for the watermark to act).
3. Abstention: accuracy on the pairs where the detector is most confident.

Reuses load / fit_temperature / first_positions / softmax from model_detector.py.

Usage:
  python model_detector_v2.py --shards wm_gemma
  python model_detector_v2.py --shards wm_gemma --weight entropy
"""

import argparse
import math
from collections import Counter, defaultdict

import numpy as np

from keyless_v2 import metrics
from model_detector import first_positions, fit_temperature, load, softmax

NAMES = ["zw_mean", "zw_total", "zu_mean", "zu_total", "zd_mean", "zd_total"]


def run(texts, pairs, h, T, min_n, max_p, weight):
    kept, n_all, n_kept = {}, 0, 0
    for k in texts:
        P = softmax(texts[k][2].astype(np.float32) / T)
        top = P.max(1)
        ent = -(P * np.log(P + 1e-12)).sum(1)
        fp = first_positions(texts[k][0], h)
        kept[k] = [(ctx, t, j, float(ent[j])) for ctx, t, j in fp if top[j] <= max_p]
        n_all += len(fp)
        n_kept += len(kept[k])

    tok_for = defaultdict(set)
    for lst in kept.values():
        for ctx, t, _, _ in lst:
            tok_for[ctx].add(t)

    def contrib(k):
        """ctx -> (chosen token, {candidate token: model prob}, entropy) for one text."""
        ti = texts[k][1]
        P = softmax(texts[k][2].astype(np.float32) / T)
        out = {}
        for ctx, t, j, ent in kept[k]:
            row = dict(zip(ti[j].tolist(), P[j].tolist()))
            out[ctx] = (t, {u: row[u] for u in tok_for[ctx] if u in row}, ent)
        return out

    N = [Counter(), Counter()]
    O = [Counter(), Counter()]
    E = [defaultdict(float), defaultdict(float)]
    for p in pairs:
        for c in (0, 1):
            for ctx, (t, q, _) in contrib((p, c)).items():
                N[c][ctx] += 1
                O[c][(ctx, t)] += 1
                for u, v in q.items():
                    E[c][(ctx, u)] += v

    def agg(zs):
        if not zs:
            return [0.0, 0.0]
        num = sum(w * z for z, w in zs)
        return [num / sum(w for _, w in zs), num / math.sqrt(sum(w * w for _, w in zs))]

    fw, fu = [], []
    for p in pairs:
        own = {0: contrib((p, 0)), 1: contrib((p, 1))}

        def score(cx):
            zw, zu, zd = [], [], []
            for ctx, (t, _, ent) in cx.items():
                w = ent if weight == "entropy" else 1.0
                if w <= 0:
                    continue
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
                    zw.append((zc[0], w))
                if 1 in zc:
                    zu.append((zc[1], w))
                if 0 in zc and 1 in zc:
                    zd.append((zc[0] - zc[1], w))
            return agg(zw) + agg(zu) + agg(zd)

        fw.append(score(own[0]))
        fu.append(score(own[1]))
    return np.array(fw), np.array(fu), n_kept / max(n_all, 1)


def abstain(sw, su, fracs=(0.10, 0.25, 0.50)):
    margin = np.abs(sw - su)
    order = np.argsort(-margin)
    parts = []
    for f in fracs:
        k = max(1, int(len(order) * f))
        sel = order[:k]
        acc = float(np.mean((sw[sel] > su[sel]) + 0.5 * (sw[sel] == su[sel])))
        parts.append(f"most confident {int(f * 100)}%: {acc:.3f} (n={k})")
    return " | ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True)
    ap.add_argument("--h", type=int, default=4)
    ap.add_argument("--max-p", default="1.0,0.95,0.9,0.8",
                    help="skip positions where Gemma's top prob is above this (1.0 = keep all)")
    ap.add_argument("--weight", choices=["none", "entropy"], default="none")
    ap.add_argument("--min-n", type=int, default=1)
    args = ap.parse_args()

    texts, pairs = load(args.shards)
    T = fit_temperature(texts, pairs)
    print(f"\nh={args.h} | weight={args.weight} | leave-one-pair-out | chance = 0.500\n", flush=True)

    for mp in [float(x) for x in args.max_p.split(",")]:
        fw, fu, kept = run(texts, pairs, args.h, T, args.min_n, mp, args.weight)
        print(f"--- max_p={mp:.2f} (keeps {kept:.0%} of positions) ---")
        for j, nm in enumerate(NAMES):
            auc, acc, lo, hi = metrics(fw[:, j], fu[:, j])
            print(f"  {nm:<9} AUC={auc:.3f}  paired acc={acc:.3f} [{lo:.3f}-{hi:.3f}]")
        print("  zd_total abstention: " + abstain(fw[:, 5], fu[:, 5]), flush=True)
        print(flush=True)


if __name__ == "__main__":
    main()