"""Keyless SynthID detector v2: context-conditioned token counting, plus
(1) a combined score over several context lengths,
(2) a count-aware key score (--stat z) next to the original log ratio,
(3) a data-scaling curve on nested subsets (--sizes).

Usage:
  python keyless_v2.py --data synthid-text/data/human_eval.jsonl
  python keyless_v2.py --data synthid-text/data/human_eval.jsonl --stat logratio
  python keyless_v2.py --data synthid-text/data/human_eval.jsonl --sizes 500,1000,2000,0 --h 4
  (size 0 = all pairs)
"""

import argparse
import json
import math
import random
from collections import Counter

import numpy as np
from sklearn.metrics import roc_auc_score


def find_cols(row, wm_col, unwm_col):
    if wm_col and unwm_col:
        return wm_col, unwm_col
    keys = list(row)
    unwm = [k for k in keys if "unwatermarked" in k.lower()]
    wm = [k for k in keys if "watermarked" in k.lower() and "unwatermarked" not in k.lower()]
    if len(unwm) != 1 or len(wm) != 1:
        raise SystemExit(f"could not guess columns from {keys}; pass --wm-col and --unwm-col")
    return wm[0], unwm[0]


def load_pairs(path, wm_col, unwm_col, tokenizer_name):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    wc, uc = find_cols(rows[0], wm_col, unwm_col)
    use_ids = f"{wc}_ids" in rows[0] and f"{uc}_ids" in rows[0]
    print(f"{len(rows)} rows | wm={wc!r} unwm={uc!r} | "
          f"{'saved token ids' if use_ids else 'tokenizer ' + tokenizer_name}")
    tok = None
    if not use_ids:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(tokenizer_name)

    pairs = []
    for idx, r in enumerate(rows):
        w, u = r.get(wc), r.get(uc)
        if not w or not u:
            continue
        if use_ids:
            wi, ui = r[f"{wc}_ids"], r[f"{uc}_ids"]
        else:
            wi = tok.encode(w, add_special_tokens=False)
            ui = tok.encode(u, add_special_tokens=False)
        n = min(len(wi), len(ui))  # equal length per pair, so length can't leak the label
        if n < 20:
            continue
        pairs.append((r.get("q_id", r.get("pid", idx)), wi[:n], ui[:n]))
    print(f"{len(pairs)} usable pairs")
    return pairs


def keys_of(ids, h):
    """Unique (context, token) keys in one text, so repeated phrases count once."""
    return set(tuple(ids[i - h:i + 1]) for i in range(h, len(ids)))


def wilson(p, n, z=1.96):
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - m, c + m


def key_score(cw, cu, stat, alpha):
    if stat == "z":
        return (cw - cu) / math.sqrt(cw + cu)  # more evidence -> bigger weight
    return math.log((cw + alpha) / (cu + alpha))


def oof_scores(pairs, h, fold_of, folds, stat, alpha, min_count):
    """Out-of-fold score for every text: each text is scored with counts from other folds only."""
    kw = [keys_of(w, h) for _, w, _ in pairs]
    ku = [keys_of(u, h) for _, _, u in pairs]
    gw, gu = Counter(), Counter()
    fw = [Counter() for _ in range(folds)]
    fu = [Counter() for _ in range(folds)]
    for i in range(len(pairs)):
        gw.update(kw[i])
        gu.update(ku[i])
        fw[fold_of[i]].update(kw[i])
        fu[fold_of[i]].update(ku[i])

    def score(keys, k):
        tot, used = 0.0, 0
        for key in keys:
            cw = gw[key] - fw[k][key]
            cu = gu[key] - fu[k][key]
            if cw + cu < min_count:
                continue
            tot += key_score(cw, cu, stat, alpha)
            used += 1
        return (tot / used if used else 0.0), used / max(len(keys), 1)

    sw, su, cov = [], [], []
    for i in range(len(pairs)):
        a, ca = score(kw[i], fold_of[i])
        b, cb = score(ku[i], fold_of[i])
        sw.append(a)
        su.append(b)
        cov += [ca, cb]
    return np.array(sw), np.array(su), float(np.mean(cov))


def metrics(sw, su):
    auc = roc_auc_score(np.r_[np.ones(len(sw)), np.zeros(len(su))], np.r_[sw, su])
    acc = float(np.mean((sw > su) + 0.5 * (sw == su)))
    lo, hi = wilson(acc, len(sw))
    return auc, acc, lo, hi


def zs(x):
    return (x - x.mean()) / (x.std() + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--tokenizer", default="google/gemma-7b-it")
    ap.add_argument("--wm-col")
    ap.add_argument("--unwm-col")
    ap.add_argument("--h", default="0,1,2,3,4,5,6", help="context lengths to report")
    ap.add_argument("--combine", default="3,4,5,6", help="context lengths to add up ('' = none)")
    ap.add_argument("--stat", choices=["z", "logratio"], default="z")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=1.0, help="smoothing for logratio")
    ap.add_argument("--min-count", type=int, default=2, help="min train occurrences to use a key")
    ap.add_argument("--sizes", default="0", help="comma list of pair counts, 0 = all")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pairs = load_pairs(args.data, args.wm_col, args.unwm_col, args.tokenizer)
    random.Random(args.seed).shuffle(pairs)  # nested subsets: size 500 is inside size 1000
    report = [int(x) for x in args.h.split(",") if x != ""]
    combine = [int(x) for x in args.combine.split(",") if x != ""]

    for size in [int(s) for s in args.sizes.split(",")]:
        sub = pairs[:size] if size else pairs
        groups = sorted({g for g, _, _ in sub}, key=str)
        random.Random(args.seed).shuffle(groups)
        gfold = {g: i % args.folds for i, g in enumerate(groups)}
        fold_of = [gfold[g] for g, _, _ in sub]

        print(f"\n=== {len(sub)} pairs | {args.folds}-fold CV by prompt | stat={args.stat} "
              f"| chance = 0.500 ===")
        scores = {}
        for h in sorted(set(report) | set(combine)):
            sw, su, cov = oof_scores(sub, h, fold_of, args.folds, args.stat,
                                     args.alpha, args.min_count)
            scores[h] = (sw, su)
            if h in report:
                auc, acc, lo, hi = metrics(sw, su)
                print(f"h={h}  AUC={auc:.3f}  paired acc={acc:.3f} "
                      f"[95% CI {lo:.3f}-{hi:.3f}]  key coverage={cov:.2f}", flush=True)

        if len(combine) > 1:
            n = len(sub)
            total = sum(zs(np.r_[scores[h][0], scores[h][1]]) for h in combine)
            auc, acc, lo, hi = metrics(total[:n], total[n:])
            print(f"combined h={args.combine}  AUC={auc:.3f}  paired acc={acc:.3f} "
                  f"[95% CI {lo:.3f}-{hi:.3f}]", flush=True)


if __name__ == "__main__":
    main()