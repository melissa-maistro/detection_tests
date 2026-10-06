"""Keyless SynthID detector baseline: context-conditioned token counting.

For each context length h, learn from training pairs which (previous h tokens, next token)
combos are more frequent in watermarked than unwatermarked text, then score test texts by
how often they use the favored combos. If the watermark uses ngram_len = n, the signal
should peak at h = n - 1. h = 0 is a control: SynthID barely changes plain token
frequencies, so it should stay near chance.

Works on DeepMind's data/human_eval.jsonl and on our Qwen pairs.jsonl (where it uses the
exact saved token ids instead of re-tokenizing).

Usage:
  pip install transformers scikit-learn
  huggingface-cli login   # Gemma is gated: accept the license on its HF page first
  python keyless_ngram_detector.py --data synthid-text/data/human_eval.jsonl
  python keyless_ngram_detector.py --data watermark_qwen/pairs.jsonl --wm-col wm --unwm-col unwm --max-pairs 5000
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


def load_pairs(path, wm_col, unwm_col, tokenizer_name, max_pairs, seed):
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
        group = r.get("q_id", r.get("pid", idx))
        pairs.append((group, wi[:n], ui[:n]))

    if max_pairs and len(pairs) > max_pairs:
        pairs = random.Random(seed).sample(pairs, max_pairs)
    toks = sum(len(w) for _, w, _ in pairs)
    print(f"{len(pairs)} usable pairs, {toks} tokens per class "
          f"(mean {toks / max(len(pairs), 1):.0f} per text)")
    return pairs


def keys_of(ids, h):
    """Unique (context, token) keys in one text, so repeated phrases count once."""
    return set(tuple(ids[i - h:i + 1]) for i in range(h, len(ids)))


def wilson(p, n, z=1.96):
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - m, c + m


def run(pairs, h, fold_of, folds, alpha, min_count):
    kw = [keys_of(w, h) for _, w, _ in pairs]
    ku = [keys_of(u, h) for _, _, u in pairs]

    # global counts minus the test fold's counts = training counts, without rebuilding
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
            tot += math.log((cw + alpha) / (cu + alpha))
            used += 1
        return (tot / used if used else 0.0), used / max(len(keys), 1)

    sw, su, cov = [], [], []
    for i in range(len(pairs)):
        a, ca = score(kw[i], fold_of[i])
        b, cb = score(ku[i], fold_of[i])
        sw.append(a)
        su.append(b)
        cov += [ca, cb]

    sw, su = np.array(sw), np.array(su)
    auc = roc_auc_score(np.r_[np.ones(len(sw)), np.zeros(len(su))], np.r_[sw, su])
    acc = float(np.mean((sw > su) + 0.5 * (sw == su)))
    lo, hi = wilson(acc, len(sw))
    return auc, acc, lo, hi, float(np.mean(cov))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--tokenizer", default="google/gemma-7b-it")
    ap.add_argument("--wm-col")
    ap.add_argument("--unwm-col")
    ap.add_argument("--h", default="0,1,2,3,4,5,6", help="context lengths to test")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=1.0, help="smoothing for the log ratio")
    ap.add_argument("--min-count", type=int, default=2, help="min train occurrences to use a key")
    ap.add_argument("--max-pairs", type=int, default=0, help="random subset, for data-scaling curves")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pairs = load_pairs(args.data, args.wm_col, args.unwm_col, args.tokenizer,
                       args.max_pairs, args.seed)

    # folds by question/prompt, so a prompt is never in both train and test
    groups = sorted({g for g, _, _ in pairs}, key=str)
    random.Random(args.seed).shuffle(groups)
    gfold = {g: i % args.folds for i, g in enumerate(groups)}
    fold_of = [gfold[g] for g, _, _ in pairs]

    print(f"\n{args.folds}-fold CV, split by prompt. Chance = 0.500")
    for h in [int(x) for x in args.h.split(",")]:
        auc, acc, lo, hi, cov = run(pairs, h, fold_of, args.folds, args.alpha, args.min_count)
        print(f"h={h}  AUC={auc:.3f}  paired acc={acc:.3f} [95% CI {lo:.3f}-{hi:.3f}]  "
              f"key coverage={cov:.2f}", flush=True)


if __name__ == "__main__":
    main()
