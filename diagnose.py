"""Three checks on where the ~55% keyless signal comes from.

A. Label shuffle: randomly swap wm/unwm labels. Must drop to ~0.50, else the code leaks.
B. Commonness (no labels used): does a text reuse patterns that are common in OTHER texts?
   If this alone gives ~0.55, our detector mostly measures "generic phrasing", not the key.
C. Repetition inside one text (no labels used): share of repeated n-grams within the text.

Usage:
  python diagnose.py --data synthid-text/data/human_eval.jsonl
"""

import argparse
import math
import random
from collections import Counter

import numpy as np

from keyless_v2 import keys_of, load_pairs, metrics, oof_scores


def show(name, sw, su):
    auc, acc, lo, hi = metrics(np.array(sw), np.array(su))
    print(f"{name:<34} AUC={auc:.3f}  paired acc={acc:.3f} [95% CI {lo:.3f}-{hi:.3f}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--tokenizer", default="google/gemma-7b-it")
    ap.add_argument("--h", type=int, default=4)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pairs = load_pairs(args.data, None, None, args.tokenizer)
    h = args.h
    print(f"\nchecks at h={h} | chance = 0.500 | AUC above 0.5 means wm scores higher\n")

    # A. label shuffle
    rng = random.Random(args.seed)
    shuf = [(g, u, w) if rng.random() < 0.5 else (g, w, u) for g, w, u in pairs]
    groups = sorted({g for g, _, _ in shuf}, key=str)
    random.Random(args.seed).shuffle(groups)
    gfold = {g: i % args.folds for i, g in enumerate(groups)}
    fold_of = [gfold[g] for g, _, _ in shuf]
    sw, su, _ = oof_scores(shuf, h, fold_of, args.folds, "z", 1.0, 2)
    show("A. shuffled labels (expect 0.50)", sw, su)

    # B. commonness of a text's patterns in all other texts (labels never used)
    kw = [keys_of(w, h) for _, w, _ in pairs]
    ku = [keys_of(u, h) for _, _, u in pairs]
    pooled = Counter()
    for s in kw + ku:
        pooled.update(s)

    def common(keys):
        return float(np.mean([math.log1p(pooled[k] - 1) for k in keys])) if keys else 0.0

    show("B. commonness (no labels)", [common(k) for k in kw], [common(k) for k in ku])

    # C. repetition inside the text (labels never used)
    def rep(ids):
        grams = [tuple(ids[i - h:i + 1]) for i in range(h, len(ids))]
        return 1 - len(set(grams)) / max(len(grams), 1)

    show("C. repetition inside text", [rep(w) for _, w, _ in pairs], [rep(u) for _, _, u in pairs])


if __name__ == "__main__":
    main()