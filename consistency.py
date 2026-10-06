"""Is the ~55% signal style or key?

For every text we measure how common its (context, token) patterns are in reference sets
that never include the text itself or its pair partner.

Style test: commonness against UNWATERMARKED texts only (no key in the reference).
  wm > unwm here means watermarked text is more generic in general.
Key test: (commonness against WATERMARKED texts) minus (commonness against UNWATERMARKED texts).
  wm > unwm here means watermarked texts agree with each other beyond general style,
  which is what a fixed key would cause. Should peak near h = ngram_len - 1.

Usage:
  python consistency.py --data synthid-text/data/human_eval.jsonl
"""

import argparse
import math
from collections import Counter

import numpy as np

from keyless_v2 import keys_of, load_pairs, metrics


def fmt(sw, su):
    auc, acc, lo, hi = metrics(np.array(sw), np.array(su))
    return f"AUC={auc:.3f} acc={acc:.3f} [{lo:.3f}-{hi:.3f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--tokenizer", default="google/gemma-7b-it")
    ap.add_argument("--h", default="1,2,3,4,5,6")
    args = ap.parse_args()

    pairs = load_pairs(args.data, None, None, args.tokenizer)
    print("\nchance = 0.500 | above 0.5 means watermarked texts score higher\n")

    for h in [int(x) for x in args.h.split(",")]:
        kw = [keys_of(w, h) for _, w, _ in pairs]
        ku = [keys_of(u, h) for _, _, u in pairs]
        cw, cu = Counter(), Counter()
        for s in kw:
            cw.update(s)
        for s in ku:
            cu.update(s)

        def common(keys, ref, exclude):
            # ref count minus the excluded text's own contribution (0 or 1, keys are sets)
            if not keys:
                return 0.0
            return float(np.mean([math.log1p(ref[k] - (k in exclude)) for k in keys]))

        style_w, style_u, key_w, key_u = [], [], [], []
        for i in range(len(pairs)):
            a, b = kw[i], ku[i]
            # wm text a: drop itself from wm ref, drop its partner b from unwm ref
            a_vs_un = common(a, cu, b)
            a_vs_wm = common(a, cw, a)
            # unwm text b: drop itself from unwm ref, drop its partner a from wm ref
            b_vs_un = common(b, cu, b)
            b_vs_wm = common(b, cw, a)
            style_w.append(a_vs_un)
            style_u.append(b_vs_un)
            key_w.append(a_vs_wm - a_vs_un)
            key_u.append(b_vs_wm - b_vs_un)

        print(f"h={h}  style: {fmt(style_w, style_u)}   key: {fmt(key_w, key_u)}", flush=True)


if __name__ == "__main__":
    main()