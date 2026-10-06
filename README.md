# Keyless SynthID detection: code

Code behind `keyless_detection_tests.html`. All tests run on Google DeepMind's
`human_eval.jsonl` (Gemma-7B-IT answers, watermarked and unwatermarked pairs).

## Start here

**`model_detector_v2.py`** is the best detector. It gives the top results in the report
(rows #17 to #19): 0.601 paired accuracy for zw, 0.598 paired / 0.585 AUC for zd, and
0.792 on the 10% most confident pairs.

It needs the Gemma probability files from **`gemma_topk.py`** (run once on a GPU).
Everything else in this folder is earlier steps or diagnostics, kept so every row of
the report can be reproduced.

## How the best detector works, in short

1. `gemma_topk.py` runs Gemma-7B-IT over every answer and saves, for every token
   position, the model's top-40 next-token predictions. This is what the model would
   pick with no watermark.
2. `model_detector_v2.py` takes every "previous 4 tokens + next token" combination and
   compares how often that token was actually chosen after that context in *other*
   watermarked texts with how often Gemma expected it there. Tokens the key favors show
   up more than expected. A text that uses many of them gets a high score.
3. Positions where Gemma was almost certain (top probability above 0.9) are skipped,
   because the watermark can barely change those tokens.
4. Each pair is scored without itself or its partner (leave-one-pair-out).

Scores it prints:
- **zw**: compared with other watermarked texts. Needs no unwatermarked texts, so it is
  the version closest to a real deployment.
- **zu**: same comparison with unwatermarked texts. Control: must stay near 0.5.
- **zd**: zw minus zu. Cancels baseline errors that affect both classes.

## Files

| File | What it is | Report rows |
|---|---|---|
| `model_detector_v2.py` | **Best detector.** Model baseline + skipping near-certain positions + entropy weighting option + abstention | #17 to #19 |
| `gemma_topk.py` | Runs Gemma-7B-IT on a GPU (Colab T4 works) and saves top-40 logits as shards | input for #13 to #19 |
| `model_detector.py` | First model-baseline detector, h=3 to 5, plus stacking. Also provides functions used by `model_detector_v2.py` | #13 to #16 |
| `keyless_v3.py` | Counting detector with leave-one-pair-out counts and stacked logistic regression | #11, #12 |
| `consistency.py` | Style test vs key-agreement test | #9, #10 |
| `diagnose.py` | Shuffled labels, label-free commonness, within-text repetition | #6 to #8 |
| `keyless_v2.py` | Counting detector with z score, h combination and data-scaling curve. Also provides functions used by most other scripts | #3 to #5 |
| `keyless_ngram_detector.py` | First counting detector (log ratio, h=0 to 6) | #1, #2 |
| `keyless_detection_tests.html` | Results report | |

Dependencies between files: most scripts import from `keyless_v2.py`, and
`model_detector_v2.py` also imports from `model_detector.py`. Keep them in the same folder.

## Setup

```bash
git clone https://github.com/google-deepmind/synthid-text.git
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
hf auth login   # Gemma is gated: accept the license on its Hugging Face page first
```

## Reproduce the best result

**Step 1, on a GPU (once, about 30 to 60 min on a Colab T4):**

```bash
pip install -U transformers accelerate bitsandbytes
python gemma_topk.py --data synthid-text/data/human_eval.jsonl --out wm_gemma
```

Produces 15 files `wm_gemma/shard_*.npz` (about 300 MB). Resumable: rerun the same
command if it stops. Use `--batch 2` or `--batch 1` if the GPU runs out of memory.

**Step 2, on any laptop (no GPU, about 10 to 15 min):**

```bash
python model_detector_v2.py --shards wm_gemma --max-p 0.9
```

Add `--weight entropy` to try entropy weighting instead (same results in our runs).
Without `--max-p`, it runs thresholds 1.0, 0.95, 0.9 and 0.8; 1.0 reproduces
`model_detector.py` at h=4.

## Run the earlier tests

All of these run on a laptop and only need the tokenizer (no GPU):

```bash
python keyless_ngram_detector.py --data synthid-text/data/human_eval.jsonl
python keyless_v2.py --data synthid-text/data/human_eval.jsonl
python keyless_v2.py --data synthid-text/data/human_eval.jsonl --sizes 500,1000,2000,0 --h 4
python diagnose.py --data synthid-text/data/human_eval.jsonl
python consistency.py --data synthid-text/data/human_eval.jsonl
python keyless_v3.py --data synthid-text/data/human_eval.jsonl
python model_detector.py --shards wm_gemma
```

`keyless_ngram_detector.py` and `keyless_v2.py` also work on our own Qwen `pairs.jsonl`
(they use the saved token ids): add `--wm-col wm --unwm-col unwm`.

## Things to know

- **Paired accuracy** = share of pairs where the watermarked answer scores higher than
  its unwatermarked twin. 0.5 is chance. **AUC** is the single-text version.
- 2841 of 3000 pairs are used: pairs under 20 tokens are dropped, and each pair is cut
  to the shorter answer's length so length cannot reveal the label.
- Counting scripts use 5-fold CV split by question; `keyless_v3.py` and the model
  detectors use leave-one-pair-out, which trains on more data (about +1 point in our
  runs). Compare methods on the same setup.
- The question text is not in `human_eval.jsonl`, so `gemma_topk.py` uses a placeholder
  user turn ("Answer the question."). Temperature is then fitted on unwatermarked texts
  (1.25).
- h=4, the total aggregation and the 0.9 threshold were chosen after seeing results,
  so headline numbers are mildly optimistic. Differences under about one point are noise.
