"""Generate paired answers from an open model (default Qwen2.5-1.5B-Instruct) with and
without a SynthID-Text watermark under OUR OWN key, for calibrating keyless detectors.

Why a local model, not an API: the watermark is applied inside the sampling loop, so
we must run the model ourselves (Hugging Face transformers' SynthID implementation).
Note: HF's hashing differs from the Gemini app's, and our key differs from Google's or
Anthropic's - what transfers is the method and data-scaling curve, not a learned key.

Each prompt gets one watermarked and one unwatermarked answer (same sampling params,
different seeds). Output: <out>/pairs.jsonl, one line per pair with texts, exact
generated token ids, and the key-based mean g-value score of each text (the "with key"
detector, i.e. the ceiling a keyless detector could approach). Resumable: prompts
already in pairs.jsonl are skipped. <out>/config.json stores the key.

Usage (Colab T4 or local):
    python qwen_synthid_generate.py --out /content/drive/MyDrive/watermark_qwen --n-pairs 60000
    python tools/qwen_synthid_generate.py --out .tmp/qwen_smoke --n-pairs 4 --model Qwen/Qwen2.5-0.5B-Instruct --max-new-tokens 40
"""

import argparse
import json
import os
import random
import re
import secrets
import time
from pathlib import Path

import torch

SYSTEM = ("You are a helpful assistant. Answer in natural prose (no code, no lists unless "
          "essential), about 150-300 words.")
CODE_RE = re.compile(r"\b(code|python|javascript|sql|function|program|script|regex|html|css|java|c\+\+)\b", re.I)


def load_prompts(n, seed=1234):
    """Prose-oriented instructions from Alpaca and Dolly (both ungated on HF)."""
    from datasets import load_dataset
    raw = []
    for r in load_dataset("tatsu-lab/alpaca", split="train"):
        p = r["instruction"].strip() + (f"\n\n{r['input'].strip()}" if r["input"].strip() else "")
        raw.append(("alpaca", p))
    for r in load_dataset("databricks/databricks-dolly-15k", split="train"):
        p = r["instruction"].strip() + (f"\n\n{r['context'].strip()}" if r["context"].strip() else "")
        raw.append(("dolly", p))
    seen, out = set(), []
    for src, p in raw:
        key = p.lower()
        if CODE_RE.search(p) or not 15 <= len(p) <= 1500 or key in seen:
            continue
        seen.add(key)
        out.append((src, p))
    random.Random(seed).shuffle(out)  # fixed order, so resuming maps pid -> same prompt
    return out[:n]


def load_or_make_config(out, args):
    path = out / "config.json"
    if path.exists():
        cfg = json.loads(path.read_text())
        print(f"resuming with existing config/key from {path}")
        return cfg
    cfg = {
        "model": args.model, "ngram_len": args.ngram_len,
        "keys": [secrets.randbelow(2**31) for _ in range(args.n_keys)],
        "temperature": args.temperature, "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens, "system_prompt": SYSTEM,
        "sampling_table_size": 2**16, "sampling_table_seed": 0, "context_history_size": 1024,
    }
    path.write_text(json.dumps(cfg, indent=2))
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-pairs", type=int, default=60000)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--max-new-tokens", type=int, default=320)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--ngram-len", type=int, default=5)
    ap.add_argument("--n-keys", type=int, default=30)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer, SynthIDTextWatermarkingConfig

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_or_make_config(out, args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} model={cfg['model']}")

    tok = AutoTokenizer.from_pretrained(cfg["model"], padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"], dtype=torch.float16 if device == "cuda" else torch.float32).to(device).eval()
    wm_cfg = SynthIDTextWatermarkingConfig(
        keys=cfg["keys"], ngram_len=cfg["ngram_len"],
        sampling_table_size=cfg["sampling_table_size"], sampling_table_seed=cfg["sampling_table_seed"],
        context_history_size=cfg["context_history_size"])
    scorer = wm_cfg.construct_processor(vocab_size=len(tok), device=device)

    stop_ids = {tok.pad_token_id, tok.eos_token_id}
    gen_eos = model.generation_config.eos_token_id
    stop_ids |= set(gen_eos if isinstance(gen_eos, list) else [gen_eos])
    stop_ids.discard(None)

    def chat(p):
        return tok.apply_chat_template([{"role": "system", "content": cfg["system_prompt"]},
                                        {"role": "user", "content": p}],
                                       tokenize=False, add_generation_prompt=True)

    def generate(prompts, watermark, seed):
        enc = tok([chat(p) for p in prompts], return_tensors="pt", padding=True).to(device)
        torch.manual_seed(seed)
        kw = dict(do_sample=True, temperature=cfg["temperature"], top_k=cfg["top_k"], top_p=1.0,
                  max_new_tokens=cfg["max_new_tokens"], pad_token_id=tok.pad_token_id)
        if watermark:
            kw["watermarking_config"] = wm_cfg
        with torch.inference_mode():
            seqs = model.generate(**enc, **kw)[:, enc["input_ids"].shape[1]:].tolist()
        ids, texts, finished = [], [], []
        for s in seqs:
            cut = next((j for j, t in enumerate(s) if t in stop_ids), None)
            s = s[:cut] if cut is not None else s
            ids.append(s)
            texts.append(tok.decode(s, skip_special_tokens=True))
            finished.append(cut is not None)
        return ids, texts, finished

    def g_score(ids):
        """Key-based detector: mean g-value over non-repeated n-grams (0.5 = no mark)."""
        if len(ids) < cfg["ngram_len"] + 1:
            return None
        x = torch.tensor([ids], device=device)
        g = scorer.compute_g_values(x).float().mean(-1)          # (1, L-n+1)
        mask = scorer.compute_context_repetition_mask(x).float()  # (1, L-n+1)
        return float((g * mask).sum() / mask.sum().clamp(min=1))

    prompts = load_prompts(args.n_pairs)
    pairs_path = out / "pairs.jsonl"
    done = set()
    if pairs_path.exists():
        with pairs_path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["pid"])
                except (ValueError, KeyError):
                    pass  # partial last line from an interrupted write
    todo = [i for i in range(len(prompts)) if i not in done]
    # batch similar-length prompts together to waste less padding
    todo.sort(key=lambda i: len(prompts[i][1]))
    print(f"{len(prompts)} prompts, {len(done)} already done, {len(todo)} to generate", flush=True)

    t0, made = time.time(), 0
    with pairs_path.open("a", encoding="utf-8") as f:
        for b in range(0, len(todo), args.batch):
            idx = todo[b:b + args.batch]
            ps = [prompts[i][1] for i in idx]
            wi, wt, wf = generate(ps, True, seed=idx[0])
            ui, ut, uf = generate(ps, False, seed=10**7 + idx[0])
            for j, i in enumerate(idx):
                f.write(json.dumps({
                    "pid": i, "source": prompts[i][0], "prompt": prompts[i][1],
                    "wm": wt[j], "unwm": ut[j], "wm_ids": wi[j], "unwm_ids": ui[j],
                    "wm_finished": wf[j], "unwm_finished": uf[j],
                    "wm_gscore": g_score(wi[j]), "unwm_gscore": g_score(ui[j]),
                }, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
            made += len(idx)
            el = time.time() - t0
            print(f"  {len(done) + made}/{len(prompts)} pairs  ({el / 60:.1f} min, "
                  f"~{el / made * (len(todo) - made) / 60:.0f} min left)", flush=True)

    # sanity check: the key-based detector must separate the two groups clearly
    rows = [json.loads(l) for l in pairs_path.open(encoding="utf-8") if l.strip()]
    gw = [r["wm_gscore"] for r in rows if r["wm_gscore"] is not None]
    gu = [r["unwm_gscore"] for r in rows if r["unwm_gscore"] is not None]
    wins = sum(r["wm_gscore"] > r["unwm_gscore"] for r in rows
               if r["wm_gscore"] is not None and r["unwm_gscore"] is not None)
    print(f"done: {len(rows)} pairs. mean g-score wm={sum(gw) / len(gw):.4f} "
          f"unwm={sum(gu) / len(gu):.4f}; key-based paired accuracy={wins / len(rows):.3f}")


if __name__ == "__main__":
    main()
