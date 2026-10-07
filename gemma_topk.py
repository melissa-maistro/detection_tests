"""Gemma-7B-it top-K next-token logits for every position of every text in DeepMind's
human_eval.jsonl: what the model would predict with no watermark. Shards go to --out
(use Google Drive on Colab); finished shards are skipped on restart.

Token ids use the same tokenization and per-pair truncation as keyless_v2.load_pairs.

Usage (GPU needed, Colab T4 works):
  pip install -U transformers accelerate bitsandbytes
  python gemma_topk.py --data synthid-text/data/human_eval.jsonl --out wm_gemma
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_texts(path, tok):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    texts, pair = [], 0
    for r in rows:
        w, u = r.get("watermarked_model_response"), r.get("unwatermarked_model_response")
        if not w or not u:
            continue
        wi = tok.encode(w, add_special_tokens=False)
        ui = tok.encode(u, add_special_tokens=False)
        n = min(len(wi), len(ui))
        if n < 20:
            continue
        texts.append((pair, 0, wi[:n]))  # cls 0 = watermarked
        texts.append((pair, 1, ui[:n]))  # cls 1 = unwatermarked
        pair += 1
    return texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="google/gemma-7b-it")
    ap.add_argument("--k", type=int, default=40)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--shard", type=int, default=400, help="texts per shard file")
    ap.add_argument("--prompt", default="Answer the question.",
                    help="placeholder user turn, since the original questions are not in the file")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    texts = load_texts(args.data, tok)
    print(f"{len(texts)} texts ({len(texts) // 2} pairs)", flush=True)

    prefix = tok.apply_chat_template([{"role": "user", "content": args.prompt}],
                                     add_generation_prompt=True, tokenize=True)
    if not isinstance(prefix, list):
        prefix = prefix["input_ids"]
    P = len(prefix)

    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="auto",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_compute_dtype=torch.float16)).eval()
    dev = next(model.parameters()).device
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0

    t0 = time.time()
    for s in range(0, len(texts), args.shard):
        path = out / f"shard_{s // args.shard:04d}.npz"
        if path.exists():
            continue
        chunk = texts[s:s + args.shard]
        ids_l, ti_l, tl_l = [], [], []
        for b in range(0, len(chunk), args.batch):
            bt = chunk[b:b + args.batch]
            L = P + max(len(t[2]) for t in bt)
            inp = torch.full((len(bt), L), pad, dtype=torch.long)
            att = torch.zeros_like(inp)
            for j, t in enumerate(bt):
                seq = prefix + t[2]
                inp[j, :len(seq)] = torch.tensor(seq)
                att[j, :len(seq)] = 1
            with torch.inference_mode():
                logits = model(input_ids=inp.to(dev), attention_mask=att.to(dev)).logits
                for j, t in enumerate(bt):
                    n = len(t[2])
                    lg = logits[j, P - 1:P - 1 + n].float()  # row k predicts t[2][k]
                    v, ix = lg.topk(args.k, dim=-1)
                    ids_l.append(np.array(t[2], dtype=np.int32))
                    ti_l.append(ix.cpu().numpy().astype(np.int32))
                    tl_l.append(v.cpu().numpy().astype(np.float16))
            del logits
        tmp = out / (path.name + ".tmp")
        with open(tmp, "wb") as f:
            np.savez(f, ids=np.concatenate(ids_l), topk_ids=np.concatenate(ti_l),
                     topk_logits=np.concatenate(tl_l),
                     lengths=np.array([len(t[2]) for t in chunk], dtype=np.int32),
                     pair=np.array([t[0] for t in chunk], dtype=np.int32),
                     cls=np.array([t[1] for t in chunk], dtype=np.int8))
        tmp.rename(path)
        done = min(s + args.shard, len(texts))
        print(f"  {done}/{len(texts)} texts, {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
