"""Tied LM-head probe: do the added-token rows become output logits that win?

Llama-3.2-1B ties input embeddings to the LM head (tie_word_embeddings=true;
checkpoints store only model.embed_tokens.weight). finetune.py appends
<|predictor_i|> tokens that appear only in the JEPA text branch — never in the
LM branch and never as a label — so their rows are trained by lbd*jepa_loss
alone, yet each row is also an output-logit row. Per checkpoint this measures:

  static   (reads one tensor): norm of every added row and of the stop/header
           specials, relative to the regular-vocab norm distribution; cosine to
           the vocab mean direction; drift of base rows from the pretrained
           weights (<|eot_id|> specifically, and the vocab median).
  forward  (--forward_examples N, GPU): over the first N examples of
           --input_file, the rank / prob / top-1 rate of those same tokens in the
           next-token distribution at two positions: right after the prompt (the
           first generated token) and right after prompt+gold (where <|eot_id|>
           should win).

Read-only over checkpoints. Writes one tidy CSV row per (cell, step, token).
Cells come from --cells and/or --runs_dir + --cell_glob; every
checkpoints/checkpoint-<step> (plus the final model with --include_final).
Work is sharded over (cell, step) with --shard_id/--num_shards; --merge
concatenates shard CSVs.
"""

import argparse
import csv
import glob
import json
import math
import os
import sys

import torch
from safetensors import safe_open

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

if not torch.cuda.is_available():
    # See eval_generations.py: evaluate.py gates a print on torch.cuda.current_device().
    torch.cuda.current_device = lambda: 0

import evaluate as ev  # noqa: E402 — repo-root evaluate.py, imported, never modified

EMBED_KEY = "model.embed_tokens.weight"
BASE_VOCAB = 128256   # Llama-3 vocab; finetune.py's added tokens start here
REGULAR_END = 128000  # ids >= this are Llama-3 specials / reserved (many untrained)
NAMED = {128001: "<|end_of_text|>", 128006: "<|start_header_id|>",
         128007: "<|end_header_id|>", 128008: "<|eom_id|>", 128009: "<|eot_id|>"}

FIELDS = ["cell", "dataset", "lbd", "k", "seed", "step", "token_id", "token",
          "predictor_index", "predictor_used", "norm", "norm_rel_median", "cos_to_vocab_mean",
          "drift_rel", "vocab_norm_median", "vocab_norm_p99", "vocab_norm_max", "vocab_drift_median",
          "fwd_n", "fwd_first_rank_median", "fwd_first_prob_mean", "fwd_first_top1_frac",
          "fwd_end_rank_median", "fwd_end_prob_mean", "fwd_end_top1_frac"]


def discover(args):
    cells = list(args.cells or [])
    if args.runs_dir:
        cells += sorted(glob.glob(os.path.join(args.runs_dir, args.cell_glob)))
    jobs = []
    for cell in cells:
        cfg_path = os.path.join(cell, "config.json")
        ck = os.path.join(cell, "checkpoints")
        if not (os.path.exists(cfg_path) and os.path.isdir(ck)):
            continue
        with open(cfg_path) as f:
            cfg = json.load(f)
        for d in glob.glob(os.path.join(ck, "checkpoint-*")):
            if os.path.exists(os.path.join(d, "model.safetensors")):
                jobs.append((cell, cfg, int(d.rsplit("-", 1)[1]), d))
        if args.include_final and os.path.exists(os.path.join(ck, "model.safetensors")):
            jobs.append((cell, cfg, "final", ck))
    jobs.sort(key=lambda j: (j[0], math.inf if j[2] == "final" else j[2]))
    return jobs


def read_embed(path):
    with safe_open(path, framework="pt") as f:
        return f.get_tensor(EMBED_KEY).float()


def base_embed(repo_id):
    from huggingface_hub import snapshot_download
    snap = snapshot_download(repo_id, local_files_only=True)
    for path in sorted(glob.glob(os.path.join(snap, "*.safetensors"))):
        with safe_open(path, framework="pt") as f:
            if EMBED_KEY in f.keys():
                return f.get_tensor(EMBED_KEY).float()
    raise SystemExit(f"no {EMBED_KEY} in {snap}")


def token_names(ckpt_dir):
    with open(os.path.join(ckpt_dir, "tokenizer.json")) as f:
        added = {t["id"]: t["content"] for t in json.load(f)["added_tokens"]}
    return {**added, **NAMED}


def static_metrics(E, E0, ids):
    reg = E[:REGULAR_END]
    norms = reg.norm(dim=1)
    med = norms.median()
    mu = reg.mean(0)
    mu = mu / mu.norm()
    base0 = E0[:REGULAR_END]
    drift = (reg - base0).norm(dim=1) / base0.norm(dim=1).clamp_min(1e-8)
    vocab = {"vocab_norm_median": float(med), "vocab_norm_p99": float(torch.quantile(norms, 0.99)),
             "vocab_norm_max": float(norms.max()), "vocab_drift_median": float(drift.median())}
    per = {}
    for i in ids:
        v = E[i]
        n = v.norm()
        per[i] = {
            "norm": float(n),
            "norm_rel_median": float(n / med),
            "cos_to_vocab_mean": float(v @ mu / n.clamp_min(1e-8)),
            "drift_rel": float((v - E0[i]).norm() / E0[i].norm().clamp_min(1e-8)) if i < BASE_VOCAB else "",
        }
    return vocab, per


@torch.no_grad()
def forward_metrics(model, tokenizer, examples, ids, original_model_name):
    ids_t = torch.tensor(ids, device=model.device)
    acc = {pos: {"rank": [], "prob": [], "top1": []} for pos in ("first", "end")}
    for ex in examples:
        msgs = ex["messages"]
        prompt = ev.format_conversation(ev.get_messages(original_model_name, msgs), tokenizer)
        p_ids = tokenizer(prompt, truncation=True, max_length=512, add_special_tokens=True)["input_ids"]
        g_ids = tokenizer(msgs[2]["content"], add_special_tokens=False)["input_ids"]
        logits = model(input_ids=torch.tensor([p_ids + g_ids], device=model.device)).logits[0].float()
        for pos, at in (("first", len(p_ids) - 1), ("end", len(p_ids) + len(g_ids) - 1)):
            row = logits[at]
            sel = row[ids_t]
            acc[pos]["rank"].append((row[None, :] > sel[:, None]).sum(-1).cpu())
            acc[pos]["prob"].append((sel - torch.logsumexp(row, -1)).exp().cpu())
            acc[pos]["top1"].append((ids_t == row.argmax()).cpu())
    out = {}
    for j, i in enumerate(ids):
        d = {"fwd_n": len(examples)}
        for pos in ("first", "end"):
            d[f"fwd_{pos}_rank_median"] = float(torch.stack(acc[pos]["rank"])[:, j].float().median())
            d[f"fwd_{pos}_prob_mean"] = float(torch.stack(acc[pos]["prob"])[:, j].mean())
            d[f"fwd_{pos}_top1_frac"] = float(torch.stack(acc[pos]["top1"])[:, j].float().mean())
        out[i] = d
    return out


def run(args):
    jobs = discover(args)
    mine = [j for n, j in enumerate(jobs) if n % args.num_shards == args.shard_id]
    print(f"[probe shard {args.shard_id}/{args.num_shards}] {len(mine)} of {len(jobs)} checkpoints")
    E0 = base_embed(args.original_model_name)
    examples = []
    if args.forward_examples:
        with open(args.input_file) as f:
            examples = [json.loads(line) for line in f if line.strip()][:args.forward_examples]
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for cell, cfg, step, ckpt in mine:
            names = token_names(ckpt)
            fwd = {}
            if args.forward_examples:
                model, tokenizer = ev.load_model_and_tokenizer(ckpt, args.original_model_name,
                                                               device_map=args.device_map)
                E = model.get_input_embeddings().weight.detach().float().cpu()
            else:
                E = read_embed(os.path.join(ckpt, "model.safetensors"))
            ids = sorted(i for i in names if i < E.shape[0] and (i >= BASE_VOCAB or i in NAMED))
            if args.forward_examples:
                fwd = forward_metrics(model, tokenizer, examples, ids, args.original_model_name)
                del model
                torch.cuda.empty_cache()
            vocab, per = static_metrics(E, E0, ids)
            for i in ids:
                tok = names[i]
                pidx = tok[len("<|predictor_"):-2] if tok.startswith("<|predictor_") else ""
                w.writerow({
                    "cell": os.path.basename(cell.rstrip("/")), "dataset": cfg.get("dataset", ""),
                    "lbd": cfg["lbd"], "k": cfg["k"], "seed": cfg["seed"], "step": step,
                    "token_id": i, "token": tok, "predictor_index": pidx,
                    "predictor_used": (int(pidx) <= int(cfg["k"])) if pidx else "",
                    **vocab, **per[i], **fwd.get(i, {}),
                })
            fh.flush()
            print(f"  {os.path.basename(cell)} step={step}: {len(ids)} tokens")


def merge(args):
    rows = []
    for path in sorted(args.inputs):
        with open(path) as f:
            rows += list(csv.DictReader(f))
    tmp = args.output_file + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, args.output_file)
    print(f"[probe] merged {len(rows)} rows from {len(args.inputs)} files -> {args.output_file}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cells", nargs="*", help="Cell dirs (<runs_dir>/<lbd>_<k>_<seed>)")
    p.add_argument("--runs_dir", default=None)
    p.add_argument("--cell_glob", default="*_*_*")
    p.add_argument("--include_final", action="store_true", help="Also probe checkpoints/model.safetensors")
    p.add_argument("--original_model_name", default="meta-llama/Llama-3.2-1B-Instruct")
    p.add_argument("--input_file", default="datasets/synth_test.jsonl")
    p.add_argument("--forward_examples", type=int, default=0, help="0 = static only (no GPU needed)")
    p.add_argument("--device_map", default="cuda:0")
    p.add_argument("--shard_id", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--output_file", required=True)
    p.add_argument("--merge", action="store_true")
    p.add_argument("--inputs", nargs="*", default=[])
    args = p.parse_args()
    merge(args) if args.merge else run(args)


if __name__ == "__main__":
    main()
