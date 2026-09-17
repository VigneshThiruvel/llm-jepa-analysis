"""Does the first optimizer step flip the Text pool into the repeated-token regime?

Stage 3 replay (λ=2): k<=14 converge (step-10 jepa_loss <=0.007), but k=15 *rises* after
the first update (0.918 -> 0.982) and is still 0.30 at step 10. Stage 2 (step 0): the
pooled predictor state is in regime A for k<=15 and switches to regime B — a layer-2
massive activation, cos(Text, Code) ~0.02 at layer 2 — over k~18-24. Hypothesis: the
first, clipped update moves that boundary below 15.

Test: the full-FT model at init, then a few real AdamW steps on lm + λ*jepa (lr 2e-5,
clip 1.0, 64 examples/step = 4 GPUs x bs 4 x accum 4, no weight decay, as the Trainer
defaults), and after each step the regime fraction, jepa_loss and norms of the Text
pool on an eval sample disjoint from the training examples — per train k. Batches go
through the same finetune/k_scan_init path (collator, RepresentationTrainer.forward,
bf16 autocast).
"""

import argparse
import json
import math
import os
import random
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import k_scan_init as ksi  # noqa: E402  (puts the repo root on sys.path too)
import finetune  # noqa: E402
from transformers import DataCollatorForLanguageModeling  # noqa: E402


def regime_stats(s):
    t2 = s["tnorm_layers"][:, 2]
    return {"jepa_loss": s["jepa_loss"], "cos_mean": float(s["cos_train"].mean()),
            "fracB_norm": float((t2 > 10).float().mean()),          # layer-2 |Text| ~2-3 (A) vs ~21 (B)
            "fracB_cos": float((s["cos_layers"][:, 2] < 0.07).float().mean()),
            "text_norm_L2": float(t2.mean()), "text_norm_L16": float(s["tnorm_layers"][:, -1].mean()),
            # LM branch, teacher-forced: does <|eot_id|> win right after the gold answer?
            "lm_logp_eot": float(s["lm_logp_eot"].mean()),
            "lm_eot_top1": float((s["lm_eot_rank"] == 0).float().mean())}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_name", default="meta-llama/Llama-3.2-1B-Instruct")
    p.add_argument("--train_file", default="datasets/synth_train.jsonl")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--ks", default="8,14,15,16,17,20,24,32")
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--lbd", type=float, default=2.0)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--step_n", type=int, default=64)
    p.add_argument("--micro", type=int, default=4)
    p.add_argument("--n_eval", type=int, default=256)
    p.add_argument("--eval_every", type=int, default=1, help="eval at step 0, every N steps and the last")
    p.add_argument("--total_steps", type=int, default=500,
                   help="linear lr decay horizon, as the Trainer (4 epochs x 125 steps, no warmup)")
    p.add_argument("--seed", type=int, default=82)
    p.add_argument("--max_length", type=int, default=512)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    ks = ksi.parse_ks(args.ks)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    model, tok = finetune.setup_model_and_tokenizer(args.model_name, use_lora=False, seed=args.seed)
    model.eval()  # Llama has no dropout; keeps scan and step identical in mode
    devices = {q.device for q in model.parameters()}
    assert len(devices) == 1, f"model sharded over {devices}; expose one GPU (CUDA_VISIBLE_DEVICES)"
    device = devices.pop()
    init_state = {n: v.detach().cpu().clone() for n, v in model.state_dict().items()}
    eot_id = tok.convert_tokens_to_ids("<|eot_id|>")
    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False, pad_to_multiple_of=None)
    t = ksi.make_trainer_shim()

    with open(args.train_file) as f:
        all_ex = [json.loads(line)["messages"] for line in f]
    eval_idx = random.Random(args.seed).sample(range(len(all_ex)), args.n_eval)  # = k_scan_init's sample
    held = set(eval_idx)
    train_idx = [i for i in range(len(all_ex)) if i not in held]
    random.Random(args.seed + 1).shuffle(train_idx)
    eval_bld = ksi.Builder(tok, args.model_name, [all_ex[i] for i in eval_idx], args.out_dir, args.max_length)

    results = {}
    for k in ks:
        model.load_state_dict(init_state)
        for q in model.parameters():
            q.requires_grad_(True)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 1.0 - s / args.total_steps))
        eval_ds = eval_bld.build(k=k)
        rows = [regime_stats(ksi.scan(model, t, eval_ds, collator, 16, device, eot_id))]
        rows[0].update(step=0)
        print(f"k={k:<3} step 0  jepa {rows[0]['jepa_loss']:.4f}  fracB(norm) {rows[0]['fracB_norm']:.2f}  "
              f"|Text|L2 {rows[0]['text_norm_L2']:.1f}  eot@1 {rows[0]['lm_eot_top1']:.3f} "
              f"logp_eot {rows[0]['lm_logp_eot']:.3f}", flush=True)
        for step in range(1, args.steps + 1):
            ex = [all_ex[train_idx[j % len(train_idx)]]  # cycles past one pass over the pool
                  for j in range((step - 1) * args.step_n, step * args.step_n)]
            ds = ksi.Builder(tok, args.model_name, ex, args.out_dir, args.max_length).build(k=k)
            opt.zero_grad(set_to_none=True)
            n = math.ceil(len(ds) / args.micro)
            lm_acc = j_acc = 0.0
            for b in ksi.batches(ds, collator, args.micro, device):
                B = b["input_ids"].shape[0]
                with ksi.amp():
                    res = t.forward(model, b)
                    hs = res["main_outputs"].hidden_states
                    ar = torch.arange(B, device=device)
                    x = hs[-1][B:][ar, t._last_token_user]
                    y = hs[-1][B:][ar, t._last_token_assistant]
                    jl = 1.0 - F.cosine_similarity(x, y, dim=-1).mean()
                    lm = res["main_outputs"].loss
                    loss = (lm + args.lbd * jl) / n
                loss.backward()
                lm_acc += lm.item() / n
                j_acc += jl.item() / n
                del res, hs
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            opt.step()
            sched.step()
            if step % args.eval_every and step != args.steps:
                print(f"k={k:<3} step {step}  train jepa {j_acc:.4f} lm {lm_acc:.3f}  gn {float(gn):.1f}", flush=True)
                continue
            r = regime_stats(ksi.scan(model, t, eval_ds, collator, 16, device, eot_id))
            r.update(step=step, train_lm=lm_acc, train_jepa=j_acc, grad_norm=float(gn))
            rows.append(r)
            print(f"k={k:<3} step {step}  jepa {r['jepa_loss']:.4f}  fracB(norm) {r['fracB_norm']:.2f}  "
                  f"fracB(cos) {r['fracB_cos']:.2f}  |Text|L2 {r['text_norm_L2']:.1f}  "
                  f"|Text|L16 {r['text_norm_L16']:.1f}  eot@1 {r['lm_eot_top1']:.3f} logp_eot {r['lm_logp_eot']:.3f}  "
                  f"train jepa {j_acc:.4f} lm {lm_acc:.3f}  gn {float(gn):.1f}",
                  flush=True)
        results[k] = rows
        del opt
        torch.cuda.empty_cache()

    evald = [r["step"] for r in results[ks[0]]]
    for name, key, fmt in (("eval jepa_loss", "jepa_loss", "{:>7.4f}"),
                           ("fracB (layer-2 |Text| > 10)", "fracB_norm", "{:>7.2f}"),
                           ("eot top-1 after the gold answer (LM branch, teacher-forced)", "lm_eot_top1", "{:>7.3f}")):
        print(f"\n{name}, lbd={args.lbd}")
        print(f"{'k':>3} " + " ".join(f"{'s' + str(s):>7}" for s in evald))
        for k in ks:
            print(f"{k:>3} " + " ".join(fmt.format(r[key]) for r in results[k]))
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump({"args": vars(args), "results": {str(k): v for k, v in results.items()}}, f, indent=1)


if __name__ == "__main__":
    main()
