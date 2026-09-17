"""Step-0 scan of the JEPA term over k: the loss, its ingredients, and what makes k=16 special.

The model is built exactly as training starts it (finetune.setup_model_and_tokenizer:
full fine-tune, predictor rows from transformers' mean-resizing) and the JEPA batch is
built by finetune.load_and_prepare_dataset + the training collator and pushed through
RepresentationTrainer.forward / build_with_additive_mask, so the pooled indices, the
packed Code branch and the 4-D additive mask are the training ones, not a copy. No
optimiser step is ever taken.

  1. setup facts  predictor-row geometry; EOS label supervision in the collated batch
  2. k scan       per k: jepa_loss (checked against compute_loss on batch 0), cos,
                  |Text|, |Code|, per-layer cos / norm, top-1 next token at the Text
                  pool, cross-k similarity of the Text embedding
  3. controls     the k-token suffix replaced by k copies of <|predictor_1|>, k copies
                  of an ordinary word, and k distinct ordinary words
  4. gradients    at init, for --grad_ks: |grad jepa|, |grad lm| and the first-order
                  change of log p(<|eot_id|> | last gold token) in the LM branch under
                  one SGD / Adam step on lm + lbd*jepa

MAX_PREDICTORS is raised to --max_predictors (default 64) so k up to 64 stays one
special token per marker; the added rows are the same mean-resized vector either way.
"""

import argparse
import contextlib
import copy
import io
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict

import torch
import torch.nn.functional as F

if not torch.cuda.is_available():  # CPU smoke test: finetune gates prints on current_device()
    torch.cuda.current_device = lambda: 0

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import finetune  # noqa: E402
from transformers import DataCollatorForLanguageModeling  # noqa: E402

WORDS = ("apple river stone music light paper green table window garden number market "
         "silver winter summer forest bridge letter orange doctor animal friend school "
         "island mountain engine flower pencil yellow planet button camera coffee dinner "
         "family finger hammer jacket kitchen ladder mirror needle office pepper rabbit "
         "rocket saddle shadow singer spider sugar temple ticket tunnel valley wagon "
         "wallet weather wizard butter carpet castle cotton dragon empire farmer glass "
         "honey jungle lemon magnet monkey").split()


def parse_ks(spec):
    ks = []
    for part in spec.split(","):
        if "-" in part:
            a, b = map(int, part.split("-"))
            ks.extend(range(a, b + 1))
        else:
            ks.append(int(part))
    return sorted(set(ks))


def pred_suffix(k):
    # finetune appends <|predictor_k|> ... <|predictor_1|>
    return "".join(f"<|predictor_{j}|>" for j in range(k, 0, -1))


class Builder:
    """JEPA datasets via finetune.load_and_prepare_dataset on a temp jsonl."""

    def __init__(self, tok, model_name, examples, tmpdir, max_length):
        self.tok, self.model_name, self.examples = tok, model_name, examples
        self.path = os.path.join(tmpdir, "examples.jsonl")
        self.max_length = max_length

    def build(self, k=0, suffix=""):
        with open(self.path, "w") as f:
            for m in self.examples:
                m = copy.deepcopy(m)
                m[1]["content"] += suffix
                f.write(json.dumps({"messages": m}) + "\n")
        with contextlib.redirect_stdout(io.StringIO()):
            return finetune.load_and_prepare_dataset(self.path, self.tok, self.model_name,
                                                     self.max_length, predictors=k)


def make_trainer_shim():
    t = object.__new__(finetune.RepresentationTrainer)  # methods only, no Trainer state
    t.lbd, t.gamma, t.last_token, t.debug = 1.0, 1.0, -2, 0
    t.additive_mask, t.jepa_l2, t.jepa_mse, t.infonce, t.jepa_ratio = True, False, False, False, -1.0
    # Same function as training, fed CPU copies: on GPU tensors its per-element Python
    # loop costs ~25k device syncs per batch.
    orig = finetune.RepresentationTrainer._last_token_index
    t._last_token_index = lambda ids, lab, am: orig(t, ids.cpu(), lab.cpu(), am.cpu()).to(ids.device)
    return t


def batches(ds, collator, bs, device):
    for s in range(0, len(ds), bs):
        b = collator([ds[i] for i in range(s, min(s + bs, len(ds)))])
        yield {k: v.to(device) for k, v in b.items()}


# Trainer(bf16=True) runs compute_loss under CUDA bf16 autocast. That is what makes
# finetune's float32 additive mask legal for SDPA (bias dtype must match the query),
# and it computes the JEPA cosine in fp32. Every forward here does the same.
def amp():
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def scan(model, *args, **kwargs):
    """_scan with every parameter frozen for its duration. RepresentationTrainer.forward
    wraps the model call in set_grad_enabled(True), which overrides no_grad: with
    trainable parameters a 32x512 scan batch builds a full graph (~85 GB, OOM)."""
    req = [q.requires_grad for q in model.parameters()]
    for q in model.parameters():
        q.requires_grad_(False)
    try:
        return _scan(model, *args, **kwargs)
    finally:
        for q, r in zip(model.parameters(), req):
            q.requires_grad_(r)


@torch.no_grad()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def _scan(model, t, ds, collator, bs, device, eot_id, keep=0, check_loss=False):
    """Per-example Text/Code pooled stats at every layer, from the training forward."""
    acc = defaultdict(list)
    lm_losses = []
    for bi, b in enumerate(batches(ds, collator, bs, device)):
        B = b["input_ids"].shape[0]
        if check_loss and bi == 0:
            t.compute_loss(model, {k: v.clone() for k, v in b.items()})
            acc["compute_loss_jepa"].append(torch.tensor([t._last_jepa_loss]))
        res = t.forward(model, b)
        hs = res["main_outputs"].hidden_states
        lm_losses.append(float(res["main_outputs"].loss))
        ar = torch.arange(B, device=device)
        iu, ia = t._last_token_user, t._last_token_assistant
        assert torch.isfinite(hs[-1][B:][ar, iu]).all() and torch.isfinite(hs[-1][B:][ar, ia]).all(), \
            f"non-finite pooled hidden states (batch {bi})"
        packed = b["input_ids_user"]  # packed (Text | Code) row after build_with_additive_mask
        acc["text_tok"].append(packed[ar, iu].cpu())
        acc["code_tok"].append(packed[ar, ia].cpu())
        cos_l, tn_l, cn_l, tmax_l, tdim_l = [], [], [], [], []
        for h in hs:
            x, y = h[B:][ar, iu].float(), h[B:][ar, ia].float()
            cos_l.append(F.cosine_similarity(x, y, dim=-1))
            tn_l.append(x.norm(dim=-1))
            cn_l.append(y.norm(dim=-1))
            m = x.abs().max(dim=-1)
            tmax_l.append(m.values)
            tdim_l.append(m.indices)
        acc["cos_layers"].append(torch.stack(cos_l, 1).cpu())
        acc["tnorm_layers"].append(torch.stack(tn_l, 1).cpu())
        acc["cnorm_layers"].append(torch.stack(cn_l, 1).cpu())
        acc["tmax_layers"].append(torch.stack(tmax_l, 1).cpu())
        acc["tmaxdim_layers"].append(torch.stack(tdim_l, 1).cpu())
        text_last, code_last = hs[-1][B:][ar, iu], hs[-1][B:][ar, ia]
        acc["cos_train"].append(F.cosine_similarity(text_last, code_last, dim=-1).float().cpu())
        acc["text_next_top1"].append(model.lm_head(text_last).argmax(-1).cpu())
        # LM branch: log p(<|eot_id|>) at the last gold token (k-independent by construction)
        idx = t._last_token_index(b["input_ids"], b["labels"], b["attention_mask"])
        assert bool((b["input_ids"][ar, idx + 1] == eot_id).all()), "last gold token not followed by eot"
        lp = torch.log_softmax(model.lm_head(hs[-1][:B][ar, idx]).float(), -1)
        acc["lm_logp_eot"].append(lp[:, eot_id].cpu())
        acc["lm_eot_rank"].append((lp > lp[:, eot_id:eot_id + 1]).sum(-1).cpu())
        if keep:
            acc["text_vec"].append(text_last.float().cpu())
            acc["code_vec"].append(code_last.float().cpu())
    out = {k: torch.cat(v) for k, v in acc.items()}
    if keep:
        out["text_vec"], out["code_vec"] = out["text_vec"][:keep], out["code_vec"][:keep]
    out["lm_loss"] = sum(lm_losses) / len(lm_losses)
    out["jepa_loss"] = float(1 - out["cos_train"].mean())
    return out


def flat_grad(params):
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in params])


def chunked(fn, *vs, chunk=1 << 26):
    tot = 0.0
    for s in range(0, vs[0].numel(), chunk):
        tot += float(fn(*[v[s:s + chunk].float() for v in vs]))
    return tot


def grads_for(model, t, ds, collator, micro, device, params, eot_id, which):
    """Gradient (bf16, flattened) of the batch-mean of `which` in {'jepa','lm','logp_eot'}."""
    for p in params:
        p.grad = None
    n = math.ceil(len(ds) / micro)
    for b in batches(ds, collator, micro, device):
        B = b["input_ids"].shape[0]
        with amp():
            res = t.forward(model, b)
            hs = res["main_outputs"].hidden_states
            ar = torch.arange(B, device=device)
            if which == "jepa":  # exactly compute_loss's default branch
                x = hs[-1][B:][ar, t._last_token_user]
                y = hs[-1][B:][ar, t._last_token_assistant]
                val = 1.0 - F.cosine_similarity(x, y, dim=-1).mean()
            elif which == "lm":
                val = res["main_outputs"].loss
            else:
                idx = t._last_token_index(b["input_ids"], b["labels"], b["attention_mask"])
                lp = torch.log_softmax(model.lm_head(hs[-1][:B][ar, idx]).float(), -1)
                val = lp[:, eot_id].mean()
        (val / n).backward()
        del res, hs
    g = flat_grad(params).detach().clone()
    for p in params:
        p.grad = None
    return g


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_name", default="meta-llama/Llama-3.2-1B-Instruct")
    p.add_argument("--train_file", default="datasets/synth_train.jsonl")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--n", type=int, default=256, help="examples for the forward scan")
    p.add_argument("--ks", default="0-64")
    p.add_argument("--report_ks", default="0,1,2,4,8,16,32,64")
    p.add_argument("--control_ks", default="0-40,48,56,64")
    p.add_argument("--grad_ks", default="0,1,2,4,8,16,32,64", help="empty string skips gradients")
    p.add_argument("--grad_n", type=int, default=64, help="one optimizer step = 4 GPUs x bs 4 x accum 4")
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--micro", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--lbds", default="0,0.5,1,2,4,8")
    p.add_argument("--seed", type=int, default=82)
    p.add_argument("--max_predictors", type=int, default=64)
    p.add_argument("--max_length", type=int, default=512)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    ks, report_ks, control_ks = parse_ks(args.ks), parse_ks(args.report_ks), parse_ks(args.control_ks)
    grad_ks = parse_ks(args.grad_ks) if args.grad_ks else []
    lbds = [float(x) for x in args.lbds.split(",")]

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    finetune.MAX_PREDICTORS = args.max_predictors
    model, tok = finetune.setup_model_and_tokenizer(args.model_name, use_lora=False, seed=args.seed)
    model.eval()
    for q in model.parameters():
        q.requires_grad_(False)
    devices = {q.device for q in model.parameters()}
    assert len(devices) == 1, f"model sharded over {devices}; expose one GPU (CUDA_VISIBLE_DEVICES)"
    device = devices.pop()
    eot_id = tok.convert_tokens_to_ids("<|eot_id|>")
    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False, pad_to_multiple_of=None)
    t = make_trainer_shim()
    summary = {"args": vars(args), "device": str(device)}

    with open(args.train_file) as f:
        all_ex = [json.loads(line)["messages"] for line in f]
    examples = random.Random(args.seed).sample(all_ex, args.n)
    bld = Builder(tok, args.model_name, examples, args.out_dir, args.max_length)

    # ---------------- 1. setup facts ----------------
    print("=" * 100 + "\n1. SETUP FACTS\n" + "=" * 100)
    E = model.get_input_embeddings().weight.float()
    pids = tok.convert_tokens_to_ids([f"<|predictor_{i}|>" for i in range(1, args.max_predictors + 1)])
    P = E[pids]
    Pn = F.normalize(P, dim=-1)
    off = Pn @ Pn.T
    vocab_med = E[:128000].norm(dim=-1).median().item()
    mean_row = E[:128000].mean(0)
    facts = {
        "vocab_rows": E.shape[0], "pad_token": tok.pad_token, "pad_id": tok.pad_token_id, "eot_id": eot_id,
        "pred_norm_mean": P.norm(dim=-1).mean().item(), "vocab_norm_median": vocab_med,
        "pred_pairwise_cos_min": off.min().item(),
        "pred_rows_bitwise_identical": bool((P == P[0]).all()),
        "pred_cos_to_vocab_mean": F.cosine_similarity(P, mean_row[None], dim=-1).mean().item(),
        "lm_head_tied": model.lm_head.weight.data_ptr() == model.get_input_embeddings().weight.data_ptr(),
    }
    ds0 = bld.build(k=0)
    b = next(batches(ds0, collator, len(ds0), "cpu"))
    lab, ids = b["labels"], b["input_ids"]
    real = b["attention_mask"].bool()
    facts["eot_tokens_in_real_positions"] = int(((ids == eot_id) & real).sum())
    facts["eot_tokens_with_label"] = int(((ids == eot_id) & (lab != -100)).sum())
    facts["labelled_tokens_per_example"] = float((lab != -100).sum(1).float().mean())
    ds_labels = torch.tensor([ds0[i]["labels"] for i in range(len(ds0))])
    facts["dataset_answer_tokens_per_example"] = float((ds_labels != -100).sum(1).float().mean())
    summary["setup"] = facts
    for k_, v in facts.items():
        print(f"  {k_:38s} {v}")
    del E, P, Pn, off

    # suffix path must reproduce finetune's predictor insertion exactly
    for k in (1, 16):
        a, c = bld.build(k=k), bld.build(k=0, suffix=pred_suffix(k))
        assert a[0]["input_ids_user"] == c[0]["input_ids_user"], f"suffix path != finetune at k={k}"

    # ---------------- 2. k scan ----------------
    print("\n" + "=" * 100 + "\n2. k SCAN (predictors as trained)\n" + "=" * 100)
    scans = {}
    for k in ks:
        scans[k] = scan(model, t, bld.build(k=k), collator, args.bs, device, eot_id,
                        keep=64, check_loss=(k in report_ks))
    L = scans[ks[0]]["cos_layers"].shape[1] - 1

    print(f"\n{'k':>3} {'jepa_loss':>10} {'cos_mean':>8} {'cos_p10':>8} {'cos_min':>8} "
          f"{'|Text|':>7} {'|Code|':>7} {'lm_loss':>7} {'text pool tok':>18} {'top1 next @Text':>22}")
    rows = {}
    for k in ks:
        s = scans[k]
        c = s["cos_train"]
        tt = Counter(tok.convert_ids_to_tokens(s["text_tok"].tolist())).most_common(1)[0]
        nx = Counter(tok.convert_ids_to_tokens(s["text_next_top1"].tolist())).most_common(1)[0]
        rows[k] = {"jepa_loss": s["jepa_loss"], "cos_mean": float(c.mean()), "cos_p10": float(c.quantile(0.1)),
                   "cos_min": float(c.min()), "text_norm": float(s["tnorm_layers"][:, -1].mean()),
                   "code_norm": float(s["cnorm_layers"][:, -1].mean()), "lm_loss": s["lm_loss"],
                   "text_pool_tok": tt, "text_next_top1": nx,
                   "cos_layers_mean": s["cos_layers"].mean(0).tolist(),
                   "tnorm_layers_mean": s["tnorm_layers"].mean(0).tolist(),
                   "tmax_layers_mean": s["tmax_layers"].mean(0).tolist(),
                   "tmaxdim_layers_mode": [Counter(s["tmaxdim_layers"][:, l].tolist()).most_common(1)[0]
                                           for l in range(L + 1)]}
        if "compute_loss_jepa" in s:
            rows[k]["compute_loss_jepa_batch0"] = float(s["compute_loss_jepa"][0])
            rows[k]["mine_jepa_batch0"] = float(1 - s["cos_train"][:args.bs].mean())
        print(f"{k:>3} {s['jepa_loss']:>10.4f} {rows[k]['cos_mean']:>8.4f} {rows[k]['cos_p10']:>8.4f} "
              f"{rows[k]['cos_min']:>8.4f} {rows[k]['text_norm']:>7.2f} {rows[k]['code_norm']:>7.2f} "
              f"{s['lm_loss']:>7.3f} {tt[0]!s:>14}x{tt[1]:<3} {nx[0]!r:>16}x{nx[1]:<3}")
    summary["scan"] = {str(k): v for k, v in rows.items()}

    print("\ncompute_loss (training path) vs this script's jepa on batch 0:")
    for k in report_ks:
        if k in rows:
            print(f"  k={k:<3} compute_loss={rows[k]['compute_loss_jepa_batch0']:.6f}  "
                  f"here={rows[k]['mine_jepa_batch0']:.6f}")

    k0 = ks[0]
    lpe = scans[k0]["lm_logp_eot"]
    summary["lm_eot_at_init"] = {"logp_mean": float(lpe.mean()), "p_median": float(lpe.exp().median()),
                                 "top1_frac": float((scans[k0]["lm_eot_rank"] == 0).float().mean()),
                                 "k_independent": all(torch.allclose(scans[k]["lm_logp_eot"], lpe, atol=1e-3)
                                                      for k in ks)}
    print(f"\nLM branch at init: log p(eot | last gold) mean {summary['lm_eot_at_init']['logp_mean']:.3f}, "
          f"eot top-1 {summary['lm_eot_at_init']['top1_frac']:.3f}, "
          f"k-independent: {summary['lm_eot_at_init']['k_independent']}")

    print("\nIngredients, example 0 (last layer, first 6 dims):")
    for k in report_ks:
        if k in scans:
            tv, cv = scans[k]["text_vec"][0], scans[k]["code_vec"][0]
            print(f"  k={k:<3} Text {[round(v, 3) for v in tv[:6].tolist()]} |T|={tv.norm():.2f}   "
                  f"Code {[round(v, 3) for v in cv[:6].tolist()]} |C|={cv.norm():.2f}   "
                  f"cos={F.cosine_similarity(tv, cv, dim=0):.4f}")

    print("\nPer-layer mean cos(Text, Code):")
    hdr = "layer " + " ".join(f"{'k=' + str(k):>7}" for k in report_ks if k in rows)
    print(hdr)
    for l in range(1, L + 1):
        print(f"{l:>5} " + " ".join(f"{rows[k]['cos_layers_mean'][l]:>7.3f}" for k in report_ks if k in rows))
    print("\nPer-layer mean |Text| (layer L is post-final-norm) and max|x_d| (dim):")
    print(hdr)
    for l in range(1, L + 1):
        print(f"{l:>5} " + " ".join(f"{rows[k]['tnorm_layers_mean'][l]:>7.1f}" for k in report_ks if k in rows)
              + "   | " + " ".join(f"{rows[k]['tmax_layers_mean'][l]:.0f}({rows[k]['tmaxdim_layers_mode'][l][0]})"
                              for k in report_ks if k in rows))

    # cross-k similarity of the Text embedding itself
    V = torch.stack([F.normalize(scans[k]["text_vec"], dim=-1) for k in ks])  # [K, 64, D]
    M = torch.einsum("aid,bid->abi", V, V).mean(-1)
    summary["text_cross_k_cos"] = {"ks": ks, "matrix": M.tolist()}
    print("\nText embedding vs its neighbours in k (mean cos over 64 examples):")
    for i, k in enumerate(ks):
        nb = [f"{M[i, j]:.3f}" for j in (i - 1, i + 1) if 0 <= j < len(ks)]
        ref = {r: M[i, ks.index(r)].item() for r in (8, 32) if r in ks}
        print(f"  k={k:<3} prev/next {'/'.join(nb):<13} vs k=8 {ref.get(8, float('nan')):.3f}  "
              f"vs k=32 {ref.get(32, float('nan')):.3f}")
    torch.save({k: {"text_vec": scans[k]["text_vec"], "code_vec": scans[k]["code_vec"],
                    "cos_train": scans[k]["cos_train"], "cos_layers": scans[k]["cos_layers"]} for k in ks},
               os.path.join(args.out_dir, "scan_vectors.pt"))

    # ---------------- 3. controls ----------------
    print("\n" + "=" * 100 + "\n3. CONTROLS: what the k-token suffix is made of\n" + "=" * 100)
    single = [w for w in WORDS if len(tok.encode(" " + w, add_special_tokens=False)) == 1]
    assert len(single) >= max(control_ks), f"only {len(single)} single-token words"
    variants = {
        "pred": pred_suffix,
        "same_pred1": lambda k: "<|predictor_1|>" * k,
        "rep_the": lambda k: " the" * k,
        "distinct_words": lambda k: "".join(" " + w for w in single[:k]),
    }
    base_len = len(tok.encode(examples[0][1]["content"], add_special_tokens=False))
    for name, fn in variants.items():
        n_tok = len(tok.encode(examples[0][1]["content"] + fn(24), add_special_tokens=False)) - base_len
        assert n_tok == 24, f"{name}: suffix of 24 gave {n_tok} tokens"
    ctrl = {}
    for name, fn in variants.items():
        ctrl[name] = {}
        for k in control_ks:
            if name == "pred" and k in scans:
                s = scans[k]
            else:
                s = scan(model, t, bld.build(k=0, suffix=fn(k)), collator, args.bs, device, eot_id)
            ctrl[name][k] = {"jepa_loss": s["jepa_loss"], "text_norm_prefinal": float(s["tnorm_layers"][:, -2].mean()),
                             "cos_layers_mean": s["cos_layers"].mean(0).tolist()}
    summary["controls"] = {n: {str(k): v for k, v in d.items()} for n, d in ctrl.items()}
    print(f"\njepa_loss at init (1 - mean cos, last layer) by suffix type:")
    print(f"{'k':>3} " + " ".join(f"{n:>15}" for n in variants))
    for k in control_ks:
        print(f"{k:>3} " + " ".join(f"{ctrl[n][k]['jepa_loss']:>15.4f}" for n in variants))
    print(f"\n|Text| at layer {L - 1} (pre-final-norm residual) by suffix type:")
    print(f"{'k':>3} " + " ".join(f"{n:>15}" for n in variants))
    for k in control_ks:
        print(f"{k:>3} " + " ".join(f"{ctrl[n][k]['text_norm_prefinal']:>15.1f}" for n in variants))

    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=str)

    # ---------------- 4. gradients ----------------
    if not grad_ks:
        return
    print("\n" + "=" * 100 + "\n4. GRADIENTS AT INIT: does the JEPA pull point against EOS?\n" + "=" * 100)
    for q in model.parameters():
        q.requires_grad_(True)
    params = [q for q in model.parameters() if q.requires_grad]
    gsub = examples[:args.grad_n]
    gbld = Builder(tok, args.model_name, gsub, args.out_dir, args.max_length)
    gE = grads_for(model, t, gbld.build(k=0), collator, args.micro, device, params, eot_id, "logp_eot")
    nE = math.sqrt(chunked(lambda a: (a * a).sum(), gE))
    grows = {}
    print(f"\n|grad log p_eot| = {nE:.3e}   (Delta = first-order change in mean log p(eot|last gold), lr={args.lr})")
    print(f"{'k':>3} {'jepa':>7} {'|gJ|':>9} {'|gL|':>9} {'cos(gJ,gE)':>10} {'cos(gL,gE)':>10} {'cos(gJ,gL)':>10}  "
          + " ".join(f"{'Adam l=' + str(l):>11}" for l in lbds))
    for k in grad_ks:
        ds = gbld.build(k=k)
        gJ = grads_for(model, t, ds, collator, args.micro, device, params, eot_id, "jepa")
        gL = grads_for(model, t, ds, collator, args.micro, device, params, eot_id, "lm")
        with torch.no_grad():
            jl = scan(model, t, ds, collator, args.bs, device, eot_id)["jepa_loss"]
        nJ = math.sqrt(chunked(lambda a: (a * a).sum(), gJ))
        nL = math.sqrt(chunked(lambda a: (a * a).sum(), gL))
        dJE = chunked(lambda a, b: (a * b).sum(), gJ, gE)
        dLE = chunked(lambda a, b: (a * b).sum(), gL, gE)
        dJL = chunked(lambda a, b: (a * b).sum(), gJ, gL)
        adam = {l: -args.lr * chunked(lambda j, g, e: (torch.sign(g + l * j) * e).sum(), gJ, gL, gE) for l in lbds}
        sgd = {l: -args.lr * (dLE + l * dJE) for l in lbds}
        grows[k] = {"jepa_loss": jl, "gJ_norm": nJ, "gL_norm": nL, "gE_norm": nE,
                    "cos_gJ_gE": dJE / (nJ * nE + 1e-30), "cos_gL_gE": dLE / (nL * nE + 1e-30),
                    "cos_gJ_gL": dJL / (nJ * nL + 1e-30), "delta_logp_eot_adam": adam, "delta_logp_eot_sgd": sgd}
        r = grows[k]
        print(f"{k:>3} {jl:>7.4f} {nJ:>9.3e} {nL:>9.3e} {r['cos_gJ_gE']:>10.4f} {r['cos_gL_gE']:>10.4f} "
              f"{r['cos_gJ_gL']:>10.4f}  " + " ".join(f"{adam[l]:>11.2e}" for l in lbds))
        del gJ, gL
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    summary["gradients"] = {str(k): v for k, v in grows.items()}
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=str)


if __name__ == "__main__":
    main()
