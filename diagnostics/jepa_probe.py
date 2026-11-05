"""Why does JEPA do nothing? Phase A probes on finished checkpoints (no training).

Two subcommands, both read-only on the cells they probe:

  embed   A1 + A2, one checkpoint. Re-embeds Text and Code exactly as the JEPA term
          saw them in training (finetune.py, --last_token=-2): Text = the user turn
          alone with k <|predictor_i|> markers appended, Code = the assistant turn
          alone, each through the chat template with the training date pinned and
          tokenized with add_special_tokens=True (double BOS, as finetune.py does),
          pooled at unpadded length - 2. That is <|predictor_1|> for k>0 and the last
          content token for k=0; the next token at both pooled positions is
          <|eot_id|>. Two views per cell:
            train  the cell's own k (what its loss aligned)
            raw    k=0 for every cell (comparable across cells and the base model)
          The --additive_mask packing puts Code after Text in one row under a
          block mask; RoPE is relative, so a separate pass is the same computation.
          Per layer (hidden_states[l]; l=16 is post-norm, the state JEPA aligns):
            pos_cos      mean cos(T_i, C_i)             (what 1 - jepa_loss reports)
            neg_cos      mean cos(T_i, C_j), i != j     (collapse: neg_cos -> pos_cos)
            gap          pos_cos - neg_cos
            tt_cos/cc_cos  mean pairwise cos within Text / within Code
            r1_t2c       top-1 retrieval of C_i among all C by cos(T_i, .)  (chance 1/n)
            *_ctr        the same after subtracting each side's mean vector
            rankme_t/c   effective rank (geometry.rankme)
          A2, last layer, at the pooled positions: next-token entropy, p(eot),
          argmax == eot. If the target there is ~always eot at ~zero entropy, the
          last-layer state at that position is free to collapse: in a decoder the
          final-layer state at position t reaches only the logits at t.

  pairs   A3, one shard of a list of checkpoint pairs (a, b). On the test set,
          teacher-forced over eval prompt + gold + <|eot_id|> (as eval_generations.py):
            kl_ab / kl_ba    mean per-token KL over gold tokens + the final eot
            argmax_disagree  fraction of those positions where the argmaxes differ
            nll_a / nll_b    mean gold-token NLL (A4, answer-only held-out loss)
          Weights: ||theta_a - theta_b|| per decoder layer / embeddings / final
          norm, and relative to ||theta_a||. Generations: from each cell's
          eval/generations.jsonl, exact-match agreement and response-string
          agreement (no GPU needed for those).

  python3 diagnostics/jepa_probe.py embed --cell_dir=<cell> --k=<k> --out=<jsonl>
  python3 diagnostics/jepa_probe.py pairs --pairs_file=<tsv> --shard_id=0 --num_shards=4 --out=<jsonl>

The driver is diagnostics/run_jepa_probe.sh. finetune.py, evaluate.py and
geometry.py are imported, never modified.
"""

import argparse
import glob
import json
import os
import sys

import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

if not torch.cuda.is_available():
    # evaluate.load_model_and_tokenizer gates a print on torch.cuda.current_device()
    torch.cuda.current_device = lambda: 0

import evaluate as ev  # noqa: E402
from geometry import rankme  # noqa: E402

MODEL = "meta-llama/Llama-3.2-1B-Instruct"
DATE = "22 Sep 2026"  # stage-5 train and eval date


def load_rows(path, n=None):
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return rows[:n] if n else rows


def load(cell, device):
    path = MODEL if cell == "base" else os.path.join(cell, "checkpoints")
    model, tok = ev.load_model_and_tokenizer(path, MODEL, device_map=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


# ---------------------------------------------------------------- embed (A1, A2)

def branch_ids(tok, messages, k, side):
    if side == "text":
        msgs = [dict(m) for m in messages[1:2]]  # finetune.get_user_messages
        for i in range(k, 0, -1):                # front_pred=False: k..1 appended
            msgs[0]["content"] += f"<|predictor_{i}|>"
    else:
        msgs = messages[2:3]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False,
                                   date_string=DATE)
    return tok(text, add_special_tokens=True)["input_ids"]


@torch.no_grad()
def pooled_states(model, tok, seqs, batch_size=32, last_token=-2, pool="last"):
    """[N, L+1, D] states at len+last_token, plus next-token stats from the last layer.

    pool="mean" instead averages each layer over the branch's content span, from 2 tokens
    after its last <|end_header_id|> through len+last_token (finetune_jepa_obj --jepa_pool=mean).
    """
    eot = tok.convert_tokens_to_ids("<|eot_id|>")
    states, ent, p_eot, top_eot = [], [], [], []
    for s in range(0, len(seqs), batch_size):
        chunk = seqs[s:s + batch_size]
        T = max(len(x) for x in chunk)
        ids = torch.zeros(len(chunk), T, dtype=torch.long)
        mask = torch.zeros(len(chunk), T, dtype=torch.long)
        for i, x in enumerate(chunk):
            ids[i, :len(x)] = torch.tensor(x)
            mask[i, :len(x)] = 1
        idx = torch.tensor([len(x) + last_token for x in chunk])
        out = model(input_ids=ids.to(model.device), attention_mask=mask.to(model.device),
                    output_hidden_states=True)
        r = torch.arange(len(chunk))
        if pool == "mean":
            hdr = tok.convert_tokens_to_ids("<|end_header_id|>")
            spans = []
            for x, e in zip(chunk, idx.tolist()):
                h_pos = [j for j in range(e + 1) if x[j] == hdr]
                spans.append((min(h_pos[-1] + 2, e) if h_pos else e, e))
            states.append(torch.stack([torch.stack([h[i, a:b + 1].mean(0) for i, (a, b) in enumerate(spans)])
                                       for h in out.hidden_states], 1).float().cpu())
        else:
            states.append(torch.stack([h[r, idx.to(h.device)] for h in out.hidden_states], 1).float().cpu())
        logp = out.logits[r, idx.to(out.logits.device)].float().log_softmax(-1)
        ent.append(-(logp.exp() * logp).sum(-1).cpu())
        p_eot.append(logp[:, eot].exp().cpu())
        top_eot.append((logp.argmax(-1) == eot).float().cpu())
    return torch.cat(states), torch.cat(ent), torch.cat(p_eot), torch.cat(top_eot)


def offdiag_mean(S):
    n = S.shape[0]
    return float((S.sum() - S.diagonal().sum()) / (n * (n - 1)))


def pair_metrics(X, Y, suffix=""):
    Xn, Yn = F.normalize(X, dim=-1), F.normalize(Y, dim=-1)
    S = Xn @ Yn.T
    pos = float(S.diagonal().mean())
    neg = offdiag_mean(S)
    r1 = float((S.argmax(1) == torch.arange(len(X), device=X.device)).float().mean())
    return {f"pos_cos{suffix}": pos, f"neg_cos{suffix}": neg, f"gap{suffix}": pos - neg,
            f"tt_cos{suffix}": offdiag_mean(Xn @ Xn.T), f"cc_cos{suffix}": offdiag_mean(Yn @ Yn.T),
            f"r1_t2c{suffix}": r1}


def done_keys(out):
    """(dataset, cell, split, view) already complete in any embed.*.jsonl next to `out`."""
    count = {}
    for p in glob.glob(os.path.join(os.path.dirname(out) or ".", "embed.*.jsonl")):
        with open(p) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:  # torn last line of a killed run
                    continue
                key = (r["dataset"], r["cell"], r["split"], r["view"])
                count[key] = count.get(key, 0) + 1
    return {k for k, v in count.items() if v >= 16}


def cmd_embed(a):
    meta = {"cell": os.path.basename(a.cell_dir.rstrip("/")), "dataset": a.dataset,
            "arm": a.arm, "lbd": a.lbd, "k": a.k, "seed": a.seed}
    sfx = "" if a.pool == "last" else f"-{a.pool}"
    views = [("train" + sfx, a.k)] + ([("raw" + sfx, 0)] if a.k > 0 else [])
    done = done_keys(a.out)
    todo = [(s, v) for s in ("test", "train") for v in views
            if (a.dataset, meta["cell"], s, v[0]) not in done]
    if not todo:
        print(f"[embed] {meta['cell']} ({a.dataset}): all splits/views done, skipping")
        return
    model, tok = load(a.cell_dir, a.device)
    with open(a.out, "a") as f:
        for split, path in [("test", a.test_file), ("train", a.train_file)]:
            rows = load_rows(path, a.n)
            for view, k in views:
                if (split, (view, k)) not in todo:
                    continue
                T, Te, Tp, Tt = pooled_states(model, tok, [branch_ids(tok, r["messages"], k, "text") for r in rows], pool=a.pool)
                C, Ce, Cp, Ct = pooled_states(model, tok, [branch_ids(tok, r["messages"], k, "code") for r in rows], pool=a.pool)
                for layer in range(1, T.shape[1]):
                    X, Y = T[:, layer].to(model.device), C[:, layer].to(model.device)  # metrics on GPU
                    row = {**meta, "split": split, "view": view, "k_view": k, "layer": layer, "n": len(rows),
                           **pair_metrics(X, Y), **pair_metrics(X - X.mean(0), Y - Y.mean(0), "_ctr"),
                           "rankme_t": rankme(X), "rankme_c": rankme(Y),
                           "norm_t": float(X.norm(dim=-1).mean()), "norm_c": float(Y.norm(dim=-1).mean())}
                    if layer == T.shape[1] - 1:
                        row.update({"next_entropy_t": float(Te.mean()), "p_eot_t": float(Tp.mean()),
                                    "top1_eot_t": float(Tt.mean()), "next_entropy_c": float(Ce.mean()),
                                    "p_eot_c": float(Cp.mean()), "top1_eot_c": float(Ct.mean())})
                    f.write(json.dumps(row) + "\n")
                f.flush()
                print(f"[embed] {meta['cell']} {split} {view}: last-layer pos {row['pos_cos']:.4f} "
                      f"neg {row['neg_cos']:.4f} r1 {row['r1_t2c']:.3f} r1_ctr {row['r1_t2c_ctr']:.3f} "
                      f"rank {row['rankme_t']:.1f}/{row['rankme_c']:.1f} H_t {row['next_entropy_t']:.3f} "
                      f"p_eot_t {row['p_eot_t']:.3f}", flush=True)


# ---------------------------------------------------------------- pairs (A3, A4)

@torch.no_grad()
def answer_logprobs(model, batch, vocab):
    """batch: list of (ids, n_prompt). Returns list of [n_answer+1, vocab] float32 log-probs.

    vocab = the rows both models share: the untrained base model lacks the 33 rows
    finetune.py adds (<|predictor_i|> etc.), which no gold sequence contains.
    """
    T = max(len(x) for x, _ in batch)
    ids = torch.zeros(len(batch), T, dtype=torch.long)
    mask = torch.zeros(len(batch), T, dtype=torch.long)
    for i, (x, _) in enumerate(batch):
        ids[i, :len(x)] = torch.tensor(x)
        mask[i, :len(x)] = 1
    logits = model(input_ids=ids.to(model.device), attention_mask=mask.to(model.device)).logits
    # row j predicts token j+1: positions n_prompt-1 .. len-2 predict gold + eot
    return [logits[i, n - 1:len(x) - 1, :vocab].float().log_softmax(-1) for i, (x, n) in enumerate(batch)]


def tf_sequences(tok, rows):
    eot = tok.convert_tokens_to_ids("<|eot_id|>")
    seqs = []
    for r in rows:
        m = r["messages"]
        prompt = tok.apply_chat_template([x for x in m if x["role"] != "assistant"], tokenize=False,
                                         add_generation_prompt=True, date_string=DATE)
        p = tok(prompt, add_special_tokens=True)["input_ids"]
        g = tok(m[2]["content"], add_special_tokens=False)["input_ids"]
        seqs.append((p + g + [eot], len(p)))
    return seqs


def layer_of(name):
    if ".layers." in name:
        return f"L{int(name.split('.layers.')[1].split('.')[0]):02d}"
    if "embed_tokens" in name:
        return "embed"
    return "norm"


@torch.no_grad()
def weight_dist(ma, mb):
    pa, pb = dict(ma.named_parameters()), dict(mb.named_parameters())
    d2, n2 = {}, {}
    for name, A in pa.items():
        B = pb[name]
        if A.shape != B.shape:  # base model has fewer embedding rows: compare the shared ones
            r = min(A.shape[0], B.shape[0])
            A, B = A[:r], B[:r]
        g = layer_of(name)
        d2[g] = d2.get(g, 0.0) + float((A.float() - B.float()).pow(2).sum())
        n2[g] = n2.get(g, 0.0) + float(A.float().pow(2).sum())
    tot_d, tot_n = sum(d2.values()), sum(n2.values())
    return {"dtheta": tot_d ** 0.5, "dtheta_rel": (tot_d / tot_n) ** 0.5,
            "dtheta_by_layer": {g: (d2[g] / n2[g]) ** 0.5 for g in sorted(d2)}}


def gen_agreement(ca, cb):
    pa = os.path.join(ca, "eval", "generations.jsonl")
    pb = os.path.join(cb, "eval", "generations.jsonl")
    if ca == "base" or not (os.path.exists(pa) and os.path.exists(pb)):
        return {}
    A = {r["idx"]: r for r in load_rows(pa)}
    B = {r["idx"]: r for r in load_rows(pb)}
    common = sorted(set(A) & set(B))
    n = len(common)
    both = sum(A[i]["match"] and B[i]["match"] for i in common)
    a_only = sum(A[i]["match"] and not B[i]["match"] for i in common)
    b_only = sum(B[i]["match"] and not A[i]["match"] for i in common)
    same = sum(A[i]["response"] == B[i]["response"] for i in common)
    return {"gen_n": n, "acc_a": sum(A[i]["match"] for i in common) / n,
            "acc_b": sum(B[i]["match"] for i in common) / n,
            "both_right": both / n, "a_only": a_only / n, "b_only": b_only / n,
            "match_agree": (n - a_only - b_only) / n, "response_same": same / n}


def cmd_pairs(a):
    pairs = [line.rstrip("\n").split("\t") for line in open(a.pairs_file) if line.strip()]
    mine = [p for i, p in enumerate(pairs) if i % a.num_shards == a.shard_id]
    done = set()
    if os.path.exists(a.out):
        done = {(r["a"], r["b"]) for r in load_rows(a.out)}
    print(f"[pairs] shard {a.shard_id}/{a.num_shards}: {len(mine)} pairs, {len(done)} done in {a.out}")
    cache = {}
    for kind, dataset, ca, cb in mine:
        if (ca, cb) in done:
            continue
        rows = load_rows(a.test_file if dataset == "synth" else a.test_file.replace("synth", dataset), a.n)
        for c in (ca, cb):
            if c not in cache:
                if len(cache) >= 2:  # keep at most two models resident
                    keep = {ca, cb}
                    for old in [x for x in cache if x not in keep]:
                        del cache[old]
                    torch.cuda.empty_cache()
                cache[c] = load(c, a.device)
        (ma, tok), (mb, _) = cache[ca], cache[cb]
        seqs = tf_sequences(tok, rows)
        vocab = min(ma.get_output_embeddings().weight.shape[0], mb.get_output_embeddings().weight.shape[0])
        kl_ab = kl_ba = dis = nll_a = nll_b = 0.0
        ntok = 0
        for s in range(0, len(seqs), a.batch_size):
            batch = seqs[s:s + a.batch_size]
            LA, LB = answer_logprobs(ma, batch, vocab), answer_logprobs(mb, batch, vocab)
            for (x, n), la, lb in zip(batch, LA, LB):
                tgt = torch.tensor(x[n:], device=la.device)
                kl_ab += float((la.exp() * (la - lb)).sum())
                kl_ba += float((lb.exp() * (lb - la)).sum())
                dis += float((la.argmax(-1) != lb.argmax(-1)).sum())
                r = torch.arange(len(tgt), device=la.device)
                # A4: gold tokens only (exclude the final eot), as eval_generations' gold_nll
                nll_a += float(-la[r, tgt][:-1].sum())
                nll_b += float(-lb[r, tgt][:-1].sum())
                ntok += len(tgt)
        n_gold = ntok - len(seqs)
        row = {"kind": kind, "dataset": dataset, "a": ca, "b": cb, "n": len(seqs), "n_tok": ntok,
               "kl_ab": kl_ab / ntok, "kl_ba": kl_ba / ntok, "kl_sym": (kl_ab + kl_ba) / (2 * ntok),
               "argmax_disagree": dis / ntok, "nll_a": nll_a / n_gold, "nll_b": nll_b / n_gold,
               **weight_dist(ma, mb), **gen_agreement(ca, cb)}
        with open(a.out, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[pairs] {kind} {dataset} {os.path.basename(ca)} vs {os.path.basename(cb)}: "
              f"kl_sym {row['kl_sym']:.4f} disagree {row['argmax_disagree']:.4f} "
              f"dtheta_rel {row['dtheta_rel']:.4f} match_agree {row.get('match_agree', float('nan')):.4f}",
              flush=True)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("embed")
    e.add_argument("--cell_dir", required=True, help="stage-5 cell dir, or 'base' for the untrained model")
    e.add_argument("--k", type=int, default=0)
    e.add_argument("--lbd", type=float, default=0.0)
    e.add_argument("--seed", type=int, default=-1)
    e.add_argument("--dataset", default="synth")
    e.add_argument("--arm", default="padfix")
    e.add_argument("--n", type=int, default=1000)
    e.add_argument("--pool", choices=["last", "mean"], default="last",
                   help="last = the trained -2 position (Phase A); mean = content-span mean (view gets a -mean suffix)")
    e.add_argument("--out", required=True)
    pp = sub.add_parser("pairs")
    pp.add_argument("--pairs_file", required=True, help="TSV: kind, dataset, cell_a, cell_b")
    pp.add_argument("--shard_id", type=int, default=0)
    pp.add_argument("--num_shards", type=int, default=1)
    pp.add_argument("--n", type=int, default=None)
    pp.add_argument("--batch_size", type=int, default=16)
    pp.add_argument("--out", required=True)
    for s in (e, pp):
        s.add_argument("--test_file", default="datasets/synth_test.jsonl")
        s.add_argument("--device", default="cuda:0")
    e.add_argument("--train_file", default="datasets/synth_train.jsonl")
    a = p.parse_args()
    if a.cmd == "embed":
        if a.dataset != "synth":
            a.test_file = a.test_file.replace("synth", a.dataset)
            a.train_file = a.train_file.replace("synth", a.dataset)
        cmd_embed(a)
    else:
        cmd_pairs(a)


if __name__ == "__main__":
    main()
