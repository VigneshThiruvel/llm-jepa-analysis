"""Per-example generation diagnostics — companion to evaluate.py's exact-match path.

run_sweep.sh scores every cell with `evaluate.py --split_tune_untune`, which
prints a Success Rate but writes nothing per example (its eval.jsonl is opened
and left empty). So when a cell scores exactly 0.000 — the k=16 column from
lambda=2 up — there is no record of what the model actually generated.

This script generates with the *same* prompt construction, the *same*
model.generate() call and the *same* matcher as that path (all imported from
evaluate.py, which is not modified), so its Success Rate reproduces
evaluate.py's. On top of that it records, per example:

  - the response as scored (skip_special_tokens=True) AND the raw decode — a
    <|predictor_i|> emitted through the tied LM head is invisible in the first
  - why generation stopped: 'eos' or 'cap' (ran into --max_new_tokens)
  - counts of every special / added token emitted
  - step-0 distribution: top-10 tokens, rank + prob of <|eot_id|> and of the
    best-ranked predictor token
  - teacher-forced gold: fed prompt+gold, how well does the model predict the
    gold tokens, and is <|eot_id|> the argmax right after them? This separates
    "cannot write the answer" from "writes it but cannot stop".
  - common-prefix length of generated vs gold tokens, and the tail loop period
  - for the first --trace_n examples, the full per-step trace

Two modes:
  (default)  one shard of the test set -> <out_dir>/shards/gen.<i>of<n>.jsonl
  --merge    all shards -> <out_dir>/generations.jsonl + summary.json, and a
             stdout report whose "Success Rate: <model>, <rate>" line is the one
             collect_sweep.py parses, so a cell scored this way collects like any
             sweep cell.

Sharding is by example (idx % n) and each example is generated independently at
batch size 1, exactly as evaluate.py does, so sharding cannot change outputs.
A shard file is resumable: re-running skips examples already written.
"""

import argparse
import collections
import glob
import json
import os
import re
import statistics
import sys

import torch
from tqdm import tqdm

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

if not torch.cuda.is_available():
    # evaluate.load_model_and_tokenizer calls torch.cuda.current_device() only to
    # gate a print, which raises on a CPU-only host. This lets --device_map=cpu
    # smoke tests run on a login node; it has no effect on GPU runs.
    torch.cuda.current_device = lambda: 0

import evaluate as ev  # noqa: E402 — the repo-root evaluate.py, imported, never modified

RATE_PATTERN = re.compile(r"^Success Rate: (?:\S+), (?P<rate>[\d.]+)", re.MULTILINE)
SHARD_RE = re.compile(r"gen\.(\d+)of(\d+)\.jsonl$")
SPECIAL_START = 128000  # Llama-3 special/reserved ids start here; finetune.py's added tokens follow 128255


def load_examples(path, max_examples=None):
    # Same rows in the same order as evaluate.py's load_dataset('json', ...)['train'],
    # minus the HF arrow cache (concurrent shards would race on a cold build).
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return rows[:max_examples] if max_examples else rows


def build_prompt(messages, tokenizer, original_model_name, date_string=None):
    full = ev.get_messages(original_model_name, messages)
    if date_string is None:
        return ev.format_conversation(full, tokenizer)
    # Identical to ev.format_conversation except that it pins the date the
    # Llama-3 chat template stamps into the system header ("Today Date: ...").
    # Unpinned, that is the wall-clock date, so re-evaluating an old checkpoint on
    # a later day feeds it different prompts than the eval being reproduced.
    msgs = [m for m in full if m["role"] != "assistant"]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                         date_string=date_string)


def predictor_id_map(tokenizer, n_rows):
    """{token_id: i} for every <|predictor_i|> that is a real logit row of the model."""
    out = {}
    for tid, added in tokenizer.added_tokens_decoder.items():
        m = re.fullmatch(r"<\|predictor_(\d+)\|>", getattr(added, "content", str(added)))
        if m and tid < n_rows:
            out[tid] = int(m.group(1))
    return out


def tail_period(ids, window=48, max_period=24):
    """Smallest p such that the last `window` generated ids repeat with period p, else None."""
    if len(ids) < window:
        return None
    tail = ids[-window:]
    for p in range(1, max_period + 1):
        if all(tail[j] == tail[j - p] for j in range(p, window)):
            return p
    return None


def _r(t, nd=5):
    return [round(float(x), nd) for x in t.tolist()]


@torch.no_grad()
def diagnose(idx, messages, model, tokenizer, args, pred_map, pred_ids_t, trace):
    tok = lambda i: tokenizer.convert_ids_to_tokens(int(i))  # noqa: E731
    eot = tokenizer.eos_token_id
    prompt = build_prompt(messages, tokenizer, args.original_model_name, args.date_string)

    # ---- generation: mirrors evaluate.generate_response line for line ----
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=args.max_length, add_special_tokens=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    out = model.generate(
        **inputs,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=False,
        max_new_tokens=args.max_new_tokens,
        return_dict_in_generate=True,  # added: exposes per-step scores
        output_logits=True,            # added: raw logits; greedy argmax is unaffected
    )
    n_prompt = inputs["input_ids"].shape[1]
    gen = out.sequences[0][n_prompt:]
    response = tokenizer.decode(gen, skip_special_tokens=True).strip()
    if response.endswith("<|end|>"):
        response = response[:-7].strip()
    # ---- end mirror ----
    match = bool(ev.eval(response, messages, args.input_file, args.spider_path))

    gen_ids = gen.tolist()
    n_gen = len(gen_ids)
    if n_gen and gen_ids[-1] == eot:
        stop = "eos"
    elif n_gen >= args.max_new_tokens:
        stop = "cap"
    else:
        stop = "other"

    L = torch.cat(out.logits, dim=0).float()  # [n_gen, V]
    lse = torch.logsumexp(L, dim=-1)
    steps = torch.arange(n_gen, device=L.device)
    chosen = gen.to(L.device)
    chosen_logit = L[steps, chosen]
    n_not_argmax = int((L.argmax(-1) != chosen).sum())
    eot_logit = L[:, eot]
    eot_rank = (L > eot_logit[:, None]).sum(-1)
    p_eot = (eot_logit - lse).exp()
    has_pred = pred_ids_t.numel() > 0
    if has_pred:
        mp_logit, mp_j = L[:, pred_ids_t].max(-1)
        mp_rank = (L > mp_logit[:, None]).sum(-1)
        p_mp = (mp_logit - lse).exp()
        mp_id = pred_ids_t[mp_j]

    top = torch.topk(L[0], 10)
    step0 = {
        "top10": [[int(i), tok(i), round(float(v), 3), round(float((v - lse[0]).exp()), 5)]
                  for v, i in zip(top.values, top.indices)],
        "eot_rank": int(eot_rank[0]),
        "p_eot": float(p_eot[0]),
        "maxpred_rank": int(mp_rank[0]) if has_pred else None,
        "maxpred_token": tok(mp_id[0]) if has_pred else None,
        "p_maxpred": float(p_mp[0]) if has_pred else None,
    }

    gold = messages[2]["content"]
    gold_ids = tokenizer(gold, add_special_tokens=False)["input_ids"]
    matched = 0
    for a, b in zip(gen_ids, gold_ids):
        if a != b:
            break
        matched += 1
    after_gold = None
    if matched == len(gold_ids) and n_gen > matched:
        # Wrote every gold token; what did it do where it should have stopped?
        s = matched
        after_gold = {"token": tok(gen_ids[s]), "eot_rank": int(eot_rank[s]), "p_eot": float(p_eot[s])}

    # Teacher-forced pass over prompt + gold + <|eot_id|>: row j predicts target j.
    tf_ids = torch.tensor([inputs["input_ids"][0].tolist() + gold_ids + [eot]], device=model.device)
    TL = model(input_ids=tf_ids).logits[0, n_prompt - 1:-1].float()
    tgt = tf_ids[0, n_prompt:]
    tlse = torch.logsumexp(TL, dim=-1)
    nll = tlse - TL[torch.arange(len(tgt), device=TL.device), tgt]
    targ = TL.argmax(-1)
    ng = len(gold_ids)
    end = TL[-1]
    tf = {
        "gold_token_acc": float((targ[:ng] == tgt[:ng]).float().mean()) if ng else None,
        "gold_nll": float(nll[:ng].mean()) if ng else None,
        "eot_rank": int((end > end[eot]).sum()),
        "p_eot": float((end[eot] - tlse[-1]).exp()),
        "end_argmax": tok(end.argmax()),
        "end_maxpred_rank": int((end > end[pred_ids_t].max()).sum()) if has_pred else None,
    }

    rec = {
        "idx": idx,
        "match": match,
        "gold": gold,
        "response": response,
        "response_raw": tokenizer.decode(gen_ids, skip_special_tokens=False),
        "n_prompt_tokens": n_prompt,
        "n_gen": n_gen,
        "stop": stop,
        "first_tokens": [tok(i) for i in gen_ids[:12]],
        "n_predictor_tokens": sum(1 for i in gen_ids if i in pred_map),
        "special_counts": dict(collections.Counter(tok(i) for i in gen_ids if i >= SPECIAL_START)),
        "response_startswith_gold": response.startswith(gold),
        "n_gold_tokens": ng,
        "gold_tokens_matched": matched,
        "after_gold": after_gold,
        "tail_period": tail_period(gen_ids),
        "min_eot_rank": int(eot_rank.min()),
        "max_p_eot": float(p_eot.max()),
        "n_chosen_not_argmax": n_not_argmax,
        "step0": step0,
        "tf": tf,
    }
    if trace:
        rec["trace"] = {
            "token": [tok(i) for i in gen_ids],
            "p_chosen": _r((chosen_logit - lse).exp()),
            "eot_rank": eot_rank.tolist(),
            "p_eot": _r(p_eot),
            "maxpred_rank": mp_rank.tolist() if has_pred else None,
            "maxpred_token": [tok(i) for i in mp_id.tolist()] if has_pred else None,
        }
    return rec


def run_shard(args):
    examples = load_examples(args.input_file, args.max_examples)
    shard_dir = os.path.join(args.out_dir, "shards")
    os.makedirs(shard_dir, exist_ok=True)
    path = os.path.join(shard_dir, f"gen.{args.shard_id}of{args.num_shards}.jsonl")
    mine = [i for i in range(len(examples)) if i % args.num_shards == args.shard_id]

    done, good = set(), []
    if os.path.exists(path):  # resume: keep complete lines, drop a torn last line
        with open(path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["idx"])
                    good.append(line if line.endswith("\n") else line + "\n")
                except (json.JSONDecodeError, KeyError):
                    break
        with open(path, "w") as f:
            f.writelines(good)

    model, tokenizer = ev.load_model_and_tokenizer(args.model_name, args.original_model_name,
                                                   device_map=args.device_map)
    n_rows = model.get_input_embeddings().weight.shape[0]
    pred_map = predictor_id_map(tokenizer, n_rows)
    pred_ids_t = torch.tensor(sorted(pred_map), dtype=torch.long, device=model.device)
    print(f"[shard {args.shard_id}/{args.num_shards}] {len(mine)} examples, {len(done)} already done")
    print(f"  logit rows={n_rows}  eos={tokenizer.eos_token_id} ({tokenizer.eos_token})  "
          f"pad={tokenizer.pad_token_id}  predictor rows={len(pred_map)}  date_string={args.date_string}")
    print(f"  checkpoint generation_config: {model.generation_config.to_diff_dict()}")

    todo = [i for i in mine if i not in done]
    with open(path, "a") as f:
        for i in tqdm(todo, desc=f"shard {args.shard_id}", mininterval=30):
            rec = diagnose(i, examples[i]["messages"], model, tokenizer, args,
                           pred_map, pred_ids_t, trace=i < args.trace_n)
            f.write(json.dumps(rec) + "\n")
            f.flush()


def _frac(pred, pool):
    return sum(1 for r in pool if pred(r)) / len(pool) if pool else None


def _median(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.fmean(xs) if xs else None


def summarize(rows, args):
    n = len(rows)
    correct = sum(r["match"] for r in rows)
    wrong = [r for r in rows if not r["match"]]
    ngen = sorted(r["n_gen"] for r in rows)
    specials = collections.Counter()
    for r in rows:
        specials.update(r["special_counts"])
    n_prefix = sum(1 for r in rows if r["match"] or r["response_startswith_gold"])
    return {
        "model_name": args.model_name,
        "input_file": args.input_file,
        "max_new_tokens": args.max_new_tokens,
        "date_string": args.date_string,
        "n": n,
        "n_correct": correct,
        "accuracy": correct / n,
        # Prefix match: the gold answer is a prefix of the output, i.e. the model
        # produced the right answer and then did or did not stop. Exact match
        # conflates task competence with termination; this separates them, and is
        # the same metric evaluate.py's --startswith reports.
        "n_prefix": n_prefix,
        "accuracy_prefix": n_prefix / n,
        "stop_reason": dict(collections.Counter(r["stop"] for r in rows)),
        "frac_cap": _frac(lambda r: r["stop"] == "cap", rows),
        "n_gen_mean": statistics.fmean(ngen),
        "n_gen_median": statistics.median(ngen),
        "n_gen_p90": ngen[int(0.9 * (n - 1))],
        "n_gen_max": ngen[-1],
        "frac_empty_response": _frac(lambda r: r["response"] == "", rows),
        "frac_any_predictor_emitted": _frac(lambda r: r["n_predictor_tokens"] > 0, rows),
        "mean_predictor_tokens": statistics.fmean(r["n_predictor_tokens"] for r in rows),
        "special_tokens_emitted": dict(specials.most_common()),
        "first_token_top20": collections.Counter(
            r["first_tokens"][0] for r in rows if r["first_tokens"]).most_common(20),
        "step0_eot_rank_median": _median(r["step0"]["eot_rank"] for r in rows),
        "step0_maxpred_rank_median": _median(r["step0"]["maxpred_rank"] for r in rows),
        "tf_gold_token_acc_mean": _mean(r["tf"]["gold_token_acc"] for r in rows),
        "tf_gold_nll_mean": _mean(r["tf"]["gold_nll"] for r in rows),
        "tf_eot_top1_frac": _frac(lambda r: r["tf"]["eot_rank"] == 0, rows),
        "tf_eot_rank_median": _median(r["tf"]["eot_rank"] for r in rows),
        "tf_p_eot_median": _median(r["tf"]["p_eot"] for r in rows),
        "tf_end_argmax_top10": collections.Counter(r["tf"]["end_argmax"] for r in rows).most_common(10),
        "n_wrong": len(wrong),
        "wrong_frac_startswith_gold": _frac(lambda r: r["response_startswith_gold"], wrong),
        "wrong_frac_all_gold_tokens_then_continued": _frac(lambda r: r["after_gold"] is not None, wrong),
        "wrong_after_gold_token_top10": collections.Counter(
            r["after_gold"]["token"] for r in wrong if r["after_gold"]).most_common(10),
        "wrong_frac_tail_loop": _frac(lambda r: r["tail_period"] is not None, wrong),
        "wrong_tail_period_hist": dict(collections.Counter(
            r["tail_period"] for r in wrong if r["tail_period"] is not None)),
        "sanity_steps_chosen_not_argmax": sum(r["n_chosen_not_argmax"] for r in rows),
    }


def _pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def print_report(rows, s, args):
    print(f"=== eval_generations: {args.model_name} ===")
    print(f"input={args.input_file}  n={s['n']}  max_new_tokens={s['max_new_tokens']}  "
          f"date_string={s['date_string']}")
    # The line collect_sweep.py parses — same format as evaluate.py.
    print(f"Success Rate: {args.model_name}, {s['accuracy']}")
    # Parsed by collect_sweep.py into the accuracy_prefix column.
    print(f"Prefix Rate: {args.model_name}, {s['accuracy_prefix']}")
    if args.reference_results:
        ref = None
        if os.path.exists(args.reference_results):
            m = RATE_PATTERN.search(open(args.reference_results).read())
            ref = float(m.group("rate")) if m else None
        if ref is None:
            print(f"Reference: no rate found in {args.reference_results}")
        else:
            verdict = "reproduced exactly" if ref == s["accuracy"] else f"differs by {s['accuracy'] - ref:+.4f}"
            if args.max_examples:
                verdict += f" (subset of {s['n']} examples — not comparable)"
            print(f"Reference rate {ref} ({args.reference_results}) -> {verdict}")
    st = s["stop_reason"]
    print(f"stop reason: " + " | ".join(f"{k} {v} ({_pct(v / s['n'])})" for k, v in sorted(st.items())))
    print(f"generated tokens: mean {s['n_gen_mean']:.1f} | median {s['n_gen_median']} | "
          f"p90 {s['n_gen_p90']} | max {s['n_gen_max']}")
    print(f"empty responses: {_pct(s['frac_empty_response'])} | any <|predictor_i|> emitted: "
          f"{_pct(s['frac_any_predictor_emitted'])} (mean {s['mean_predictor_tokens']:.2f}/example)")
    print(f"special tokens emitted (total): {dict(list(s['special_tokens_emitted'].items())[:12])}")
    print(f"first generated token (top 8): {s['first_token_top20'][:8]}")
    print(f"step 0: <|eot_id|> rank median {s['step0_eot_rank_median']} | "
          f"best predictor rank median {s['step0_maxpred_rank_median']}")
    print(f"teacher-forced gold: token acc {s['tf_gold_token_acc_mean']:.4f} | nll {s['tf_gold_nll_mean']:.4f} | "
          f"<|eot_id|> argmax after gold {_pct(s['tf_eot_top1_frac'])} "
          f"(rank median {s['tf_eot_rank_median']}, p median {s['tf_p_eot_median']:.4g})")
    print(f"teacher-forced argmax after gold (top 5): {s['tf_end_argmax_top10'][:5]}")
    print(f"wrong answers n={s['n_wrong']}: startswith gold {_pct(s['wrong_frac_startswith_gold'])} | "
          f"all gold tokens then continued {_pct(s['wrong_frac_all_gold_tokens_then_continued'])} | "
          f"tail loop {_pct(s['wrong_frac_tail_loop'])} {s['wrong_tail_period_hist']}")
    print(f"  token emitted where gold ended (top 5): {s['wrong_after_gold_token_top10'][:5]}")
    print(f"sanity: steps where chosen != argmax(raw logits): {s['sanity_steps_chosen_not_argmax']} "
          f"(0 expected for greedy)")
    wrong = [r for r in rows if not r["match"]][:8]
    if wrong:
        print("--- first wrong examples ---")
    for r in wrong:
        print(f"[{r['idx']}] gold:     {r['gold']!r}")
        print(f"      response: {r['response'][:200]!r}")
        print(f"      raw:      {r['response_raw'][:200]!r}")
        print(f"      stop={r['stop']} n_gen={r['n_gen']} first={r['first_tokens'][:6]} "
              f"tf_eot_rank={r['tf']['eot_rank']}")


def run_merge(args):
    examples = load_examples(args.input_file, args.max_examples)
    shard_dir = os.path.join(args.out_dir, "shards")
    files = sorted(glob.glob(os.path.join(shard_dir, "gen.*of*.jsonl")))
    if not files:
        raise SystemExit(f"no shard files in {shard_dir}")
    counts = {int(SHARD_RE.search(f).group(2)) for f in files}
    if len(counts) != 1:
        raise SystemExit(f"shards from runs with different --num_shards {sorted(counts)}; "
                         f"delete {shard_dir} and rerun")
    if len(files) != counts.pop():
        raise SystemExit(f"expected one file per shard, found {len(files)} in {shard_dir}")
    by_idx = {}
    for f in files:
        with open(f) as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    if r["idx"] in by_idx:
                        raise SystemExit(f"duplicate idx {r['idx']} across shards")
                    by_idx[r["idx"]] = r
    missing = [i for i in range(len(examples)) if i not in by_idx]
    if missing:
        raise SystemExit(f"{len(missing)} examples missing (first {missing[:5]}) — a shard did not finish")
    rows = [by_idx[i] for i in range(len(examples))]

    def write(path, write_fn):
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            write_fn(fh)
        os.replace(tmp, path)

    write(os.path.join(args.out_dir, "generations.jsonl"),
          lambda fh: fh.writelines(json.dumps(r) + "\n" for r in rows))
    s = summarize(rows, args)
    write(os.path.join(args.out_dir, "summary.json"), lambda fh: json.dump(s, fh, indent=1))
    print_report(rows, s, args)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_name", required=True, help="Checkpoint dir (what evaluate.py gets as --model_name)")
    p.add_argument("--original_model_name", required=True)
    p.add_argument("--input_file", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--max_new_tokens", type=int, required=True,
                   help="Generation budget. Required and >0: evaluate.py's -1 silently caps at 20 tokens.")
    p.add_argument("--max_length", type=int, default=512, help="Prompt truncation, as evaluate.py's --max_length")
    p.add_argument("--spider_path", default="")
    p.add_argument("--device_map", default="cuda:0")
    p.add_argument("--max_examples", type=int, default=None)
    p.add_argument("--shard_id", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--trace_n", type=int, default=50, help="Examples (by idx) that get a full per-step trace")
    p.add_argument("--date_string", default=None,
                   help='Pin the chat template date, e.g. "11 Aug 2026" (default: today, as evaluate.py)')
    p.add_argument("--merge", action="store_true", help="Merge shards and report instead of generating")
    p.add_argument("--reference_results", default=None,
                   help="A results.txt whose Success Rate this run should reproduce (merge only)")
    args = p.parse_args()
    if args.max_new_tokens <= 0:
        p.error("--max_new_tokens must be a positive budget")
    if not 0 <= args.shard_id < args.num_shards:
        p.error("need 0 <= --shard_id < --num_shards")
    run_merge(args) if args.merge else run_shard(args)


if __name__ == "__main__":
    main()
