"""Where does the JEPA loss read its two embeddings, as a function of k?

CPU only, tokenizer only. Reproduces finetune.py's tokenization for the sweep's
JEPA arm (--last_token=-2 --additive_mask, default --max_length=512, predictors
appended to the user turn as <|predictor_k|>...<|predictor_1|>) and reports the
token at each pooled index:

  text side  JepaTrainer._last_token_index on the user branch: unpadded length - 2
  code side  build_with_additive_mask packs the assistant branch after the user
             branch in the same row; the pooled position is
             clamp(last_asst + last_user + 1, max=max_length-1)

A k-dependent off-by-one here would change *what* the JEPA term aligns — and
could explain a non-monotone k axis — without touching any logged loss. Also
reports how each predictor marker tokenizes (1 token when it is a registered
special, many when it falls back to BPE), branch lengths, and truncation.
The logic is copied from finetune.py (not imported: importing it pulls in the
Trainer stack and a CUDA-gated setup); keep the two in sync.
"""

import argparse
import collections
import copy
import json
import statistics

from transformers import AutoTokenizer

EXTRA = ["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", "<|perception|>"]


def training_tokenizer(model_name, max_predictors):
    # finetune.setup_model_and_tokenizer, tokenizer half
    tok = AutoTokenizer.from_pretrained(model_name)
    special = [f"<|predictor_{i}|>" for i in range(1, max_predictors + 1)] + EXTRA
    new = [t for t in special if t not in tok.vocab]
    if new:
        tok.add_special_tokens({"additional_special_tokens": new})
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def encode(tok, text, max_length):
    return tok(text, truncation=True, max_length=max_length, padding="max_length",
               return_tensors=None, add_special_tokens=True)


def unpad_len(ids, mask):
    # JepaTrainer._last_token_index's unpad(): stop at the first pad after content
    n, can_break = 0, False
    for _, m in zip(ids, mask):
        if m != 0:
            can_break = True
        if m == 0 and can_break:
            break
        n += 1
    return n


def pooled(tok, messages, k, max_length, last_token):
    user = copy.deepcopy(messages)[1:2]  # finetune.get_user_messages
    to_add = k
    while to_add > 0:                    # front_pred=False
        user[0]["content"] += f"<|predictor_{to_add}|>"
        to_add -= 1
    tu = encode(tok, tok.apply_chat_template(user, tokenize=False, add_generation_prompt=False), max_length)
    ta = encode(tok, tok.apply_chat_template(messages[2:3], tokenize=False, add_generation_prompt=False),
                max_length)
    lu = unpad_len(tu["input_ids"], tu["attention_mask"])
    la = unpad_len(ta["input_ids"], ta["attention_mask"])
    iu, ia = lu + last_token, la + last_token
    length_user, length_asst = iu + 1, ia + 1
    lc = min(length_asst, max_length - length_user)
    packed = list(tu["input_ids"])
    packed[length_user:length_user + lc] = ta["input_ids"][:lc]
    pa = min(ia + iu + 1, max_length - 1)
    return {
        "user_len": lu, "asst_len": la, "user_idx": iu, "asst_pos": pa,
        "user_tok": tok.convert_ids_to_tokens(tu["input_ids"][iu]),
        "asst_tok": tok.convert_ids_to_tokens(packed[pa]),
        "asst_pos_ok": lc == length_asst and packed[pa] == ta["input_ids"][ia],
        "user_truncated": lu >= max_length,
        "user_tail": tok.convert_ids_to_tokens(tu["input_ids"][max(0, iu - 3):lu]),
        "user_text": tok.decode(tu["input_ids"][:lu]),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_name", default="meta-llama/Llama-3.2-1B-Instruct")
    p.add_argument("--train_file", default="datasets/synth_train.jsonl")
    p.add_argument("--n_examples", type=int, default=200)
    p.add_argument("--ks", type=int, nargs="+", default=[0, 1, 2, 4, 8, 16, 32])
    p.add_argument("--max_predictors", type=int, default=32, help="finetune.MAX_PREDICTORS (10 before 2026-08-11)")
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--last_token", type=int, default=-2)
    p.add_argument("--out_json", default=None)
    args = p.parse_args()

    tok = training_tokenizer(args.model_name, args.max_predictors)
    with open(args.train_file) as f:
        examples = [json.loads(line)["messages"] for _, line in zip(range(args.n_examples), f)]
    print(f"tokenizer: {args.model_name}  len={len(tok)}  padding_side={tok.padding_side}  "
          f"pad={tok.pad_token!r}  eos={tok.eos_token!r}  MAX_PREDICTORS={args.max_predictors}")
    tmpl = tok.chat_template or ""
    print(f"chat template stamps a date: {'strftime_now' in tmpl or 'date_string' in tmpl}")
    print(f"n_examples={len(examples)}  max_length={args.max_length}  last_token={args.last_token}\n")

    report = {}
    for k in args.ks:
        rs = [pooled(tok, m, k, args.max_length, args.last_token) for m in examples]
        marker = f"<|predictor_{k}|>" if k else None
        summary = {
            "marker_n_tokens": len(tok.encode(marker, add_special_tokens=False)) if marker else 0,
            "user_tok": collections.Counter(r["user_tok"] for r in rs).most_common(5),
            "asst_tok": collections.Counter(r["asst_tok"] for r in rs).most_common(5),
            "asst_pos_ok_frac": sum(r["asst_pos_ok"] for r in rs) / len(rs),
            "user_truncated": sum(r["user_truncated"] for r in rs),
            "user_len": [min(r["user_len"] for r in rs), statistics.median(r["user_len"] for r in rs),
                         max(r["user_len"] for r in rs)],
            "packed_len_max": max(r["asst_pos"] + 2 for r in rs),
            "example0_user_tail": rs[0]["user_tail"],
        }
        report[k] = summary
        print(f"k={k:<3} marker tokens={summary['marker_n_tokens']}  user_len min/med/max={summary['user_len']}  "
              f"packed_len_max={summary['packed_len_max']}  truncated={summary['user_truncated']}")
        print(f"      text-side pooled token : {summary['user_tok']}")
        print(f"      code-side pooled token : {summary['asst_tok']}  (packing consistent: "
              f"{100 * summary['asst_pos_ok_frac']:.0f}%)")
        print(f"      example 0 user tail    : {summary['example0_user_tail']}")
    k_show = max(args.ks)
    print(f"\n--- example 0, text branch as trained at k={k_show} (unpadded) ---")
    print(pooled(tok, examples[0], k_show, args.max_length, args.last_token)["user_text"])
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump({str(k): v for k, v in report.items()}, f, indent=1)


if __name__ == "__main__":
    main()
