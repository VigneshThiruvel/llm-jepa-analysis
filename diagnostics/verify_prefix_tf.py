#!/usr/bin/env python3
"""Verify prefix accuracy and teacher-forced (TF) gold accuracy on the stage-5 cells.

Read-only, CPU. Recomputes both metrics from the per-example rows in each cell's
eval/generations.jsonl (written by eval_generations.py) instead of trusting
summary.json, and cross-checks them three ways:

  1. recorded vs recomputed   summary.json's accuracy / accuracy_prefix /
                              tf_gold_token_acc_mean against the same quantity
                              recomputed from the rows (must be identical).
  2. string vs token prefix   response.startswith(gold) against "every gold token
                              was generated" (gold_tokens_matched == n_gold_tokens).
  3. greedy vs TF             greedy decoding is argmax at every step, so a row
                              whose greedy output starts with the gold tokens must
                              have TF argmax == gold at every gold position, and
                              vice versa. tf_seq_acc (all gold tokens argmax-correct)
                              should equal the token-level prefix rate; any
                              disagreement is a bookkeeping or numerics bug.

    python3 diagnostics/verify_prefix_tf.py                  # seed 82, both datasets
    python3 diagnostics/verify_prefix_tf.py --seed=23 --datasets=synth

Outputs: <out_dir>/cells.csv (one row per dataset x arm x cell) and report.txt.
"""
import argparse
import csv
import glob
import json
import os

ROOT = "diag_runs/k16/stage5_padfix"
TOL = 1e-9


def load_rows(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def verify_cell(ds, arm, tag, cell_dir):
    gen = os.path.join(cell_dir, "eval", "generations.jsonl")
    summ = os.path.join(cell_dir, "eval", "summary.json")
    if not (os.path.exists(gen) and os.path.exists(summ)):
        return None, [f"{ds}/{arm}/{tag}: missing generations.jsonl or summary.json"]
    s = json.load(open(summ))
    rows = load_rows(gen)
    issues = []
    n = len(rows)
    idxs = [r["idx"] for r in rows]
    if len(set(idxs)) != n:
        issues.append(f"duplicate idx ({n - len(set(idxs))})")
    if n != s["n"]:
        issues.append(f"row count {n} != summary n {s['n']}")

    exact = sum(r["match"] for r in rows)
    pre_str = sum(1 for r in rows if r["response"].startswith(r["gold"]))
    pre_flag = sum(1 for r in rows if r["match"] or r["response_startswith_gold"])
    pre_tok = sum(1 for r in rows if r["gold_tokens_matched"] == r["n_gold_tokens"])
    exact_not_prefix = sum(1 for r in rows if r["match"] and not r["response"].startswith(r["gold"]))

    tf_rows = [r for r in rows if r["tf"]["gold_token_acc"] is not None]
    tf_tok = sum(r["tf"]["gold_token_acc"] for r in tf_rows) / len(tf_rows)
    tf_all = [r["tf"]["gold_token_acc"] == 1.0 for r in tf_rows]
    tf_seq = sum(tf_all) / len(tf_rows)
    tf_nll = sum(r["tf"]["gold_nll"] for r in tf_rows) / len(tf_rows)
    tf_eot_top1 = sum(r["tf"]["eot_rank"] == 0 for r in rows) / n

    # greedy-prefix (token level) vs TF all-correct, per row
    tokpre = [r["gold_tokens_matched"] == r["n_gold_tokens"] for r in tf_rows]
    g_not_tf = sum(1 for a, b in zip(tokpre, tf_all) if a and not b)
    tf_not_g = sum(1 for a, b in zip(tokpre, tf_all) if b and not a)

    for name, rec, got in [("accuracy", s["accuracy"], exact / n),
                           ("accuracy_prefix", s["accuracy_prefix"], pre_flag / n),
                           ("tf_gold_token_acc_mean", s["tf_gold_token_acc_mean"], tf_tok)]:
        if abs(rec - got) > TOL:
            issues.append(f"{name}: summary {rec:.6f} != recomputed {got:.6f}")
    if pre_flag != pre_str:
        issues.append(f"prefix flag {pre_flag} != string recompute {pre_str}")
    if exact_not_prefix:
        issues.append(f"{exact_not_prefix} exact matches are not prefix matches")

    cfg = {}
    cfg_path = os.path.join(cell_dir, "config.json")
    if os.path.exists(cfg_path):
        cfg = json.load(open(cfg_path))
    lbd, k, seed = tag.split("_")
    row = {
        "dataset": ds, "arm": arm, "lbd": float(lbd), "k": int(k), "seed": int(seed),
        "n": n,
        "date_string": s.get("date_string"),
        "max_new_tokens": s.get("max_new_tokens"),
        "acc_exact": exact / n,
        "acc_prefix": pre_str / n,
        "acc_prefix_tok": pre_tok / n,
        "acc_prefix_recorded": s["accuracy_prefix"],
        "tf_gold_token_acc": tf_tok,
        "tf_gold_token_acc_recorded": s["tf_gold_token_acc_mean"],
        "tf_seq_acc": tf_seq,
        "tf_gold_nll": tf_nll,
        "tf_eot_top1": tf_eot_top1,
        "frac_cap": s["frac_cap"],
        "prefix_minus_exact": (pre_str - exact) / n,
        "prefix_str_minus_tok": (pre_str - pre_tok) / n,
        "greedy_prefix_not_tf_all": g_not_tf,
        "tf_all_not_greedy_prefix": tf_not_g,
        "issues": "; ".join(issues),
    }
    return row, [f"{ds}/{arm}/{tag}: {i}" for i in issues]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--seed", default="82")
    ap.add_argument("--datasets", default="synth,turk")
    ap.add_argument("--arms", default="control,padfix")
    ap.add_argument("--out_dir", default=None)
    a = ap.parse_args()
    out_dir = a.out_dir or os.path.join(a.root, f"verify_seed{a.seed}")
    os.makedirs(out_dir, exist_ok=True)

    rows, problems = [], []
    for ds in a.datasets.split(","):
        for arm in a.arms.split(","):
            for cell_dir in sorted(glob.glob(os.path.join(a.root, ds, arm, f"*_*_{a.seed}"))):
                row, issues = verify_cell(ds, arm, os.path.basename(cell_dir), cell_dir)
                problems += issues
                if row:
                    rows.append(row)
    rows.sort(key=lambda r: (r["dataset"], r["lbd"], r["k"], r["arm"]))

    with open(os.path.join(out_dir, "cells.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})

    lines = [f"stage-5 prefix / TF-gold verification, seed {a.seed}, {len(rows)} cells", ""]
    hdr = f"{'ds':5} {'cell':8} {'arm':7} {'exact':>6} {'prefix':>6} {'tfTok':>6} {'tfSeq':>6} {'eotT1':>6} {'cap':>5} {'g!tf':>4} {'tf!g':>4}"
    lines.append(hdr)
    for r in rows:
        cell = f"{r['lbd']:g}_{r['k']}"
        lines.append(f"{r['dataset']:5} {cell:8} {r['arm']:7} {r['acc_exact']:6.4f} {r['acc_prefix']:6.4f} "
                     f"{r['tf_gold_token_acc']:6.4f} {r['tf_seq_acc']:6.4f} {r['tf_eot_top1']:6.4f} {r['frac_cap']:5.3f} "
                     f"{r['greedy_prefix_not_tf_all']:4d} {r['tf_all_not_greedy_prefix']:4d}")
    lines += ["", "per dataset x arm (min-max over cells):"]
    for ds in sorted({r["dataset"] for r in rows}):
        for arm in sorted({r["arm"] for r in rows}):
            sub = [r for r in rows if r["dataset"] == ds and r["arm"] == arm]
            if not sub:
                continue
            rng = lambda key: f"{min(r[key] for r in sub):.4f}-{max(r[key] for r in sub):.4f}"
            lines.append(f"  {ds}/{arm}: exact {rng('acc_exact')} | prefix {rng('acc_prefix')} | "
                         f"tf_tok {rng('tf_gold_token_acc')} | tf_seq {rng('tf_seq_acc')}")
    dates = sorted({str(r["date_string"]) for r in rows})
    caps = sorted({str(r["max_new_tokens"]) for r in rows})
    lines += ["", f"eval date_string(s): {dates}   max_new_tokens: {caps}",
              f"consistency problems: {len(problems)}"] + [f"  {p}" for p in problems]
    report = "\n".join(lines)
    open(os.path.join(out_dir, "report.txt"), "w").write(report + "\n")
    print(report)
    print(f"\nwrote {out_dir}/cells.csv, {out_dir}/report.txt")


if __name__ == "__main__":
    main()
