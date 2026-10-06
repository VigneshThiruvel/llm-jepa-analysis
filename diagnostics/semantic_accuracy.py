#!/usr/bin/env python3
"""Semantic (language-equivalence) accuracy for synth/turk generations.

Exact match scores `(.*[A-Z].*).*(dog).*` against gold `.*([A-Z]).*(dog).*.*` as wrong.
This rescoring counts a generation as correct when it denotes the same language as the
gold regex, using the exact derivative-based DFA equivalence in regex_equiv.py (NL-RX
dialect: ~ & \\b). Before the exact check, both regexes are run on a fixed set of
~1500 sample strings; a disagreement there proves non-equivalence (a sample can never
prove equivalence, so verdicts stay exact). Pairs that exceed the exact check's time /
state budget are reported as "budget" and scored as wrong.

Degenerate golds: ~20% of NL-RX targets denote the empty language (e.g. `([A-Z])&(dog)`)
or every string (`~(...∅...)`). For those, any output with the same degenerate language
is "equivalent", including run-on junk unioned onto the answer. The report therefore
also gives every metric on the non-degenerate golds only, and split by stop reason.

Read-only, CPU. Reads <root>/<ds>/<arm>/<λ>_<k>_<seed>/eval/generations.jsonl, scores the
stored `response` (already stripped, as scored for exact match), writes under <out_dir>:
  verdicts.jsonl       cache: one verdict per unique (response, gold); reused on rerun
  gold_kind.json       per test idx: ok / empty / all
  cells.csv            per cell: exact / semantic, all and non-degenerate, \\b-as-ε variant
  equiv_pairs.jsonl    every string-different but language-equal response
  report.txt           seed means per cell + gain over the λ=0 baseline

    python3 diagnostics/semantic_accuracy.py --workers=32
"""
import argparse
import csv
import glob
import json
import os
import random
import signal
import statistics
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import regex_equiv as R  # noqa: E402

ROOT = "diag_runs/k16/stage5_padfix"
MAX_STATES = 200000
TIME_LIMIT = 20  # seconds per (pair, boundary mode); a hit counts as "budget" = not correct
MODES = ("assert", "eps")


def _samples(n=1500, seed=0):
    toks = ["dog", "truck", "ring", "lake", "a", "A", "e", "E", "b", "z", "Z", "0", "1", "9",
            " ", "_", ".", "-", "!"]
    rnd = random.Random(seed)
    out = {""}
    while len(out) < n:
        out.add("".join(rnd.choice(toks) for _ in range(rnd.randint(1, 8))))
    return sorted(out)


SAMPLES = _samples()


class _Timeout(Exception):
    pass


def _alarm(*_):
    raise _Timeout


def _accepts(r, s, bnd, memo):
    pw = False
    for ch in s:
        r = R._deriv(r, ch, pw, bnd, memo)
        pw = ch in R.WORD
        if r == R.EMPTY:
            return False
    return R._eps(r, pw, False, bnd)


def check(pair):
    resp, gold = pair
    try:
        a = R.parse(resp)
    except (R.ParseError, RecursionError):
        return pair, {m: "unparseable" for m in MODES}
    b = R.parse(gold)
    out = {}
    for bnd in MODES:
        signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(TIME_LIMIT)
        try:
            ma, mb = {}, {}
            if any(_accepts(a, s, bnd, ma) != _accepts(b, s, bnd, mb) for s in SAMPLES):
                v = False
            else:
                v = R.equivalent(a, b, boundary=bnd, max_states=MAX_STATES)
        except (RecursionError, _Timeout):
            v = None
        finally:
            signal.alarm(0)
        out[bnd] = {True: "equiv", False: "differ", None: "budget"}[v]
    return pair, out


def gold_kind(gold):
    a = R.parse(gold)
    if R.equivalent(a, R.EMPTY):
        return "empty"
    if R.equivalent(a, R.ANY_STAR):
        return "all"
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--datasets", default="synth,turk")
    ap.add_argument("--arms", default="control,padfix")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()
    out_dir = a.out_dir or os.path.join(a.root, "semantic")
    os.makedirs(out_dir, exist_ok=True)
    sys.setrecursionlimit(20000)

    cells = []
    for ds in a.datasets.split(","):
        for arm in a.arms.split(","):
            for d in sorted(glob.glob(os.path.join(a.root, ds, arm, "*_*_*"))):
                g = os.path.join(d, "eval", "generations.jsonl")
                if os.path.exists(g):
                    rows = [json.loads(x) for x in open(g) if x.strip()]
                    cells.append((ds, arm, os.path.basename(d), rows))

    # gold kinds (synth and turk share the same test targets, keyed by gold string)
    golds = sorted({r["gold"] for _, _, _, rows in cells for r in rows})
    kind = {g: gold_kind(g) for g in golds}
    json.dump(kind, open(os.path.join(out_dir, "gold_kind.json"), "w"), indent=0)

    vpath = os.path.join(out_dir, "verdicts.jsonl")
    verdict = {}
    if os.path.exists(vpath):
        for x in open(vpath):
            x = json.loads(x)
            verdict[(x["response"], x["gold"])] = x["verdict"]
    pairs = {(r["response"], r["gold"]) for _, _, _, rows in cells for r in rows if not r["match"]}
    todo = sorted(pairs - set(verdict))
    print(f"{len(cells)} cells, {len(pairs)} unique non-exact pairs, {len(todo)} to check "
          f"({len(pairs) - len(todo)} cached)", flush=True)
    with open(vpath, "a") as vf, Pool(a.workers) as pool:
        for i, (pair, v) in enumerate(pool.imap_unordered(check, todo, chunksize=8)):
            verdict[pair] = v
            vf.write(json.dumps({"response": pair[0], "gold": pair[1], "verdict": v}) + "\n")
            if (i + 1) % 2000 == 0:
                vf.flush()
                print(f"  {i + 1}/{len(todo)}", flush=True)

    def score(r, bnd):
        return "exact" if r["match"] else verdict[(r["response"], r["gold"])][bnd]

    out_rows, eq_f = [], open(os.path.join(out_dir, "equiv_pairs.jsonl"), "w")
    for ds, arm, tag, rows in cells:
        lbd, k, seed = tag.split("_")
        n = len(rows)
        ok = [r for r in rows if kind[r["gold"]] == "ok"]
        eos = [r for r in rows if r["stop"] != "cap"]
        good = lambda rs, bnd="assert": sum(score(r, bnd) in ("exact", "equiv") for r in rs)
        cnt = {}
        for r in rows:
            v = score(r, "assert")
            cnt[v] = cnt.get(v, 0) + 1
            if v == "equiv":
                eq_f.write(json.dumps({"dataset": ds, "arm": arm, "cell": tag, "idx": r["idx"],
                                       "gold_kind": kind[r["gold"]], "stop": r["stop"],
                                       "gold": r["gold"], "response": r["response"]}) + "\n")
        out_rows.append({
            "dataset": ds, "arm": arm, "lbd": float(lbd), "k": int(k), "seed": int(seed), "n": n,
            "acc_exact": cnt.get("exact", 0) / n,
            "acc_semantic": good(rows) / n,
            "acc_semantic_beps": good(rows, "eps") / n,
            "n_nondegen": len(ok),
            "acc_exact_nondegen": sum(r["match"] for r in ok) / len(ok),
            "acc_semantic_nondegen": good(ok) / len(ok),
            "frac_cap": 1 - len(eos) / n,
            "n_equiv_not_exact": cnt.get("equiv", 0),
            "n_equiv_not_exact_nondegen": sum(score(r, "assert") == "equiv" for r in ok),
            "n_equiv_not_exact_cap": sum(score(r, "assert") == "equiv" and r["stop"] == "cap" for r in rows),
            "n_differ": cnt.get("differ", 0),
            "n_unparseable": cnt.get("unparseable", 0),
            "n_budget": cnt.get("budget", 0),
            "n_budget_beps": sum(score(r, "eps") == "budget" for r in rows),
        })
    eq_f.close()
    out_rows.sort(key=lambda r: (r["dataset"], r["arm"], r["lbd"], r["k"], r["seed"]))
    with open(os.path.join(out_dir, "cells.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader()
        for r in out_rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) and k.startswith(("acc", "frac")) else v)
                        for k, v in r.items()})

    nk = {kk: sum(1 for g in kind.values() if g == kk) for kk in ("ok", "empty", "all")}
    L = [f"semantic (DFA-equivalence) accuracy, {len(cells)} cells under {a.root}",
         f"distinct test golds: {nk['ok']} ok, {nk['empty']} empty-language, {nk['all']} all-strings",
         "columns: seed means; gain* = cell minus the λ=0 baseline of the same arm and seeds;",
         "  *_nd = non-degenerate golds only; semEps = \\b treated as ε", ""]
    L.append(f"{'ds':5} {'arm':7} {'cell':7} {'seeds':>5} {'exact':>6} {'sem':>6} {'semEps':>6} {'ex_nd':>6} "
             f"{'sem_nd':>6} {'gainEx':>7} {'gainSem':>7} {'gSemND':>7} {'unpars':>6} {'budget':>6}")
    groups = {}
    for r in out_rows:
        groups.setdefault((r["dataset"], r["arm"], r["lbd"], r["k"]), []).append(r)
    mean = lambda rs, key: statistics.fmean(x[key] for x in rs)
    for (ds, arm, lbd, k), rs in sorted(groups.items()):
        base = groups.get((ds, arm, 0.0, 0))
        seeds = {x["seed"] for x in rs}
        bs = [x for x in base if x["seed"] in seeds] if base else None
        gain = lambda key: mean(rs, key) - mean(bs, key) if bs else float("nan")
        L.append(f"{ds:5} {arm:7} {f'{lbd:g}_{k}':7} {len(rs):5d} {mean(rs, 'acc_exact'):6.4f} "
                 f"{mean(rs, 'acc_semantic'):6.4f} {mean(rs, 'acc_semantic_beps'):6.4f} "
                 f"{mean(rs, 'acc_exact_nondegen'):6.4f} {mean(rs, 'acc_semantic_nondegen'):6.4f} "
                 f"{gain('acc_exact'):+7.4f} {gain('acc_semantic'):+7.4f} {gain('acc_semantic_nondegen'):+7.4f} "
                 f"{sum(x['n_unparseable'] for x in rs):6d} {sum(x['n_budget'] for x in rs):6d}")
    rep = "\n".join(L)
    open(os.path.join(out_dir, "report.txt"), "w").write(rep + "\n")
    print(rep)
    print(f"\nwrote {out_dir}/cells.csv, equiv_pairs.jsonl, verdicts.jsonl, gold_kind.json, report.txt")


if __name__ == "__main__":
    main()
