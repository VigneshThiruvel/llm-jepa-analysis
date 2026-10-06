#!/usr/bin/env python3
"""How much room is left for JEPA once equivalent regexes are scored as correct?

Reads the verdict cache written by semantic_accuracy.py and, per dataset and arm, looks
at every test example across all cells (all λ, k, seeds):
  never   no cell ever produces an equivalent regex -> ceiling no training tweak here reaches
  always  every cell gets it                        -> no room either
  mixed   some cells do, some don't                 -> the only examples a JEPA gain can come from
plus the "oracle" accuracy (any cell correct) and the baseline-vs-JEPA flip counts.
Prints a sample of never-solved examples so their labels can be eyeballed.

    python3 diagnostics/semantic_room.py [--arm=padfix]
"""
import argparse
import collections
import glob
import json
import os
import random

ROOT = "diag_runs/k16/stage5_padfix"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--arm", default="padfix")
    ap.add_argument("--datasets", default="synth,turk")
    ap.add_argument("--show", type=int, default=8)
    a = ap.parse_args()
    sem = os.path.join(a.root, "semantic")
    verdict = {}
    for x in open(os.path.join(sem, "verdicts.jsonl")):
        x = json.loads(x)
        verdict[(x["response"], x["gold"])] = x["verdict"]["assert"]
    kind = json.load(open(os.path.join(sem, "gold_kind.json")))
    L = []
    for ds in a.datasets.split(","):
        nl = {i: json.loads(l)["messages"][1]["content"] for i, l in enumerate(open(f"datasets/{ds}_test.jsonl"))}
        ok = collections.defaultdict(dict)  # idx -> {tag: bool}
        gold = {}
        for d in sorted(x for x in glob.glob(os.path.join(a.root, ds, a.arm, "*_*_*")) if os.path.isdir(x)):
            tag = os.path.basename(d)
            for line in open(os.path.join(d, "eval", "generations.jsonl")):
                r = json.loads(line)
                gold[r["idx"]] = r["gold"]
                ok[r["idx"]][tag] = r["match"] or verdict[(r["response"], r["gold"])] == "equiv"
        tags = sorted({t for v in ok.values() for t in v})
        n = len(ok)
        cls = {i: ("always" if all(v.values()) else "never" if not any(v.values()) else "mixed") for i, v in ok.items()}
        L.append(f"=== {ds} / {a.arm}: {n} examples x {len(tags)} cells ===")
        for c in ("always", "mixed", "never"):
            ids = [i for i in cls if cls[i] == c]
            byk = collections.Counter(kind[gold[i]] for i in ids)
            L.append(f"  {c:6} {len(ids):5d} ({len(ids) / n:.3f})   gold kinds {dict(byk)}")
        L.append(f"  oracle (any cell semantic-correct): {sum(c != 'never' for c in cls.values()) / n:.4f}")
        # baseline vs every JEPA cell, matched seed
        seeds = sorted({t.split("_")[2] for t in tags})
        for t in tags:
            lbd, k, s = t.split("_")
            if lbd == "0.0":
                continue
            b = f"0.0_0_{s}"
            if b not in tags:
                continue
            gain = sum(ok[i][t] and not ok[i][b] for i in ok)
            loss = sum(ok[i][b] and not ok[i][t] for i in ok)
            L.append(f"  {t:10} vs {b}: +{gain:3d} / -{loss:3d}  net {(gain - loss) / n:+.4f}")
        never = [i for i in cls if cls[i] == "never" and kind[gold[i]] == "ok"]
        random.Random(0).shuffle(never)
        L.append(f"  sample of never-solved (non-degenerate gold), {min(a.show, len(never))} of {len(never)}:")
        for i in never[: a.show]:
            L.append(f"    [{i}] NL: {nl[i]}\n         gold: {gold[i]}")
        L.append("")
    rep = "\n".join(L)
    out = os.path.join(sem, f"room_{a.arm}.txt")
    open(out, "w").write(rep + "\n")
    print(rep)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
