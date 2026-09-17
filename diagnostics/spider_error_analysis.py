"""Spider failure taxonomy over an eval_generations.py generations.jsonl.

CPU-only, read-only. Re-executes each generated query and its gold query against
the same sqlite DB that evaluate.spider_eval uses, and buckets every example:

  correct       the matcher said match (never re-derived: `match` is authoritative)
  cap           generation hit --max_new_tokens (truncated SQL)
  empty         nothing generated
  no_table      sqlite: "no such table"   -> schema hallucination
  no_column     sqlite: "no such column"  -> schema hallucination
  syntax        any other sqlite error
  wrong_result  ran cleanly, different rows
  scorer_miss   ran cleanly, byte-identical rows, yet match=False (harness bug)

Also reports the exact-string match rate, which isolates "right SQL, scored
wrong" from "wrong SQL".

  python3 diagnostics/spider_error_analysis.py --generations=<dir>/generations.jsonl
"""
import argparse
import collections
import json
import os
import re
import subprocess

DB_RE = re.compile(r"For db_id:\[(.+)\]")


def run(dbfile, sql):
    r = subprocess.run(["sqlite3", dbfile, sql], capture_output=True, timeout=60)
    return r.stdout, r.stderr.decode("utf-8", errors="replace").strip()


def norm(sql):
    return re.sub(r"\s+", " ", sql.strip().rstrip(";")).lower()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--generations", required=True)
    p.add_argument("--input_file", default="datasets/spider_test.jsonl")
    p.add_argument("--spider_path", default="spider_data/database")
    p.add_argument("--examples", type=int, default=5, help="Examples printed per bucket")
    args = p.parse_args()

    gold = [json.loads(l)["messages"] for l in open(args.input_file) if l.strip()]
    rows = [json.loads(l) for l in open(args.generations) if l.strip()]
    buckets = collections.Counter()
    shown = collections.defaultdict(list)
    exact = 0
    out_rows = []
    for r in rows:
        msgs = gold[r["idx"]]
        db = DB_RE.search(msgs[1]["content"]).group(1)
        dbfile = os.path.join(args.spider_path, db, db + ".sqlite")
        gen, gt = r["response"], msgs[2]["content"]
        exact += norm(gen) == norm(gt)
        if r["match"]:
            b = "correct"
        elif r.get("stop") == "cap":
            b = "cap"
        elif not gen:
            b = "empty"
        else:
            g_out, g_err = run(dbfile, gen)
            t_out, _ = run(dbfile, gt)
            if g_err:
                b = ("no_table" if "no such table" in g_err
                     else "no_column" if "no such column" in g_err else "syntax")
            elif g_out == t_out:
                b = "scorer_miss"
            else:
                b = "wrong_result"
        buckets[b] += 1
        out_rows.append({"idx": r["idx"], "db": db, "bucket": b, "gen": gen, "gold": gt})
        if b != "correct" and len(shown[b]) < args.examples:
            shown[b].append((db, gt, gen))

    n = len(rows)
    print(f"n={n}  execution-match={buckets['correct'] / n:.4f}  normalised-string-match={exact / n:.4f}")
    for b, c in buckets.most_common():
        print(f"  {b:13s} {c:5d}  {100 * c / n:5.1f}%")
    for b, exs in shown.items():
        print(f"\n--- {b} ---")
        for db, gt, gen in exs:
            print(f"  [{db}]\n    gold: {gt}\n    gen : {gen}")
    out = os.path.join(os.path.dirname(args.generations), "spider_buckets.jsonl")
    with open(out, "w") as f:
        f.writelines(json.dumps(x) + "\n" for x in out_rows)
    print(f"\nper-example buckets -> {out}")


if __name__ == "__main__":
    main()
