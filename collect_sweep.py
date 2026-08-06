"""Gather sweep-run outputs into tidy CSVs.

Scans <runs_dir>/<lbd>_<k>_<seed>/ (written by run_sweep.sh) for:
  - config.json + results.txt -> accuracy_tidy.csv  (one row per (lbd, k, seed))
  - geometry.jsonl            -> geometry_tidy.csv   (one row per (lbd, k, seed,
                                                      step, split, layer))

The two tables share the (lbd, k, seed) key, so the mediation regressions can
join accuracy onto the per-checkpoint geometry directly. Idempotent: rescans
everything present and rewrites both CSVs, so it is safe to run after every array
task. Read-only over the run dirs.
"""

import argparse
import csv
import glob
import json
import os
import re

RATE_PATTERN = re.compile(r"^Success Rate: (?:\S+), (?P<rate>[\d.]+)", re.MULTILINE)

GEO_ID_COLS = ["lbd", "k", "seed", "step", "split", "layer", "n"]
GEO_METRIC_COLS = [
    "rankme_text", "rankme_code",
    "uniformity_text", "uniformity_code",
    "alignment_cos", "linear_r2_text_to_code",
]


def _as_int(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return x


def main():
    parser = argparse.ArgumentParser(description="Collect sweep results into tidy CSVs")
    parser.add_argument("--runs_dir", type=str, default="sweep_runs/synth")
    args = parser.parse_args()

    accuracy_rows = []
    geometry_rows = []

    for run_dir in sorted(glob.glob(os.path.join(args.runs_dir, "*_*_*"))):
        if not os.path.isdir(run_dir):
            continue
        config_file = os.path.join(run_dir, "config.json")
        if not os.path.exists(config_file):
            print(f"[collect] skipping (no config.json): {run_dir}")
            continue
        with open(config_file) as f:
            cfg = json.load(f)
        meta = {"lbd": cfg["lbd"], "k": cfg["k"], "seed": cfg["seed"], "arm": cfg["arm"]}

        results_file = os.path.join(run_dir, "results.txt")
        if os.path.exists(results_file):
            with open(results_file) as f:
                rate_match = RATE_PATTERN.search(f.read())
            if rate_match:
                accuracy_rows.append({**meta, "accuracy": float(rate_match.group("rate"))})
            else:
                print(f"[collect] no success rate found in {results_file}")
        else:
            print(f"[collect] missing {results_file}")

        geometry_file = os.path.join(run_dir, "geometry.jsonl")
        if os.path.exists(geometry_file):
            with open(geometry_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        geometry_rows.append(json.loads(line))
        else:
            print(f"[collect] missing {geometry_file}")

    accuracy_rows.sort(key=lambda r: (r["lbd"], r["k"], r["seed"]))
    geometry_rows.sort(key=lambda r: (r["lbd"], r["k"], r["seed"],
                                      _as_int(r["step"]), r["split"], r["layer"]))

    def write_csv(path, rows, fieldnames=None):
        if not rows:
            print(f"[collect] no rows for {path}")
            return
        fieldnames = fieldnames or list(rows[0].keys())
        tmp = path + ".tmp"
        with open(tmp, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp, path)
        print(f"[collect] wrote {len(rows)} rows to {path}")

    write_csv(os.path.join(args.runs_dir, "accuracy_tidy.csv"),
              accuracy_rows, ["lbd", "k", "seed", "arm", "accuracy"])
    write_csv(os.path.join(args.runs_dir, "geometry_tidy.csv"),
              geometry_rows, GEO_ID_COLS + GEO_METRIC_COLS)


if __name__ == "__main__":
    main()
