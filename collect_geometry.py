"""Gather per-run geometry.jsonl files into one tidy CSV.

Scans <runs_dir>/<dataset>_<arm>/geometry.jsonl (written by geometry.py via
run_geometry.sh) and writes:

  geometry_tidy.csv   one row per (dataset, arm, lbd, k, seed, step, split, layer)

Each geometry.jsonl row already carries lbd/k/seed/step/split/layer plus the
metrics; this collector attaches the dataset and arm from the run's config.json
so the geometry table can be joined to final_results_seed82_long.csv on
(dataset, arm, seed). Idempotent: rescans everything present and rewrites the CSV.

Read-only over the run dirs — it touches nothing in the training/eval path.
"""

import argparse
import csv
import glob
import json
import os

# Column order: identifiers first, then the geometric scalars.
METRIC_COLS = [
    "rankme_text", "rankme_code",
    "uniformity_text", "uniformity_code",
    "alignment_cos", "linear_r2_text_to_code",
]
ID_COLS = ["dataset", "arm", "lbd", "k", "seed", "step", "split", "layer", "n"]


def main():
    parser = argparse.ArgumentParser(description="Collect per-run geometry into a tidy CSV")
    parser.add_argument("--runs_dir", type=str, default="final_runs_seed82")
    args = parser.parse_args()

    rows = []
    for geo_file in sorted(glob.glob(os.path.join(args.runs_dir, "*_*", "geometry.jsonl"))):
        run_dir = os.path.dirname(geo_file)
        config_file = os.path.join(run_dir, "config.json")
        if not os.path.exists(config_file):
            print(f"[collect] skipping (no config.json): {run_dir}")
            continue
        with open(config_file) as f:
            cfg = json.load(f)

        with open(geo_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                r["dataset"] = cfg["dataset"]
                r["arm"] = cfg["arm"]
                rows.append(r)

    if not rows:
        print(f"[collect] no geometry rows found under {args.runs_dir}")
        return

    rows.sort(key=lambda r: (r["dataset"], r["arm"], r["seed"],
                             _as_int(r["step"]), r["split"], r["layer"]))

    fieldnames = ID_COLS + METRIC_COLS
    out_path = os.path.join(args.runs_dir, "geometry_tidy.csv")
    tmp = out_path + ".tmp"
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, out_path)
    print(f"[collect] wrote {len(rows)} rows to {out_path}")


def _as_int(step):
    try:
        return int(step)
    except (TypeError, ValueError):
        return step


if __name__ == "__main__":
    main()
