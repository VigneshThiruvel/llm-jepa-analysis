"""Gather the seed-82 per-dataset config runs into result CSVs.

Scans <runs_dir>/<dataset>_<arm>/ for config.json + results.txt and writes:

  final_results_seed82.csv       one row per dataset, mirroring the config table
                                 with a baseline_accuracy / jepa_accuracy column
  final_results_seed82_long.csv  one row per (dataset, arm), with the arm's own
                                 lambda / k and a status column

Idempotent: rescans everything present and rewrites both CSVs, so it can be run
after every array task and the last writer produces the complete table.
"""

import argparse
import csv
import glob
import json
import os
import re

RATE_PATTERN = re.compile(r"^Success Rate: (?:\S+), (?P<rate>[\d.]+)", re.MULTILINE)

# Display name + row order for the wide table, matching the config table.
DATASETS = [
    ("synth", "NL-RX-SYNTH"),
    ("turk", "NL-RX-TURK"),
    ("gsm8k", "GSM8K"),
    ("spider", "Spider"),
]
DISPLAY = dict(DATASETS)
ORDER = {name: i for i, (name, _) in enumerate(DATASETS)}

# Spider is execution accuracy (run the SQL, compare result sets); the rest are
# exact match on the generated string.
METRIC = {"spider": "execution accuracy"}


def read_run(run_dir):
    """Return the run's config dict plus its accuracy (None if not finished)."""
    config_file = os.path.join(run_dir, "config.json")
    if not os.path.exists(config_file):
        print(f"[collect] skipping (no config.json): {run_dir}")
        return None
    with open(config_file) as f:
        run = json.load(f)

    results_file = os.path.join(run_dir, "results.txt")
    if not os.path.exists(results_file):
        print(f"[collect] missing {results_file}")
        run["accuracy"] = None
        run["status"] = "missing"
        return run

    with open(results_file) as f:
        rate_match = RATE_PATTERN.search(f.read())
    if not rate_match:
        print(f"[collect] no success rate found in {results_file}")
        run["accuracy"] = None
        run["status"] = "incomplete"
        return run

    run["accuracy"] = float(rate_match.group("rate"))
    run["status"] = "ok"
    return run


def write_csv(path, fieldnames, rows):
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)
    print(f"[collect] wrote {len(rows)} rows to {path}")


def main():
    parser = argparse.ArgumentParser(description="Collect seed-82 config runs into CSVs")
    parser.add_argument("--runs_dir", type=str, default="final_runs_seed82")
    args = parser.parse_args()

    runs = []
    for run_dir in sorted(glob.glob(os.path.join(args.runs_dir, "*_*"))):
        if not os.path.isdir(run_dir):
            continue
        run = read_run(run_dir)
        if run:
            runs.append(run)

    if not runs:
        print(f"[collect] no runs found under {args.runs_dir}")
        return

    runs.sort(key=lambda r: (ORDER.get(r["dataset"], 99), r["arm"]))

    long_rows = [
        {
            "dataset": DISPLAY.get(r["dataset"], r["dataset"]),
            "arm": r["arm"],
            "lr": r["lr"],
            "lambda": r["lbd"],
            "k": r["k"],
            "seed": r["seed"],
            "epochs": r["epochs"],
            "metric": METRIC.get(r["dataset"], "exact match"),
            "accuracy": r["accuracy"],
            "status": r["status"],
        }
        for r in runs
    ]
    write_csv(
        os.path.join(args.runs_dir, "final_results_seed82_long.csv"),
        list(long_rows[0].keys()),
        long_rows,
    )

    # Wide table: lambda/k come from the JEPA arm, since they are what the
    # config table specifies and the baseline arm does not use them.
    by_dataset = {}
    for r in runs:
        entry = by_dataset.setdefault(
            r["dataset"],
            {
                "dataset": DISPLAY.get(r["dataset"], r["dataset"]),
                "lr": r["lr"],
                "lambda": "",
                "k": "",
                "seed": r["seed"],
                "epochs": r["epochs"],
                "metric": METRIC.get(r["dataset"], "exact match"),
                "baseline_accuracy": "",
                "jepa_accuracy": "",
            },
        )
        if r["arm"] == "jepa":
            entry["lambda"] = r["lbd"]
            entry["k"] = r["k"]
        acc = "" if r["accuracy"] is None else r["accuracy"]
        entry[f"{r['arm']}_accuracy"] = acc

    for entry in by_dataset.values():
        base, jepa = entry["baseline_accuracy"], entry["jepa_accuracy"]
        entry["delta"] = round(jepa - base, 6) if base != "" and jepa != "" else ""

    wide_rows = [by_dataset[name] for name, _ in DATASETS if name in by_dataset]
    write_csv(
        os.path.join(args.runs_dir, "final_results_seed82.csv"),
        list(wide_rows[0].keys()),
        wide_rows,
    )


if __name__ == "__main__":
    main()
