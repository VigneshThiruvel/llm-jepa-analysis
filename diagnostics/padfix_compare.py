"""Cell-by-cell padfix vs control vs sweep table for the stage-5 pad-token test.

The tidy CSVs written by collect_sweep.py are what plot_sweep.py consumes, one pair
per (dataset, model), and the padfix runs stamp `dataset` as "<ds>_<arm>" so each arm
is its own table. This script joins those arms back together for reading: one row per
(dataset, lbd, k, seed) with the control accuracy, the padfix accuracy, their
difference, and the original sweep cell alongside. Termination columns come from each
run's eval/summary.json (frac_cap, the share of generations that hit the token cap
instead of stopping, and the teacher-forced p(eot) after the gold answer), which is
where the mechanism shows up.

    python3 diagnostics/padfix_compare.py                      # prints + writes CSV
    python3 diagnostics/padfix_compare.py --out_csv path.csv

Reads only; writes a single CSV (default under the stage-5 root).
"""

import argparse
import csv
import glob
import json
import os

ROOT = "diag_runs/k16/stage5_padfix"
ARMS = ["control", "padfix"]


def read_arm(ds, arm):
    """{(lbd, k, seed): row} from the arm's tidy CSV, with termination columns joined on."""
    out = {}
    pattern = os.path.join(ROOT, ds, arm, f"accuracy_tidy_{ds}_{arm}_*.csv")
    for path in sorted(glob.glob(pattern)):
        for r in csv.DictReader(open(path)):
            key = (float(r["lbd"]), int(r["k"]), int(r["seed"]))
            tag = f"{r['lbd']}_{r['k']}_{r['seed']}"
            summary = os.path.join(ROOT, ds, arm, tag, "eval", "summary.json")
            if os.path.exists(summary):
                s = json.load(open(summary))
                r["frac_cap"] = s.get("frac_cap")
                r["p_eot"] = s.get("tf_p_eot_median")
                r["n_gen_mean"] = s.get("n_gen_mean")
            out[key] = r
    return out


def read_sweep(ds):
    out = {}
    for path in sorted(glob.glob(f"sweep_runs/{ds}/accuracy_tidy_{ds}_*.csv")):
        for r in csv.DictReader(open(path)):
            out[(float(r["lbd"]), int(r["k"]), int(r["seed"]))] = r
    return out


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", default="synth,turk")
    p.add_argument("--out_csv", default=os.path.join(ROOT, "padfix_vs_control.csv"))
    args = p.parse_args()

    rows = []
    for ds in args.datasets.split(","):
        arms = {a: read_arm(ds, a) for a in ARMS}
        sweep = read_sweep(ds)
        for key in sorted(set().union(*[set(a) for a in arms.values()]) if any(arms.values()) else []):
            lbd, k, seed = key
            c, f = arms["control"].get(key), arms["padfix"].get(key)
            ca, fa = num(c and c.get("accuracy")), num(f and f.get("accuracy"))
            rows.append({
                "dataset": ds, "lbd": lbd, "k": k, "seed": seed,
                "sweep": num((sweep.get(key) or {}).get("accuracy")),
                "control": ca, "padfix": fa,
                "delta_padfix_minus_control": None if None in (ca, fa) else round(fa - ca, 4),
                "control_prefix": num(c and c.get("accuracy_prefix")),
                "padfix_prefix": num(f and f.get("accuracy_prefix")),
                "control_frac_cap": num(c and c.get("frac_cap")),
                "padfix_frac_cap": num(f and f.get("frac_cap")),
                "control_p_eot": num(c and c.get("p_eot")),
                "padfix_p_eot": num(f and f.get("p_eot")),
                "control_status": (c or {}).get("status"),
                "padfix_status": (f or {}).get("status"),
            })

    if not rows:
        raise SystemExit(f"no padfix tables under {ROOT} — run collect_sweep.py per arm first")

    with open(args.out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    fmt = lambda v: "  -   " if v is None else f"{v:6.4f}"  # noqa: E731
    print(f"{'dataset':6} {'lbd':>6} {'k':>3} {'seed':>4} {'sweep':>6} {'ctl':>6} {'fix':>6} "
          f"{'Δ':>7} {'ctlCap':>6} {'fixCap':>6}")
    for r in rows:
        d = r["delta_padfix_minus_control"]
        print(f"{r['dataset']:6} {r['lbd']:6} {r['k']:3} {r['seed']:4} {fmt(r['sweep'])} "
              f"{fmt(r['control'])} {fmt(r['padfix'])} {'   -   ' if d is None else f'{d:+7.4f}'} "
              f"{fmt(r['control_frac_cap'])} {fmt(r['padfix_frac_cap'])}")
    print(f"\n[padfix_compare] {len(rows)} rows -> {args.out_csv}")


if __name__ == "__main__":
    main()
