"""Gather sweep-run outputs into tidy CSVs, with a training-health verdict.

Scans <runs_dir>/<lbd>_<k>_<seed>/ (written by run_sweep.sh) for:
  - config.json + results.txt + trainer_state.json -> accuracy_tidy.csv
                                                      (one row per (lbd, k, seed))
  - geometry.jsonl -> geometry_tidy.csv  (one row per (lbd, k, seed, step, split,
                                          layer))
  - a plain-text health_report.txt listing every cell that did not train cleanly

The two tables share the (lbd, k, seed) key, so the mediation regressions can
join accuracy onto the per-checkpoint geometry directly. Idempotent: rescans
everything present and rewrites both CSVs, so it is safe to run after every array
task. Read-only over the run dirs.

WHY THE HEALTH VERDICT EXISTS
-----------------------------
At large lambda the JEPA term can break optimization outright rather than merely
hurt accuracy. A broken run still writes a checkpoint and still evals, so it
lands in the CSV as a perfectly ordinary-looking `accuracy` — 0.0, or whatever
a degenerate model happens to score — and gets averaged into a dose-response
curve as if it were a measurement. "lambda >= X breaks training" is a real
result, but it is a *different kind* of result from "accuracy fell to 0.0", and
the two must not be summed.

So: `status` classifies the run, `accuracy` is **blank unless status == 'ok'**
(so a careless `.mean()` skips it instead of averaging a broken cell in), and
`accuracy_raw` always carries the parsed number for anyone who wants it.

WHAT IS *NOT* A FAILURE SIGNAL (read before adding a rule here)
---------------------------------------------------------------
- **Absolute loss.** The Trainer logs `total = gamma*lm_loss + lbd*jepa_loss`,
  so the logged loss scales with lambda *by construction*. Any absolute-loss
  threshold would flag every high-lambda cell as broken and manufacture exactly
  the bound the sweep is supposed to test. Only scale-free comparisons (a run
  against its own first step, or lm_loss on its own) are admissible.
- **Low rankme / high alignment_cos.** Measured on the finished tranche-1 cells,
  rankme_text at the last layer falls 237 (lambda=0) -> 8.2 (lambda=0.5, k=0) ->
  3.8 (lambda=2, k=0) while alignment_cos rises 0.39 -> 0.99 -> 0.998 — and
  those cells score *above* baseline (0.695 vs 0.5945 at lambda=2). Representation
  collapse is what the JEPA term is *for*; it is the treatment, not a fault.
  Geometry is recorded here as raw numbers and is deliberately never allowed to
  set `status`.

The only admissible signals are non-finite values and a run's own objective
failing to improve — both scale-free, both lambda-invariant.
"""

import argparse
import csv
import glob
import json
import math
import os
import re

RATE_PATTERN = re.compile(r"^Success Rate: (?:\S+), (?P<rate>[\d.]+)", re.MULTILINE)

GEO_ID_COLS = ["dataset", "model", "lbd", "k", "seed", "step", "split", "layer", "n"]
GEO_METRIC_COLS = [
    "rankme_text", "rankme_code",
    "uniformity_text", "uniformity_code",
    "alignment_cos", "linear_r2_text_to_code",
]

ACC_COLS = [
    # Identity first. Every run dir writes the same basenames, so a row that
    # travels (copied to a laptop, concatenated with another dataset) must carry
    # its own dataset and model rather than relying on which folder it sat in.
    "dataset", "model",
    # key + verdict next, so the verdict is impossible to miss in a join
    "lbd", "k", "seed", "arm", "status", "accuracy", "accuracy_raw", "status_detail",
    # training diagnostics (scale-free comparisons only; see module docstring)
    "loss_first", "loss_last", "train_loss",
    "lm_loss_first", "lm_loss_last", "jepa_loss_first", "jepa_loss_last",
    "grad_norm_med", "grad_norm_max", "nonfinite_steps", "logged_steps",
    # geometry at the final checkpoint (recorded, NEVER used to set status)
    "rankme_text_final", "rankme_code_final", "alignment_cos_final",
]


def _as_int(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return x


def _finite(x):
    return isinstance(x, (int, float)) and math.isfinite(x)


def _median(xs):
    if not xs:
        return None
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def _load_trainer_state(run_dir):
    """Final trainer_state.json, falling back to the newest checkpoint's copy.

    The top-level file only appears once training finishes, so the fallback is
    what gives an in-flight or half-finished cell a partial verdict instead of
    no verdict at all.
    """
    # Cells whose checkpoints were purged (run_sweep.sh tranche 3) keep a copy of
    # the training log at the run root. Checked first: for a purged cell it is the
    # only copy left, and it is identical to the one under checkpoints/ otherwise.
    preserved = os.path.join(run_dir, "trainer_state.json")
    if os.path.exists(preserved):
        return preserved

    final = os.path.join(run_dir, "checkpoints", "trainer_state.json")
    if os.path.exists(final):
        return final
    ckpts = glob.glob(os.path.join(run_dir, "checkpoints", "checkpoint-*", "trainer_state.json"))
    if not ckpts:
        return None
    return max(ckpts, key=lambda p: _as_int(os.path.basename(os.path.dirname(p)).split("-")[-1]))


def training_health(run_dir):
    """Classify how a cell's training went, from its trainer_state.json alone.

    Returns (status, detail, diagnostics). Statuses:
      ok            — trained, objective improved, all values finite
      no_train      — no trainer_state.json (never started, or died before step 1)
      nonfinite     — NaN/Inf in a logged loss or grad_norm
      lm_diverged   — lm_loss ended above where it started (needs component logging)
      not_converged — total loss ended at or above its first logged value
                      (fallback when lm_loss was not logged for this cell)
    """
    diag = {c: None for c in ACC_COLS if c not in
            ("dataset", "model", "lbd", "k", "seed", "arm",
             "status", "accuracy", "accuracy_raw", "status_detail")}

    state_file = _load_trainer_state(run_dir)
    if state_file is None:
        return "no_train", "no trainer_state.json", diag

    with open(state_file) as f:
        history = json.load(f).get("log_history", [])

    steps = [e for e in history if "loss" in e and "train_loss" not in e]
    summary = next((e for e in history if "train_loss" in e), None)
    if not steps:
        return "no_train", "trainer_state.json has no logged steps", diag

    def series(key):
        return [e[key] for e in steps if key in e]

    losses = series("loss")
    grads = series("grad_norm")
    lm = series("lm_loss")
    jepa = series("jepa_loss")

    diag.update(
        loss_first=losses[0] if losses else None,
        loss_last=losses[-1] if losses else None,
        train_loss=summary.get("train_loss") if summary else None,
        lm_loss_first=lm[0] if lm else None,
        lm_loss_last=lm[-1] if lm else None,
        jepa_loss_first=jepa[0] if jepa else None,
        jepa_loss_last=jepa[-1] if jepa else None,
        grad_norm_med=_median(grads),
        grad_norm_max=max(grads) if grads else None,
        logged_steps=len(steps),
    )

    # 1. Non-finite anywhere — unambiguous, scale-free.
    checked = losses + grads + lm + jepa + ([diag["train_loss"]] if diag["train_loss"] is not None else [])
    bad = [v for v in checked if not _finite(v)]
    diag["nonfinite_steps"] = len(bad)
    if bad:
        return "nonfinite", f"{len(bad)} non-finite logged value(s)", diag

    # 2. The run's own objective got worse. Scale-free: each run is compared only
    #    against its own first logged step, never against another lambda's.
    #    lm_loss is the sharper test — it isolates "the LM broke" from "the JEPA
    #    term never converged" — but is only present for cells trained after the
    #    component-logging patch (2026-08-10), so fall back to total loss.
    if lm:
        if lm[-1] > lm[0]:
            return "lm_diverged", f"lm_loss {lm[0]:.3f} -> {lm[-1]:.3f} (worse than init)", diag
    elif losses and losses[-1] >= losses[0]:
        return "not_converged", f"loss {losses[0]:.3f} -> {losses[-1]:.3f} (no improvement)", diag

    return "ok", "", diag


def final_geometry(geometry_rows):
    """rankme / alignment at the last checkpoint, last layer, test split.

    Recorded for analysis only. Explicitly not part of the health verdict — see
    the module docstring on why low rankme is the treatment, not a fault.
    """
    test = [r for r in geometry_rows if r.get("split") == "test"]
    if not test:
        return {}
    last_step = max(_as_int(r["step"]) for r in test)
    last_layer = max(r["layer"] for r in test)
    match = [r for r in test
             if _as_int(r["step"]) == last_step and r["layer"] == last_layer]
    if not match:
        return {}
    r = match[0]
    return {
        "rankme_text_final": r.get("rankme_text"),
        "rankme_code_final": r.get("rankme_code"),
        "alignment_cos_final": r.get("alignment_cos"),
    }


def main():
    parser = argparse.ArgumentParser(description="Collect sweep results into tidy CSVs")
    parser.add_argument("--runs_dir", type=str, default="sweep_runs/synth")
    args = parser.parse_args()

    accuracy_rows = []
    geometry_rows = []
    unhealthy = []

    for run_dir in sorted(glob.glob(os.path.join(args.runs_dir, "*_*_*"))):
        if not os.path.isdir(run_dir):
            continue
        config_file = os.path.join(run_dir, "config.json")
        if not os.path.exists(config_file):
            print(f"[collect] skipping (no config.json): {run_dir}")
            continue
        with open(config_file) as f:
            cfg = json.load(f)
        # Fall back to the run dir's parent for pre-2026-08-16 cells whose
        # config.json predates these fields.
        ident = {
            "dataset": cfg.get("dataset") or os.path.basename(os.path.dirname(run_dir.rstrip("/"))),
            "model": cfg.get("model", "unknown"),
        }
        meta = {**ident, "lbd": cfg["lbd"], "k": cfg["k"],
                "seed": cfg["seed"], "arm": cfg["arm"]}

        cell_geometry = []
        geometry_file = os.path.join(run_dir, "geometry.jsonl")
        if os.path.exists(geometry_file):
            with open(geometry_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        # geometry.jsonl rows carry only (lbd,k,seed,step,...);
                        # stamp identity on so the table stands alone.
                        cell_geometry.append({**ident, **json.loads(line)})
            geometry_rows.extend(cell_geometry)
        else:
            print(f"[collect] missing {geometry_file}")

        results_file = os.path.join(run_dir, "results.txt")
        rate = None
        if os.path.exists(results_file):
            with open(results_file) as f:
                rate_match = RATE_PATTERN.search(f.read())
            if rate_match:
                rate = float(rate_match.group("rate"))
            else:
                print(f"[collect] no success rate found in {results_file}")
        else:
            print(f"[collect] missing {results_file}")

        if rate is None:
            continue  # cell has not been evaluated yet; nothing to report

        status, detail, diag = training_health(run_dir)
        row = {
            **meta,
            "status": status,
            # Blank unless the run trained cleanly, so a broken cell cannot be
            # silently averaged into a dose-response curve. Raw value kept below.
            "accuracy": rate if status == "ok" else None,
            "accuracy_raw": rate,
            "status_detail": detail,
            **diag,
            **final_geometry(cell_geometry),
        }
        accuracy_rows.append(row)
        if status != "ok":
            unhealthy.append(row)

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

    # One pair of CSVs per (dataset, model), with both in the filename. Every
    # runs_dir used to write the same two basenames, so copying synth's and
    # turk's tables into one folder silently clobbered one of them; a second
    # model in the same runs_dir would have done the same. Both are now
    # distinguishable by name AND self-identifying by column.
    groups = sorted({(r["dataset"], r["model"]) for r in accuracy_rows} |
                    {(r["dataset"], r["model"]) for r in geometry_rows})
    for dataset, model in groups:
        slug = f"{dataset}_{model.split('/')[-1]}"
        acc_g = [r for r in accuracy_rows if (r["dataset"], r["model"]) == (dataset, model)]
        geo_g = [r for r in geometry_rows if (r["dataset"], r["model"]) == (dataset, model)]
        write_csv(os.path.join(args.runs_dir, f"accuracy_tidy_{slug}.csv"), acc_g, ACC_COLS)
        write_csv(os.path.join(args.runs_dir, f"geometry_tidy_{slug}.csv"),
                  geo_g, GEO_ID_COLS + GEO_METRIC_COLS)

    # Pre-2026-08-16 unsuffixed files are no longer updated; say so rather than
    # leaving a stale table that looks current.
    for legacy in ("accuracy_tidy.csv", "geometry_tidy.csv"):
        p = os.path.join(args.runs_dir, legacy)
        if os.path.exists(p):
            print(f"[collect] NOTE: {p} is legacy and no longer updated — safe to delete "
                  f"(regenerated as accuracy_tidy_<dataset>_<model>.csv)")

    # Human-readable failure log: the failure mode itself, not a score standing
    # in for it. Rewritten from scratch each pass, like the CSVs.
    report_path = os.path.join(args.runs_dir, "health_report.txt")
    lines = [
        "Training-health report for " + args.runs_dir,
        f"{len(accuracy_rows)} evaluated cells, {len(unhealthy)} did not train cleanly.",
        "",
        "'accuracy' is blank for these cells in accuracy_tidy.csv; the score they",
        "happened to produce is in 'accuracy_raw'. A broken cell is evidence about",
        "the lambda bound, not a point on the accuracy curve — do not fit through it.",
        "",
    ]
    if unhealthy:
        for r in unhealthy:
            lines.append(
                f"  lbd={r['lbd']:<8g} k={r['k']:<3} seed={r['seed']:<4} "
                f"{r['status']:<14} {r['status_detail']}"
            )
            lines.append(
                f"      accuracy_raw={r['accuracy_raw']} "
                f"loss {r['loss_first']} -> {r['loss_last']} "
                f"grad_norm_max={r['grad_norm_max']}"
            )
    else:
        lines.append("  (none)")
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[collect] wrote {report_path}")

    if unhealthy:
        print(f"[collect] WARNING: {len(unhealthy)} cell(s) did not train cleanly:")
        for r in unhealthy:
            print(f"[collect]   lbd={r['lbd']:g} k={r['k']} seed={r['seed']}: "
                  f"{r['status']} — {r['status_detail']}")


if __name__ == "__main__":
    main()
