"""Per-step training-loss curves for every cell of a runs dir -> one tidy CSV.

Scans <runs_dir>/<lbd>_<k>_<seed>/ (the run_sweep.sh / run_model_grid.sh layout),
reads each cell's trainer_state.json (located exactly as collect_sweep.py does:
the copy preserved at purge time, else checkpoints/, else the newest
checkpoint-*/), and writes one row per logged training step:

  dataset, arm, model, lbd, k, seed, step, epoch, loss, lm_loss, jepa_loss,
  grad_norm, learning_rate

What the columns are (finetune.py, RepresentationTrainer.log):
  - loss       Trainer's running mean of gamma*lm_loss + lbd*jepa_loss over the
               logging interval, averaged across ranks. Scales with lbd by
               construction -- never compare it across lbd.
  - lm_loss    next-token cross-entropy of the LAST micro-batch before the log
               call, rank 0 only (not an interval mean).
  - jepa_loss  1 - cos(Pred(Enc(Text)), Enc(Code)) of that same micro-batch; blank
               on baseline cells (JEPA off).
  - lm_loss_source  'rank0_microbatch' (JEPA cells, the RepresentationTrainer value
               above) or 'trainer_loss' (baseline cells: --regular trains with the
               plain HF Trainer, which logs only `loss`; with JEPA off that IS the
               LM loss, as an all-rank mean over the step, so it is copied here).
With --eval_steps=1 (run_model_grid.sh) every optimizer step is logged, so the
curves have one point per step. Sweep cells logged every 10 steps.

Read-only over the run dirs. Usage:
  python3 diagnostics/collect_losses.py --runs_dir=diag_runs/models/openelm/synth/control
  -> <runs_dir>/loss_curves.csv
"""

import argparse
import csv
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collect_sweep import _load_trainer_state  # noqa: E402

FIELDS = ["dataset", "arm", "model", "lbd", "k", "seed", "step", "epoch",
          "loss", "lm_loss", "jepa_loss", "grad_norm", "learning_rate", "lm_loss_source"]


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--runs_dir", required=True)
    p.add_argument("--output_file", default=None, help="default: <runs_dir>/loss_curves.csv")
    args = p.parse_args()

    rows, missing = [], []
    for run_dir in sorted(glob.glob(os.path.join(args.runs_dir, "*_*_*"))):
        cfg_file = os.path.join(run_dir, "config.json")
        if not os.path.isfile(cfg_file):
            continue
        cfg = json.load(open(cfg_file))
        state_file = _load_trainer_state(run_dir)
        if state_file is None:
            missing.append(os.path.basename(run_dir))
            continue
        for h in json.load(open(state_file)).get("log_history", []):
            if "loss" not in h:  # end-of-training summary rows carry train_loss, not loss
                continue
            row = {
                "dataset": cfg.get("dataset"), "arm": cfg.get("arm"), "model": cfg.get("model"),
                "lbd": cfg.get("lbd"), "k": cfg.get("k"), "seed": cfg.get("seed"),
                **{f: h.get(f) for f in FIELDS[6:-1]},
                "step": h.get("step"), "lm_loss_source": "rank0_microbatch",
            }
            if row["lm_loss"] is None and cfg.get("arm") == "baseline":
                row["lm_loss"], row["lm_loss_source"] = h.get("loss"), "trainer_loss"
            rows.append(row)

    out = args.output_file or os.path.join(args.runs_dir, "loss_curves.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"[collect_losses] {len(rows)} step rows from "
          f"{len({(r['lbd'], r['k'], r['seed']) for r in rows})} cells -> {out}"
          + (f"; no trainer_state.json yet: {', '.join(missing)}" if missing else ""))


if __name__ == "__main__":
    main()
