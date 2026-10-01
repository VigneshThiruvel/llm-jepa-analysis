#!/bin/bash
#SBATCH --account=westai0096
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=00:40:00
#SBATCH --partition=dc-hwai
#SBATCH --output=slurm/logs/thesis/k16diag_date_%j.out
#SBATCH --error=slurm/logs/thesis/k16diag_date_%j.err
#
# Is the stage-1 vs sweep accuracy gap the eval prompt's date? Stage 1 evaluated with
# date_string=None (= its generation day, 11 Sep 2026); the sweep evaluated 2.0_8_82 on
# 10 Aug 2026. Re-eval the *stage-1* checkpoint with the date pinned to each:
#   A  "11 Sep 2026" -> must reproduce stage 1's 0.7355 (same prompt as stage 1)
#   B  "10 Aug 2026" -> same prompt as the sweep's 0.6985
# Same eval path as run_stage0.sh (eval_generations.py, sharded over 4 GPUs). Writes only
# under diag_runs/k16/date_check/; nothing in stage1/ or sweep_runs/ is touched.
#   sbatch diagnostics/run_date_check.sh

[ -f finetune.py ] && [ -d diagnostics ] || { echo "submit from llm-jepa-analysis/" >&2; exit 1; }
source diagnostics/common.sh
nvidia-smi --query-gpu=index,name,memory.used --format=csv

TAG=2.0_8_82
CKPT=diag_runs/k16/stage1/synth/${TAG}/checkpoints
OUT_ROOT=diag_runs/k16/date_check

for SPEC in "11Sep|11 Sep 2026|diag_runs/k16/stage1/synth/${TAG}/results.txt" \
            "10Aug|10 Aug 2026|sweep_runs/synth/${TAG}/results.txt"; do
  IFS='|' read -r LABEL DATE REF <<< "${SPEC}"
  EOUT=${OUT_ROOT}/${TAG}_${LABEL}
  if [ -s "${EOUT}/summary.json" ]; then echo "--- ${EOUT}: done, skipping"; continue; fi
  mkdir -p "${EOUT}"
  echo "=== $(date +%T) ${TAG} stage-1 checkpoint, date_string='${DATE}', reference ${REF} ==="
  GEN_ARGS=(--date_string="${DATE}" --trace_n=50)
  run_eval_sharded "${CKPT}" "${EOUT}" datasets/synth_test.jsonl 256 \
    --reference_results="${REF}" | tee "${EOUT}/report.txt"
done
