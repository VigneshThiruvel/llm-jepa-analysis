#!/bin/bash
#SBATCH --account=westai0096
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --partition=dc-hwai
#SBATCH --array=0-64
#SBATCH --output=slurm/logs/thesis/prefix_%A_%a.out
#SBATCH --error=slurm/logs/thesis/prefix_%A_%a.err

# Prefix-match backfill: re-score sweep cells that still have weights, adding the
# prefix column without retraining anything. Submit from the repo root:
#   NCONFIGS=1 ./diagnostics/run_prefix_backfill.sh     # prints the --array to use
#   sbatch diagnostics/run_prefix_backfill.sh
#
# Eval only — no training, no geometry (purged cells keep their geometry.jsonl
# anyway). Per cell: 4-GPU sharded eval_generations.py with the chat-template date
# pinned to that cell's original eval day, so the exact-match rate reproduces and
# the prefix rate is directly comparable to the number already in the CSV.
#
# Writes ONLY new files into each cell dir: eval_prefix/ and results_prefix.txt.
# results.txt, checkpoints/ and geometry.jsonl are never touched, so the sweep's
# own record is intact and collect_sweep.py picks the new column up on its own.
set -euo pipefail

DS=${DS:-synth}
case "${DS}" in
  synth|turk) MAX_NEW_TOKENS=256 ;;
  gsm8k)      MAX_NEW_TOKENS=512 ;;
  *) echo "unknown dataset '${DS}'" >&2; exit 1 ;;
esac
RUNS_DIR=${RUNS_DIR:-sweep_runs/${DS}}

# Only cells whose final model survived the purge — the rest cannot be scored
# without retraining them (a separate, far more expensive job).
CELLS=()
for d in $(ls -d ${RUNS_DIR}/*_*_*/ 2>/dev/null | sort); do
  [ -f "${d}checkpoints/model.safetensors" ] && CELLS+=("$(basename "${d%/}")")
done
if [ -n "${NCONFIGS:-}" ]; then
  echo "${DS}: ${#CELLS[@]} cells with weights -> --array=0-$(( ${#CELLS[@]} - 1 ))"; exit 0
fi
TASK=${SLURM_ARRAY_TASK_ID:-0}
[ "${TASK}" -lt "${#CELLS[@]}" ] || { echo "task ${TASK} >= ${#CELLS[@]} cells" >&2; exit 1; }
TAG=${CELLS[$TASK]}

cd "${SLURM_SUBMIT_DIR:-$PWD}"
[ -f finetune.py ] && [ -d diagnostics ] || { echo "submit from llm-jepa-analysis/" >&2; exit 1; }
source diagnostics/common.sh

RUN_DIR=${RUNS_DIR}/${TAG}
OUT=${RUN_DIR}/eval_prefix
DATASET=datasets/${DS}
echo "=== prefix backfill ${DS} ${TAG} ==="

if [ -z "${RERUN:-}" ] && grep -q "^Prefix Rate" "${RUN_DIR}/results_prefix.txt" 2>/dev/null; then
  echo "=== ${TAG}: already has a prefix rate, skipping ==="; exit 0
fi
[ -n "${RERUN:-}" ] && rm -rf "${OUT}"

trap 'python3 collect_sweep.py --runs_dir=${RUNS_DIR} || true' EXIT

DATE=$(date -r "${RUN_DIR}/results.txt" "+%d %b %Y")
echo "--- date_string='${DATE}' (from results.txt mtime)"
GEN_ARGS=(--date_string="${DATE}" --trace_n=0)
run_eval_sharded "${RUN_DIR}/checkpoints" "${OUT}" ${DATASET}_test.jsonl ${MAX_NEW_TOKENS} \
  --reference_results="${RUN_DIR}/results.txt" | tee "${RUN_DIR}/results_prefix.txt"

echo "=== ${TAG} done ==="
