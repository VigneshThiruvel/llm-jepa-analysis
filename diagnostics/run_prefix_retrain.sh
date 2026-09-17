#!/bin/bash
#SBATCH --account=westai0096
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=03:00:00
#SBATCH --partition=dc-hwai
#SBATCH --array=0-26
#SBATCH --output=slurm/logs/thesis/prefixrt_%A_%a.out
#SBATCH --error=slurm/logs/thesis/prefixrt_%A_%a.err

# Prefix-match for sweep cells whose weights were PURGED: retrain, then score.
# Companion to run_prefix_backfill.sh, which handles cells that still have weights.
# Submit from the repo root:
#   NCONFIGS=1 ./diagnostics/run_prefix_retrain.sh      # prints the --array to use
#   sbatch diagnostics/run_prefix_retrain.sh
#
# Default scope: every seed-82 synth cell with no surviving model (the k=16 and
# k=32 columns plus the λ=128 row) — completing the primary 78-cell grid in both
# metrics. SEEDS= / DS= widen it.
#
# Weights and generations go to ${OUT_ROOT}; the ONLY thing written into
# sweep_runs is a new results_prefix.txt per cell, so the sweep's own results.txt,
# geometry.jsonl and trainer_state.json are untouched and collect_sweep.py picks
# the prefix column up for the whole grid from one table.
#
# The exact rate in that file comes from a *fresh* training run, so it lands in
# the accuracy_exact_recheck column, never in `accuracy` — retrains reproduce to
# ~2 pts, not bit-exactly (the chat template stamps the current date into every
# prompt). Geometry is not recomputed: the original run's geometry.jsonl stands.
set -euo pipefail

DS=${DS:-synth}
case "${DS}" in
  synth|turk) LR=2e-5; MAX_NEW_TOKENS=256 ;;
  gsm8k)      LR=2e-5; MAX_NEW_TOKENS=512 ;;
  *) echo "unknown dataset '${DS}'" >&2; exit 1 ;;
esac
RUNS_DIR=${RUNS_DIR:-sweep_runs/${DS}}
OUT_ROOT=${OUT_ROOT:-diag_runs/prefix_grid/${DS}}
# Same spelling as run_sweep.sh — the tag is the on-disk directory name.
LBDS=(0.125 0.25 0.5 1.0 2.0 4.0 8.0 16.0 32.0 64.0 128.0)
KS=(0 1 2 4 8 16 32)
read -r -a SEEDS <<< "${SEEDS:-82}"

CELLS=()
add_if_purged() {
  local tag=$1
  [ -d "${RUNS_DIR}/${tag}" ] || return 0                       # never run at all
  [ -f "${RUNS_DIR}/${tag}/checkpoints/model.safetensors" ] && return 0  # has weights
  # NOTE: do NOT filter on results_prefix.txt here. The list must be identical for
  # every array task; filtering on work already finished makes it shrink while the
  # array runs, so later indices point at different cells or past the end (that is
  # what killed tasks 15-26 of job 15617829 after ~19s each). The
  # already-done check belongs after the index is resolved, below.
  CELLS+=("${tag}")
}
for S in "${SEEDS[@]}"; do
  add_if_purged "0.0_0_${S}"
  for L in "${LBDS[@]}"; do for K in "${KS[@]}"; do add_if_purged "${L}_${K}_${S}"; done; done
done
if [ -n "${NCONFIGS:-}" ]; then
  echo "${DS} seeds '${SEEDS[*]}': ${#CELLS[@]} purged cells -> --array=0-$(( ${#CELLS[@]} - 1 ))"
  printf '  %s\n' "${CELLS[@]}"; exit 0
fi
TASK=${SLURM_ARRAY_TASK_ID:-0}
[ "${TASK}" -lt "${#CELLS[@]}" ] || { echo "task ${TASK} >= ${#CELLS[@]} cells" >&2; exit 1; }
TAG=${CELLS[$TASK]}
IFS=_ read -r LBD K SEED <<< "${TAG}"
if [ -z "${RERUN:-}" ] && grep -q "^Prefix Rate" "${RUNS_DIR}/${TAG}/results_prefix.txt" 2>/dev/null; then
  echo "=== ${TAG}: already has a prefix rate, skipping ==="; exit 0
fi

cd "${SLURM_SUBMIT_DIR:-$PWD}"
[ -f finetune.py ] && [ -d diagnostics ] || { echo "submit from llm-jepa-analysis/" >&2; exit 1; }
source diagnostics/common.sh
export MASTER_ADDR=$(hostname -I | awk '{print $1}')
export MASTER_PORT=$((29800 + TASK))

EPOCHS=4
DATASET=datasets/${DS}
RUN_DIR=${OUT_ROOT}/${TAG}
CKPT_DIR=${RUN_DIR}/checkpoints
if [ "${LBD}" = "0.0" ]; then ARM=baseline; else ARM=jepa; fi
mkdir -p "${RUN_DIR}"
echo "=== prefix retrain ${DS} ${TAG}: arm=${ARM} lbd=${LBD} k=${K} seed=${SEED} ==="

cat > "${RUN_DIR}/config.json" <<EOF
{"dataset": "${DS}", "arm": "${ARM}", "lr": "${LR}", "lbd": ${LBD}, "k": ${K},
 "seed": ${SEED}, "epochs": ${EPOCHS}, "model": "${MODEL}", "max_new_tokens": ${MAX_NEW_TOKENS},
 "stage": "prefix_grid", "git_commit": "$(git rev-parse --short HEAD 2>/dev/null || echo unknown)",
 "max_predictors": $(grep -oP '^MAX_PREDICTORS = \K\d+' finetune.py)}
EOF

trap 'python3 collect_sweep.py --runs_dir=${RUNS_DIR} || true' EXIT

if [ ! -f "${CKPT_DIR}/model.safetensors" ]; then
  if [ "${ARM}" = "baseline" ]; then
    torchrun --nproc_per_node=4 finetune.py \
      --train_file ${DATASET}_train.jsonl --output_dir=${CKPT_DIR} \
      --num_epochs=${EPOCHS} --finetune_seed=${SEED} \
      --model_name=${MODEL} --learning_rate=${LR} \
      --predictors=0 --regular
  else
    torchrun --nproc_per_node=4 finetune.py \
      --train_file ${DATASET}_train.jsonl --output_dir=${CKPT_DIR} \
      --num_epochs=${EPOCHS} --finetune_seed=${SEED} \
      --model_name=${MODEL} --learning_rate=${LR} \
      --lbd=${LBD} --predictors=${K} --last_token=-2 --additive_mask
  fi
else
  echo "=== ${TAG}: model present, skipping training ==="
fi

GEN_ARGS=(--trace_n=0)
run_eval_sharded "${CKPT_DIR}" "${RUN_DIR}/eval" ${DATASET}_test.jsonl ${MAX_NEW_TOKENS} \
  --reference_results="${RUNS_DIR}/${TAG}/results.txt" | tee "${RUN_DIR}/results.txt"
# The only write into sweep_runs: a new sidecar next to the untouched results.txt.
cp -f "${RUN_DIR}/results.txt" "${RUNS_DIR}/${TAG}/results_prefix.txt"

echo "=== ${TAG} done ($(du -sh "${RUN_DIR}" | cut -f1)) ==="
