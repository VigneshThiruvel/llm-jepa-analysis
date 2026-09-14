#!/bin/bash
#SBATCH --account=westai0096
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=03:00:00
#SBATCH --partition=dc-hwai
#SBATCH --array=0-5
#SBATCH --output=slurm/logs/thesis/k16diag_s1_%A_%a.out
#SBATCH --error=slurm/logs/thesis/k16diag_s1_%A_%a.err

# k=16 diagnosis, stage 1 — retrain a small ladder of cells, KEEP the weights,
# and score them with the per-example eval. Submit from the repo root:
#   sbatch diagnostics/run_stage1.sh
#
# Default ladder, seed 82 (brackets the k=16 transition, a control on each side):
#   k=16 at λ 0.5 (0.454) / 1.0 (0.077) / 2.0 (0.000) / 4.0 (0.000)
#   k=8  at λ 2.0 (0.709)   healthy neighbour
#   k=32 at λ 2.0 (0.626)   the non-monotone-in-k control
# Widen later without editing this file, e.g. more seeds:
#   CELLS="2.0:16:23 2.0:16:37" sbatch --array=0-1 diagnostics/run_stage1.sh
#   (NCONFIGS=1 CELLS="..." ./diagnostics/run_stage1.sh prints the --array to use)
#
# Per cell: train (identical command to run_sweep.sh) → eval_generations.py over
# 4 GPUs (results.txt carries the Success Rate line) → tied-head probe over every
# checkpoint → geometry.py per checkpoint (as run_sweep.sh) → collect_sweep.py.
# Writes only under ${OUT_ROOT}; sweep_runs/ is read (as the reference) never
# written. No purge: ~12 GB per cell stays on disk so later stages can re-probe.
# Skip guards per stage; RERUN=1 redoes the cell.
set -euo pipefail

DS=${DS:-synth}
case "${DS}" in
  synth|turk) LR=2e-5; MAX_NEW_TOKENS=256 ;;
  gsm8k)      LR=2e-5; MAX_NEW_TOKENS=512 ;;
  *) echo "unknown dataset '${DS}'" >&2; exit 1 ;;
esac
# λ strings must keep run_sweep.sh's spelling (0.5 / 1.0 / 2.0) so the tag matches
# the sweep cell used as the reference.
read -r -a CELLS <<< "${CELLS:-0.5:16:82 1.0:16:82 2.0:16:82 4.0:16:82 2.0:8:82 2.0:32:82}"
if [ -n "${NCONFIGS:-}" ]; then
  echo "${#CELLS[@]} cells -> --array=0-$(( ${#CELLS[@]} - 1 ))"; exit 0
fi
TASK=${SLURM_ARRAY_TASK_ID:-0}
[ "${TASK}" -lt "${#CELLS[@]}" ] || { echo "task ${TASK} >= ${#CELLS[@]} cells" >&2; exit 1; }
IFS=: read -r LBD K SEED <<< "${CELLS[$TASK]}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
[ -f finetune.py ] && [ -d diagnostics ] || { echo "submit from llm-jepa-analysis/" >&2; exit 1; }
source diagnostics/common.sh
nvidia-smi --query-gpu=index,name,memory.used --format=csv

export MASTER_ADDR=$(hostname -I | awk '{print $1}')
export MASTER_PORT=$((29700 + TASK))

EPOCHS=4
DATASET=datasets/${DS}
OUT_ROOT=${OUT_ROOT:-diag_runs/k16/stage1/${DS}}
TAG="${LBD}_${K}_${SEED}"
RUN_DIR=${OUT_ROOT}/${TAG}
CKPT_DIR=${RUN_DIR}/checkpoints
REF=sweep_runs/${DS}/${TAG}/results.txt
if [ "${LBD}" = "0.0" ]; then ARM=baseline; else ARM=jepa; fi

[ -n "${RERUN:-}" ] && rm -rf "${RUN_DIR}"
mkdir -p "${RUN_DIR}"
echo "=== stage1 ${DS} ${TAG}: arm=${ARM} lbd=${LBD} k=${K} seed=${SEED} lr=${LR} -> ${RUN_DIR} ==="

cat > "${RUN_DIR}/config.json" <<EOF
{"dataset": "${DS}", "arm": "${ARM}", "lr": "${LR}", "lbd": ${LBD}, "k": ${K},
 "seed": ${SEED}, "epochs": ${EPOCHS}, "model": "${MODEL}", "max_new_tokens": ${MAX_NEW_TOKENS},
 "stage": "k16_stage1", "git_commit": "$(git rev-parse --short HEAD 2>/dev/null || echo unknown)",
 "max_predictors": $(grep -oP '^MAX_PREDICTORS = \K\d+' finetune.py)}
EOF

trap 'python3 collect_sweep.py --runs_dir=${OUT_ROOT} || true' EXIT

# --- 1. Train: same command as run_sweep.sh ---
if [ ! -f "${CKPT_DIR}/model.safetensors" ]; then
  if [ "${ARM}" = "baseline" ]; then
    torchrun --nproc_per_node=4 finetune.py \
      --train_file ${DATASET}_train.jsonl --output_dir=${CKPT_DIR} \
      --num_epochs=${EPOCHS} --finetune_seed=${SEED} \
      --model_name=${MODEL} --learning_rate=${LR} \
      --predictors=0 --regular --save_epoch_checkpoints
  else
    torchrun --nproc_per_node=4 finetune.py \
      --train_file ${DATASET}_train.jsonl --output_dir=${CKPT_DIR} \
      --num_epochs=${EPOCHS} --finetune_seed=${SEED} \
      --model_name=${MODEL} --learning_rate=${LR} \
      --lbd=${LBD} --predictors=${K} --last_token=-2 --additive_mask \
      --save_epoch_checkpoints
  fi
else
  echo "=== ${TAG}: final model present, skipping training ==="
fi
cp -f "${CKPT_DIR}/trainer_state.json" "${RUN_DIR}/trainer_state.json" 2>/dev/null || true

# --- 2. Per-example eval (4 GPUs); results.txt gets the Success Rate line ---
if ! grep -q "^Success Rate" "${RUN_DIR}/results.txt" 2>/dev/null || [ ! -s "${RUN_DIR}/eval/summary.json" ]; then
  GEN_ARGS=(--trace_n=50)
  run_eval_sharded "${CKPT_DIR}" "${RUN_DIR}/eval" ${DATASET}_test.jsonl ${MAX_NEW_TOKENS} \
    --reference_results="${REF}" | tee "${RUN_DIR}/results.txt"
else
  echo "=== ${TAG}: eval done, skipping ==="
fi

# --- 3. Tied-head probe over every checkpoint, forward pass on 64 test prompts ---
if [ ! -s "${RUN_DIR}/tied_head_probe.csv" ]; then
  python3 diagnostics/tied_head_probe.py --cells "${RUN_DIR}" --include_final \
    --input_file=${DATASET}_test.jsonl --forward_examples=64 \
    --output_file="${RUN_DIR}/tied_head_probe.csv"
fi

# --- 4. Geometry, one checkpoint per GPU (same geometry.py call as run_sweep.sh) ---
GEO_FILE=${RUN_DIR}/geometry.jsonl
if [ ! -s "${GEO_FILE}" ]; then
  mkdir -p "${RUN_DIR}/geo_parts"
  pids=(); g=0
  for ckpt in $(ls -d ${CKPT_DIR}/checkpoint-* | sort -t- -k2 -n); do
    STEP=$(basename "${ckpt}" | sed 's/checkpoint-//')
    CUDA_VISIBLE_DEVICES=${GPUS[$(( g % NGPU ))]} python3 geometry.py \
      --model_name="${ckpt}" --original_model_name="${MODEL}" \
      --input_files="train:${DATASET}_train.jsonl,test:${DATASET}_test.jsonl" \
      --output_file="${RUN_DIR}/geo_parts/${STEP}.jsonl" --max_examples=500 \
      --lbd=${LBD} --k=${K} --seed=${SEED} --step="${STEP}" \
      > "${RUN_DIR}/geo_parts/log.${STEP}.txt" 2>&1 &
    pids+=($!); g=$((g + 1))
  done
  rc=0; for p in "${pids[@]}"; do wait "${p}" || rc=1; done
  [ ${rc} -eq 0 ] || { tail -n 5 "${RUN_DIR}"/geo_parts/log.*.txt >&2; exit 1; }
  for f in $(ls "${RUN_DIR}"/geo_parts/*.jsonl | sort -t/ -k1 -V); do cat "${f}"; done > "${GEO_FILE}"
fi

echo "=== stage1 ${TAG} done ($(du -sh "${RUN_DIR}" | cut -f1)) ==="
