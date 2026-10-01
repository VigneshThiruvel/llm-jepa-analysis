#!/bin/bash
#SBATCH --account=westai0096
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=03:30:00
#SBATCH --partition=dc-hwai
#SBATCH --output=slurm/logs/thesis/k16diag_padfix_%A_%a.out
#SBATCH --error=slurm/logs/thesis/k16diag_padfix_%A_%a.err

# k=16 diagnosis, stage 5 — is the collapse caused by padding with the stop token?
# finetune.py pads with <|eot_id|> (= eos), and the LM collator masks every pad id
# out of the labels, so eot is never a training target. diagnostics/finetune_padfix.py
# runs finetune.py unchanged but pads with <|finetune_right_pad_id|>, which puts
# eot back into the loss and changes nothing else (same input_ids, attention masks
# and JEPA views; checked on 64 synth rows: the only label difference is the 3 eot
# per example).
#
# Two arms per cell, same driver, same day-independent prompts:
#   padfix   --pad_token=<|finetune_right_pad_id|>
#   control  --pad_token=eos  (= upstream behaviour; separates the fix from retrain
#            noise, which is ~0.03 on synth at a fixed seed)
# Training and eval prompts are both pinned to ${DATE}, so every arm, seed and
# dataset sees identical text whenever it runs. Reference = the sweep cell.
#
# Per task: train -> per-example eval (eval_generations.py, 4 GPUs) -> geometry.py
# per epoch checkpoint (as run_sweep.sh) -> collect_sweep.py. config.json stamps
# `dataset` as "<ds>_<arm>" (e.g. synth_padfix), which is what collect_sweep.py names
# its CSVs after and what plot_sweep.py names its figure directory and titles after,
# so padfix and control tables/figures never collide with the sweep's.
#
#   sbatch --array=0-47 diagnostics/run_padfix.sh              # synth, 8 cells x 3 seeds x 2 arms
#   DS=turk sbatch --array=0-53 diagnostics/run_padfix.sh      # turk, 9 cells x 3 seeds x 2 arms
#   NCONFIGS=1 [DS=turk] ./diagnostics/run_padfix.sh           # print the --array to use
#   SEEDS=82 ARMS=padfix CELLS=2.0:16 sbatch --array=0 diagnostics/run_padfix.sh
#
# Cells (sweep accuracy, seed 82; turk adds 1.0_32, its grid best):
#   k=16 collapse   synth 1.0_16 0.094  2.0_16 0.000  8.0_16 0.000  partial 0.5_16 0.481
#                   turk  1.0_16 0.007  2.0_16 0.000  8.0_16 0.000  partial 0.5_16 0.208
#   high-λ k=4      synth 64.0_4 0.061   turk 64.0_4 0.026    (the other collapse family)
#   healthy         synth 0.0_0 0.595 / 2.0_8 0.699 / 8.0_0 0.778 (grid best)
#                   turk  0.0_0 0.231 / 2.0_8 0.310 / 8.0_0 0.222 / 1.0_32 0.343 (grid best)
# Seeds 23/37 exist in the synth sweep; turk was only ever run at 82, so there the
# control arm is the only same-day reference.
#
# Writes only under ${OUT_ROOT}; sweep_runs/ is read as the reference, never written.
# A cell is redone from scratch when its final model is missing, or when geometry is
# wanted and neither geometry.jsonl nor the epoch checkpoints are on disk (that is the
# state the first seed-82 synth array left behind — it purged checkpoints after eval),
# so every artefact of a cell always comes from one single training run.
# GEOMETRY=0 skips the geometry stage; KEEP_EPOCHS=1 keeps the epoch checkpoints.
set -euo pipefail

DS=${DS:-synth}
case "${DS}" in
  synth|turk) LR=2e-5; MAX_NEW_TOKENS=256 ;;
  *) echo "unknown dataset '${DS}'" >&2; exit 1 ;;
esac
DATE=${DATE:-22 Sep 2026}
GEOMETRY=${GEOMETRY:-1}
DEFAULT_CELLS="2.0:16 1.0:16 8.0:16 0.5:16 64.0:4 0.0:0 2.0:8 8.0:0"
[ "${DS}" = "turk" ] && DEFAULT_CELLS="${DEFAULT_CELLS} 1.0:32"
read -r -a CELLS <<< "${CELLS:-${DEFAULT_CELLS}}"
read -r -a SEEDS <<< "${SEEDS:-82 23 37}"
read -r -a ARMS <<< "${ARMS:-padfix control}"
N=$(( ${#SEEDS[@]} * ${#CELLS[@]} * ${#ARMS[@]} ))
if [ -n "${NCONFIGS:-}" ]; then echo "${N} tasks -> --array=0-$(( N - 1 ))"; exit 0; fi
TASK=${SLURM_ARRAY_TASK_ID:-0}
[ "${TASK}" -lt "${N}" ] || { echo "task ${TASK} >= ${N}" >&2; exit 1; }
# arm innermost, so a cell's two arms are adjacent tasks and land together
ARM=${ARMS[$(( TASK % ${#ARMS[@]} ))]}
REST=$(( TASK / ${#ARMS[@]} ))
IFS=: read -r LBD K <<< "${CELLS[$(( REST % ${#CELLS[@]} ))]}"
SEED=${SEEDS[$(( REST / ${#CELLS[@]} ))]}
case "${ARM}" in
  padfix)  PAD='<|finetune_right_pad_id|>' ;;
  control) PAD=eos ;;
  *) echo "unknown arm '${ARM}'" >&2; exit 1 ;;
esac

cd "${SLURM_SUBMIT_DIR:-$PWD}"
[ -f finetune.py ] && [ -d diagnostics ] || { echo "submit from llm-jepa-analysis/" >&2; exit 1; }
source diagnostics/common.sh
nvidia-smi --query-gpu=index,name,memory.used --format=csv

export MASTER_ADDR=$(hostname -I | awk '{print $1}')
export MASTER_PORT=$((29800 + TASK))

EPOCHS=4
DATASET=datasets/${DS}
OUT_ROOT=${OUT_ROOT:-diag_runs/k16/stage5_padfix/${DS}}
TAG="${LBD}_${K}_${SEED}"
RUN_DIR=${OUT_ROOT}/${ARM}/${TAG}
CKPT_DIR=${RUN_DIR}/checkpoints
GEO_FILE=${RUN_DIR}/geometry.jsonl
REF=sweep_runs/${DS}/${TAG}/results.txt
if [ "${LBD}" = "0.0" ]; then SWEEP_ARM=baseline; else SWEEP_ARM=jepa; fi

# Redo the cell when its artefacts cannot all come from one training run.
NEED_GEO_CKPT=0
if [ "${GEOMETRY}" = "1" ] && [ ! -s "${GEO_FILE}" ] && ! compgen -G "${CKPT_DIR}/checkpoint-*" > /dev/null; then
  NEED_GEO_CKPT=1
fi
if [ -n "${RERUN:-}" ] || [ "${NEED_GEO_CKPT}" = "1" ]; then
  [ -d "${RUN_DIR}" ] && echo "=== ${TAG}/${ARM}: redoing from scratch (RERUN=${RERUN:-} need_geo_ckpt=${NEED_GEO_CKPT}) ==="
  rm -rf "${RUN_DIR}"
fi
mkdir -p "${RUN_DIR}"
echo "=== stage5 ${DS} ${TAG} arm=${ARM} pad=${PAD} date='${DATE}' lr=${LR} geometry=${GEOMETRY} -> ${RUN_DIR} ==="

# `dataset` carries the arm: it is what collect_sweep.py names its CSVs after and
# what plot_sweep.py uses for the figure directory and titles. `arm` keeps the
# sweep's baseline/jepa meaning so the tables stay comparable.
cat > "${RUN_DIR}/config.json" <<EOF
{"dataset": "${DS}_${ARM}", "base_dataset": "${DS}", "arm": "${SWEEP_ARM}",
 "pad_arm": "${ARM}", "pad_token": "${PAD}", "date_string": "${DATE}",
 "lr": "${LR}", "lbd": ${LBD}, "k": ${K}, "seed": ${SEED}, "epochs": ${EPOCHS},
 "model": "${MODEL}", "max_new_tokens": ${MAX_NEW_TOKENS}, "stage": "k16_stage5_padfix",
 "git_commit": "$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"}
EOF

trap 'python3 collect_sweep.py --runs_dir=${OUT_ROOT}/${ARM} || true' EXIT

# --- 1. Train: run_sweep.sh's command, through the padfix wrapper ---
PADFIX_ARGS=(--pad_token="${PAD}" --train_date_string="${DATE}")
if [ ! -f "${CKPT_DIR}/model.safetensors" ]; then
  if [ "${LBD}" = "0.0" ]; then
    torchrun --nproc_per_node=4 diagnostics/finetune_padfix.py \
      --train_file ${DATASET}_train.jsonl --output_dir=${CKPT_DIR} \
      --num_epochs=${EPOCHS} --finetune_seed=${SEED} \
      --model_name=${MODEL} --learning_rate=${LR} \
      --predictors=0 --regular --save_epoch_checkpoints "${PADFIX_ARGS[@]}"
  else
    torchrun --nproc_per_node=4 diagnostics/finetune_padfix.py \
      --train_file ${DATASET}_train.jsonl --output_dir=${CKPT_DIR} \
      --num_epochs=${EPOCHS} --finetune_seed=${SEED} \
      --model_name=${MODEL} --learning_rate=${LR} \
      --lbd=${LBD} --predictors=${K} --last_token=-2 --additive_mask \
      --save_epoch_checkpoints "${PADFIX_ARGS[@]}"
  fi
else
  echo "=== ${TAG}/${ARM}: final model present, skipping training ==="
fi
cp -f "${CKPT_DIR}/trainer_state.json" "${RUN_DIR}/trainer_state.json" 2>/dev/null || true

# --- 2. Per-example eval, date pinned (stop reason eos/cap is in summary.json) ---
if ! grep -q "^Success Rate" "${RUN_DIR}/results.txt" 2>/dev/null || [ ! -s "${RUN_DIR}/eval/summary.json" ]; then
  GEN_ARGS=(--date_string="${DATE}" --trace_n=50)
  REF_ARGS=()
  [ -f "${REF}" ] && REF_ARGS=(--reference_results="${REF}")
  run_eval_sharded "${CKPT_DIR}" "${RUN_DIR}/eval" ${DATASET}_test.jsonl ${MAX_NEW_TOKENS} \
    "${REF_ARGS[@]}" | tee "${RUN_DIR}/results.txt"
else
  echo "=== ${TAG}/${ARM}: eval done, skipping ==="
fi

# --- 3. Geometry, one checkpoint per GPU (same geometry.py call as run_sweep.sh) ---
if [ "${GEOMETRY}" = "1" ] && [ ! -s "${GEO_FILE}" ]; then
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

[ -n "${KEEP_EPOCHS:-}" ] || rm -rf "${CKPT_DIR}"/checkpoint-*
echo "=== stage5 ${TAG}/${ARM} done ($(du -sh "${RUN_DIR}" | cut -f1)) ==="
