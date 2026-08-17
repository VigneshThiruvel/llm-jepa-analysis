#!/bin/bash
#SBATCH --account=westai0096
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=08:00:00
#SBATCH --partition=dc-hwai
#SBATCH --array=0-233
# Dataset is a runtime knob (DS=...), which #SBATCH cannot interpolate — so the
# filename carries the job id, not "synth", or turk/gsm8k logs land under a
# synth name. The dataset is echoed in the first lines of each log.
#SBATCH --output=slurm/logs/thesis/sweep_%A_%a.out
#SBATCH --error=slurm/logs/thesis/sweep_%A_%a.err

# Observational sweep — SYNTH, per CLAUDE.md. Two tranches, one array.
#
# Tranche 1 (done, job 15488886): baseline (λ=0, once per seed) + JEPA
#   λ∈{0.5,1,2} × k∈{0,1}, seeds {82,23,37} — 21 runs, the seed-precision arm.
# Tranche 2 (this submission): extreme dose-response on an exponential scale at
#   the single primary seed 82 — λ∈{0.125,0.25,0.5,1,2,4,8,16,32,64} ×
#   k∈{0,1,2,4,8}, minus the 6 cells tranche 1 already covers at seed 82 — 44 runs.
#   Other seeds get added only if the extremes turn up something worth the GPU time.
# Tranche 3 (this submission): k up to 32 and lambda up to 128, seed 82 — the
#   k in {16,32} columns across every existing lambda, plus a full lambda=128 row.
#   27 runs. These carry a 4th CONFIGS field, "purge": checkpoints are deleted
#   once accuracy and geometry are captured, so the tranche costs ~0 disk.
#
# Each (λ,k,seed) is an independent run: train → per-epoch trajectory checkpoints
# → accuracy eval → geometry on every checkpoint → tidy CSVs.
#
# Append-only: each stage has its own skip guard, so re-submitting never redoes
# finished work and densifying the grid later reuses everything already run. The
# 21 tranche-1 cells are kept in the list on purpose — they cost ~20 s each to
# skip and keep this file the single source of truth for the whole grid.
# RERUN=1 forces a full redo of the cell.

set -euo pipefail

# --- Grid definition -------------------------------------------------------
# Kept above the module/GPU setup so `NCONFIGS=1 ./run_sweep.sh` can report the
# array size from a login node, and so a bad --array fails before booking a node.
#
# Dataset and seed set are the only knobs; the λ×k grid is identical for every
# (dataset, seed). lr is fixed per dataset — only λ, k, seed vary.
#   synth, 3 seeds (default):  sbatch --array=0-233 run_sweep.sh
#   turk,  1 seed:             DS=turk SEEDS=82 sbatch --array=0-77 run_sweep.sh
#   gsm8k, 1 seed:             DS=gsm8k SEEDS=82 sbatch --array=0-77 run_sweep.sh
# Outputs stay namespaced per dataset under sweep_runs/<DS>/, so each dataset
# gets its own accuracy_tidy.csv / geometry_tidy.csv with no renaming; seed is
# already a column, so extra seeds just add rows to that dataset's existing CSVs.
DS=${DS:-synth}
case "${DS}" in
  synth|turk) LR=2e-5; MAX_NEW_TOKENS=256 ;;
  # gsm8k needs the chain-of-thought before "#### <answer>", hence the larger budget.
  gsm8k)      LR=2e-5; MAX_NEW_TOKENS=512 ;;
  *) echo "unknown dataset '${DS}' (expected synth|turk|gsm8k)" >&2; exit 1 ;;
esac

# The grid, generated per seed: one no-JEPA baseline (λ=0, k is a no-op there)
# plus every λ×k combination. λ>0 includes k=0, which decouples k from the
# JEPA-on/off contrast — k=0 is JEPA *without* predictor tokens, not the baseline.
#
# λ are STRINGS, not numbers, and must keep this exact spelling (0.5 / 1.0 / 2.0,
# never .5 / 1 / 2): the tag <λ>_<k>_<seed> is the on-disk directory name, so a
# reformatted λ misses the skip guard and silently retrains a finished cell.
# k ≤ MAX_PREDICTORS (32) in finetune.py — above that the marker is not a special
# token and BPEs into ordinary text, silently changing what the k axis means.
LBDS=(0.125 0.25 0.5 1.0 2.0 4.0 8.0 16.0 32.0 64.0 128.0)
KS=(0 1 2 4 8 16 32)
read -r -a SEEDS <<< "${SEEDS:-82 23 37}"

# 4th field "purge": delete checkpoints once accuracy AND geometry are captured
# (~12 GB/cell otherwise, and this expansion is 220 new cells). Finished cells
# exit at the completeness guard further down before ever reaching the purge
# step, so the tranche-1/2 seed-82 checkpoints already on disk are not touched.
CONFIGS=()
for S in "${SEEDS[@]}"; do
  CONFIGS+=("0.0 0 ${S} purge")
  for L in "${LBDS[@]}"; do
    for K in "${KS[@]}"; do
      CONFIGS+=("${L} ${K} ${S} purge")
    done
  done
done

# Submitting with a wrong --array silently truncates or overruns the grid, so
# fail loudly instead: `NCONFIGS=1 ./run_sweep.sh` prints the size to use.
if [ -n "${NCONFIGS:-}" ]; then
  echo "${DS}: ${#CONFIGS[@]} cells for seeds '${SEEDS[*]}' -> --array=0-$(( ${#CONFIGS[@]} - 1 ))"
  exit 0
fi
if [ "${SLURM_ARRAY_TASK_ID:-0}" -ge "${#CONFIGS[@]}" ]; then
  echo "task ${SLURM_ARRAY_TASK_ID} >= ${#CONFIGS[@]} configs for DS=${DS} seeds='${SEEDS[*]}'" >&2
  exit 1
fi

VENV_DIR=/p/project1/westai0096/vignesh_thesis/jepa/.venv
HF_CACHE_DIR=/p/project1/westai0096/vignesh_thesis/jepa/.cache/huggingface

export PYTHONPATH="$SLURM_SUBMIT_DIR":${PYTHONPATH:-}

module load Stages
module load Python/3.13.5
source "${VENV_DIR}/bin/activate"

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

export TRANSFORMERS_OFFLINE=1
# Per-array-task datasets cache. evaluate.py and geometry.py both call
# load_dataset('json', ...) single-process; concurrent array tasks sharing one
# cache dir race on the cold build (ArrowInvalid: schema null/length 0 — a task
# memory-maps a half-written arrow). SLURM_JOB_ID is unique per array task, so
# this isolates each. Model weights are unaffected (they come from HF_HUB_CACHE).
export HF_DATASETS_CACHE=${TMPDIR:-/tmp}/hf_datasets_${SLURM_JOB_ID:-$$}
export HF_HUB_CACHE=${HF_CACHE_DIR}/hub
export HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

export MASTER_ADDR=$(hostname -I | awk '{print $1}')
export MASTER_PORT=$((29500 + SLURM_ARRAY_TASK_ID))

MODEL=meta-llama/Llama-3.2-1B-Instruct
EPOCHS=4
DATASET=datasets/${DS}
RUNS_DIR=sweep_runs/${DS}

read -r LBD K SEED KEEP <<< "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

if [ "${LBD}" = "0.0" ]; then ARM=baseline; else ARM=jepa; fi

TAG="${LBD}_${K}_${SEED}"
RUN_DIR=${RUNS_DIR}/${TAG}
CKPT_DIR=${RUN_DIR}/checkpoints
GEO_FILE=${RUN_DIR}/geometry.jsonl
mkdir -p ${RUN_DIR}

echo "=== ${DS} ${TAG}: arm=${ARM} lbd=${LBD} k=${K} seed=${SEED} lr=${LR} epochs=${EPOCHS} ==="

# Config sidecar for the collector.
cat > ${RUN_DIR}/config.json <<EOF
{"dataset": "${DS}", "arm": "${ARM}", "lr": "${LR}", "lbd": ${LBD}, "k": ${K},
 "seed": ${SEED}, "epochs": ${EPOCHS}, "model": "${MODEL}", "max_new_tokens": ${MAX_NEW_TOKENS}}
EOF

# Always leave the tidy CSVs regenerated from whatever finished, even if this
# task dies partway.
trap 'python3 collect_sweep.py --runs_dir=${RUNS_DIR} || true' EXIT

# Whole cell already complete? (accuracy + geometry both present)
if [ -z "${RERUN:-}" ] && grep -q "Success Rate" "${RUN_DIR}/results.txt" 2>/dev/null && [ -s "${GEO_FILE}" ]; then
  echo "=== ${TAG}: already complete (results.txt + geometry.jsonl), skipping ==="
  exit 0
fi

# --- 1. Train (per-epoch model-only trajectory checkpoints) ---
# Skip if the final model is already saved (train succeeded on a prior attempt).
if [ -n "${RERUN:-}" ] || [ ! -f "${CKPT_DIR}/model.safetensors" ]; then
  if [ "${ARM}" = "baseline" ]; then
    torchrun --nproc_per_node=4 finetune.py \
      --train_file ${DATASET}_train.jsonl \
      --output_dir=${CKPT_DIR} \
      --num_epochs=${EPOCHS} --finetune_seed=${SEED} \
      --model_name=${MODEL} --learning_rate=${LR} \
      --predictors=0 --regular \
      --save_epoch_checkpoints
  else
    torchrun --nproc_per_node=4 finetune.py \
      --train_file ${DATASET}_train.jsonl \
      --output_dir=${CKPT_DIR} \
      --num_epochs=${EPOCHS} --finetune_seed=${SEED} \
      --model_name=${MODEL} --learning_rate=${LR} \
      --lbd=${LBD} --predictors=${K} --last_token=-2 \
      --additive_mask \
      --save_epoch_checkpoints
  fi
else
  echo "=== ${TAG}: final model present, skipping training ==="
fi

# --- 2. Accuracy eval on the final model (synth = exact match) ---
if [ -n "${RERUN:-}" ] || ! grep -q "Success Rate" "${RUN_DIR}/results.txt" 2>/dev/null; then
  rm -f ${RUN_DIR}/eval.jsonl
  python3 evaluate.py \
    --model_name=${CKPT_DIR} \
    --input_file=${DATASET}_test.jsonl \
    --output_file=${RUN_DIR}/eval.jsonl \
    --split_tune_untune \
    --original_model_name=${MODEL} \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --nosplit_data | tee ${RUN_DIR}/results.txt
else
  echo "=== ${TAG}: results.txt already has a success rate, skipping eval ==="
fi

# --- 3. Geometry on every trajectory checkpoint (train + test splits) ---
if [ -n "${RERUN:-}" ] || [ ! -s "${GEO_FILE}" ]; then
  rm -f ${GEO_FILE}
  for ckpt in $(ls -d ${CKPT_DIR}/checkpoint-* 2>/dev/null | sort -t- -k2 -n); do
    STEP=$(basename "${ckpt}" | sed 's/checkpoint-//')
    echo "--- geometry ${TAG} step=${STEP} ---"
    python3 geometry.py \
      --model_name="${ckpt}" \
      --original_model_name="${MODEL}" \
      --input_files="train:${DATASET}_train.jsonl,test:${DATASET}_test.jsonl" \
      --output_file="${GEO_FILE}" \
      --max_examples=500 \
      --lbd=${LBD} --k=${K} --seed=${SEED} --step="${STEP}"
  done
else
  echo "=== ${TAG}: geometry.jsonl already present, skipping geometry ==="
fi

# --- 4. Purge checkpoints (tranche 3: keep the measurements, not the weights) ---
# ~12 GB/cell of trajectory checkpoints exist only to be measured. Training still
# writes them so geometry keeps its per-epoch trajectory; they are deleted once
# both measurements are safely on disk. Guarded on accuracy AND geometry having
# succeeded, so a half-finished cell keeps its weights and can resume instead of
# retraining. results.txt + geometry.jsonl survive, so the cell-complete guard at
# the top still recognises a purged cell as done.
if [ "${KEEP:-}" = "purge" ] \
   && grep -q "Success Rate" "${RUN_DIR}/results.txt" 2>/dev/null \
   && [ -s "${GEO_FILE}" ]; then
  # Keep the training log before deleting the weights — collect_sweep.py's health
  # verdict reads it, and without this every purged cell would classify no_train
  # and have its accuracy blanked.
  cp -f "${CKPT_DIR}/trainer_state.json" "${RUN_DIR}/trainer_state.json" 2>/dev/null || true
  echo "=== ${TAG}: purging checkpoints ($(du -sh ${CKPT_DIR} 2>/dev/null | cut -f1)) ==="
  rm -rf "${CKPT_DIR}"
elif [ "${KEEP:-}" = "purge" ]; then
  echo "=== ${TAG}: NOT purging — accuracy or geometry missing, weights kept for retry ==="
fi

echo "=== ${DS} ${TAG} done ==="
