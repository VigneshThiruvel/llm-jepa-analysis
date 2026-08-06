#!/bin/bash
#SBATCH --account=westai0096
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=08:00:00
#SBATCH --partition=dc-hwai
#SBATCH --array=0-20
#SBATCH --output=slurm/logs/thesis/sweep_synth_%A_%a.out
#SBATCH --error=slurm/logs/thesis/sweep_synth_%A_%a.err

# Observational sweep — SYNTH first tranche (21 runs), per CLAUDE.md.
#
# Grid: baseline (λ=0, once per seed) + JEPA λ∈{0.5,1,2} × k∈{0,1}, seeds
# {82,23,37}. Each (λ,k,seed) is an independent run: train → per-epoch trajectory
# checkpoints → accuracy eval → geometry on every checkpoint → tidy CSVs.
#
# Append-only: each stage has its own skip guard, so re-submitting never redoes
# finished work and densifying the grid later reuses everything already run.
# RERUN=1 forces a full redo of the cell.

set -euo pipefail

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
DS=synth
DATASET=datasets/${DS}
LR=2e-5                 # SYNTH lr is fixed across all cells (only λ, k, seed vary)
EPOCHS=4
MAX_NEW_TOKENS=256
RUNS_DIR=sweep_runs/${DS}

# "lbd k seed". λ=0 rows are the no-JEPA baseline (k is a no-op there, run once
# per seed). λ>0 rows are JEPA, including k=0 at nonzero λ (decouples k from the
# JEPA-on/off contrast, as the pilot flagged).
CONFIGS=(
  "0.0 0 82"  "0.0 0 23"  "0.0 0 37"
  "0.5 0 82"  "0.5 0 23"  "0.5 0 37"
  "0.5 1 82"  "0.5 1 23"  "0.5 1 37"
  "1.0 0 82"  "1.0 0 23"  "1.0 0 37"
  "1.0 1 82"  "1.0 1 23"  "1.0 1 37"
  "2.0 0 82"  "2.0 0 23"  "2.0 0 37"
  "2.0 1 82"  "2.0 1 23"  "2.0 1 37"
)
read -r LBD K SEED <<< "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

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

echo "=== ${DS} ${TAG} done ==="
