#!/bin/bash
#SBATCH --account=westai0096
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --partition=dc-hwai
#SBATCH --array=0-7
#SBATCH --output=slurm/logs/thesis/final_seed82_%A_%a.out
#SBATCH --error=slurm/logs/thesis/final_seed82_%A_%a.err

# Per-dataset selected configs at seed 82, baseline (no-JEPA) vs JEPA.
#
#   Dataset        lr      lbd   k
#   NL-RX-SYNTH    2e-5    1     1
#   NL-RX-TURK     2e-5    1     1
#   GSM8K          2e-5    0.5   4
#   Spider         4e-5    1     3
#
# Spider lr is 4e-5 (paper Table 13). It was 1e-5 until 2026-09-17, taken from the
# paper's Fig 7b label and upstream run.sh; at 1e-5 the model never memorises the
# schemas and scores 0.21 vs the paper's 0.475 (LOGBOOK 2026-09-17, job 15632321).
#
# One array task per (dataset, arm) = 8 tasks. Each task trains, evaluates
# accuracy, then re-gathers every finished run into the result CSVs, so the
# tables are complete once the last task lands (and usable before that).
#
# Re-running the array is safe: a run whose results.txt already holds a
# success rate is skipped. Set RERUN=1 to force retraining everything.

set -euo pipefail

# --- Environment ---------------------------------------------------------
# llm-jepa ships no venv of its own; reuse the JURECA venv (transformers 5.3.0,
# which drives finetune.py/evaluate.py here). Point HF_* at the same cache so the
# offline Llama-3.2-1B-Instruct weights resolve. Change these two if you build a
# dedicated environment for this repo.
VENV_DIR=/p/project1/westai0096/vignesh_thesis/jepa/.venv
HF_CACHE_DIR=/p/project1/westai0096/vignesh_thesis/jepa/.cache/huggingface

export PYTHONPATH="$SLURM_SUBMIT_DIR":${PYTHONPATH:-}

module load Stages
module load Python/3.13.5
source "${VENV_DIR}/bin/activate"

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_CACHE=${HF_CACHE_DIR}/datasets
export HF_HUB_CACHE=${HF_CACHE_DIR}/hub
export HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

export MASTER_ADDR=$(hostname -I | awk '{print $1}')
export MASTER_PORT=$((29600 + SLURM_ARRAY_TASK_ID))

MODEL=meta-llama/Llama-3.2-1B-Instruct
EPOCHS=4
SEED=82
RUNS_DIR=final_runs_seed82

# "dataset arm lr lbd k" — arm is `baseline` (--regular, no JEPA) or `jepa`.
# Baseline rows carry lbd=0 k=0: the JEPA loss is off, so those knobs are unused.
CONFIGS=(
  "synth  baseline 2e-5 0   0"
  "synth  jepa     2e-5 1   1"
  "turk   baseline 2e-5 0   0"
  "turk   jepa     2e-5 1   1"
  "gsm8k  baseline 2e-5 0   0"
  "gsm8k  jepa     2e-5 0.5 4"
  "spider baseline 4e-5 0   0"
  "spider jepa     4e-5 1   3"
)
read -r DS ARM LR LBD K <<< "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

DATASET=datasets/${DS}

# Generation budget and eval extras differ per dataset. Never pass
# --max_new_tokens=-1 here: that leaves max_new_tokens unset, and the base
# model's generation_config has no max_length, so HF falls back to 20 total
# tokens and every generation comes out empty.
case "${DS}" in
  synth|turk) MAX_NEW_TOKENS=256; EVAL_EXTRA=() ;;
  gsm8k)      MAX_NEW_TOKENS=512; EVAL_EXTRA=() ;;                                # chain-of-thought before "#### <answer>"
  spider)     MAX_NEW_TOKENS=256; EVAL_EXTRA=(--spider_path=spider_data/database) ;;  # execution-accuracy needs the sqlite DBs
  *) echo "unknown dataset ${DS}" >&2; exit 1 ;;
esac

TAG="${DS}_${ARM}"
RUN_DIR=${RUNS_DIR}/${TAG}
CKPT_DIR=${RUN_DIR}/checkpoints
mkdir -p ${RUN_DIR}

echo "=== ${TAG}: dataset=${DS} arm=${ARM} lr=${LR} lbd=${LBD} k=${K} seed=${SEED} epochs=${EPOCHS} ==="

# Config sidecar: the collector reads this instead of re-parsing the dir name.
cat > ${RUN_DIR}/config.json <<EOF
{"dataset": "${DS}", "arm": "${ARM}", "lr": "${LR}", "lbd": ${LBD}, "k": ${K},
 "seed": ${SEED}, "epochs": ${EPOCHS}, "model": "${MODEL}", "max_new_tokens": ${MAX_NEW_TOKENS}}
EOF

# Always leave the CSVs regenerated from whatever completed, even if this task dies.
trap 'python3 collect_final_configs.py --runs_dir=${RUNS_DIR} || true' EXIT

if [ -z "${RERUN:-}" ] && grep -q "Success Rate" "${RUN_DIR}/results.txt" 2>/dev/null; then
  echo "=== ${TAG}: results.txt already has a success rate, skipping (set RERUN=1 to force) ==="
  exit 0
fi

# --- 1. Train ---
if [ "${ARM}" = "baseline" ]; then
  torchrun --nproc_per_node=4 finetune.py \
    --train_file ${DATASET}_train.jsonl \
    --output_dir=${CKPT_DIR} \
    --num_epochs=${EPOCHS} --finetune_seed=${SEED} \
    --model_name=${MODEL} --learning_rate=${LR} \
    --predictors=0 --regular
else
  torchrun --nproc_per_node=4 finetune.py \
    --train_file ${DATASET}_train.jsonl \
    --output_dir=${CKPT_DIR} \
    --num_epochs=${EPOCHS} --finetune_seed=${SEED} \
    --model_name=${MODEL} --learning_rate=${LR} \
    --lbd=${LBD} --predictors=${K} --last_token=-2 \
    --additive_mask
fi

# --- 2. Accuracy eval on the final model ---
# Stale rows in eval.jsonl would be skipped rather than regenerated.
rm -f ${RUN_DIR}/eval.jsonl
python3 evaluate.py \
  --model_name=${CKPT_DIR} \
  --input_file=${DATASET}_test.jsonl \
  --output_file=${RUN_DIR}/eval.jsonl \
  --split_tune_untune \
  --original_model_name=${MODEL} \
  --max_new_tokens ${MAX_NEW_TOKENS} \
  "${EVAL_EXTRA[@]+"${EVAL_EXTRA[@]}"}" \
  --nosplit_data | tee ${RUN_DIR}/results.txt

echo "=== ${TAG} done ==="
