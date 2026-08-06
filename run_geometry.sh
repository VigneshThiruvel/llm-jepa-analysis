#!/bin/bash
#SBATCH --account=westai0096
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --partition=dc-hwai
#SBATCH --array=0-7
#SBATCH --output=slurm/logs/thesis/geometry_%A_%a.out
#SBATCH --error=slurm/logs/thesis/geometry_%A_%a.err

# Geometry data collection over the trajectory checkpoints of each finished run.
#
# This is a *read-only* companion to run_final_configs_seed82.sh: it loads each
# saved checkpoint frozen, in eval mode, and writes per-(split, layer) geometric
# scalars. It never trains, never touches accuracy, and writes only geometry.jsonl
# — so it cannot affect the replication results.
#
# One array task per run dir. Each task computes geometry on every
# checkpoint-<step> of that run (train + test splits) and appends tidy rows to
# <run>/geometry.jsonl. Re-running is safe: a run whose geometry.jsonl already has
# rows is skipped unless RERUN=1.

set -euo pipefail

VENV_DIR=/p/project1/westai0096/vignesh_thesis/jepa/.venv
HF_CACHE_DIR=/p/project1/westai0096/vignesh_thesis/jepa/.cache/huggingface

export PYTHONPATH="$SLURM_SUBMIT_DIR":${PYTHONPATH:-}

module load Stages
module load Python/3.13.5
source "${VENV_DIR}/bin/activate"

export TRANSFORMERS_OFFLINE=1
# Per-array-task datasets cache so concurrent geometry tasks don't race on the
# cold load_dataset('json', ...) build (ArrowInvalid: schema null/length 0).
# SLURM_JOB_ID is unique per array task. Model weights come from HF_HUB_CACHE.
export HF_DATASETS_CACHE=${TMPDIR:-/tmp}/hf_datasets_${SLURM_JOB_ID:-$$}
export HF_HUB_CACHE=${HF_CACHE_DIR}/hub
export HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

RUNS_DIR=final_runs_seed82
MODEL=meta-llama/Llama-3.2-1B-Instruct
MAX_EXAMPLES=500          # examples sampled per split (matches the jepa pilot)

# Same run order as run_final_configs_seed82.sh so the array indices line up.
TAGS=(synth_baseline synth_jepa turk_baseline turk_jepa gsm8k_baseline gsm8k_jepa spider_baseline spider_jepa)
TAG=${TAGS[$SLURM_ARRAY_TASK_ID]}
RUN_DIR=${RUNS_DIR}/${TAG}
CKPT_DIR=${RUN_DIR}/checkpoints
GEO_FILE=${RUN_DIR}/geometry.jsonl

if [ ! -d "${CKPT_DIR}" ]; then
  echo "=== ${TAG}: no checkpoints dir, nothing to measure ==="; exit 0
fi

# Pull dataset / lbd / k / seed from the config sidecar the training run wrote.
read -r DS LBD K SEED <<< "$(python3 -c "import json;c=json.load(open('${RUN_DIR}/config.json'));print(c['dataset'],c['lbd'],c['k'],c['seed'])")"
DATASET=datasets/${DS}
SPLITS="train:${DATASET}_train.jsonl,test:${DATASET}_test.jsonl"

echo "=== geometry ${TAG}: dataset=${DS} lbd=${LBD} k=${K} seed=${SEED} ==="

if [ -z "${RERUN:-}" ] && [ -s "${GEO_FILE}" ]; then
  echo "=== ${TAG}: ${GEO_FILE} already has rows, skipping (set RERUN=1 to force) ==="
  exit 0
fi
rm -f "${GEO_FILE}"

# One frozen forward pass per trajectory checkpoint. "step" is the checkpoint
# number, which gives the temporal ordering the mediation analysis relies on.
for ckpt in $(ls -d ${CKPT_DIR}/checkpoint-* 2>/dev/null | sort -t- -k2 -n); do
  STEP=$(basename "${ckpt}" | sed 's/checkpoint-//')
  echo "--- ${TAG} step=${STEP} ---"
  python3 geometry.py \
    --model_name="${ckpt}" \
    --original_model_name="${MODEL}" \
    --input_files="${SPLITS}" \
    --output_file="${GEO_FILE}" \
    --max_examples=${MAX_EXAMPLES} \
    --lbd=${LBD} --k=${K} --seed=${SEED} --step="${STEP}"
done

# Re-gather every finished run's geometry into the tidy CSV.
python3 collect_geometry.py --runs_dir=${RUNS_DIR} || true

echo "=== geometry ${TAG} done ==="
