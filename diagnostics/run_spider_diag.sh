#!/bin/bash
#SBATCH --account=westai0096
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=03:00:00
#SBATCH --partition=dc-hwai
#SBATCH --array=0-5
#SBATCH --output=slurm/logs/thesis/spiderdiag_%A_%a.out
#SBATCH --error=slurm/logs/thesis/spiderdiag_%A_%a.err

# Spider level gap: ours 20.6 / 21.6 (seed 82, lr 1e-5) vs paper Table 13
# 47.52 / 50.55. Submit from the repo root:  sbatch diagnostics/run_spider_diag.sh
#
# Tasks 0-1 [A] per-example eval of the EXISTING replication checkpoints
#   (final_runs_seed82/spider_{baseline,jepa}), date pinned to their eval day, with
#   a fidelity check against their results.txt, then a failure taxonomy
#   (schema hallucination / syntax / wrong rows / truncation / scorer miss).
# Tasks 2-5 [B] lr probe, seed 82: the paper gives Spider lr = 4e-5 in Table 13
#   but 1e-5 in Fig 7b (same 50.55% cell); replication used 1e-5. Trains
#   {baseline, jepa λ1 k3} at lr {2e-5, 4e-5} with the exact
#   run_final_configs_seed82.sh commands (+ per-epoch checkpoints) and evaluates.
#
# Read-only over final_runs_seed82/; writes only under diag_runs/spider/.
set -euo pipefail
TASKS=(
  "eval  baseline 1e-5"
  "eval  jepa     1e-5"
  "train baseline 2e-5"
  "train jepa     2e-5"
  "train baseline 4e-5"
  "train jepa     4e-5"
)
TASK=${SLURM_ARRAY_TASK_ID:-0}
read -r MODE ARM LR <<< "${TASKS[$TASK]}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
[ -f finetune.py ] && [ -d diagnostics ] || { echo "submit from llm-jepa-analysis/" >&2; exit 1; }
source diagnostics/common.sh
export MASTER_ADDR=$(hostname -I | awk '{print $1}')
export MASTER_PORT=$((29800 + TASK))

SEED=82; EPOCHS=4; LBD=1; K=3; MNT=256
DATASET=datasets/spider
GEN_ARGS=(--spider_path=spider_data/database --trace_n=0)

if [ "${MODE}" = "eval" ]; then
  SRC=final_runs_seed82/spider_${ARM}
  OUT=diag_runs/spider/replication_eval/${ARM}
  mkdir -p "${OUT}"
  DATE=$(date -r "${SRC}/results.txt" "+%d %b %Y")
  GEN_ARGS+=(--date_string="${DATE}")
  echo "=== [A] ${ARM}: ${SRC} date_string='${DATE}' -> ${OUT} ==="
  if [ ! -s "${OUT}/summary.json" ]; then
    run_eval_sharded "${SRC}/checkpoints" "${OUT}" ${DATASET}_test.jsonl ${MNT} \
      --reference_results="${SRC}/results.txt" | tee "${OUT}/report.txt"
  fi
  python3 diagnostics/spider_error_analysis.py --generations="${OUT}/generations.jsonl" \
    | tee "${OUT}/taxonomy.txt"
  exit 0
fi

RUN_DIR=diag_runs/spider/lr_probe/${LR}_${ARM}
CKPT_DIR=${RUN_DIR}/checkpoints
mkdir -p "${RUN_DIR}"
echo "=== [B] ${ARM} lr=${LR} seed=${SEED} -> ${RUN_DIR} ==="
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
fi
cp -f "${CKPT_DIR}/trainer_state.json" "${RUN_DIR}/" 2>/dev/null || true
DATE=$(date "+%d %b %Y"); GEN_ARGS+=(--date_string="${DATE}")
if [ ! -s "${RUN_DIR}/eval/summary.json" ]; then
  run_eval_sharded "${CKPT_DIR}" "${RUN_DIR}/eval" ${DATASET}_test.jsonl ${MNT} | tee "${RUN_DIR}/results.txt"
fi
python3 diagnostics/spider_error_analysis.py --generations="${RUN_DIR}/eval/generations.jsonl" \
  | tee "${RUN_DIR}/taxonomy.txt"
echo "=== ${RUN_DIR} done ==="
