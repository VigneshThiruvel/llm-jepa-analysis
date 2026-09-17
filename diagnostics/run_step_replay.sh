#!/bin/bash
#SBATCH --account=westai0096
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=03:00:00
#SBATCH --partition=dc-hwai
#SBATCH --output=slurm/logs/thesis/k16diag_s3_%j.out
#SBATCH --error=slurm/logs/thesis/k16diag_s3_%j.err
#
# Stage 3 of the k=16 diagnosis: per-step replay of the first epoch around k=16.
# Stage 2 showed jepa_loss at step 0 is ~0.8 for every k, so the step-10 gap
# (k=16 ~0.27 vs ~0.005 at k=8/32) is made in the first optimizer steps. Same
# command as run_stage1.sh (finetune.py unmodified), 1 epoch, logging every step,
# --debug=5 prints rank-0 lm/jepa loss per micro-batch. Weights are deleted; only
# trainer_state.json and train.log are kept.
#   sbatch diagnostics/run_step_replay.sh
#   KS="16" SEED=23 sbatch diagnostics/run_step_replay.sh

[ -f finetune.py ] && [ -d diagnostics ] || { echo "submit from llm-jepa-analysis/" >&2; exit 1; }
source diagnostics/common.sh
nvidia-smi --query-gpu=index,name,memory.used --format=csv

export MASTER_ADDR=$(hostname -I | awk '{print $1}')
LBD=${LBD:-2.0}
SEED=${SEED:-82}
LR=${LR:-2e-5}
KS=${KS:-"4 8 12 14 15 16 17 18 20 24 32"}
OUT_ROOT=${OUT_ROOT:-diag_runs/k16/stage3_steps}

i=0
for K in ${KS}; do
  RUN=${OUT_ROOT}/${LBD}_${K}_${SEED}
  mkdir -p "${RUN}"
  if [ -f "${RUN}/trainer_state.json" ]; then echo "=== ${RUN}: done, skipping ==="; continue; fi
  export MASTER_PORT=$((29800 + i)); i=$((i + 1))
  echo "=== $(date +%T) ${RUN}: lbd=${LBD} k=${K} seed=${SEED} ==="
  torchrun --nproc_per_node=4 finetune.py \
    --train_file datasets/synth_train.jsonl --output_dir="${RUN}/ckpt" \
    --num_epochs=1 --finetune_seed=${SEED} \
    --model_name=${MODEL} --learning_rate=${LR} \
    --lbd=${LBD} --predictors=${K} --last_token=-2 --additive_mask \
    --eval_steps=1 --debug=5 --save_epoch_checkpoints > "${RUN}/train.log" 2>&1 \
    || echo "!!! training failed for ${RUN}"
  st=$(ls "${RUN}"/ckpt/trainer_state.json "${RUN}"/ckpt/checkpoint-*/trainer_state.json 2>/dev/null | head -1)
  [ -n "${st}" ] && cp -f "${st}" "${RUN}/trainer_state.json"
  rm -rf "${RUN}/ckpt"
done
