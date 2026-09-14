#!/bin/bash
#SBATCH --account=westai0096
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
#SBATCH --partition=dc-hwai
#SBATCH --output=slurm/logs/thesis/k16diag_s0_%j.out
#SBATCH --error=slurm/logs/thesis/k16diag_s0_%j.err

# k=16 diagnosis, stage 0 — measurements on data already on disk, no training.
# Submit from the repo root:   sbatch diagnostics/run_stage0.sh
#
#   [A] pooling-index check (CPU): which token each JEPA embedding is read at, per k
#   [B] per-example eval (eval_generations.py) on surviving sweep checkpoints —
#       a healthy cell (fidelity check against its recorded Success Rate), a
#       mid-collapse cell and a near-zero cell (what degenerate generations are)
#   [C] tied-head probe over every surviving seed-82 checkpoint (k<=8, all λ≤64,
#       all 4 epochs): added-row norms + first-token / end-of-answer ranks
#
# Read-only over sweep_runs/; writes only under ${OUT}. RERUN=1 redoes [B].
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[ -f finetune.py ] && [ -d diagnostics ] || { echo "submit from llm-jepa-analysis/" >&2; exit 1; }
source diagnostics/common.sh

OUT=${OUT:-diag_runs/k16/stage0}
SRC_RUNS=sweep_runs/synth
read -r -a EVAL_CELLS <<< "${EVAL_CELLS:-8.0_8_82 16.0_4_82 64.0_4_82}"
PROBE_GLOB=${PROBE_GLOB:-*_82}
FORWARD_N=${FORWARD_N:-64}
mkdir -p "${OUT}"
echo "=== stage 0: OUT=${OUT} GPUs=${GPUS[*]} eval cells=${EVAL_CELLS[*]} ==="

echo "=== [A] pooling-index check ==="
python3 diagnostics/check_pooling_index.py --model_name="${MODEL}" \
  --train_file=datasets/synth_train.jsonl --n_examples=200 \
  --out_json="${OUT}/pooling_index.json" | tee "${OUT}/pooling_index.txt"

echo "=== [B] per-example eval on surviving checkpoints ==="
for TAG in "${EVAL_CELLS[@]}"; do
  CELL=${SRC_RUNS}/${TAG}
  EOUT=${OUT}/eval/${TAG}
  if [ -z "${RERUN:-}" ] && [ -s "${EOUT}/summary.json" ]; then
    echo "--- ${TAG}: done, skipping"; continue
  fi
  [ -n "${RERUN:-}" ] && rm -rf "${EOUT}"
  mkdir -p "${EOUT}"
  # Pin the chat template's "Today Date" to the day the reference eval ran
  # (results.txt mtime), so the fidelity check compares identical prompts.
  DATE=$(date -r "${CELL}/results.txt" "+%d %b %Y")
  echo "--- ${TAG}: date_string='${DATE}'"
  GEN_ARGS=(--date_string="${DATE}" --trace_n=50)
  run_eval_sharded "${CELL}/checkpoints" "${EOUT}" datasets/synth_test.jsonl 256 \
    --reference_results="${CELL}/results.txt" | tee "${EOUT}/report.txt"
done

echo "=== [C] tied-head probe (forward N=${FORWARD_N}) ==="
mkdir -p "${OUT}/probe"
pids=()
for ((g = 0; g < NGPU; g++)); do
  CUDA_VISIBLE_DEVICES=${GPUS[$g]} python3 diagnostics/tied_head_probe.py \
    --runs_dir="${SRC_RUNS}" --cell_glob="${PROBE_GLOB}" \
    --input_file=datasets/synth_test.jsonl --forward_examples="${FORWARD_N}" \
    --shard_id=${g} --num_shards=${NGPU} --output_file="${OUT}/probe/probe.${g}.csv" \
    > "${OUT}/probe/log.${g}.txt" 2>&1 &
  pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait "${p}" || rc=1; done
[ ${rc} -eq 0 ] || { tail -n 5 "${OUT}"/probe/log.*.txt >&2; exit 1; }
python3 diagnostics/tied_head_probe.py --merge --inputs "${OUT}"/probe/probe.*.csv \
  --output_file="${OUT}/tied_head_probe.csv"

echo "=== stage 0 done: ${OUT} ==="
