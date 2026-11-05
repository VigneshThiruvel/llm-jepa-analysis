#!/bin/bash
#SBATCH --account=westai0096
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --partition=dc-hwai
#SBATCH --output=slurm/logs/thesis/jepa_obj_%A_%a.out
#SBATCH --error=slurm/logs/thesis/jepa_obj_%A_%a.err

# Phase C of "why does JEPA do nothing" (plan: ~/.claude/plans/i-want-to-now-scalable-cupcake.md).
# Phase A: the cosine JEPA term is solved by collapse. Here the JEPA term is swapped for objectives
# that cannot be (diagnostics/finetune_jepa_obj.py), one change at a time, all on top of padfix
# (pad = <|finetune_right_pad_id|>, train/eval date pinned) and at batch layout bs 16 x accum 1
# x 4 GPUs (= 64, as the sweep's 4 x 4 x 4) so InfoNCE sees 64-way negatives.
#
# An arm is a spec  name:lbd:k:obj:layer:pool  (lbd 0.0 = baseline, --regular). Steps:
#   step 0  base:0.0:0:-:-:-   cos:1.0:1:cos:-1:last
#   step 1  nce0.25:0.25:1:infonce:-1:last   nce1:1.0:1:infonce:-1:last
#   step 2  ARMS="nce1mean:1.0:1:infonce:-1:mean nce1L8:1.0:1:infonce:8:last"   (after step 1)
#   step 3  ARMS="vic...:...:vicreg:..."                                        (conditional)
# Per task: train -> eval_generations (sharded, date pinned) -> jepa_probe embed (last + mean).
# POST=1 (no array): pairs vs the same-seed base arm + report. Output diag_runs/jepa_null/phaseC/.
#
#   NCONFIGS=1 ./diagnostics/run_jepa_obj.sh                 # print the --array to use
#   sbatch --array=0-3 diagnostics/run_jepa_obj.sh           # steps 0+1, seed 82
#   POST=1 sbatch --dependency=afterok:<job> diagnostics/run_jepa_obj.sh
set -euo pipefail

DS=${DS:-synth}
LR=2e-5; MAX_NEW_TOKENS=256; EPOCHS=4
DATE=${DATE:-22 Sep 2026}
PAD='<|finetune_right_pad_id|>'
OUT_ROOT=${OUT_ROOT:-diag_runs/jepa_null/phaseC/${DS}}
DEFAULT_ARMS="base:0.0:0:-:-:- cos:1.0:1:cos:-1:last nce0.25:0.25:1:infonce:-1:last nce1:1.0:1:infonce:-1:last"
read -r -a ARMS <<< "${ARMS:-${DEFAULT_ARMS}}"
read -r -a SEEDS <<< "${SEEDS:-82}"
N=$(( ${#ARMS[@]} * ${#SEEDS[@]} ))
if [ -n "${NCONFIGS:-}" ]; then echo "${N} tasks -> --array=0-$(( N - 1 ))"; exit 0; fi

cd "${SLURM_SUBMIT_DIR:-$PWD}"
[ -f finetune.py ] && [ -d diagnostics ] || { echo "submit from llm-jepa-analysis/" >&2; exit 1; }
source diagnostics/common.sh
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

if [ -n "${POST:-}" ]; then
  # pairs: every non-base arm vs base at the same seed, and seed pairs within an arm
  python3 - "${OUT_ROOT}" "${DS}" <<'EOF'
import glob, itertools, os, sys
root, ds = sys.argv[1], sys.argv[2]
cells = {}
for d in sorted(glob.glob(f"{root}/*/*_*_*")):
    if os.path.exists(f"{d}/checkpoints/model.safetensors"):
        cells.setdefault(d.split("/")[-2], {})[int(d.rsplit("_", 1)[1])] = d
base = cells.get("base", {})
pairs = []
for arm, seeds in sorted(cells.items()):
    for a, b in itertools.combinations(sorted(seeds), 2):
        pairs.append(("base_seed" if arm == "base" else "jepa_seed", ds, seeds[a], seeds[b]))
    if arm != "base":
        pairs += [("jepa_vs_base", ds, base[s], d) for s, d in sorted(seeds.items()) if s in base]
open(f"{root}/pairs.tsv", "w").writelines("\t".join(p) + "\n" for p in pairs)
print(f"{len(pairs)} pairs")
EOF
  pids=()
  for ((g = 0; g < NGPU; g++)); do
    CUDA_VISIBLE_DEVICES=${GPUS[$g]} python3 diagnostics/jepa_probe.py pairs --pairs_file="${OUT_ROOT}/pairs.tsv" \
      --test_file=datasets/${DS}_test.jsonl --shard_id=${g} --num_shards=${NGPU} \
      --out="${OUT_ROOT}/pairs.${g}.jsonl" > "${OUT_ROOT}/pairs_log.${g}.txt" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "${p}"; done
  python3 diagnostics/jepa_probe_report.py --dir="${OUT_ROOT}" --phase=C
  exit 0
fi

TASK=${SLURM_ARRAY_TASK_ID:-0}
[ "${TASK}" -lt "${N}" ] || { echo "task ${TASK} >= ${N}" >&2; exit 1; }
SPEC=${ARMS[$(( TASK % ${#ARMS[@]} ))]}
SEED=${SEEDS[$(( TASK / ${#ARMS[@]} ))]}
IFS=: read -r ARM LBD K OBJ LAYER POOL <<< "${SPEC}"

export MASTER_ADDR=$(hostname -I | awk '{print $1}')
export MASTER_PORT=$((29900 + TASK))

TAG="${LBD}_${K}_${SEED}"
RUN_DIR=${OUT_ROOT}/${ARM}/${TAG}
CKPT_DIR=${RUN_DIR}/checkpoints
[ -n "${RERUN:-}" ] && rm -rf "${RUN_DIR}"
mkdir -p "${RUN_DIR}"
echo "=== phaseC ${DS} ${ARM} ${TAG} obj=${OBJ} layer=${LAYER} pool=${POOL} -> ${RUN_DIR} ==="
nvidia-smi --query-gpu=index,name,memory.used --format=csv

cat > "${RUN_DIR}/config.json" <<EOF
{"dataset": "${DS}_phaseC_${ARM}", "base_dataset": "${DS}", "arm": "${ARM}", "spec": "${SPEC}",
 "jepa_obj": "${OBJ}", "jepa_layer": "${LAYER}", "jepa_pool": "${POOL}",
 "pad_token": "${PAD}", "date_string": "${DATE}", "lr": "${LR}", "lbd": ${LBD}, "k": ${K},
 "seed": ${SEED}, "epochs": ${EPOCHS}, "batch_size": 16, "grad_accum": 1,
 "model": "${MODEL}", "max_new_tokens": ${MAX_NEW_TOKENS}, "stage": "jepa_null_phaseC"}
EOF

# --- 1. Train ---
COMMON=(--train_file datasets/${DS}_train.jsonl --output_dir=${CKPT_DIR} --num_epochs=${EPOCHS}
        --finetune_seed=${SEED} --model_name=${MODEL} --learning_rate=${LR}
        --batch_size=16 --grad_accum=1 --save_epoch_checkpoints
        --pad_token="${PAD}" --train_date_string="${DATE}")
if [ ! -f "${CKPT_DIR}/model.safetensors" ]; then
  if [ "${LBD}" = "0.0" ]; then
    torchrun --nproc_per_node=4 diagnostics/finetune_jepa_obj.py "${COMMON[@]}" --predictors=0 --regular
  else
    torchrun --nproc_per_node=4 diagnostics/finetune_jepa_obj.py "${COMMON[@]}" \
      --lbd=${LBD} --predictors=${K} --last_token=-2 --additive_mask \
      --jepa_obj=${OBJ} --jepa_layer=${LAYER} --jepa_pool=${POOL}
  fi
else
  echo "=== ${ARM}/${TAG}: final model present, skipping training ==="
fi
cp -f "${CKPT_DIR}/trainer_state.json" "${RUN_DIR}/trainer_state.json" 2>/dev/null || true
rm -rf "${CKPT_DIR}"/checkpoint-*

# --- 2. Per-example eval, date pinned ---
if [ ! -s "${RUN_DIR}/eval/summary.json" ]; then
  GEN_ARGS=(--date_string="${DATE}" --trace_n=20)
  run_eval_sharded "${CKPT_DIR}" "${RUN_DIR}/eval" datasets/${DS}_test.jsonl ${MAX_NEW_TOKENS} \
    | tee "${RUN_DIR}/results.txt"
fi

# --- 3. Probe: last-token and content-mean pooling, all layers (GPU 0 and 1 in parallel) ---
P=(--cell_dir="${RUN_DIR}" --dataset=${DS} --arm=${ARM} --lbd=${LBD} --k=${K} --seed=${SEED} --n=1000)
CUDA_VISIBLE_DEVICES=${GPUS[0]} python3 diagnostics/jepa_probe.py embed "${P[@]}" --pool=last \
  --out="${RUN_DIR}/embed.last.jsonl" > "${RUN_DIR}/probe_last.log" 2>&1 &
p1=$!
CUDA_VISIBLE_DEVICES=${GPUS[1]} python3 diagnostics/jepa_probe.py embed "${P[@]}" --pool=mean \
  --out="${RUN_DIR}/embed.mean.jsonl" > "${RUN_DIR}/probe_mean.log" 2>&1 &
p2=$!
wait ${p1}; wait ${p2}
grep -h "^\[embed\]" "${RUN_DIR}"/probe_*.log
echo "=== phaseC ${ARM}/${TAG} done ($(du -sh "${RUN_DIR}" | cut -f1)) ==="
