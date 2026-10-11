#!/bin/bash
#SBATCH --account=westai0096
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=08:00:00
#SBATCH --partition=dc-hwai
#SBATCH --output=slurm/logs/thesis/models_%A_%a.out
#SBATCH --error=slurm/logs/thesis/models_%A_%a.err

# Small λ×k grid on a model other than Llama-3.2-1B — does the pad=eos label bug
# (memory k16-zero-diagnosis / stage 5) show up there too, and do we land on the
# paper's numbers first? Same cell pipeline as run_sweep.sh: train with per-epoch
# checkpoints -> evaluate.py (the recorded accuracy) -> per-example generations
# (stop reason eos/cap, p(eos) after gold) -> geometry on every epoch checkpoint
# -> tidy CSVs (accuracy_tidy / geometry_tidy via collect_sweep.py, per-step
# loss_curves via diagnostics/collect_losses.py).
#
# Grid per seed = the pilot grid (jepa/run_basic.sh, job 15434597): one no-JEPA
# baseline + λ{0.5,2.0} × k{1,2} -> 5 cells; default seeds {82,23} -> 10 cells.
# Seeds are the paper's (CLAUDE.md seed hygiene), not the pilot's retired 42/123.
#
# Knobs: MODEL_KEY (openelm|gemma), ARM (control = upstream pad = eos | padfix = per-model PADFIX_TOKEN), DS (synth|turk),
# SEEDS, CELLS (explicit "<λ>_<k>_<seed>" tags, replaces the grid), GRAD_ACCUM (default 4
# = finetune.py's default; 8 gives effective batch 4×8×4 GPUs = 128, the authors' upstream
# run.sh which trains on 8 GPUs × 4 × 4; outputs then go to <ARM>_bs128/), RERUN=1. Check the array size first:
#   NCONFIGS=1 [LIST=1] MODEL_KEY=openelm ./diagnostics/run_model_grid.sh
#   MODEL_KEY=openelm ARM=control sbatch --array=0-9 diagnostics/run_model_grid.sh
# Submit from the repo root (relative paths). Outputs:
#   diag_runs/models/<MODEL_KEY>/<DS>/<ARM>/<λ>_<k>_<seed>/
# Checkpoints are kept (~11 GB/cell for OpenELM) for the later padfix comparison.

set -euo pipefail

MODEL_KEY=${MODEL_KEY:-openelm}
ARM=${ARM:-control}
DS=${DS:-synth}

# Per-model settings, all from the paper / upstream code:
#   LR          paper's tuned baseline lr for (model, SYNTH) — Table 12 / Fig 7.
#   LAST_TOKEN  upstream run.sh's per-model --last_token (commented block). For
#               OpenELM, -4 is the last predictor token (or the last Text token at
#               k=0) only under transformers 4.55's Llama-2 tokenizer, which emits
#               '▁' '<0x0A>' after every </s>; transformers 5.3 emits '<0x0A>' alone.
#   VENV        OpenELM's remote modeling code cannot load on transformers 5.3 (meta-
#               device init breaks its rotary buffers; it calls cache APIs 5.x
#               removed), so it runs on the authors' pinned stack: torch 2.7.1+cu126,
#               transformers 4.55.2, peft 0.17.0, numpy 2.3.2 (setup.sh).
#   JEPA_EXTRA  no --additive_mask for OpenELM: its _update_causal_mask discards any
#               4-D mask and rebuilds a plain causal one, so the packed Code block
#               would attend to Text. Upstream run.sh's run_jepa does not pass it
#               either; finetune.py then encodes Text and Code in separate passes.
case "${MODEL_KEY}" in
  openelm)
    MODEL=apple/OpenELM-1_1B-Instruct
    LR=8e-5             # Table 12: baseline 12.07±1.81, JEPA 25.40±2.40 (λ4 k3); Fig 7d
    LAST_TOKEN=-4
    VENV_DIR=/p/project1/westai0096/vignesh_thesis/venv-tf4552
    # Its prepare_inputs_for_generation reads DynamicCache.seen_tokens / get_max_length(),
    # gone in 4.55.2 too -> every generate() raises. diagnostics/compat_openelm/
    # sitecustomize.py restores the two accessors (cached == uncached greedy, 4/4 checked).
    COMPAT_PYTHONPATH=diagnostics/compat_openelm
    # padfix pad: Llama-2 tokenizer's <unk> (id 0) — in the vocab with an embedding row,
    # never produced by byte-fallback SentencePiece, so it never occurs in the data.
    PADFIX_TOKEN='<unk>'
    DEFAULT_PAD=eos     # Llama-2 tokenizer has no pad -> finetune.py uses </s>: collides
    JEPA_EXTRA=() ;;
  gemma)
    MODEL=google/gemma-2-2b-it
    LR=1e-5             # Table 12: baseline 33.65±3.24, JEPA 43.12±2.61 (λ2 k4); Fig 7c
    # -2 = <end_of_turn> of the Text / Code turn (gemma ends a turn with <end_of_turn> \n;
    # tokenizer pads left, which _last_token_index handles), after any predictor tokens.
    LAST_TOKEN=-2
    VENV_DIR=/p/project1/westai0096/vignesh_thesis/venv-tf4552   # authors' stack, as for OpenELM
    COMPAT_PYTHONPATH=
    # Default pad is <pad> (0) != <eos> (1) != <end_of_turn> (107): <end_of_turn> is labelled
    # (400/400 on 200 rows). <eos> never occurs in the data, but the IT prior emits it right
    # after <end_of_turn>, which is what evaluate.py stops on. No padfix arm for gemma.
    PADFIX_TOKEN=
    DEFAULT_PAD='<pad>'
    JEPA_EXTRA=() ;;
  *) echo "unknown MODEL_KEY '${MODEL_KEY}' (expected openelm|gemma)" >&2; exit 1 ;;
esac
EVAL_ONLY=; EVAL_CMD=(python3); EOS_LABEL=false; EVAL_STOP=
case "${ARM}" in
  control) TRAIN_SCRIPT=finetune.py; PAD_ARGS=() ;;   # upstream: pad = eos, so eos is never labelled
  # finetune.py unchanged, pad swapped after setup (diagnostics/finetune_pad.py), so eos is
  # labelled. The checkpoint's saved tokenizer carries the new pad into eval/geometry.
  padfix)  [ -n "${PADFIX_TOKEN}" ] || { echo "no padfix arm for ${MODEL_KEY}: its default pad is not the turn terminator" >&2; exit 1; }
           TRAIN_SCRIPT=diagnostics/finetune_pad.py; PAD_ARGS=(--pad_token="${PADFIX_TOKEN}") ;;
  # Test arms (gemma termination, LOGBOOK 2026-10-09):
  # addmask  = control + --additive_mask on JEPA cells (the paper's 2-pass packing). With a
  #            LEFT-padding tokenizer (gemma) the packing pools a <pad> as Code and unmasks the
  #            left pads. Baseline cells (--regular) never use it: identical to control.
  addmask)  TRAIN_SCRIPT=finetune.py; PAD_ARGS=(); JEPA_EXTRA+=(--additive_mask) ;;
  # eoslabel = tokenizer eos appended to every full training conversation, so the token
  #            evaluate.py stops on is a training target (diagnostics/finetune_eos.py).
  eoslabel) TRAIN_SCRIPT=diagnostics/finetune_eos.py; PAD_ARGS=(); EOS_LABEL=true ;;
  # stopeot  = eval-only: the control<suffix> checkpoints re-scored with generation stopping
  #            on the turn marker instead of the tokenizer eos (diagnostics/eval_stop.py).
  #            Weights identical to control, so trainer_state + geometry are copied.
  stopeot)  [ "${MODEL_KEY}" = gemma ] || { echo "stopeot is defined for gemma only" >&2; exit 1; }
            EVAL_ONLY=control; EVAL_STOP='<end_of_turn>'
            EVAL_CMD=(python3 diagnostics/eval_stop.py --stop_token="${EVAL_STOP}") ;;
  *) echo "ARM '${ARM}' not implemented (expected control|padfix|addmask|eoslabel|stopeot)" >&2; exit 1 ;;
esac
case "${DS}" in
  synth|turk) MAX_NEW_TOKENS=256 ;;
  *) echo "DS '${DS}' not supported here (expected synth|turk)" >&2; exit 1 ;;
esac

# λ are strings in on-disk spelling (0.5 / 2.0) — the tag is the cell directory name.
LBDS=(0.5 2.0)
KS=(1 2)
read -r -a SEEDS <<< "${SEEDS:-82 23}"
GRAD_ACCUM=${GRAD_ACCUM:-4}
BATCH=$(( 4 * GRAD_ACCUM * 4 ))   # per-device batch 4 (finetune.py default) × accum × 4 GPUs
RUNS_SUFFIX=$([ "${GRAD_ACCUM}" = 4 ] && echo "" || echo "_bs${BATCH}")
CONFIGS=()
read -r -a CELL_TAGS <<< "${CELLS:-}"
for T in "${CELL_TAGS[@]+"${CELL_TAGS[@]}"}"; do CONFIGS+=("${T//_/ }"); done
[ ${#CONFIGS[@]} -gt 0 ] || for S in "${SEEDS[@]}"; do
  CONFIGS+=("0.0 0 ${S}")
  for L in "${LBDS[@]}"; do
    for K in "${KS[@]}"; do
      CONFIGS+=("${L} ${K} ${S}")
    done
  done
done

if [ -n "${NCONFIGS:-}" ]; then
  echo "${MODEL_KEY}/${DS}/${ARM}${RUNS_SUFFIX} (batch ${BATCH}): ${#CONFIGS[@]} cells -> --array=0-$(( ${#CONFIGS[@]} - 1 ))"
  [ -n "${LIST:-}" ] && printf '%s\n' "${CONFIGS[@]}" | nl -v0
  exit 0
fi
if [ "${SLURM_ARRAY_TASK_ID:-0}" -ge "${#CONFIGS[@]}" ]; then
  echo "task ${SLURM_ARRAY_TASK_ID} >= ${#CONFIGS[@]} configs" >&2; exit 1
fi

HF_CACHE_DIR=/p/project1/westai0096/vignesh_thesis/jepa/.cache/huggingface
REPO=${SLURM_SUBMIT_DIR:-$PWD}
export PYTHONPATH="${COMPAT_PYTHONPATH:+${REPO}/${COMPAT_PYTHONPATH}:}${REPO}":${PYTHONPATH:-}
module load Stages/2026 GCCcore/.14.3.0 Python/3.13.5
source "${VENV_DIR}/bin/activate"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi
python3 -c "import torch, transformers, sys; print('torch', torch.__version__, 'transformers', transformers.__version__, 'shim', 'sitecustomize' in sys.modules)"

export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_HUB_CACHE=${HF_CACHE_DIR}/hub
# Per-task caches: concurrent tasks race on a cold datasets build, and remote-code
# models are copied into the modules cache on every load.
export HF_DATASETS_CACHE=${TMPDIR:-/tmp}/hf_datasets_${SLURM_JOB_ID:-$$}
export HF_MODULES_CACHE=${TMPDIR:-/tmp}/hf_modules_${SLURM_JOB_ID:-$$}
export MASTER_ADDR=$(hostname -I | awk '{print $1}')
export MASTER_PORT=$((29500 + ${SLURM_ARRAY_TASK_ID:-0}))

EPOCHS=4
DATASET=datasets/${DS}
RUNS_DIR=diag_runs/models/${MODEL_KEY}/${DS}/${ARM}${RUNS_SUFFIX}
read -r LBD K SEED <<< "${CONFIGS[${SLURM_ARRAY_TASK_ID:-0}]}"
if [ "${LBD}" = "0.0" ]; then CELL_ARM=baseline; else CELL_ARM=jepa; fi
TAG="${LBD}_${K}_${SEED}"
RUN_DIR=${RUNS_DIR}/${TAG}
CKPT_DIR=${RUN_DIR}/checkpoints
GEO_FILE=${RUN_DIR}/geometry.jsonl
mkdir -p "${RUN_DIR}"

echo "=== ${MODEL_KEY} ${DS} ${ARM} ${TAG}: ${CELL_ARM} lbd=${LBD} k=${K} seed=${SEED} lr=${LR} last_token=${LAST_TOKEN} batch=${BATCH} ==="
cat > "${RUN_DIR}/config.json" <<EOF
{"dataset": "${DS}", "arm": "${CELL_ARM}", "pad_arm": "${ARM}", "lr": "${LR}", "lbd": ${LBD}, "k": ${K},
 "seed": ${SEED}, "epochs": ${EPOCHS}, "model": "${MODEL}", "last_token": ${LAST_TOKEN},
 "pad_token": "$([ "${ARM}" = padfix ] && echo "${PADFIX_TOKEN}" || echo "${DEFAULT_PAD}")", "grad_accum": ${GRAD_ACCUM}, "effective_batch": ${BATCH},
 "additive_mask": $([ ${#JEPA_EXTRA[@]} -gt 0 ] && [ "${CELL_ARM}" = jepa ] && echo true || echo false), "train_eos_label": ${EOS_LABEL},
 "eval_stop_token": "${EVAL_STOP:-eos}", "weights_from": "${EVAL_ONLY:+${EVAL_ONLY}${RUNS_SUFFIX}}", "max_new_tokens": ${MAX_NEW_TOKENS}}
EOF

trap 'python3 collect_sweep.py --runs_dir=${RUNS_DIR} || true; python3 diagnostics/collect_losses.py --runs_dir=${RUNS_DIR} || true' EXIT

if [ -z "${RERUN:-}" ] && grep -q "Success Rate" "${RUN_DIR}/results.txt" 2>/dev/null \
   && [ -s "${GEO_FILE}" ] && [ -s "${RUN_DIR}/generations/summary.json" ]; then
  echo "=== ${TAG}: already complete, skipping ==="; exit 0
fi

# A finished save is model.safetensors, or a shard index when the weights exceed the
# 5 GB shard size (gemma-2-2b: model-0000{1,2}-of-00002.safetensors + index).
has_weights() { [ -f "$1/model.safetensors" ] || [ -f "$1/model.safetensors.index.json" ]; }

# --- 0. Eval-only arms reuse another arm's finished cell (same weights).
if [ -n "${EVAL_ONLY}" ]; then
  SRC=diag_runs/models/${MODEL_KEY}/${DS}/${EVAL_ONLY}${RUNS_SUFFIX}/${TAG}
  has_weights "${SRC}/checkpoints" && [ -s "${SRC}/geometry.jsonl" ] \
    || { echo "eval-only arm needs a finished ${SRC}" >&2; exit 1; }
  ln -sfn "$(realpath "${SRC}/checkpoints")" "${CKPT_DIR}"
  cp -f "${SRC}/trainer_state.json" "${SRC}/geometry.jsonl" "${RUN_DIR}/"
fi

# --- 1. Train. --eval_steps=1 only sets logging_steps: lm_loss/jepa_loss every step.
if [ -n "${EVAL_ONLY}" ]; then
  echo "=== ${TAG}: eval-only arm, weights from ${SRC} ==="
elif [ -n "${RERUN:-}" ] || ! has_weights "${CKPT_DIR}"; then
  COMMON=(--train_file "${DATASET}_train.jsonl" --output_dir="${CKPT_DIR}"
          --num_epochs=${EPOCHS} --finetune_seed=${SEED}
          --model_name=${MODEL} --learning_rate=${LR}
          --grad_accum=${GRAD_ACCUM} --eval_steps=1 --save_epoch_checkpoints
          "${PAD_ARGS[@]+"${PAD_ARGS[@]}"}")
  if [ "${CELL_ARM}" = "baseline" ]; then
    torchrun --nproc_per_node=4 ${TRAIN_SCRIPT} "${COMMON[@]}" --predictors=0 --regular
  else
    torchrun --nproc_per_node=4 ${TRAIN_SCRIPT} "${COMMON[@]}" \
      --lbd=${LBD} --predictors=${K} --last_token=${LAST_TOKEN} "${JEPA_EXTRA[@]+"${JEPA_EXTRA[@]}"}"
  fi
  cp -f "${CKPT_DIR}/trainer_state.json" "${RUN_DIR}/trainer_state.json" 2>/dev/null \
    || cp -f "$(ls -d ${CKPT_DIR}/checkpoint-* | sort -t- -k2 -n | tail -1)/trainer_state.json" "${RUN_DIR}/trainer_state.json"
else
  echo "=== ${TAG}: final model present, skipping training ==="
fi

# --- 2. Accuracy (exact match), exactly as run_sweep.sh scores a cell.
if [ -n "${RERUN:-}" ] || ! grep -q "Success Rate" "${RUN_DIR}/results.txt" 2>/dev/null; then
  rm -f "${RUN_DIR}/eval.jsonl"
  "${EVAL_CMD[@]}" evaluate.py --model_name="${CKPT_DIR}" --input_file="${DATASET}_test.jsonl" \
    --output_file="${RUN_DIR}/eval.jsonl" --split_tune_untune --original_model_name=${MODEL} \
    --max_new_tokens ${MAX_NEW_TOKENS} --nosplit_data | tee "${RUN_DIR}/results.txt"
fi

# --- 3. Per-example generations: stop reason (eos vs cap), p(eos) right after the
# gold answer, prefix match. Same prompts/generate()/matcher as evaluate.py; the
# report says whether the rate reproduces results.txt. No date pinning: the
# OpenELM template has no date in it.
GEN_DIR=${RUN_DIR}/generations
if [ -n "${RERUN:-}" ] || [ ! -s "${GEN_DIR}/summary.json" ]; then
  # Shards are resumable (eval_generations skips examples already written), so a
  # killed task keeps them; only RERUN starts from scratch.
  if [ -n "${RERUN:-}" ]; then rm -rf "${GEN_DIR}"; fi
  rm -f "${GEN_DIR}/generations.jsonl" "${GEN_DIR}/summary.json" "${GEN_DIR}/report.txt"
  mkdir -p "${GEN_DIR}/shards"
  IFS=, read -r -a GEN_GPUS <<< "${CUDA_VISIBLE_DEVICES}"
  GEN_ARGS=(--model_name="${CKPT_DIR}" --original_model_name="${MODEL}"
            --input_file="${DATASET}_test.jsonl" --out_dir="${GEN_DIR}"
            --max_new_tokens="${MAX_NEW_TOKENS}" --trace_n=0)
  pids=()
  for g in "${!GEN_GPUS[@]}"; do
    CUDA_VISIBLE_DEVICES=${GEN_GPUS[$g]} "${EVAL_CMD[@]}" diagnostics/eval_generations.py "${GEN_ARGS[@]}" \
      --shard_id=${g} --num_shards=${#GEN_GPUS[@]} --device_map=cuda:0 \
      > "${GEN_DIR}/shards/log.${g}.txt" 2>&1 &
    pids+=($!)
  done
  rc=0; for p in "${pids[@]}"; do wait "${p}" || rc=1; done
  if [ ${rc} -ne 0 ]; then tail -n 20 "${GEN_DIR}"/shards/log.*.txt >&2; echo "generation shard failed" >&2; exit 1; fi
  "${EVAL_CMD[@]}" diagnostics/eval_generations.py "${GEN_ARGS[@]}" --merge \
    --reference_results="${RUN_DIR}/results.txt" | tee "${GEN_DIR}/report.txt"
  rm -rf "${GEN_DIR}/shards"
fi

# --- 4. Geometry on every epoch checkpoint (train + test, 500 examples each).
if [ -n "${RERUN:-}" ] || [ ! -s "${GEO_FILE}" ]; then
  rm -f "${GEO_FILE}"
  for ckpt in $(ls -d ${CKPT_DIR}/checkpoint-* 2>/dev/null | sort -t- -k2 -n); do
    STEP=$(basename "${ckpt}" | sed 's/checkpoint-//')
    echo "--- geometry ${TAG} step=${STEP} ---"
    python3 geometry.py --model_name="${ckpt}" --original_model_name="${MODEL}" \
      --input_files="train:${DATASET}_train.jsonl,test:${DATASET}_test.jsonl" \
      --output_file="${GEO_FILE}" --max_examples=500 \
      --lbd=${LBD} --k=${K} --seed=${SEED} --step="${STEP}"
  done
fi

echo "=== ${MODEL_KEY} ${DS} ${ARM} ${TAG} done ==="
