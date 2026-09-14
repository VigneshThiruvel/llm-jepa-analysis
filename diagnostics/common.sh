# Sourced by diagnostics/run_stage*.sh (from the repo root). Same environment as
# run_sweep.sh, except the module line names the stage explicitly: bare
# `module load Stages; module load Python/3.13.5` only resolves where the default
# stage and GCCcore happen to be preloaded (it fails on the login nodes).

module load Stages/2026 GCCcore/.14.3.0 Python/3.13.5
source /p/project1/westai0096/vignesh_thesis/jepa/.venv/bin/activate

export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_HUB_CACHE=/p/project1/westai0096/vignesh_thesis/jepa/.cache/huggingface/hub
export HF_DATASETS_CACHE=${TMPDIR:-/tmp}/hf_datasets_${SLURM_JOB_ID:-$$}

MODEL=meta-llama/Llama-3.2-1B-Instruct
IFS=, read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES:-0}"
NGPU=${#GPUS[@]}

# run_eval_sharded <ckpt_dir> <out_dir> <input_file> <max_new_tokens> [merge args...]
# One eval_generations.py shard per GPU, then a merge whose stdout is the
# report (callers tee it). Extra shard args come from the global array GEN_ARGS.
run_eval_sharded() {
  local ckpt=$1 out=$2 input=$3 mnt=$4
  shift 4
  mkdir -p "${out}/shards"
  local pids=() g rc=0
  for ((g = 0; g < NGPU; g++)); do
    CUDA_VISIBLE_DEVICES=${GPUS[$g]} python3 diagnostics/eval_generations.py \
      --model_name="${ckpt}" --original_model_name="${MODEL}" --input_file="${input}" \
      --out_dir="${out}" --max_new_tokens="${mnt}" --shard_id=${g} --num_shards=${NGPU} \
      --device_map=cuda:0 "${GEN_ARGS[@]}" > "${out}/shards/log.${g}.txt" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "${p}" || rc=1; done
  if [ ${rc} -ne 0 ]; then
    echo "eval shard failed for ${ckpt}:" >&2
    tail -n 5 "${out}"/shards/log.*.txt >&2
    return 1
  fi
  python3 diagnostics/eval_generations.py --merge --model_name="${ckpt}" \
    --original_model_name="${MODEL}" --input_file="${input}" --out_dir="${out}" \
    --max_new_tokens="${mnt}" "${GEN_ARGS[@]}" "$@"
}
