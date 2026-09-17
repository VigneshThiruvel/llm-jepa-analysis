#!/bin/bash
#SBATCH --account=westai0096
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --partition=dc-hwai
#SBATCH --output=slurm/logs/thesis/k16diag_s2_%j.out
#SBATCH --error=slurm/logs/thesis/k16diag_s2_%j.err
#
# Stage 2 of the k=16 diagnosis: step-0 JEPA scan over k, no training.
#   sbatch diagnostics/run_k_scan_init.sh [extra k_scan_init.py args]
# Output: diag_runs/k16/stage2_init/{report.txt,summary.json,scan_vectors.pt}

[ -f finetune.py ] && [ -d diagnostics ] || { echo "submit from llm-jepa-analysis/" >&2; exit 1; }
source diagnostics/common.sh
# dc-hwai nodes are not shared, so --gres=gpu:1 still exposes all 4 GPUs, and
# finetune.setup_model_and_tokenizer's device_map="auto" would shard the model.
export CUDA_VISIBLE_DEVICES=$(cut -d, -f1 <<< "${CUDA_VISIBLE_DEVICES:-0}")
nvidia-smi --query-gpu=index,name,memory.total --format=csv

OUT=${OUT:-diag_runs/k16/stage2_init}
mkdir -p "${OUT}"
python3 -u diagnostics/k_scan_init.py --out_dir="${OUT}" "$@" 2>&1 | tee "${OUT}/report.txt"
exit "${PIPESTATUS[0]}"
