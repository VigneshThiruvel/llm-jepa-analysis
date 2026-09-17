#!/bin/bash
#SBATCH --account=westai0096
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --partition=dc-hwai
#SBATCH --output=slurm/logs/thesis/k16diag_s4_%j.out
#SBATCH --error=slurm/logs/thesis/k16diag_s4_%j.err
#
# Stage 4 of the k=16 diagnosis: does the first optimizer step flip the Text pool
# into the repeated-token regime? A few real AdamW steps from init per k.
#   sbatch diagnostics/run_k_step_flip.sh [extra k_step_flip.py args]
# Output: diag_runs/k16/stage4_flip/{report.txt,summary.json}

[ -f finetune.py ] && [ -d diagnostics ] || { echo "submit from llm-jepa-analysis/" >&2; exit 1; }
source diagnostics/common.sh
# dc-hwai nodes are not shared: --gres=gpu:1 still exposes all 4 GPUs (see run_k_scan_init.sh)
export CUDA_VISIBLE_DEVICES=$(cut -d, -f1 <<< "${CUDA_VISIBLE_DEVICES:-0}")
nvidia-smi

OUT=${OUT:-diag_runs/k16/stage4_flip}
mkdir -p "${OUT}"
python3 -u diagnostics/k_step_flip.py --out_dir="${OUT}" "$@" 2>&1 | tee "${OUT}/report.txt"
exit "${PIPESTATUS[0]}"
