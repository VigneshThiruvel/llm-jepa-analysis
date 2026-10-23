#!/bin/bash
#SBATCH --account=westai0096
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --partition=dc-hwai
#SBATCH --output=slurm/logs/thesis/jepa_probe_%j.out
#SBATCH --error=slurm/logs/thesis/jepa_probe_%j.err

# Why does JEPA do nothing? Phase A (no training): diagnostics/jepa_probe.py over the
# stage-5 padfix checkpoints (synth + turk, seeds 82/23/37) and the untrained base model.
#   embed  A1 collapse vs alignment, A2 next-token stats at the pooled positions
#   pairs  A3 JEPA-vs-baseline against seed-vs-seed (KL, argmax disagreement, ||dtheta||,
#          generation agreement), A4 answer-only test NLL
# Output: diag_runs/jepa_null/phaseA/{embed,pairs}.<gpu>.jsonl, pairs.tsv, cells.tsv
#
#   sbatch diagnostics/run_jepa_probe.sh
#   DATASETS=synth N_EMBED=64 N_PAIRS=64 MAX_CELLS=2 ./diagnostics/run_jepa_probe.sh   # smoke

set -euo pipefail
cd /p/project1/westai0096/vignesh_thesis/llm-jepa-analysis
source diagnostics/common.sh
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8  # 4 workers on 32 cores; torch defaults to every core each

ROOT=diag_runs/k16/stage5_padfix
OUT=${OUT:-diag_runs/jepa_null/phaseA}
DATASETS=${DATASETS:-"synth turk"}
N_EMBED=${N_EMBED:-1000}
N_PAIRS=${N_PAIRS:-}
MAX_CELLS=${MAX_CELLS:-}
mkdir -p "${OUT}"

# cells.tsv: dataset, cell_dir|base, lbd, k, seed
python3 - "${ROOT}" "${OUT}" "${DATASETS}" "${MAX_CELLS}" <<'EOF'
import glob, itertools, os, sys
root, out, datasets, max_cells = sys.argv[1], sys.argv[2], sys.argv[3].split(), sys.argv[4]
cells, pairs = [], []
for ds in datasets:
    dirs = sorted(d for d in glob.glob(f"{root}/{ds}/padfix/*_*_*")
                  if os.path.exists(f"{d}/checkpoints/model.safetensors"))
    if max_cells:
        dirs = dirs[:int(max_cells)]
    cells.append((ds, "base", 0.0, 0, -1))
    by = {}
    for d in dirs:
        l, k, s = os.path.basename(d).split("_")
        cells.append((ds, d, float(l), int(k), int(s)))
        by.setdefault(f"{l}_{k}", {})[int(s)] = d
    base = by.get("0.0_0", {})
    for s, d in sorted(base.items()):
        pairs.append(("ft_vs_init", ds, "base", d))
    for tag, seeds in sorted(by.items()):
        for a, b in itertools.combinations(sorted(seeds), 2):
            pairs.append(("base_seed" if tag == "0.0_0" else "jepa_seed", ds, seeds[a], seeds[b]))
        if tag != "0.0_0":
            for s, d in sorted(seeds.items()):
                if s in base:
                    pairs.append(("jepa_vs_base", ds, base[s], d))
with open(f"{out}/cells.tsv", "w") as f:
    f.writelines("\t".join(map(str, c)) + "\n" for c in cells)
with open(f"{out}/pairs.tsv", "w") as f:
    f.writelines("\t".join(p) + "\n" for p in pairs)
print(f"{len(cells)} embed cells, {len(pairs)} pairs")
EOF

worker() {
  local g=$1 i=0 ds cell lbd k seed
  while IFS=$'\t' read -r ds cell lbd k seed; do
    if (( i++ % NGPU == g )); then
      CUDA_VISIBLE_DEVICES=${GPUS[$g]} python3 diagnostics/jepa_probe.py embed --cell_dir="${cell}" \
        --dataset="${ds}" --lbd="${lbd}" --k="${k}" --seed="${seed}" --n="${N_EMBED}" \
        --out="${OUT}/embed.${g}.jsonl" < /dev/null
    fi
  done < "${OUT}/cells.tsv"
  CUDA_VISIBLE_DEVICES=${GPUS[$g]} python3 diagnostics/jepa_probe.py pairs --pairs_file="${OUT}/pairs.tsv" \
    --shard_id="${g}" --num_shards="${NGPU}" ${N_PAIRS:+--n=${N_PAIRS}} --out="${OUT}/pairs.${g}.jsonl"
}

pids=()
for ((g = 0; g < NGPU; g++)); do
  worker "${g}" > "${OUT}/log.${g}.txt" 2>&1 &
  pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait "${p}" || rc=1; done
grep -h "^\[embed\]\|^\[pairs\]" "${OUT}"/log.*.txt | sort || true
exit ${rc}
