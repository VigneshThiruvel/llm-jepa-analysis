#!/bin/bash
# Build the plotting dataset for the stage-5 pad-token test and render its figures.
#
# Each padfix run stamps config.json's `dataset` as "<ds>_<arm>" (synth_padfix,
# synth_control, turk_padfix, turk_control), so collect_sweep.py writes one tidy pair
# per arm — accuracy_tidy_<ds>_<arm>_<model>.csv + geometry_tidy_<ds>_<arm>_<model>.csv —
# and plot_sweep.py renders each into its own directory, <out>/<ds>_<arm>_<model>/,
# with the arm in every figure title. Nothing collides with the sweep's own tables.
#
#   ./diagnostics/make_padfix_dataset.sh                  # collect + assemble + plot
#   WITH_SWEEP=1 ./diagnostics/make_padfix_dataset.sh     # also copy the sweep tables in,
#                                                         # so the sweep figures render beside
#                                                         # the padfix ones for comparison
#   DATASETS=synth ./diagnostics/make_padfix_dataset.sh
#   PLOT=0 ./diagnostics/make_padfix_dataset.sh           # dataset only, no figures
#
# Outputs (all under the gitignored stage-5 root):
#   <root>/plot_data/       the tidy CSVs, ready for `plot_sweep.py --data-dir`
#   <root>/figures/<ds>_<arm>_<model>/  fig1..fig6 .png + each figure's _data.csv
#   <root>/padfix_vs_control.csv        cell-by-cell padfix vs control vs sweep
set -euo pipefail

[ -f plot_sweep.py ] && [ -d diagnostics ] || { echo "run from llm-jepa-analysis/" >&2; exit 1; }
source diagnostics/common.sh

ROOT=${ROOT:-diag_runs/k16/stage5_padfix}
DATA=${DATA:-${ROOT}/plot_data}
FIGS=${FIGS:-${ROOT}/figures}
read -r -a DATASETS <<< "${DATASETS:-synth turk}"
read -r -a ARMS <<< "${ARMS:-padfix control}"

mkdir -p "${DATA}"
for ds in "${DATASETS[@]}"; do
  for arm in "${ARMS[@]}"; do
    RUNS=${ROOT}/${ds}/${arm}
    [ -d "${RUNS}" ] || { echo "--- ${RUNS}: not present, skipping"; continue; }
    echo "=== collect ${ds}/${arm} ==="
    python3 collect_sweep.py --runs_dir="${RUNS}"
    # Only the <dataset>_<model>-stamped pair travels; collect_sweep.py also writes
    # legacy accuracy_tidy.csv / geometry_tidy.csv basenames, and copying those in
    # would make plot_sweep.py render one unnamed extra set from whichever arm
    # happened to be copied last.
    for f in "${RUNS}"/accuracy_tidy_"${ds}_${arm}"_*.csv "${RUNS}"/geometry_tidy_"${ds}_${arm}"_*.csv; do
      [ -f "${f}" ] && cp -f "${f}" "${DATA}/"
    done
  done
  if [ -n "${WITH_SWEEP:-}" ]; then
    for f in sweep_runs/${ds}/accuracy_tidy_${ds}_*.csv sweep_runs/${ds}/geometry_tidy_${ds}_*.csv; do
      [ -f "${f}" ] && cp -f "${f}" "${DATA}/"
    done
  fi
done

echo "=== dataset in ${DATA} ==="
ls -la "${DATA}"
python3 diagnostics/padfix_compare.py --datasets="$(IFS=,; echo "${DATASETS[*]}")" \
  --out_csv="${ROOT}/padfix_vs_control.csv"

if [ "${PLOT:-1}" = "1" ]; then
  echo "=== figures -> ${FIGS} ==="
  # k-fix 16 is the collapse column, so the per-layer and trajectory figures are
  # about the cells this test is for; the sweep's own runs used the default k=0.
  python3 plot_sweep.py --data-dir="${DATA}" --out-dir="${FIGS}" --k-fix="${KFIX:-16}" \
    ${SEEDS_ARG:+--seeds="${SEEDS_ARG}"}
  find "${FIGS}" -name '*.png' | sort
fi
