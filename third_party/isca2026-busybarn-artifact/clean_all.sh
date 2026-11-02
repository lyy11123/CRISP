#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

for d in fig_communication \
         fig_intra_ch \
         fig_intra_coreshape \
         fig_intra_power \
         fig_intra_multifaults \
         fig_endtoend \
         fig_convergence \
         fig_heatmap \
         fig_ablation; do
  echo "Cleaning $d"
  (cd "$d" && make clean)
done

for d in models/*/; do
  [ -d "$d" ] || continue
  echo "Cleaning $d"
  (cd "$d" && make clean)
done

echo "All clean."
