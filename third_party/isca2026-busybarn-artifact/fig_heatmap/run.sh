#!/usr/bin/env bash

set -euo pipefail

STEPS="${1:-1000}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p ./results

echo "Running plot_hot_transformer_block.py with --steps $STEPS"
python plot_hot_transformer_block.py --steps "$STEPS"

echo "Done. Figures in results/"
