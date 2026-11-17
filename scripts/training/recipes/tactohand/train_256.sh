#!/usr/bin/env bash
set -euo pipefail

# Portable single-node entry; GPUS=8 reproduces the referenced campaign size.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${PROJECT_ROOT}"

GPUS="${GPUS:-1}"
if [[ ! "${GPUS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "GPUS must be a positive integer" >&2
  exit 2
fi
STRATEGY=auto
if (( GPUS > 1 )); then
  STRATEGY=ddp
fi

exec bash scripts/training/run_experiment.sh tactohand_256 \
  --data tactohand \
  --name "${RUN_NAME:-tactohand_256}" \
  --log-dir "${LOG_DIR:-./logs}" \
  --python "${PYTHON_BIN:-python}" \
  "training.devices=${GPUS}" \
  "training.gpu_ids=null" \
  "training.num_nodes=1" \
  "training.strategy=${STRATEGY}" \
  "$@"
