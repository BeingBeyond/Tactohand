#!/usr/bin/env bash
set -euo pipefail

# Failure localizer: any nonzero exit reports the exact failing command +
# line, so a silent job death can be attributed to this wrapper or not.
# Does not fire on the final exec into python training.
trap 'rc=$?; if [[ $rc -ne 0 ]]; then echo "[run_experiment] FAILED rc=${rc} at line ${LINENO}: ${BASH_COMMAND}" >&2; fi' EXIT

usage() {
  cat <<'EOF'
Usage:
  bash scripts/training/run_experiment.sh EXPERIMENT [options] [hydra_overrides...]

Options:
  --data NAME       Hydra data group (default: tactohand)
  --name NAME       Human-readable suffix appended to the run directory
  --log-dir PATH    Root directory for logs (default: ./logs)
  --python PATH     Python executable or command on PATH (default: python3)
  --no-tee          Save train.log only; do not mirror training output to terminal
  --dry-run         Write artifacts and resolved config, then exit before training
  -h, --help        Show this help

Examples:
  bash scripts/training/run_experiment.sh tactohand_384
  bash scripts/training/run_experiment.sh tactohand_384 --name dexycb_smoke data.datasets=[dexycb] training.gpu_ids='[0]'
  bash scripts/training/run_experiment.sh tactohand_384 data.batch_size=16 training.strategy=auto

Artifacts:
  logs/<timestamp>_<experiment>[_name]/
    train.log
    command.txt
    resolved_config.yaml
    git_status.txt
    git_diff.patch
    env.txt
EOF
}

if [[ $# -eq 0 ]]; then
  usage
  exit 1
fi

EXPERIMENT_NAME=""
DATA_NAME="tactohand"
RUN_NAME=""
LOG_DIR="./logs"
PYTHON_BIN="${PYTHON_BIN:-python3}"
USE_TEE=1
DRY_RUN=0
OVERRIDES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --data)
      DATA_NAME="$2"
      shift 2
      ;;
    --name)
      RUN_NAME="$2"
      shift 2
      ;;
    --log-dir)
      LOG_DIR="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --no-tee)
      USE_TEE=0
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    name=*)
      RUN_NAME="${1#name=}"
      shift
      ;;
    data=*)
      DATA_NAME="${1#data=}"
      shift
      ;;
    *)
      if [[ -z "$EXPERIMENT_NAME" && "$1" != *=* && "$1" != +*=* && "$1" != "~"* ]]; then
        EXPERIMENT_NAME="$1"
      else
        OVERRIDES+=("$1")
      fi
      shift
      ;;
  esac
done

if [[ -z "$EXPERIMENT_NAME" ]]; then
  echo "[run_experiment] Missing EXPERIMENT." >&2
  usage
  exit 1
fi

if [[ "$PYTHON_BIN" != */* ]]; then
  PYTHON_BIN="$(command -v "$PYTHON_BIN" || true)"
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[run_experiment] Python executable not found or not executable: $PYTHON_BIN" >&2
  exit 1
fi

echo "[run_experiment] wrapper entered: experiment=${EXPERIMENT_NAME} data=${DATA_NAME} python=${PYTHON_BIN} log_dir=${LOG_DIR}" >&2

sanitize() {
  printf '%s' "$1" | tr -c 'A-Za-z0-9_.=-' '_' | sed 's/^_*//; s/_*$//'
}

timestamp="$(date +%Y%m%d_%H%M%S)"
run_suffix="$(sanitize "$EXPERIMENT_NAME")"
if [[ -n "$RUN_NAME" ]]; then
  run_suffix="${run_suffix}_$(sanitize "$RUN_NAME")"
fi

if [[ -z "${V2C_RUN_VERSION:-}" ]]; then
  export V2C_RUN_VERSION="${timestamp}_${run_suffix}"
else
  export V2C_RUN_VERSION
fi
EXPERIMENT_DIR="${LOG_DIR}/${V2C_RUN_VERSION}"
echo "[run_experiment] creating run dir: ${EXPERIMENT_DIR}" >&2
mkdir -p "$EXPERIMENT_DIR"

ARTIFACT_SUFFIX=""
if [[ "${V2C_MULTINODE:-0}" == "1" && -n "${NODE_RANK:-}" ]]; then
  ARTIFACT_SUFFIX=".node${NODE_RANK}"
fi
STDOUT_LOG="${STDOUT_LOG:-${EXPERIMENT_DIR}/train${ARTIFACT_SUFFIX}.log}"

write_exit_status() {
  local status="$1"
  local signal_name="${2:-}"
  {
    printf 'time: %s\n' "$(date --iso-8601=seconds 2>/dev/null || date)"
    if [[ -n "$signal_name" ]]; then
      printf 'signal: %s\n' "$signal_name"
    fi
    printf 'exit_code: %s\n' "$status"
  } > "$EXPERIMENT_DIR/exit_status${ARTIFACT_SUFFIX}.txt"
}

trap 'write_exit_status 143 TERM; exit 143' TERM
trap 'write_exit_status 130 INT; exit 130' INT
trap 'write_exit_status 129 HUP; exit 129' HUP

CMD=(
  "$PYTHON_BIN" -u training/train.py
  "experiment=$EXPERIMENT_NAME"
  "data=$DATA_NAME"
  "logging.save_dir=$LOG_DIR"
)
if [[ ${#OVERRIDES[@]} -gt 0 ]]; then
  CMD+=("${OVERRIDES[@]}")
fi

{
  printf 'cwd: %s\n' "$(pwd)"
  printf 'time: %s\n' "$(date --iso-8601=seconds 2>/dev/null || date)"
  printf 'V2C_RUN_VERSION: %s\n' "$V2C_RUN_VERSION"
  printf 'command:'
  printf ' %q' "${CMD[@]}"
  printf '\n'
	} > "$EXPERIMENT_DIR/command${ARTIFACT_SUFFIX}.txt"

git status --short > "$EXPERIMENT_DIR/git_status${ARTIFACT_SUFFIX}.txt" 2>&1 || true
git diff -- . > "$EXPERIMENT_DIR/git_diff${ARTIFACT_SUFFIX}.patch" 2>&1 || true

{
  printf 'python: %s\n' "$PYTHON_BIN"
  "$PYTHON_BIN" --version
  "$PYTHON_BIN" - <<'PY'
import platform
print("platform:", platform.platform())
try:
    import torch
    print("torch:", torch.__version__)
    print("cuda_available:", torch.cuda.is_available())
    print("cuda:", torch.version.cuda)
    print("gpu_count:", torch.cuda.device_count())
    for idx in range(torch.cuda.device_count()):
        print(f"gpu[{idx}]:", torch.cuda.get_device_name(idx))
except Exception as exc:
    print("torch_info_error:", repr(exc))
try:
    import torchcodec
    print("torchcodec:", getattr(torchcodec, "__version__", "unknown"))
except Exception as exc:
    print("torchcodec_error:", repr(exc))
PY
} > "$EXPERIMENT_DIR/env${ARTIFACT_SUFFIX}.txt" 2>&1 || true

CONFIG_CMD=("${CMD[@]}" "--cfg" "job")
if ! "${CONFIG_CMD[@]}" > "$EXPERIMENT_DIR/resolved_config${ARTIFACT_SUFFIX}.yaml" 2>"$EXPERIMENT_DIR/resolved_config${ARTIFACT_SUFFIX}.err"; then
  echo "[run_experiment] Warning: failed to dump resolved config; see $EXPERIMENT_DIR/resolved_config${ARTIFACT_SUFFIX}.err" >&2
fi

echo "[run_experiment] experiment=$EXPERIMENT_NAME data=$DATA_NAME"
echo "[run_experiment] run_dir=$EXPERIMENT_DIR"
echo "[run_experiment] command:"
printf '  %q' "${CMD[@]}"
echo

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "[run_experiment] dry-run complete; training was not started."
  exit 0
fi

set +e
if [[ "$USE_TEE" -eq 1 ]]; then
  set -o pipefail
  "${CMD[@]}" 2>&1 | tee "$STDOUT_LOG"
  status="${PIPESTATUS[0]}"
else
  "${CMD[@]}" > "$STDOUT_LOG" 2>&1
  status="$?"
fi
set -e
write_exit_status "$status"
exit "$status"
