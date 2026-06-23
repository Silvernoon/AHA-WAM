#!/usr/bin/env bash
set -euo pipefail

NPROC_PER_NODE="${1:?Usage: bash scripts/train_zero2.sh <nproc_per_node> [hydra_overrides...]}"
shift

EXTRA_ARGS=("$@")
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  case "${EXTRA_ARGS[0]}" in
    causal)
      echo "Error: causal training variants are not included in this release. Use task=<task> with model=ahawam for retained chunk-local training." >&2
      exit 1
      ;;
    standard)
      EXTRA_ARGS=("${EXTRA_ARGS[@]:1}")
      ;;
  esac
fi
NUM_MACHINES="${NNODES:-1}"
MACHINE_RANK="${NODE_RANK:-0}"
MAIN_PROCESS_IP="${MASTER_ADDR:-127.0.0.1}"
MAIN_PROCESS_PORT="${MASTER_PORT:-29500}"

DEFAULT_TASK="robotwin_ahawam"
DEFAULT_MODEL="ahawam"
is_integer() {
  [[ "${1}" =~ ^[0-9]+$ ]]
}

if ! is_integer "${NUM_MACHINES}" || ! is_integer "${MACHINE_RANK}"; then
  echo "Error: NUM_MACHINES (${NUM_MACHINES}) and MACHINE_RANK (${MACHINE_RANK}) must be integers." >&2
  exit 1
fi

extract_task_basename() {
  local cfg="$1"
  if [[ "${cfg}" == task/* ]]; then
    local name="${cfg#task/}"
    name="${name%.yaml}"
    echo "${name}"
    return 0
  fi
  return 1
}

# Read the model specified in `- override /model: <model>` from a task yaml file.
# Returns the model name, or empty string if not found.
extract_model_from_task_yaml() {
  local task_name="$1"
  local yaml_path="./configs/task/${task_name}.yaml"
  if [[ ! -f "${yaml_path}" ]]; then
    echo ""
    return 0
  fi
  local model_name
  model_name="$(grep -m1 '^\s*-\s*override\s*/model\s*:' "${yaml_path}" \
    | sed 's/.*\/model\s*:\s*//' \
    | tr -d '[:space:]')"
  echo "${model_name}"
}

TASK_BASENAME="${DEFAULT_TASK}"
HAS_TASK_OVERRIDE=0
HAS_MODEL_OVERRIDE=0
for ((i = 0; i < ${#EXTRA_ARGS[@]}; i++)); do
  arg="${EXTRA_ARGS[$i]}"
  case "${arg}" in
    --config-name)
      if ((i + 1 < ${#EXTRA_ARGS[@]})); then
        next="${EXTRA_ARGS[$((i + 1))]}"
        if parsed="$(extract_task_basename "${next}")"; then
          TASK_BASENAME="${parsed}"
        fi
      fi
      ;;
    --config-name=*)
      cfg="${arg#--config-name=}"
      if parsed="$(extract_task_basename "${cfg}")"; then
        TASK_BASENAME="${parsed}"
      fi
      ;;
    task=*)
      cfg="${arg#task=}"
      cfg="${cfg%.yaml}"
      TASK_BASENAME="${cfg}"
      HAS_TASK_OVERRIDE=1
      ;;
    model=*)
      HAS_MODEL_OVERRIDE=1
      ;;
  esac
done

if [[ "${HAS_TASK_OVERRIDE}" -eq 0 ]]; then
  EXTRA_ARGS=("task=${DEFAULT_TASK}" "${EXTRA_ARGS[@]}")
fi

if [[ "${HAS_MODEL_OVERRIDE}" -eq 0 ]]; then
  YAML_MODEL="$(extract_model_from_task_yaml "${TASK_BASENAME}")"
  if [[ -n "${YAML_MODEL}" ]]; then
    echo "[model] resolved from task yaml (${TASK_BASENAME}.yaml): model=${YAML_MODEL}"
    EXTRA_ARGS=("model=${YAML_MODEL}" "${EXTRA_ARGS[@]}")
  else
    echo "[model] task yaml not found or no /model override in it, falling back to default: model=${DEFAULT_MODEL}"
    EXTRA_ARGS=("model=${DEFAULT_MODEL}" "${EXTRA_ARGS[@]}")
  fi
fi

if [[ -z "${RUN_ID:-}" ]]; then
  if (( NUM_MACHINES <= 1 )); then
    RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)"
  else
    RUN_ID_SYNC_TIMEOUT="${RUN_ID_SYNC_TIMEOUT:-180}"
    RUN_ID_SYNC_PORT="${RUN_ID_SYNC_PORT:-$((MAIN_PROCESS_PORT + 11))}"

    export RUN_ID_SYNC_HOST="${MAIN_PROCESS_IP}"
    export RUN_ID_SYNC_PORT
    export RUN_ID_SYNC_TIMEOUT
    export RUN_ID_SYNC_MACHINE_RANK="${MACHINE_RANK}"
    export RUN_ID_SYNC_NUM_MACHINES="${NUM_MACHINES}"
    export RUN_ID_SYNC_TASK_BASENAME="${TASK_BASENAME}"

    RUN_ID="$(
      python - <<'PY'
import datetime
import os
from datetime import timedelta

import torch.distributed as dist

host = os.environ["RUN_ID_SYNC_HOST"]
port = int(os.environ["RUN_ID_SYNC_PORT"])
timeout_s = int(os.environ["RUN_ID_SYNC_TIMEOUT"])
machine_rank = int(os.environ["RUN_ID_SYNC_MACHINE_RANK"])
num_machines = int(os.environ["RUN_ID_SYNC_NUM_MACHINES"])
task_basename = os.environ.get("RUN_ID_SYNC_TASK_BASENAME", "train")

store = dist.TCPStore(
    host_name=host,
    port=port,
    world_size=num_machines,
    is_master=(machine_rank == 0),
    timeout=timedelta(seconds=timeout_s),
)
key = f"run_id::{task_basename}"
if machine_rank == 0:
    run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    store.set(key, run_id)
run_id = store.get(key).decode("utf-8")
print(run_id)
PY
    )"

    echo "[run_id_sync] mode=tcpstore host=${RUN_ID_SYNC_HOST} port=${RUN_ID_SYNC_PORT} timeout_s=${RUN_ID_SYNC_TIMEOUT} run_id=${RUN_ID}"
  fi
fi

echo "[launch] nproc_per_node=${NPROC_PER_NODE} num_machines=${NUM_MACHINES} machine_rank=${MACHINE_RANK} task=${TASK_BASENAME} run_id=${RUN_ID}"

#   "output_dir=./runs/${TASK_BASENAME}/${RUN_ID}" \

HYDRA_FULL_ERROR=1 accelerate launch \
  --config_file scripts/accelerate_configs/accelerate_zero2_ds.yaml \
  --num_processes "${NPROC_PER_NODE}" \
  scripts/train.py \
  "wandb.name=${TASK_BASENAME}" \
  "${EXTRA_ARGS[@]}"