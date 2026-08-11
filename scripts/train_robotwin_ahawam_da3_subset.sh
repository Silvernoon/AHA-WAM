#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-/tank/test/sivn/robotwin2.0-fastwam/robotwin2.0}"
STATS_PATH="${STATS_PATH:-/tank/test/sivn/robotwin2.0-fastwam/dataset_stats.json}"
TEXT_CACHE="${TEXT_CACHE:-/tank/test/sivn/robotwin2.0-fastwam/text_embeds_subset_32_8}"
OUTPUT_DIR="${OUTPUT_DIR:-/tank/test/sivn/results/ahawam_da3_subset_stage1}"
MAX_STEPS="${MAX_STEPS:-500}"
TASK_CONFIG="${TASK_CONFIG:-robotwin_ahawam_da3_subset}"
LOG_EVERY="${LOG_EVERY:-10}"
SAVE_EVERY="${SAVE_EVERY:-100}"
EVAL_EVERY="${EVAL_EVERY:-100}"

export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-/tank/test/sivn/wan_models}"
export DIFFSYNTH_SKIP_DOWNLOAD="${DIFFSYNTH_SKIP_DOWNLOAD:-true}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SCRIPT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${SCRIPT_ROOT}/src:${SCRIPT_ROOT}/../depth-anything-3/src${PYTHONPATH:+:${PYTHONPATH}}"

EXTRA_OVERRIDES=()
if [[ -n "${RESUME:-}" ]]; then
  EXTRA_OVERRIDES+=("resume=${RESUME}" "init_checkpoint=null")
fi

exec "${PYTHON_BIN}" "${SCRIPT_ROOT}/scripts/train.py" \
  "task=${TASK_CONFIG}" \
  "data.train.dataset_dirs=[${DATA_ROOT}]" \
  "data.val.dataset_dirs=[${DATA_ROOT}]" \
  "data.train.pretrained_norm_stats=${STATS_PATH}" \
  "data.val.pretrained_norm_stats=${STATS_PATH}" \
  "data.train.text_embedding_cache_dir=${TEXT_CACHE}" \
  "data.val.text_embedding_cache_dir=${TEXT_CACHE}" \
  "output_dir=${OUTPUT_DIR}" \
  "max_steps=${MAX_STEPS}" \
  "log_every=${LOG_EVERY}" \
  "save_every=${SAVE_EVERY}" \
  "eval_every=${EVAL_EVERY}" \
  wandb.enabled=false \
  "${EXTRA_OVERRIDES[@]}"
