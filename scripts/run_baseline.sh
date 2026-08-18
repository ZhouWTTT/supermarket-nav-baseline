#!/usr/bin/env bash
set -eo pipefail

baseline_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ -f /opt/ros/humble/setup.bash ]; then
    # ROS setup scripts may reference unset variables, so enable nounset only
    # after sourcing the official environment.
    source /opt/ros/humble/setup.bash
fi
set -u

export PYTHONPATH="${baseline_root}${PYTHONPATH:+:${PYTHONPATH}}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-99}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
export SUPERMARKET_PATH_MEMORY="${SUPERMARKET_PATH_MEMORY:-1}"
export SUPERMARKET_PATH_MEMORY_FILE="${SUPERMARKET_PATH_MEMORY_FILE:-/root/.cache/supermarket_path_memory.json}"

runner=(
    python3
    "${baseline_root}/examples/supermarket_sorting/competition_runner.py"
    --weights "${SUPERMARKET_BASELINE_WEIGHTS:-${baseline_root}/examples/supermarket_sorting/perception/checkpoints/best.pt}"
    --max-scan-cycles "${SUPERMARKET_MAX_SCAN_CYCLES:-2}"
    --max-attempts "${SUPERMARKET_MAX_ATTEMPTS:-2}"
    --memory-confirmations "${SUPERMARKET_MEMORY_CONFIRMATIONS:-${SUPERMARKET_INVENTORY_CONFIRMATIONS:-3}}"
    --memory-confidence-threshold "${SUPERMARKET_MEMORY_CONFIDENCE:-0.90}"
    --inference-hz "${SUPERMARKET_INFERENCE_HZ:-12}"
    --device "${SUPERMARKET_DEVICE:-cpu}"
    --order-timeout "${SUPERMARKET_ORDER_TIMEOUT:-0}"
    --match-timeout "${SUPERMARKET_MATCH_TIMEOUT:-570}"
    --target-time "${SUPERMARKET_TARGET_TIME:-400}"
)

if [ "${SUPERMARKET_SHOW:-0}" = "1" ]; then
    runner+=(--show)
fi

exec "${runner[@]}"
