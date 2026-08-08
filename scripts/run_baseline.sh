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

runner=(
    python3
    "${baseline_root}/examples/supermarket_sorting/competition_runner.py"
    --weights "${SUPERMARKET_BASELINE_WEIGHTS:-${baseline_root}/examples/supermarket_sorting/perception/checkpoints/best.pt}"
    --max-scan-cycles "${SUPERMARKET_MAX_SCAN_CYCLES:-2}"
    --max-attempts "${SUPERMARKET_MAX_ATTEMPTS:-2}"
    --inventory-confirmations "${SUPERMARKET_INVENTORY_CONFIRMATIONS:-3}"
    --order-timeout "${SUPERMARKET_ORDER_TIMEOUT:-0}"
    --match-timeout "${SUPERMARKET_MATCH_TIMEOUT:-570}"
)

if [ "${SUPERMARKET_SHOW:-0}" = "1" ]; then
    runner+=(--show)
fi

exec "${runner[@]}"
