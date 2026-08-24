#!/usr/bin/env bash
set -eo pipefail

baseline_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

source /opt/ros/humble/setup.bash
source "${baseline_root}/ros2_ws/install/setup.bash"

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-99}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
export SUPERMARKET_NAV_BACKEND=nav2
export SUPERMARKET_BASE_CMD_TOPIC=/motion/manip_cmd_vel

ros2 launch supermarket_bringup candidate_navigation.launch.py &
nav_pid=$!

cleanup() {
    kill -TERM "${nav_pid}" 2>/dev/null || true
    wait "${nav_pid}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Lifecycle activation normally finishes in about one second with the Server
# online.  Bound startup so a malformed Nav2 configuration cannot leave the
# business runner waiting indefinitely.
ready=0
for _ in $(seq 1 50); do
    if ros2 lifecycle get /collision_monitor 2>/dev/null | grep -q '^active'; then
        ready=1
        break
    fi
    sleep 0.1
done
if [ "${ready}" != "1" ]; then
    echo "candidate navigation failed to become active within 5 seconds" >&2
    exit 1
fi

"${baseline_root}/scripts/run_baseline.sh"
