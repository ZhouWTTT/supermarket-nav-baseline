#!/usr/bin/env bash
set -eo pipefail

baseline_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

source /opt/ros/humble/setup.bash
source "${baseline_root}/ros2_ws/install/setup.bash"

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-99}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"

# This entry point intentionally starts only the candidate navigation stack.
# The formal run_baseline.sh remains on the validated legacy chain until the
# candidate passes the release gates documented in the refactor plan.
exec ros2 launch supermarket_bringup candidate_navigation.launch.py "$@"
