#!/usr/bin/env bash
set -eo pipefail

baseline_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

source /opt/ros/humble/setup.bash
source "${baseline_root}/ros2_ws/install/setup.bash"

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-99}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
export SUPERMARKET_REPO_ROOT="${baseline_root}"
export SUPERMARKET_NAV_BACKEND=nav2
export SUPERMARKET_BASE_CMD_TOPIC=/motion/manip_cmd_vel

# Independent stage-6 candidate.  The formal run_baseline.sh and the
# transitional run_candidate.sh remain unchanged until full-order gates pass.
exec ros2 launch supermarket_bringup candidate_system.launch.py "$@"
