#!/usr/bin/env bash
# 3 rounds x 5 orders, 30 min per round (forced restart). No log monitoring.
set -u
LOGDIR=/home/zwt/runner_test_logs/auto3x5_$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOGDIR"
exec >"$LOGDIR/manager.log" 2>&1
echo "manager start $(date '+%F %T') logdir=$LOGDIR"

for round in 1 2 3; do
  echo "=== ROUND $round start $(date '+%F %T') ==="
  TASKS=$(shuf -i 1-45 -n 5 | awk '{printf "%sproduct_%03d", sep, $1; sep=","} END {print ""}')
  echo "TASKS=$TASKS"
  echo zwt | sudo -S docker rm -f supermarket_sorting_client supermarket_sorting_server >/dev/null 2>&1
  xhost +local:docker >/dev/null 2>&1
  echo zwt | sudo -S docker run --rm -d --runtime=nvidia --network host --ipc host --name supermarket_sorting_server \
    -e "DISPLAY=${DISPLAY:-:1}" -e MUJOCO_GL=glfw -e SUPERMARKET_HEADLESS=0 -e SUPERMARKET_ENABLE_RENDER=1 -e SUPERMARKET_USE_GS=1 \
    -e "SUPERMARKET_TASKS=${TASKS}" -v /tmp/.X11-unix:/tmp/.X11-unix:rw -v supermarket_sorting_cache:/root/.cache \
    supermarket_sorting:server bash -lc 'source /opt/ros/humble/setup.bash && cd /workspace/supermarket_sorting_task && python3 examples/supermarket_sorting/supermarket_sorting_server.py'
  sleep 25
  echo zwt | sudo -S docker run -d --runtime=nvidia --network host --ipc host --name supermarket_sorting_client \
    -e ROS_DOMAIN_ID=99 -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp -e SUPERMARKET_PATH_MEMORY=1 \
    -e SUPERMARKET_PATH_MEMORY_FILE=/root/.cache/supermarket_path_memory_test.json -e YOLO_CONFIG_DIR=/tmp/Ultralytics \
    -e SUPERMARKET_MATCH_TIMEOUT=1800 \
    -v /home/zwt/baseline_dev:/workspace/baseline:ro -v supermarket_sorting_cache:/root/.cache \
    supermarket_sorting:client bash -lc 'source /opt/ros/humble/setup.bash && cd /workspace/baseline && ./scripts/run_baseline.sh'
  echo "round $round client started $(date '+%T'); waiting 30 min"
  sleep 1800
  echo "=== ROUND $round forced restart $(date '+%T') ==="
  echo zwt | sudo -S docker rm -f supermarket_sorting_client supermarket_sorting_server >/dev/null 2>&1
done
echo "=== ALL 3 ROUNDS DONE $(date '+%F %T') ==="
