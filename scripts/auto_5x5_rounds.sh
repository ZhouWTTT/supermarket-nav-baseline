#!/usr/bin/env bash
# 5 rounds x 5 orders, 30 min per round (forced restart), 5-min checks that
# persist results to disk. Full docker logs saved before containers are removed.
set -u
LOGDIR=/home/zwt/runner_test_logs/auto5x5_$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOGDIR"
exec >"$LOGDIR/manager.log" 2>&1
echo "manager start $(date '+%F %T') logdir=$LOGDIR"

save_full() {
  # $1 = round number
  echo zwt | sudo -S docker logs supermarket_sorting_client 2>&1 >"$LOGDIR/round${1}_full.log"
  echo "saved round $1 full log ($(wc -l <"$LOGDIR/round${1}_full.log") lines)"
}

for round in 1 2 3 4 5; do
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
  RUNPREFIX=$(echo zwt | sudo -S docker logs supermarket_sorting_server 2>&1 | grep -oE 'run_[a-f0-9]+' | tail -1)
  echo "RUNPREFIX=$RUNPREFIX"
  echo zwt | sudo -S docker run -d --runtime=nvidia --network host --ipc host --name supermarket_sorting_client \
    -e ROS_DOMAIN_ID=99 -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp -e SUPERMARKET_PATH_MEMORY=1 \
    -e SUPERMARKET_PATH_MEMORY_FILE=/root/.cache/supermarket_path_memory_test.json -e YOLO_CONFIG_DIR=/tmp/Ultralytics \
    -e SUPERMARKET_MATCH_TIMEOUT=1800 \
    -v /home/zwt/baseline_dev:/workspace/baseline:ro -v supermarket_sorting_cache:/root/.cache \
    supermarket_sorting:client bash -lc 'source /opt/ros/humble/setup.bash && cd /workspace/baseline && ./scripts/run_baseline.sh'
  echo "round $round client started $(date '+%T')"

  for i in $(seq 1 6); do
    sleep 300
    if ! echo zwt | sudo -S docker ps --format '{{.Names}}' 2>/dev/null | grep -qx supermarket_sorting_client; then
      echo "round $round client exited early at check $i $(date '+%T'); finishing round"
      break
    fi
    echo "--- round $round check $i $(date '+%T') ---"
    echo zwt | sudo -S docker logs supermarket_sorting_client 2>&1 | grep -E "order id=|PLACE COMPLETE|match finished|delivered|failed|starting order" | tail -8 >>"$LOGDIR/round${round}_progress.log"
    echo zwt | sudo -S docker logs supermarket_sorting_client 2>&1 | grep -E "\[fatal\]|Traceback|\[ERROR\]" | tail -6 >>"$LOGDIR/round${round}_issues.log"
  done

  save_full "$round"
  echo "=== ROUND $round forced restart $(date '+%T') ==="
  echo zwt | sudo -S docker rm -f supermarket_sorting_client supermarket_sorting_server >/dev/null 2>&1
done
echo "=== ALL 5 ROUNDS DONE $(date '+%F %T') ==="
