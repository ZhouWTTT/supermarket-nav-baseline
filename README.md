# Supermarket Sorting Competition Client

This repository is the client-side entry for the DG-202606 supermarket
sorting task.  The formal entry waits for `/supermarket_sorting/task`, validates
the version-1 JSON document, and executes every anonymous order as a
pick-deliver cycle.  A failed item is retried without terminating the match.

## Official Client image

Mount the repository at `/workspace/baseline` with the ROS domain settings
required by the organizer:

```bash
docker run --rm -dit \
  --gpus all --network host --ipc host \
  --name supermarket_sorting_client \
  -e ROS_DOMAIN_ID=99 \
  -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
  -v "$(pwd):/workspace/baseline:ro" \
  supermarket_sorting:client

docker exec -it supermarket_sorting_client \
  bash -lc 'cd /workspace/baseline && ./scripts/run_baseline.sh'
```

The entry script supports these optional environment variables:

- `SUPERMARKET_BASELINE_WEIGHTS`: YOLO checkpoint path.
- `SUPERMARKET_MAX_SCAN_CYCLES`: shelf scan cycles per attempt, default `2`.
- `SUPERMARKET_MAX_ATTEMPTS`: attempts per anonymous order, default `2`.
- `SUPERMARKET_INVENTORY_CONFIRMATIONS`: synchronized observations required
  before reusing a kind-to-ArUco mapping, default `3`.
- `SUPERMARKET_ORDER_TIMEOUT`: per-order timeout in seconds, default `300`;
  `0` disables it.
- `SUPERMARKET_MATCH_TIMEOUT`: safe match deadline, default `3600`.
- `SUPERMARKET_CLOSE_RECHECK=1`: give the level-aligned camera a short ArUco
  preference window before falling back to YOLO + depth; default `1`.
- `SUPERMARKET_SHOW=1`: request the optional OpenCV result window.

Runtime summaries are written below `/tmp/supermarket_competition/<run_prefix>`
inside the Client container.  The formal path never enables fixed-layout or
ground-truth diagnostic options.

## GUI launcher

宿主机运行项目根目录的 `gui_competition_runner.py`，可一键启动官方 Server
与当前正式比赛入口，并实时查看双方日志、3×15 记忆矩阵、每格观测证据和
比赛摘要：

```bash
python3 gui_competition_runner.py
```

GUI 会把本目录挂载到 Client 容器的 `/workspace/baseline`，执行
`examples/supermarket_sorting/competition_runner.py`，并将运行产物写到
`logs/competition_runner/<run_prefix>/`，便于宿主机每秒刷新。需要宿主机已
安装 `python3-tk` 与 Docker，并已拉取官方 final `:server` / `:client`
镜像。也可用 `SUPERMARKET_GUI_SERVER_IMAGE` 和
`SUPERMARKET_GUI_CLIENT_IMAGE` 环境变量覆盖镜像名。

## Submission image

The same entry can be packaged on top of the organizer's Client image:

```bash
docker build -f docker/Dockerfile.client \
  -t your-team/supermarket-sorting:submission .
```

The image defaults to `scripts/run_baseline.sh`; Server communication still
requires host networking, `ROS_DOMAIN_ID=99`, CycloneDDS, and GPU access.

## Development checks

The formal entry script can be checked on the host without ROS:

```bash
bash -n scripts/run_baseline.sh
```

End-to-end validation must use the organizer's Server and Client images with
randomized products and five randomized corridor obstacles.
