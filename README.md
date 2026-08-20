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
- `SUPERMARKET_ORDER_TIMEOUT`: per-order timeout in seconds; `0` disables it
  and is the default.
- `SUPERMARKET_MATCH_TIMEOUT`: safe match deadline, default `570`.
- `SUPERMARKET_SHOW=1`: request the optional OpenCV result window.

Runtime summaries are written below `/tmp/supermarket_competition/<run_prefix>`
inside the Client container.  The formal path never enables fixed-layout or
ground-truth diagnostic options.

## GUI launcher

宿主机运行项目根目录的 `gui_snapshot_pick.py` 可一键启动 Server、记忆矩阵
录入和连续多单抓取客户端（固定使用官方 final 镜像）：

```bash
python3 gui_snapshot_pick.py
```

它会挂载本目录到客户端容器的 `/workspace/supermarket_sorting_task` 并执行
`examples/supermarket_sorting/snapshot_pick_client.py`：先逐架行走录入并显示
3×15 记忆矩阵，再按记忆处理每单的「抓取 → 送货 → 放置 → 返回」。GUI 在
宿主机生成订单列表并通过 `--orders` 传给客户端。

正式比赛入口（任务话题驱动、带完整放桌流程）仍可通过 `scripts/run_baseline.sh`
运行，两者互不影响。

## Submission image

The same entry can be packaged on top of the organizer's Client image:

```bash
docker build -f docker/Dockerfile.client \
  -t your-team/supermarket-sorting:submission .
```

The image defaults to `scripts/run_baseline.sh`; Server communication still
requires host networking, `ROS_DOMAIN_ID=99`, CycloneDDS, and GPU access.

## Development checks

The task model has no ROS dependency and can be checked on the host:

```bash
python3 tests/test_competition_task.py
bash -n scripts/run_baseline.sh
```

End-to-end validation must use the organizer's Server and Client images with
randomized products and five randomized corridor obstacles.
