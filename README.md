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
- `SUPERMARKET_MATCH_TIMEOUT`: safe match deadline, default `570`.
- `SUPERMARKET_SHOW=1`: request the optional OpenCV result window.

Runtime summaries are written below `/tmp/supermarket_competition/<run_prefix>`
inside the Client container.  The formal path never enables fixed-layout or
ground-truth diagnostic options.

## GUI launcher

宿主机运行项目根目录的 `gui_snapshot_pick.py` 可一键启动 Server 与
快照优先多单客户端（固定使用官方 final 镜像）：

```bash
python3 gui_snapshot_pick.py
```

它会挂载本目录到客户端容器的 `/workspace/supermarket_sorting_task` 并执行
`examples/supermarket_sorting/snapshot_pick_client.py`：可选先正式行走逐架
录入记忆矩阵，之后每单按记忆直达对应货架/层抓取，执行
「抓货区抓取 → 终点直接扔货 → 返回抓货区」。GUI 在宿主机生成订单列表并
通过 `--orders` 传给客户端，自动录像写入 `logs/`。

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

The formal entry script can be checked on the host without ROS:

```bash
bash -n scripts/run_baseline.sh
```

End-to-end validation must use the organizer's Server and Client images with
randomized products and five randomized corridor obstacles.
