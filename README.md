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
- `SUPERMARKET_ORDER_TIMEOUT`: per-order timeout in seconds, default `150`.
- `SUPERMARKET_MATCH_TIMEOUT`: safe match deadline, default `570`.
- `SUPERMARKET_SHOW=1`: request the optional OpenCV result window.

Runtime summaries are written below `/tmp/supermarket_competition/<run_prefix>`
inside the Client container.  The formal path never enables fixed-layout or
ground-truth diagnostic options.

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
