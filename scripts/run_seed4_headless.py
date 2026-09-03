#!/usr/bin/env python3
"""Headless equivalent of the GUI launcher for obstacle-seed 4 validation.

Starts the official Server (obstacles pinned to seed 4) and the Client runner
from the current repository, then polls until the match summary appears or a
deadline passes.  Usage::

    python3 scripts/run_seed4_headless.py [--orders k,v] [--order-timeout S]
        [--match-timeout S]
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SERVER_NAME = "supermarket_runner_gui_server"
CLIENT_NAME = "supermarket_runner_gui_client"
SERVER_IMAGE = os.environ.get(
    "SUPERMARKET_GUI_SERVER_IMAGE",
    "crpi-1pzq998p9m7w0auy.cn-hangzhou.personal.cr.aliyuncs.com/"
    "challengecup/supermarket_sorting_final:server")
CLIENT_IMAGE = os.environ.get(
    "SUPERMARKET_GUI_CLIENT_IMAGE",
    "crpi-1pzq998p9m7w0auy.cn-hangzhou.personal.cr.aliyuncs.com/"
    "challengecup/supermarket_sorting_final:client")
CONTAINER_ROOT = "/workspace/baseline"
RUNTIME_DIR_CONTAINER = f"{CONTAINER_ROOT}/logs/competition_runner"
RUNTIME_DIR_HOST = REPO_ROOT / "logs" / "competition_runner"
TORCH_CACHE = "/root/.cache/torch_extensions/cu128"


def run(args, timeout=30.0):
    result = subprocess.run(
        args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    return result.returncode, result.stdout.decode(), result.stderr.decode()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--orders", default="product_016,product_020")
    parser.add_argument("--obstacle-seed", type=int, default=4)
    parser.add_argument(
        "--gui-seed", type=int, default=None,
        help="replicate the GUI launcher: SUPERMARKET_SEED=N, obstacle seed "
             "auto = N+1000003 (no SUPERMARKET_OBSTACLE_SEED), and the task "
             "list generated exactly like gui_competition_runner.generate_tasks")
    parser.add_argument(
        "--product-seed", type=int, default=None,
        help="SUPERMARKET_SEED for reproducible product placement")
    parser.add_argument("--order-timeout", type=int, default=300)
    parser.add_argument("--match-timeout", type=int, default=1200)
    parser.add_argument("--target-time", type=int, default=600)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--max-scan-cycles", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--inference-hz", type=float, default=12.0)
    parser.add_argument("--deadline-s", type=float, default=1800.0)
    args = parser.parse_args()

    for name in (CLIENT_NAME, SERVER_NAME):
        run(["docker", "rm", "-f", name], timeout=15.0)

    seed_env = []
    if args.product_seed is not None:
        seed_env = ["-e", f"SUPERMARKET_SEED={args.product_seed}"]
    orders = args.orders
    obstacle_seed = args.obstacle_seed
    if args.gui_seed is not None:
        # 与 GUI 完全一致：只传 SUPERMARKET_SEED，订单列表用同一个抽样算法，
        # 不传 SUPERMARKET_OBSTACLE_SEED（Server 端自动 obstacle=N+1000003）。
        import random as _random
        indexes = sorted(_random.Random(args.gui_seed).sample(
            range(1, 46), 5))
        orders = ",".join(f"product_{i:03d}" for i in indexes)
        seed_env = ["-e", f"SUPERMARKET_SEED={args.gui_seed}"]
        obstacle_seed = None
    server_args = [
        "docker", "run", "--rm", "-d", "--name", SERVER_NAME,
        "--gpus", "all", "--network", "host", "--ipc", "host",
        "-e", "ROS_DOMAIN_ID=99",
        "-e", "RMW_IMPLEMENTATION=rmw_cyclonedds_cpp",
        "-e", "MUJOCO_GL=egl", "-e", "SUPERMARKET_HEADLESS=1",
        "-e", "SUPERMARKET_ENABLE_RENDER=1",
        "-e", "SUPERMARKET_ENABLE_LIDAR=1",
        "-e", "SUPERMARKET_USE_GS=1",
        "-e", "SUPERMARKET_RANDOMIZE=1",
        "-e", "SUPERMARKET_RANDOMIZE_OBSTACLES=1",
        *([] if obstacle_seed is None
          else ["-e", f"SUPERMARKET_OBSTACLE_SEED={obstacle_seed}"]),
        "-e", f"SUPERMARKET_TASKS={orders}",
        "-e", f"TORCH_EXTENSIONS_DIR={TORCH_CACHE}",
        *seed_env,
        "-v", "supermarket_sorting_cache:/root/.cache",
        SERVER_IMAGE,
        "bash", "-lc",
        "cd /workspace/supermarket_sorting_task && "
        "source /opt/ros/humble/setup.bash && "
        "python3 examples/supermarket_sorting/supermarket_sorting_server.py",
    ]
    code, out, err = run(server_args, timeout=30.0)
    if code != 0:
        print("server start failed:", err)
        return 1
    print("server started")

    runner_cmd = [
        "python3", "examples/supermarket_sorting/competition_runner.py",
        "--weights", f"{CONTAINER_ROOT}/examples/supermarket_sorting/"
                     "perception/checkpoints/best.pt",
        "--max-scan-cycles", str(args.max_scan_cycles),
        "--max-attempts", str(args.max_attempts),
        "--memory-confirmations", "3",
        "--memory-confidence-threshold", "0.95",
        "--grab-policy", "nearest",
        "--inference-hz", f"{args.inference_hz:g}",
        "--device", args.device,
        "--order-timeout", str(args.order_timeout),
        "--match-timeout", str(args.match_timeout),
        "--target-time", str(args.target_time),
        "--runtime-dir", RUNTIME_DIR_CONTAINER,
        "--record-everywhere",
        "--perception-always-on",
        "--dynamic-direct",
        "--close-recheck",
    ]
    client_args = [
        "docker", "run", "--rm", "-d", "--name", CLIENT_NAME,
        "--gpus", "all", "--network", "host", "--ipc", "host",
        "-e", "ROS_DOMAIN_ID=99",
        "-e", "RMW_IMPLEMENTATION=rmw_cyclonedds_cpp",
        "-e", "YOLO_CONFIG_DIR=/tmp/Ultralytics",
        "-e", "SUPERMARKET_PATH_MEMORY=1",
        "-e",
        "SUPERMARKET_PATH_MEMORY_FILE=/root/.cache/supermarket_path_memory.json",
        "-e", f"PYTHONPATH={CONTAINER_ROOT}",
        "-e", f"TORCH_EXTENSIONS_DIR={TORCH_CACHE}",
        "-v", f"{REPO_ROOT}:{CONTAINER_ROOT}",
        "-v", "supermarket_sorting_cache:/root/.cache",
        CLIENT_IMAGE,
        "bash", "-lc",
        f"cd {shlex.quote(CONTAINER_ROOT)} && "
        "source /opt/ros/humble/setup.bash && "
        f"mkdir -p {shlex.quote(RUNTIME_DIR_CONTAINER)} && "
        f"exec {shlex.join(runner_cmd)}",
    ]
    code, out, err = run(client_args, timeout=30.0)
    if code != 0:
        print("client start failed:", err)
        run(["docker", "rm", "-f", SERVER_NAME])
        return 1
    print("client started")

    deadline = time.monotonic() + args.deadline_s
    baseline = max(
        (p.stat().st_mtime for p in RUNTIME_DIR_HOST.glob("*/summary.json")),
        default=0.0)
    summary_seen = None
    while time.monotonic() < deadline:
        time.sleep(5.0)
        _code, logs, _ = run(
            ["docker", "logs", "--tail", "200", SERVER_NAME], timeout=10.0)
        if "randomized corridor obstacles (seed=4" in logs:
            print("server: obstacles pinned to seed 4")
        newest = sorted(RUNTIME_DIR_HOST.glob("*/summary.json"),
                        key=lambda p: p.stat().st_mtime)
        if newest:
            candidate = newest[-1]
            import json as _json
            try:
                doc = _json.loads(candidate.read_text())
            except (ValueError, OSError):
                doc = {}
            # 忽略“任务受理/每单完成”的瞬时快照（accepted/worker_finished）；
            # 只认终局摘要（全部订单终结或比赛超时/致命失败）。
            reason = doc.get("reason")
            terminal = reason in {
                "orders_terminal", "match_timeout", "fatal_worker"}
            orders = doc.get("orders") or []
            finished_orders = sum(
                o.get("status") in {"delivered", "failed"} for o in orders)
            all_done = bool(orders) and finished_orders == len(orders)
            if ((terminal or all_done)
                    and float(doc.get("elapsed_s", 0.0)) > 20.0
                    and candidate.stat().st_mtime > baseline + 10.0):
                summary_seen = candidate
                break
        client_running = run(
            ["docker", "inspect", "-f", "{{.State.Running}}", CLIENT_NAME],
            timeout=10.0)[1].strip()
        if client_running == "false":
            print("client container exited before summary; checking logs")
            break

    if summary_seen is None:
        print("no fresh summary within deadline")
        _code, tail, _ = run(
            ["docker", "logs", "--tail", "60", CLIENT_NAME], timeout=10.0)
        print(tail[-4000:])
        return 2

    import json
    data = json.loads(summary_seen.read_text())
    print("summary:", summary_seen)
    print(json.dumps(data, ensure_ascii=False, indent=2)[:3000])
    print("run dir:", summary_seen.parent)
    return 0


if __name__ == "__main__":
    sys.exit(main())
