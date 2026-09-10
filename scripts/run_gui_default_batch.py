#!/usr/bin/env python3
"""Headless re-run of the GUI default configuration for several seeds.

This mirrors ``gui_competition_runner.py`` exactly (same Server env, same
Runner CLI flags, same task-list sampling) but runs the official Server in
headless mode so it can be driven from a non-interactive shell.

For each requested seed it:

1. generates the order list with the GUI sampling algorithm,
2. starts the official Server with ``SUPERMARKET_SEED=<seed>``,
3. starts the Client runner with the GUI default flags,
4. waits for the terminal match summary,
5. saves both container logs and the fresh runtime directory.

Usage::

    python3 scripts/run_gui_default_batch.py --seeds 101,202,303
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
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

# GUI defaults (gui_competition_runner.py _build_controls).
COUNT = 5
CYCLES = 2
ATTEMPTS = 2
CONFIRMATIONS = 3
MEMORY_CONF = 0.95
POLICY = "nearest"
INFERENCE_HZ = 8.0
DEVICE = "cuda"
ORDER_TIMEOUT = 300
MATCH_TIMEOUT = 3600
TARGET_TIME = 400


def run(args, timeout=30.0):
    result = subprocess.run(
        args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    return result.returncode, result.stdout.decode(), result.stderr.decode()


def generate_tasks(count: int, seed: int) -> str:
    indexes = sorted(random.Random(seed).sample(range(1, 46), count))
    return ",".join(f"product_{i:03d}" for i in indexes)


def cleanup():
    for name in (CLIENT_NAME, SERVER_NAME):
        run(["docker", "rm", "-f", name], timeout=20.0)


def start_server(seed: int, orders: str):
    server_args = [
        "docker", "run", "--rm", "-d", "--name", SERVER_NAME,
        "--gpus", "all", "--network", "host", "--ipc", "host",
        "-e", "ROS_DOMAIN_ID=99",
        "-e", "RMW_IMPLEMENTATION=rmw_cyclonedds_cpp",
        # GUI default enables the render window; headless keeps the same
        # renderer via EGL because this shell has no interactive X session.
        "-e", "MUJOCO_GL=egl", "-e", "SUPERMARKET_HEADLESS=1",
        "-e", "SUPERMARKET_ENABLE_RENDER=1",
        "-e", "SUPERMARKET_ENABLE_LIDAR=1",
        "-e", "SUPERMARKET_USE_GS=1",
        "-e", "SUPERMARKET_RGB_CAMERAS=head",
        "-e", "SUPERMARKET_RENDER_FPS=12",
        "-e", "SUPERMARKET_GS_SEQUENTIAL=0",
        "-e", "SUPERMARKET_RANDOMIZE=1",
        "-e", "SUPERMARKET_RANDOMIZE_OBSTACLES=1",
        "-e", f"SUPERMARKET_SEED={seed}",
        "-e", f"SUPERMARKET_TASKS={orders}",
        "-e", f"TORCH_EXTENSIONS_DIR={TORCH_CACHE}",
        "-v", "supermarket_sorting_cache:/root/.cache",
        "-v", (
            f"{REPO_ROOT / 'examples/supermarket_sorting/supermarket_sorting_server_render.py'}:"
            "/tmp/supermarket_sorting_server_render.py:ro"),
        SERVER_IMAGE,
        "bash", "-lc",
        "cd /workspace/supermarket_sorting_task && "
        "source /opt/ros/humble/setup.bash && "
        "python3 /tmp/supermarket_sorting_server_render.py",
    ]
    return run(server_args, timeout=40.0)


def start_client():
    runner = [
        "python3", "examples/supermarket_sorting/competition_runner.py",
        "--weights", f"{CONTAINER_ROOT}/examples/supermarket_sorting/"
                     "perception/checkpoints/best.pt",
        "--max-scan-cycles", str(CYCLES),
        "--max-attempts", str(ATTEMPTS),
        "--memory-confirmations", str(CONFIRMATIONS),
        "--memory-confidence-threshold", f"{MEMORY_CONF:.3f}",
        "--grab-policy", POLICY,
        "--inference-hz", f"{INFERENCE_HZ:g}",
        "--device", DEVICE,
        "--order-timeout", str(ORDER_TIMEOUT),
        "--match-timeout", str(MATCH_TIMEOUT),
        "--target-time", str(TARGET_TIME),
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
        f"exec {shlex.join(runner)}",
    ]
    return run(client_args, timeout=40.0)


def newest_summary_baseline() -> float:
    return max(
        (p.stat().st_mtime for p in RUNTIME_DIR_HOST.glob("*/summary.json")),
        default=0.0)


def wait_for_summary(baseline: float, deadline: float):
    while time.monotonic() < deadline:
        time.sleep(5.0)
        newest = sorted(RUNTIME_DIR_HOST.glob("*/summary.json"),
                        key=lambda p: p.stat().st_mtime)
        if not newest:
            continue
        candidate = newest[-1]
        if candidate.stat().st_mtime <= baseline + 10.0:
            continue
        try:
            doc = json.loads(candidate.read_text())
        except (ValueError, OSError):
            continue
        reason = doc.get("reason")
        orders = doc.get("orders") or []
        finished = sum(
            o.get("status") in {"delivered", "failed"} for o in orders)
        all_done = bool(orders) and finished == len(orders)
        terminal = reason in {"orders_terminal", "match_timeout", "fatal_worker"}
        if (terminal or all_done) and float(doc.get("elapsed_s", 0.0)) > 20.0:
            return candidate
        running = run(
            ["docker", "inspect", "-f", "{{.State.Running}}", CLIENT_NAME],
            timeout=10.0)[1].strip()
        if running == "false":
            return candidate
    return None


def save_logs(dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    for name, out in ((CLIENT_NAME, "client.log"), (SERVER_NAME, "server.log")):
        code, logs, err = run(
            ["docker", "logs", name], timeout=30.0)
        (dest / out).write_text(logs + ("\n--- stderr ---\n" + err if err else ""))


def run_one(seed: int, out_root: Path, deadline_s: float) -> dict:
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = out_root / f"{stamp}_seed{seed}"
    orders = generate_tasks(COUNT, seed)
    print(f"[seed {seed}] orders={orders} -> {dest}", flush=True)

    cleanup()
    baseline = newest_summary_baseline()
    code, _out, err = start_server(seed, orders)
    if code != 0:
        print(f"[seed {seed}] server start failed: {err}", flush=True)
        cleanup()
        return {"seed": seed, "orders": orders, "status": "server_failed"}
    code, _out, err = start_client()
    if code != 0:
        print(f"[seed {seed}] client start failed: {err}", flush=True)
        save_logs(dest)
        cleanup()
        return {"seed": seed, "orders": orders, "status": "client_failed"}

    summary = wait_for_summary(baseline, time.monotonic() + deadline_s)
    save_logs(dest)
    result = {"seed": seed, "orders": orders, "dir": str(dest)}
    if summary is not None:
        result["summary"] = str(summary)
        result["run_dir"] = str(summary.parent)
        try:
            doc = json.loads(summary.read_text())
            result["reason"] = doc.get("reason")
            result["elapsed_s"] = doc.get("elapsed_s")
            result["delivered"] = doc.get("delivered")
            result["failed"] = doc.get("failed")
            result["status"] = "ok"
        except (ValueError, OSError):
            result["status"] = "summary_unreadable"
    else:
        result["status"] = "no_summary"
    cleanup()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="101,202,303,404,505,606,707")
    parser.add_argument("--deadline-s", type=float, default=1500.0)
    parser.add_argument("--out", default=str(REPO_ROOT / "logs" / "gui_default_batch"))
    args = parser.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    results = []
    for i, seed in enumerate(seeds, 1):
        print(f"=== run {i}/{len(seeds)} seed {seed} "
              f"{dt.datetime.now():%F %T} ===", flush=True)
        res = run_one(seed, out_root, args.deadline_s)
        results.append(res)
        print(f"[seed {seed}] result: {json.dumps(res, ensure_ascii=False)}",
              flush=True)
        (out_root / "batch_results.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2))

    print("=== ALL DONE ===", flush=True)
    for res in results:
        print(json.dumps(res, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
