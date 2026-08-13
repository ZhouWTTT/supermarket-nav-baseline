#!/usr/bin/env python3
"""进程级日志落盘：把 stdout/stderr 同时写控制台和带时间戳的日志文件。

ROS2/rclpy 的控制台输出由 C 层直接写文件描述符 1/2，Python 层替换
``sys.stdout``/``sys.stderr`` 拦不到它。因此这里把底层文件描述符重定向到
一个管道，由后台线程同时写回原控制台流和日志文件，保证所有输出（含 rclpy
日志、print、第三方库输出）都不丢失。

用法::

    from run_log import start_run_log
    start_run_log("gui_client_continuous")

日志默认写到仓库 ``logs/`` 目录（容器内挂载即宿主机 logs/），可用
``SUPERMARKET_LOG_DIR`` 环境变量覆盖；``SUPERMARKET_RUN_LOG=0`` 可关闭。
"""

from __future__ import annotations

import datetime
import os
import sys
import threading
from pathlib import Path


def _writable_dir(candidates: list[Path]) -> Path:
    for candidate in candidates:
        try:
            path = Path(candidate)
            path.mkdir(parents=True, exist_ok=True)
            probe = path / ".run_log_probe"
            probe.write_text("", encoding="utf-8")
            probe.unlink()
            return path
        except OSError:
            continue
    return Path("/tmp")


def _tee_fd(fd: int, log_file) -> None:
    """把 fd 通过管道重定向；后台线程把数据同时写回原流和日志文件。"""
    try:
        saved = os.dup(fd)
    except OSError:
        return
    read_fd, write_fd = os.pipe()
    os.dup2(write_fd, fd)
    os.close(write_fd)

    def pump() -> None:
        try:
            while True:
                chunk = os.read(read_fd, 8192)
                if not chunk:
                    break
                try:
                    os.write(saved, chunk)
                except OSError:
                    pass
                try:
                    log_file.write(
                        chunk.decode("utf-8", errors="replace"))
                    log_file.flush()
                except (OSError, ValueError):
                    pass
        finally:
            try:
                os.close(read_fd)
            except OSError:
                pass
            try:
                os.close(saved)
            except OSError:
                pass

    threading.Thread(
        target=pump, daemon=True, name=f"run-log-tee-{fd}").start()


def start_run_log(prefix: str = "client", log_dir: str | None = None) -> Path | None:
    """把进程 stdout/stderr 落地到带时间戳的日志文件，返回日志路径。

    同一个进程重复调用是安全的（只安装一次）。
    """
    if os.environ.get("SUPERMARKET_RUN_LOG", "1") == "0":
        return None
    if getattr(start_run_log, "_installed", False):
        return getattr(start_run_log, "_path")

    repo_root = Path(__file__).resolve().parents[2]
    candidates: list[Path] = []
    env_dir = os.environ.get("SUPERMARKET_LOG_DIR")
    if env_dir:
        candidates.append(Path(env_dir))
    if log_dir is not None:
        candidates.append(Path(log_dir))
    candidates.append(repo_root / "logs")
    candidates.append(Path("/tmp"))
    directory = _writable_dir(candidates)

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = directory / f"{prefix}_{stamp}.log"
    handle = open(path, "a", encoding="utf-8", buffering=1)
    for fd in (1, 2):
        _tee_fd(fd, handle)
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except (AttributeError, OSError):
        pass

    start_run_log._installed = True
    start_run_log._path = path
    print(f"[run-log] saving run log to {path}", flush=True)
    return path
