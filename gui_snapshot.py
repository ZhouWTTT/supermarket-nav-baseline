#!/usr/bin/env python3
"""快照实验 GUI：只负责随机种子控制、启动 Server+快照客户端、展示记忆矩阵。

不含任何商品抓取/订单功能。矩阵数据来自客户端写出的
``logs/memory_matrix.json``（3×15 精确格 + 无码近似记录）。
"""

from __future__ import annotations

import datetime
import json
import os
import subprocess
import tkinter as tk
from tkinter import messagebox, ttk


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
SERVER_NAME = "supermarket_snapshot_server"
CLIENT_NAME = "supermarket_snapshot_client"
ROS_DOMAIN_ID = "99"
RMW = "rmw_cyclonedds_cpp"
TORCH_CACHE = "/root/.cache/torch_extensions/cu128"

OFFICIAL_PREFIX = (
    "crpi-1pzq998p9m7w0auy.cn-hangzhou.personal.cr.aliyuncs.com/"
    "challengecup/supermarket_sorting_final")
SERVER_IMAGE = f"{OFFICIAL_PREFIX}:server"
CLIENT_IMAGE = f"{OFFICIAL_PREFIX}:client"

KINDS = [
    "kele", "maidong", "heweidao", "shupian", "zhijin",
    "kouxiangtang", "sanmingzhi", "pingguo", "chengzi",
]
KIND_SHORT = {
    "kele": "可乐", "maidong": "脉动", "heweidao": "核桃刀",
    "shupian": "薯片", "zhijin": "纸巾", "kouxiangtang": "口香糖",
    "sanmingzhi": "三明治", "pingguo": "苹果", "chengzi": "橙子",
}
KIND_COLORS = {
    "kele": "#FF6B6B", "maidong": "#4ECDC4", "heweidao": "#FFE066",
    "shupian": "#FFA94D", "zhijin": "#B197FC", "kouxiangtang": "#69DB7C",
    "sanmingzhi": "#74C0FC", "pingguo": "#FF9F9F", "chengzi": "#FFD43B",
}
MATRIX_EMPTY_COLOR = "#F1F3F5"
MATRIX_CONSUMED_COLOR = "#ADB5BD"
MATRIX_ROWS = ("L1", "L2", "L3")
MATRIX_COLS = [
    f"{shelf}{col}"
    for shelf in ("A", "B", "C", "D", "E")
    for col in ("1", "2", "3")]


def run_cmd(args, **kwargs):
    proc = subprocess.run(
        args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        **kwargs)
    return proc.returncode, proc.stdout, proc.stderr


class SnapshotApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("超市分拣 - 整架快照实验")
        self.root.geometry("1000x720")
        self.root.minsize(860, 600)

        self._last_log = {SERVER_NAME: "", CLIENT_NAME: ""}
        self._poll_job = None
        self._matrix_job = None
        self.seed_var = tk.StringVar(value="")
        self.window_var = tk.BooleanVar(value=True)

        self._build_controls()
        self._build_tabs()
        self._poll_status()
        self._poll_memory_matrix()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---------------- UI ----------------
    def _build_controls(self):
        top = ttk.LabelFrame(self.root, text="实验控制", padding=8)
        top.pack(fill="x", padx=10, pady=(10, 4))

        ttk.Label(top, text="随机种子(留空=随机):").pack(side="left")
        self.seed_entry = ttk.Entry(top, textvariable=self.seed_var, width=8)
        self.seed_entry.pack(side="left", padx=(4, 16))

        ttk.Checkbutton(
            top, text="显示仿真窗口",
            variable=self.window_var).pack(side="left", padx=(0, 16))

        ttk.Button(
            top, text="开始快照", width=10,
            command=self._start).pack(side="left", padx=4)
        ttk.Button(
            top, text="停止", width=6,
            command=self._stop).pack(side="left", padx=4)
        self.status_var = tk.StringVar(value="Server: 未启动     Client: 未启动")
        ttk.Label(top, textvariable=self.status_var).pack(side="right")

    def _build_tabs(self):
        tabs = ttk.Notebook(self.root)
        tabs.pack(fill="both", expand=True, padx=10, pady=(4, 10))

        self.server_text = self._make_log_tab(tabs, "Server 日志")
        self.client_text = self._make_log_tab(tabs, "Client 日志")
        self._build_memory_matrix_tab(tabs)

    def _make_log_tab(self, tabs, title):
        frame = ttk.Frame(tabs)
        text = tk.Text(frame, wrap="none", state="disabled",
                       font=("DejaVu Sans Mono", 9))
        scroll = ttk.Scrollbar(frame, orient="vertical", command=text.yview)
        text.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        text.pack(side="left", fill="both", expand=True)
        tabs.add(frame, text=title)
        return text

    def _build_memory_matrix_tab(self, tabs):
        frame = ttk.Frame(tabs)
        header = ttk.Frame(frame)
        header.pack(fill="x", padx=8, pady=(8, 4))
        ttk.Label(
            header, text="记忆矩阵（行=层 L1/L2/L3，列=货架×列 A1..E3）",
            font=("DejaVu Sans", 10, "bold")).pack(side="left")
        self.matrix_status = ttk.Label(header, text="等待数据...")
        self.matrix_status.pack(side="right")

        grid_frame = ttk.Frame(frame)
        grid_frame.pack(padx=8, pady=2)
        ttk.Label(grid_frame, text="", width=5).grid(row=0, column=0)
        for col_index, col_label in enumerate(MATRIX_COLS):
            ttk.Label(
                grid_frame, text=col_label, width=6,
                anchor="center").grid(row=0, column=col_index + 1)
        self.matrix_labels: dict[str, list[tk.Label]] = {}
        for row_index, level in enumerate(MATRIX_ROWS):
            ttk.Label(
                grid_frame, text=level, width=5,
                anchor="center").grid(row=row_index + 1, column=0)
            row_labels = []
            for col_index in range(len(MATRIX_COLS)):
                label = tk.Label(
                    grid_frame, text="", width=6, height=1,
                    relief="ridge", bg=MATRIX_EMPTY_COLOR,
                    font=("DejaVu Sans", 9))
                label.grid(
                    row=row_index + 1, column=col_index + 1,
                    padx=1, pady=1, sticky="nsew")
                row_labels.append(label)
            self.matrix_labels[level] = row_labels

        self.matrix_approx_label = ttk.Label(
            frame, text="近似记录(无码): 无", foreground="#555555")
        self.matrix_approx_label.pack(fill="x", padx=8, pady=(6, 0))

        legend = ttk.Frame(frame)
        legend.pack(fill="x", padx=8, pady=(6, 8))
        for kind in KINDS:
            swatch = tk.Label(
                legend, text="  ", bg=KIND_COLORS[kind], width=2)
            swatch.pack(side="left", padx=(6, 1))
            ttk.Label(legend, text=KIND_SHORT[kind]).pack(side="left")
        tabs.add(frame, text="记忆矩阵")

    # ---------------- docker ----------------
    @staticmethod
    def _docker_running(name: str) -> bool:
        code, _, _ = run_cmd([
            "docker", "inspect", "-f", "{{.State.Running}}", name])
        return code == 0

    @staticmethod
    def _check_image(image: str) -> bool:
        code, _, _ = run_cmd(["docker", "image", "inspect", image])
        return code == 0

    def _seed_value(self):
        seed_text = self.seed_var.get().strip()
        if not seed_text:
            return None
        try:
            return int(seed_text)
        except ValueError:
            return None

    def _server_args(self):
        args = [
            "docker", "run", "--rm", "-d",
            "--name", SERVER_NAME,
            "--runtime=nvidia", "--network", "host", "--ipc", "host",
            "-e", f"ROS_DOMAIN_ID={ROS_DOMAIN_ID}",
            "-e", f"RMW_IMPLEMENTATION={RMW}",
        ]
        if self.window_var.get():
            subprocess.run(["xhost", "+local:docker"],
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
            args += [
                "-e", f"DISPLAY={os.environ.get('DISPLAY', '')}",
                "-e", "MUJOCO_GL=glfw",
                "-e", "SUPERMARKET_HEADLESS=0",
                "-e", "SUPERMARKET_DISPLAY_CAMERA=top_gs",
                "-v", "/tmp/.X11-unix:/tmp/.X11-unix:rw",
            ]
        else:
            args += [
                "-e", "MUJOCO_GL=egl",
                "-e", "SUPERMARKET_HEADLESS=1",
            ]
        args += [
            "-e", "SUPERMARKET_ENABLE_RENDER=1",
            "-e", "SUPERMARKET_ENABLE_LIDAR=1",
            "-e", "SUPERMARKET_USE_GS=1",
            "-e", "SUPERMARKET_RANDOMIZE=1",
            "-e", "SUPERMARKET_RANDOMIZE_OBSTACLES=1",
        ]
        seed = self._seed_value()
        if seed is not None:
            args += ["-e", f"SUPERMARKET_SEED={seed}"]
        args += [
            "-e", f"TORCH_EXTENSIONS_DIR={TORCH_CACHE}",
            "-v", "supermarket_sorting_cache:/root/.cache",
            SERVER_IMAGE,
            "bash", "-lc",
            "cd /workspace/supermarket_sorting_task && "
            "mkdir -p logs && source /opt/ros/humble/setup.bash && "
            "python3 examples/supermarket_sorting/"
            "supermarket_sorting_server.py 2>&1 | "
            "tee logs/snapshot_server_$(date +%H%M%S).log",
        ]
        return args

    def _client_args(self):
        args = [
            "docker", "run", "--rm", "-d",
            "--name", CLIENT_NAME,
            "--runtime=nvidia", "--network", "host", "--ipc", "host",
            "-e", f"ROS_DOMAIN_ID={ROS_DOMAIN_ID}",
            "-e", f"RMW_IMPLEMENTATION={RMW}",
            "-e", "YOLO_CONFIG_DIR=/tmp/Ultralytics",
            "-e", "PYTHONPATH=/workspace/supermarket_sorting_task",
            "-e", f"TORCH_EXTENSIONS_DIR={TORCH_CACHE}",
            "-v", f"{REPO_ROOT}:/workspace/supermarket_sorting_task",
            "-v", "supermarket_sorting_cache:/root/.cache",
            CLIENT_IMAGE,
            "bash", "-lc",
            "cd /workspace/supermarket_sorting_task && "
            "source /opt/ros/humble/setup.bash && "
            "python3 examples/supermarket_sorting/"
            "snapshot_client.py 2>&1 | "
            "tee logs/snapshot_client_$(date +%H%M%S).log",
        ]
        return args

    def _start(self):
        if not self._check_image(SERVER_IMAGE) or \
                not self._check_image(CLIENT_IMAGE):
            messagebox.showerror(
                "镜像缺失",
                f"请先拉取官方镜像:\n  {SERVER_IMAGE}\n  {CLIENT_IMAGE}")
            return
        if self._docker_running(SERVER_NAME):
            run_cmd(["docker", "rm", "-f", SERVER_NAME])
        if self._docker_running(CLIENT_NAME):
            run_cmd(["docker", "rm", "-f", CLIENT_NAME])
        self._append_log(
            self.server_text,
            f"启动 Server (seed={self._seed_value()})...\n")
        code, _, err = run_cmd(self._server_args())
        if code != 0:
            self._append_log(self.server_text, f"Server 启动失败: {err}\n")
            return
        self._append_log(self.client_text, "启动快照客户端...\n")
        code, _, err = run_cmd(self._client_args())
        if code != 0:
            self._append_log(self.client_text, f"Client 启动失败: {err}\n")

    def _stop(self):
        for name in (CLIENT_NAME, SERVER_NAME):
            if self._docker_running(name):
                run_cmd(["docker", "stop", "-t", "3", name])
                run_cmd(["docker", "rm", "-f", name])
        self._append_log(self.client_text, "已停止。\n")

    # ---------------- polling ----------------
    def _poll_status(self):
        server_on = self._docker_running(SERVER_NAME)
        client_on = self._docker_running(CLIENT_NAME)
        self.status_var.set(
            f"Server: {'运行中' if server_on else '未启动'}     "
            f"Client: {'运行中' if client_on else '未启动'}")
        self._poll_logs()
        self._poll_job = self.root.after(1500, self._poll_status)

    def _poll_logs(self):
        for name, widget in ((SERVER_NAME, self.server_text),
                             (CLIENT_NAME, self.client_text)):
            code, out, _ = run_cmd(["docker", "logs", "--tail", "300", name])
            if code != 0:
                continue
            if out != self._last_log[name]:
                self._last_log[name] = out
                self._append_log(widget, out if out else "(暂无输出)\n")

    def _poll_memory_matrix(self):
        path = os.path.join(REPO_ROOT, "logs", "memory_matrix.json")
        try:
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, ValueError):
            self.matrix_status.config(text="暂无数据")
            self._matrix_job = self.root.after(1000, self._poll_memory_matrix)
            return
        grid = data.get("grid", {})
        for level in MATRIX_ROWS:
            row = grid.get(level) or []
            for col_index in range(len(MATRIX_COLS)):
                label = self.matrix_labels[level][col_index]
                cell = (
                    row[col_index]
                    if col_index < len(row) else None)
                if not isinstance(cell, dict):
                    label.config(text="", bg=MATRIX_EMPTY_COLOR)
                    continue
                kind = str(cell.get("kind", "?"))
                consumed = bool(cell.get("consumed", False))
                text = KIND_SHORT.get(kind, kind)
                if consumed:
                    text = "✓" + text
                label.config(
                    text=text,
                    bg=(MATRIX_CONSUMED_COLOR if consumed
                        else KIND_COLORS.get(kind, "#DEE2E6")))
        # YOLO-only 近似记录直接铺到网格：无精确记录的格子显示 "≈种类"，
        # 便于观察深度回退记录了哪些货架/层。
        approx = data.get("approx", {})
        for key, records in approx.items():
            try:
                level, shelf = key.split("|")
            except ValueError:
                continue
            if level not in MATRIX_ROWS or shelf not in "ABCDE":
                continue
            col_base = MATRIX_COLS.index(f"{shelf}1")
            kinds = sorted(records)
            if not kinds:
                continue
            for i in range(3):
                label = self.matrix_labels[level][col_base + i]
                current = str(label.cget("text"))
                if current and "≈" not in current:
                    continue  # 已有精确记录，不覆盖
                kind = kinds[i % len(kinds)]
                label.config(
                    text="≈" + KIND_SHORT.get(kind, kind),
                    bg=KIND_COLORS.get(kind, "#DEE2E6"))
        try:
            updated = float(data.get("updated_at", 0.0))
            stamp = datetime.datetime.fromtimestamp(
                updated).strftime("%H:%M:%S")
            self.matrix_status.config(text=f"更新于 {stamp}")
        except (TypeError, ValueError, OSError):
            self.matrix_status.config(text="已更新")
        approx_parts = []
        for key, records in sorted(approx.items()):
            try:
                level, shelf = key.split("|")
            except ValueError:
                continue
            kinds = "/".join(
                KIND_SHORT.get(str(kind), str(kind))
                for kind in sorted(records))
            approx_parts.append(f"{shelf}-{level}: {kinds}")
        self.matrix_approx_label.config(
            text=(
                "近似记录(无码): " + ("；".join(approx_parts)
                                    if approx_parts else "无")))
        self._matrix_job = self.root.after(1000, self._poll_memory_matrix)

    def _append_log(self, widget, text):
        widget.configure(state="normal")
        widget.insert("end", text)
        widget.see("end")
        widget.configure(state="disabled")

    def _on_close(self):
        if self._poll_job is not None:
            self.root.after_cancel(self._poll_job)
        if self._matrix_job is not None:
            self.root.after_cancel(self._matrix_job)
        self.root.destroy()


def main():
    root = tk.Tk()
    SnapshotApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
