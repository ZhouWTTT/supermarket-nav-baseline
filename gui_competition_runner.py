#!/usr/bin/env python3
"""超市分拣正式流程的一键启动与记忆矩阵监视 GUI。

在宿主机启动官方 Server/Client 容器，Client 运行当前仓库中的
``competition_runner.py``。界面轮询容器日志以及写回仓库的
``logs/competition_runner/<run_prefix>/memory_matrix.json``、``summary.json``。

依赖：Python 3、tkinter、Docker、官方 server/client 镜像。
"""

from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
import random
import shlex
import shutil
import subprocess
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent
SERVER_NAME = "supermarket_runner_gui_server"
CLIENT_NAME = "supermarket_runner_gui_client"
ROS_DOMAIN_ID = "99"
RMW_IMPLEMENTATION = "rmw_cyclonedds_cpp"
TORCH_CACHE = "/root/.cache/torch_extensions/cu128"

OFFICIAL_PREFIX = (
    "crpi-1pzq998p9m7w0auy.cn-hangzhou.personal.cr.aliyuncs.com/"
    "challengecup/supermarket_sorting_final"
)
SERVER_IMAGE = os.environ.get(
    "SUPERMARKET_GUI_SERVER_IMAGE", f"{OFFICIAL_PREFIX}:server")
CLIENT_IMAGE = os.environ.get(
    "SUPERMARKET_GUI_CLIENT_IMAGE", f"{OFFICIAL_PREFIX}:client")

CONTAINER_ROOT = "/workspace/baseline"
RUNTIME_DIR_CONTAINER = f"{CONTAINER_ROOT}/logs/competition_runner"
RUNTIME_DIR_HOST = REPO_ROOT / "logs" / "competition_runner"
WEIGHTS_CONTAINER = (
    f"{CONTAINER_ROOT}/examples/supermarket_sorting/"
    "perception/checkpoints/best.pt"
)

KINDS = (
    "kele", "maidong", "heweidao", "shupian", "zhijin",
    "kouxiangtang", "sanmingzhi", "pingguo", "chengzi",
)
KIND_NAMES = {
    "kele": "可乐",
    "maidong": "脉动",
    "heweidao": "核桃刀",
    "shupian": "薯片",
    "zhijin": "纸巾",
    "kouxiangtang": "口香糖",
    "sanmingzhi": "三明治",
    "pingguo": "苹果",
    "chengzi": "橙子",
}
KIND_COLORS = {
    "kele": "#ff8787",
    "maidong": "#63e6be",
    "heweidao": "#ffe066",
    "shupian": "#ffa94d",
    "zhijin": "#b197fc",
    "kouxiangtang": "#8ce99a",
    "sanmingzhi": "#74c0fc",
    "pingguo": "#ff6b6b",
    "chengzi": "#ffd43b",
}
EMPTY_COLOR = "#f1f3f5"
CONSUMED_COLOR = "#adb5bd"
UPDATED_COLOR = "#f08c00"
SELECTED_COLOR = "#1971c2"
LEVELS = ("L1", "L2", "L3")
COLUMNS = tuple(
    f"{shelf}{column}"
    for shelf in "ABCDE"
    for column in "123"
)


def run_command(args: list[str], timeout: float = 8.0) -> tuple[int, str, str]:
    """Run a short host command without leaving the GUI blocked forever."""
    try:
        result = subprocess.run(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        return 127, "", str(exc)
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        return 124, stdout, stderr or f"命令超时：{args[0]}"
    return result.returncode, result.stdout, result.stderr


def generate_tasks(count: int, seed: int | None) -> list[str]:
    """Generate at most five distinct fixed product bodies."""
    count = max(1, min(int(count), 5))
    indexes = sorted(random.Random(seed).sample(range(1, 46), count))
    return [f"product_{index:03d}" for index in indexes]


def slot_key(level: str, column: str) -> str:
    return f"{level}|{column[0]}|{column[1:]}"


def cell_signature(document: dict[str, Any], key: str) -> tuple[Any, ...] | None:
    """Fields that indicate a visible/evidential update to one matrix cell."""
    cell = (document.get("cells") or {}).get(key)
    if not isinstance(cell, dict):
        return None
    kind = str(cell.get("kind", ""))
    candidate = ((document.get("candidates") or {}).get(key) or {}).get(
        kind, {})
    return (
        kind,
        bool(cell.get("consumed")),
        cell.get("confidence"),
        candidate.get("observations"),
        candidate.get("sample_count", cell.get("sample_count")),
        candidate.get("last_seen", cell.get("last_seen")),
    )


def format_cell_details(document: dict[str, Any], key: str) -> str:
    """Render the primary evidence and retained alternatives for a cell."""
    cell = (document.get("cells") or {}).get(key)
    display_key = key.replace("|", "-")
    if not isinstance(cell, dict):
        return f"货位 {display_key}：尚无已确认记忆。"

    kind = str(cell.get("kind", "?"))
    candidates = (document.get("candidates") or {}).get(key) or {}
    primary = candidates.get(kind, {}) if isinstance(candidates, dict) else {}

    def number(value: Any, digits: int = 3) -> str:
        try:
            return f"{float(value):.{digits}f}"
        except (TypeError, ValueError):
            return "-"

    last_seen = primary.get("last_seen", cell.get("last_seen"))
    try:
        seen_text = dt.datetime.fromtimestamp(float(last_seen)).strftime(
            "%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError, OSError):
        seen_text = "-"

    lines = [
        f"货位 {display_key}    主类别：{KIND_NAMES.get(kind, kind)} ({kind})"
        f"    状态：{'已取走' if cell.get('consumed') else '有效'}",
        "置信度："
        f"{number(cell.get('confidence'))}    "
        f"观测批次：{primary.get('observations', '-')}    "
        "样本数："
        f"{primary.get('sample_count', cell.get('sample_count', '-'))}    "
        f"最近距离：{number(cell.get('closest_distance'))} m",
        "货物世界坐标："
        f"({number(cell.get('world_x'))}, {number(cell.get('world_y'))}, "
        f"{number(cell.get('world_z'))})    "
        "机器人观察位："
        f"({number(cell.get('observer_x'))}, {number(cell.get('observer_y'))})",
        f"最近更新：{seen_text}",
    ]
    alternatives = []
    if isinstance(candidates, dict):
        for candidate_kind, candidate in sorted(candidates.items()):
            if candidate_kind == kind or not isinstance(candidate, dict):
                continue
            alternatives.append(
                f"{KIND_NAMES.get(candidate_kind, candidate_kind)}: "
                f"conf={number(candidate.get('confidence'))}, "
                f"批次={candidate.get('observations', '-')}, "
                f"样本={candidate.get('sample_count', '-')}"
            )
    if alternatives:
        lines.append("同格历史候选：" + "；".join(alternatives))
    return "\n".join(lines)


class LauncherApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("超市分拣：正式 Runner 与记忆矩阵")
        root.geometry("1120x790")
        root.minsize(900, 650)

        self._last_logs = {SERVER_NAME: "", CLIENT_NAME: ""}
        self._matrix_document: dict[str, Any] = {}
        self._matrix_signatures: dict[str, tuple[Any, ...] | None] = {}
        self._matrix_path: Path | None = None
        self._selected_key: str | None = None
        self._changed_until: dict[str, float] = {}
        self._poll_jobs: dict[str, str] = {}

        self._build_controls()
        self._build_tabs()
        self._regenerate_tasks()
        self._schedule(self._poll_containers, 250)
        self._schedule(self._poll_matrix, 300)
        self._schedule(self._poll_summary, 500)
        root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _schedule(self, callback, delay_ms: int) -> None:
        job = self.root.after(delay_ms, callback)
        self._poll_jobs[callback.__name__] = job

    def _build_controls(self) -> None:
        panel = ttk.LabelFrame(self.root, text="启动配置", padding=8)
        panel.pack(fill="x", padx=10, pady=(10, 4))

        self.count_var = tk.IntVar(value=5)
        self.seed_var = tk.StringVar()
        self.tasks_var = tk.StringVar()
        self.cycles_var = tk.IntVar(value=2)
        self.attempts_var = tk.IntVar(value=2)
        self.confirmations_var = tk.IntVar(value=3)
        self.memory_conf_var = tk.DoubleVar(value=0.95)
        self.policy_var = tk.StringVar(value="nearest")
        self.inference_hz_var = tk.DoubleVar(value=12.0)
        self.device_var = tk.StringVar(value="cpu")
        self.order_timeout_var = tk.IntVar(value=300)
        self.match_timeout_var = tk.IntVar(value=3600)
        self.target_time_var = tk.IntVar(value=400)

        self.obstacles_var = tk.BooleanVar(value=True)
        self.sim_window_var = tk.BooleanVar(value=True)
        self.yolo_window_var = tk.BooleanVar(value=False)
        self.record_everywhere_var = tk.BooleanVar(value=True)
        self.perception_always_var = tk.BooleanVar(value=True)
        self.dynamic_direct_var = tk.BooleanVar(value=True)
        self.close_recheck_var = tk.BooleanVar(value=True)

        first = ttk.Frame(panel)
        first.pack(fill="x")
        ttk.Label(first, text="订单数").pack(side="left")
        ttk.Spinbox(
            first, from_=1, to=5, width=4, textvariable=self.count_var,
            command=self._regenerate_tasks,
        ).pack(side="left", padx=(4, 12))
        ttk.Label(first, text="随机种子（留空=随机）").pack(side="left")
        seed_entry = ttk.Entry(first, width=9, textvariable=self.seed_var)
        seed_entry.pack(side="left", padx=(4, 12))
        seed_entry.bind("<Return>", self._regenerate_tasks)
        ttk.Label(first, text="任务列表（逗号分隔）").pack(side="left")
        ttk.Entry(first, textvariable=self.tasks_var).pack(
            side="left", fill="x", expand=True, padx=(4, 8))
        ttk.Button(first, text="随机生成", command=self._regenerate_tasks).pack(
            side="left")

        second = ttk.Frame(panel)
        second.pack(fill="x", pady=(6, 0))
        for label, variable, start, end, width in (
            ("扫描轮数", self.cycles_var, 1, 10, 4),
            ("最大尝试", self.attempts_var, 1, 5, 4),
            ("确认帧", self.confirmations_var, 1, 10, 4),
        ):
            ttk.Label(second, text=label).pack(side="left")
            ttk.Spinbox(
                second, from_=start, to=end, width=width,
                textvariable=variable,
            ).pack(side="left", padx=(4, 12))
        ttk.Label(second, text="记忆置信度").pack(side="left")
        ttk.Spinbox(
            second, from_=0.5, to=1.0, increment=0.05, width=5,
            textvariable=self.memory_conf_var,
        ).pack(side="left", padx=(4, 12))
        ttk.Label(second, text="抓取策略").pack(side="left")
        ttk.Combobox(
            second, width=9, state="readonly", textvariable=self.policy_var,
            values=("nearest", "sequence"),
        ).pack(side="left", padx=(4, 12))
        ttk.Label(second, text="推理 Hz").pack(side="left")
        ttk.Spinbox(
            second, from_=1, to=30, width=5,
            textvariable=self.inference_hz_var,
        ).pack(side="left", padx=(4, 12))
        ttk.Label(second, text="设备").pack(side="left")
        ttk.Combobox(
            second, width=6, state="readonly", textvariable=self.device_var,
            values=("auto", "cpu", "cuda"),
        ).pack(side="left", padx=(4, 0))

        third = ttk.Frame(panel)
        third.pack(fill="x", pady=(6, 0))
        for label, variable, start, end, step in (
            ("单订单超时(s)", self.order_timeout_var, 0, 3600, 30),
            ("比赛超时(s)", self.match_timeout_var, 60, 7200, 60),
            ("目标用时(s)", self.target_time_var, 60, 3600, 30),
        ):
            ttk.Label(third, text=label).pack(side="left")
            ttk.Spinbox(
                third, from_=start, to=end, increment=step, width=7,
                textvariable=variable,
            ).pack(side="left", padx=(4, 12))
        ttk.Checkbutton(
            third, text="随机障碍物", variable=self.obstacles_var,
        ).pack(side="left", padx=(0, 10))
        ttk.Checkbutton(
            third, text="仿真窗口", variable=self.sim_window_var,
        ).pack(side="left", padx=(0, 10))
        ttk.Checkbutton(
            third, text="YOLO窗口", variable=self.yolo_window_var,
        ).pack(side="left")

        fourth = ttk.Frame(panel)
        fourth.pack(fill="x", pady=(6, 0))
        ttk.Checkbutton(
            fourth, text="全程录入矩阵", variable=self.record_everywhere_var,
        ).pack(side="left", padx=(0, 12))
        ttk.Checkbutton(
            fourth, text="感知常开", variable=self.perception_always_var,
        ).pack(side="left", padx=(0, 12))
        ttk.Checkbutton(
            fourth, text="矩阵直达/动态改道", variable=self.dynamic_direct_var,
        ).pack(side="left", padx=(0, 12))
        ttk.Checkbutton(
            fourth, text="ArUco优先近距复核（YOLO兜底）",
            variable=self.close_recheck_var,
        ).pack(side="left")

        buttons = ttk.Frame(panel)
        buttons.pack(fill="x", pady=(8, 0))
        ttk.Button(
            buttons, text="启动 Server + Runner", command=self.start_all,
        ).pack(side="left")
        ttk.Button(
            buttons, text="仅启动 Server", command=self.start_server,
        ).pack(side="left", padx=(8, 0))
        ttk.Button(
            buttons, text="仅启动 Runner", command=self.start_runner,
        ).pack(side="left", padx=(8, 0))
        ttk.Button(
            buttons, text="停止全部", command=self.stop_all,
        ).pack(side="left", padx=(8, 0))
        ttk.Button(
            buttons, text="停止 Server",
            command=lambda: self.stop_one(SERVER_NAME),
        ).pack(side="left", padx=(8, 0))
        ttk.Button(
            buttons, text="停止 Runner",
            command=lambda: self.stop_one(CLIENT_NAME),
        ).pack(side="left", padx=(8, 0))
        ttk.Button(
            buttons, text="打开运行目录", command=self.open_runtime_dir,
        ).pack(side="left", padx=(8, 0))
        ttk.Button(
            buttons, text="清空界面日志", command=self.clear_log_views,
        ).pack(side="left", padx=(8, 0))

        self.status_var = tk.StringVar(value="Server：检查中    Runner：检查中")
        ttk.Label(
            panel, textvariable=self.status_var, foreground="#087f23",
        ).pack(anchor="w", pady=(6, 0))
        ttk.Label(
            panel,
            text=(
                "产物目录：logs/competition_runner/<run_prefix>/；关闭界面不会"
                "自动停止容器。镜像可用 SUPERMARKET_GUI_SERVER_IMAGE / "
                "SUPERMARKET_GUI_CLIENT_IMAGE 覆盖。"
            ),
            foreground="#666666",
        ).pack(anchor="w")

    def _build_tabs(self) -> None:
        tabs = ttk.Notebook(self.root)
        tabs.pack(fill="both", expand=True, padx=10, pady=(4, 10))
        self.server_log = self._make_text_tab(tabs, "Server 日志")
        self.runner_log = self._make_text_tab(tabs, "Runner 日志")
        self._build_matrix_tab(tabs)
        self.summary_text = self._make_text_tab(tabs, "比赛摘要", wrap="word")

    @staticmethod
    def _make_text_tab(
        tabs: ttk.Notebook, title: str, wrap: str = "none",
    ) -> tk.Text:
        frame = ttk.Frame(tabs)
        text = tk.Text(
            frame, wrap=wrap, state="disabled",
            font=("DejaVu Sans Mono", 9),
        )
        yscroll = ttk.Scrollbar(frame, orient="vertical", command=text.yview)
        xscroll = ttk.Scrollbar(frame, orient="horizontal", command=text.xview)
        text.configure(
            yscrollcommand=yscroll.set,
            xscrollcommand=xscroll.set if wrap == "none" else None,
        )
        yscroll.pack(side="right", fill="y")
        if wrap == "none":
            xscroll.pack(side="bottom", fill="x")
        text.pack(fill="both", expand=True)
        tabs.add(frame, text=title)
        return text

    def _build_matrix_tab(self, tabs: ttk.Notebook) -> None:
        frame = ttk.Frame(tabs)
        header = ttk.Frame(frame)
        header.pack(fill="x", padx=8, pady=(8, 4))
        ttk.Label(
            header,
            text="记忆矩阵（行=层，列=货架×列；橙框=最近 3 秒有更新）",
            font=("DejaVu Sans", 10, "bold"),
        ).pack(side="left")
        self.matrix_status_var = tk.StringVar(value="等待数据……")
        ttk.Label(header, textvariable=self.matrix_status_var).pack(side="right")

        grid = ttk.Frame(frame)
        grid.pack(fill="x", padx=8, pady=2)
        ttk.Label(grid, text="", width=4).grid(row=0, column=0)
        for index, column in enumerate(COLUMNS, start=1):
            ttk.Label(grid, text=column, width=6, anchor="center").grid(
                row=0, column=index)

        self.matrix_labels: dict[str, tk.Label] = {}
        for row, level in enumerate(LEVELS, start=1):
            ttk.Label(grid, text=level, width=4, anchor="center").grid(
                row=row, column=0)
            for column_index, column in enumerate(COLUMNS, start=1):
                key = slot_key(level, column)
                label = tk.Label(
                    grid,
                    text="",
                    width=6,
                    height=2,
                    relief="ridge",
                    background=EMPTY_COLOR,
                    cursor="hand2",
                    highlightthickness=2,
                    highlightbackground=EMPTY_COLOR,
                    font=("DejaVu Sans", 9),
                )
                label.grid(
                    row=row, column=column_index,
                    padx=1, pady=1, sticky="nsew",
                )
                label.bind(
                    "<Button-1>",
                    lambda _event, selected=key: self._select_cell(selected),
                )
                self.matrix_labels[key] = label

        self.matrix_stats_var = tk.StringVar(
            value="已记录 0/45    有效 0    已取走 0    保留候选 0")
        ttk.Label(
            frame, textvariable=self.matrix_stats_var, foreground="#555555",
        ).pack(fill="x", padx=8, pady=(6, 0))

        detail = ttk.LabelFrame(frame, text="货位证据（点击格子查看）", padding=6)
        detail.pack(fill="x", padx=8, pady=(6, 0))
        self.matrix_detail_var = tk.StringVar(
            value="尚未选择货位。矩阵数据每秒自动刷新。")
        ttk.Label(
            detail,
            textvariable=self.matrix_detail_var,
            justify="left",
            wraplength=1040,
        ).pack(fill="x")

        legend = ttk.Frame(frame)
        legend.pack(fill="x", padx=8, pady=(7, 8))
        for kind in KINDS:
            tk.Label(
                legend, text="  ", width=2, background=KIND_COLORS[kind],
            ).pack(side="left", padx=(5, 1))
            ttk.Label(legend, text=KIND_NAMES[kind]).pack(side="left")
        tk.Label(
            legend, text="  ", width=2, background=CONSUMED_COLOR,
        ).pack(side="left", padx=(12, 1))
        ttk.Label(legend, text="已取走").pack(side="left")
        tabs.add(frame, text="记忆矩阵")

    def _seed(self) -> int | None:
        text = self.seed_var.get().strip()
        if not text:
            return None
        return int(text)

    def _regenerate_tasks(self, _event=None) -> None:
        try:
            count = self.count_var.get()
            seed = self._seed()
        except (tk.TclError, ValueError):
            messagebox.showerror("输入错误", "订单数和随机种子必须是整数。")
            return
        self.tasks_var.set(",".join(generate_tasks(count, seed)))

    def _tasks(self) -> list[str]:
        return [item.strip() for item in self.tasks_var.get().split(",") if item.strip()]

    def _validate(self) -> bool:
        if shutil.which("docker") is None:
            messagebox.showerror("Docker 不可用", "宿主机未找到 docker 命令。")
            return False
        try:
            self._seed()
            values = (
                self.cycles_var.get(), self.attempts_var.get(),
                self.confirmations_var.get(), self.memory_conf_var.get(),
                self.inference_hz_var.get(), self.order_timeout_var.get(),
                self.match_timeout_var.get(), self.target_time_var.get(),
            )
        except (tk.TclError, ValueError):
            messagebox.showerror("配置无效", "请检查所有数字配置项。")
            return False
        if not self._tasks():
            messagebox.showerror("任务为空", "请随机生成或手动填写任务列表。")
            return False
        if len(self._tasks()) > 5:
            messagebox.showwarning(
                "任务超过五单",
                "当前只有五个交付槽位，建议任务数量不超过五个。",
            )
        if not (values[0] >= 1 and values[1] >= 1 and values[2] >= 1):
            messagebox.showerror("配置无效", "轮数、尝试次数和确认帧必须大于零。")
            return False
        if not 0.0 <= float(values[3]) <= 1.0:
            messagebox.showerror("配置无效", "记忆置信度必须位于 0 到 1。")
            return False
        return True

    @staticmethod
    def _allow_x11() -> None:
        if shutil.which("xhost"):
            run_command(["xhost", "+local:docker"], timeout=3.0)

    def _server_args(self) -> list[str]:
        args = [
            "docker", "run", "--rm", "-d", "--name", SERVER_NAME,
            "--gpus", "all", "--network", "host", "--ipc", "host",
            "-e", f"ROS_DOMAIN_ID={ROS_DOMAIN_ID}",
            "-e", f"RMW_IMPLEMENTATION={RMW_IMPLEMENTATION}",
        ]
        if self.sim_window_var.get():
            self._allow_x11()
            args.extend([
                "-e", f"DISPLAY={os.environ.get('DISPLAY', '')}",
                "-e", "MUJOCO_GL=glfw",
                "-e", "SUPERMARKET_HEADLESS=0",
                "-e", "SUPERMARKET_DISPLAY_CAMERA=top_gs",
                "-v", "/tmp/.X11-unix:/tmp/.X11-unix:rw",
            ])
        else:
            args.extend(["-e", "MUJOCO_GL=egl", "-e", "SUPERMARKET_HEADLESS=1"])
        args.extend([
            "-e", "SUPERMARKET_ENABLE_RENDER=1",
            "-e", "SUPERMARKET_ENABLE_LIDAR=1",
            "-e", "SUPERMARKET_USE_GS=1",
            "-e", "SUPERMARKET_RANDOMIZE=1",
            "-e", f"SUPERMARKET_RANDOMIZE_OBSTACLES={int(self.obstacles_var.get())}",
            "-e", f"SUPERMARKET_TASKS={','.join(self._tasks())}",
            "-e", f"TORCH_EXTENSIONS_DIR={TORCH_CACHE}",
        ])
        seed = self._seed()
        if seed is not None:
            args.extend(["-e", f"SUPERMARKET_SEED={seed}"])
        obstacle_seed = os.environ.get("SUPERMARKET_OBSTACLE_SEED", "").strip()
        if obstacle_seed:
            args.extend(["-e", f"SUPERMARKET_OBSTACLE_SEED={obstacle_seed}"])
        args.extend([
            "-v", "supermarket_sorting_cache:/root/.cache",
            SERVER_IMAGE,
            "bash", "-lc",
            "cd /workspace/supermarket_sorting_task && "
            "source /opt/ros/humble/setup.bash && "
            "python3 examples/supermarket_sorting/supermarket_sorting_server.py",
        ])
        return args

    def _runner_args(self) -> list[str]:
        runner = [
            "python3", "examples/supermarket_sorting/competition_runner.py",
            "--weights", WEIGHTS_CONTAINER,
            "--max-scan-cycles", str(self.cycles_var.get()),
            "--max-attempts", str(self.attempts_var.get()),
            "--memory-confirmations", str(self.confirmations_var.get()),
            "--memory-confidence-threshold", f"{self.memory_conf_var.get():.3f}",
            "--grab-policy", self.policy_var.get(),
            "--inference-hz", f"{self.inference_hz_var.get():g}",
            "--device", self.device_var.get(),
            "--order-timeout", str(self.order_timeout_var.get()),
            "--match-timeout", str(self.match_timeout_var.get()),
            "--target-time", str(self.target_time_var.get()),
            "--runtime-dir", RUNTIME_DIR_CONTAINER,
        ]
        for enabled, flag in (
            (self.record_everywhere_var.get(), "--record-everywhere"),
            (self.perception_always_var.get(), "--perception-always-on"),
            (self.dynamic_direct_var.get(), "--dynamic-direct"),
            (self.yolo_window_var.get(), "--show"),
        ):
            if enabled:
                runner.append(flag)
        runner.append(
            "--close-recheck"
            if self.close_recheck_var.get()
            else "--no-close-recheck")

        args = [
            "docker", "run", "--rm", "-d", "--name", CLIENT_NAME,
            "--gpus", "all", "--network", "host", "--ipc", "host",
            "-e", f"ROS_DOMAIN_ID={ROS_DOMAIN_ID}",
            "-e", f"RMW_IMPLEMENTATION={RMW_IMPLEMENTATION}",
            "-e", "YOLO_CONFIG_DIR=/tmp/Ultralytics",
            "-e", "SUPERMARKET_PATH_MEMORY=1",
            "-e", "SUPERMARKET_PATH_MEMORY_FILE=/root/.cache/supermarket_path_memory.json",
            "-e", f"PYTHONPATH={CONTAINER_ROOT}",
            "-e", f"TORCH_EXTENSIONS_DIR={TORCH_CACHE}",
        ]
        if self.yolo_window_var.get():
            self._allow_x11()
            args.extend([
                "-e", f"DISPLAY={os.environ.get('DISPLAY', '')}",
                "-v", "/tmp/.X11-unix:/tmp/.X11-unix:rw",
            ])
        args.extend([
            "-v", f"{REPO_ROOT}:{CONTAINER_ROOT}",
            "-v", "supermarket_sorting_cache:/root/.cache",
            CLIENT_IMAGE,
            "bash", "-lc",
            f"cd {shlex.quote(CONTAINER_ROOT)} && "
            "source /opt/ros/humble/setup.bash && "
            f"mkdir -p {shlex.quote(RUNTIME_DIR_CONTAINER)} && "
            f"exec {shlex.join(runner)}",
        ])
        return args

    @staticmethod
    def _container_running(name: str) -> bool:
        code, stdout, _ = run_command(
            ["docker", "inspect", "-f", "{{.State.Running}}", name],
            timeout=3.0,
        )
        return code == 0 and stdout.strip().lower() == "true"

    @staticmethod
    def _image_available(image: str) -> bool:
        return run_command(["docker", "image", "inspect", image])[0] == 0

    def _remove_container(self, name: str) -> None:
        run_command(["docker", "rm", "-f", name], timeout=12.0)
        self._last_logs[name] = ""

    def _start_container(self, args: list[str], title: str) -> bool:
        code, _stdout, stderr = run_command(args, timeout=30.0)
        if code != 0:
            messagebox.showerror(f"{title} 启动失败", stderr.strip() or "未知错误")
            return False
        return True

    def _check_image(self, image: str) -> bool:
        if self._image_available(image):
            return True
        messagebox.showerror(
            "镜像缺失",
            f"本机没有镜像：\n{image}\n\n请先使用 docker pull 拉取。",
        )
        return False

    def start_server(self) -> bool:
        if not self._validate() or not self._check_image(SERVER_IMAGE):
            return False
        self._remove_container(SERVER_NAME)
        self._replace_text(self.server_log, "正在启动 Server……\n")
        started = self._start_container(self._server_args(), "Server")
        if started:
            self._replace_text(self.server_log, "Server 容器已创建，等待仿真初始化……\n")
        return started

    def start_runner(self) -> bool:
        if not self._validate() or not self._check_image(CLIENT_IMAGE):
            return False
        self._remove_container(CLIENT_NAME)
        RUNTIME_DIR_HOST.mkdir(parents=True, exist_ok=True)
        self._replace_text(self.runner_log, "正在启动 Runner，等待任务话题……\n")
        started = self._start_container(self._runner_args(), "Runner")
        if started:
            self._replace_text(self.runner_log, "Runner 容器已创建。\n")
        return started

    def start_all(self) -> None:
        if not self._validate():
            return
        if not self._check_image(SERVER_IMAGE) or not self._check_image(CLIENT_IMAGE):
            return
        self._remove_container(CLIENT_NAME)
        self._remove_container(SERVER_NAME)
        self._replace_text(self.server_log, "正在启动 Server……\n")
        if not self._start_container(self._server_args(), "Server"):
            return
        self._replace_text(self.server_log, "Server 容器已创建，等待仿真初始化……\n")
        RUNTIME_DIR_HOST.mkdir(parents=True, exist_ok=True)
        self._replace_text(self.runner_log, "正在启动 Runner，等待任务话题……\n")
        if not self._start_container(self._runner_args(), "Runner"):
            self._remove_container(SERVER_NAME)
            return
        self._replace_text(self.runner_log, "Runner 容器已创建。\n")

    def stop_all(self) -> None:
        self._remove_container(CLIENT_NAME)
        self._remove_container(SERVER_NAME)
        self._replace_text(self.server_log, "Server 已停止。\n")
        self._replace_text(self.runner_log, "Runner 已停止。\n")

    def stop_one(self, name: str) -> None:
        self._remove_container(name)
        if name == SERVER_NAME:
            self._replace_text(self.server_log, "Server 已停止。\n")
        else:
            self._replace_text(self.runner_log, "Runner 已停止。\n")

    def open_runtime_dir(self) -> None:
        RUNTIME_DIR_HOST.mkdir(parents=True, exist_ok=True)
        opener = shutil.which("xdg-open")
        if opener is None:
            messagebox.showinfo("运行目录", str(RUNTIME_DIR_HOST))
            return
        try:
            subprocess.Popen(
                [opener, str(RUNTIME_DIR_HOST)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError as exc:
            messagebox.showerror("无法打开目录", str(exc))

    def clear_log_views(self) -> None:
        for name in (SERVER_NAME, CLIENT_NAME):
            code, output, error = run_command(
                ["docker", "logs", "--tail", "500", name], timeout=3.0)
            self._last_logs[name] = output + error if code == 0 else ""
        self._replace_text(self.server_log, "")
        self._replace_text(self.runner_log, "")

    @staticmethod
    def _replace_text(widget: tk.Text, value: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("end", value)
        widget.see("end")
        widget.configure(state="disabled")

    def _poll_containers(self) -> None:
        if shutil.which("docker") is None:
            self.status_var.set("Docker：未安装或不在 PATH 中")
        else:
            server = self._container_running(SERVER_NAME)
            runner = self._container_running(CLIENT_NAME)
            self.status_var.set(
                f"Server：{'运行中' if server else '未启动'}    "
                f"Runner：{'运行中' if runner else '未启动'}"
            )
            for name, widget in (
                (SERVER_NAME, self.server_log),
                (CLIENT_NAME, self.runner_log),
            ):
                code, output, error = run_command(
                    ["docker", "logs", "--tail", "500", name], timeout=3.0)
                combined = output + error
                if code == 0 and combined != self._last_logs[name]:
                    self._last_logs[name] = combined
                    self._replace_text(widget, combined or "（暂无输出）\n")
        self._schedule(self._poll_containers, 1500)

    @staticmethod
    def _newest_artifact(filename: str) -> Path | None:
        if not RUNTIME_DIR_HOST.is_dir():
            return None
        matches = list(RUNTIME_DIR_HOST.glob(f"*/{filename}"))
        if not matches:
            return None
        return max(matches, key=lambda path: path.stat().st_mtime)

    def _poll_matrix(self) -> None:
        path = self._newest_artifact("memory_matrix.json")
        if path is None:
            self.matrix_status_var.set("暂无矩阵文件")
            self._schedule(self._poll_matrix, 1000)
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.matrix_status_var.set("矩阵正在写入或读取失败")
            self._schedule(self._poll_matrix, 1000)
            return
        if not isinstance(data, dict):
            self.matrix_status_var.set("矩阵格式错误")
            self._schedule(self._poll_matrix, 1000)
            return

        now = dt.datetime.now().timestamp()
        run_changed = path != self._matrix_path
        previous = {} if run_changed else self._matrix_signatures
        signatures = {
            key: cell_signature(data, key) for key in self.matrix_labels
        }
        for key, signature in signatures.items():
            if signature is not None and signature != previous.get(key):
                self._changed_until[key] = now + 3.0
        if run_changed:
            self._changed_until.clear()
            for key, signature in signatures.items():
                if signature is not None:
                    self._changed_until[key] = now + 3.0
        self._matrix_document = data
        self._matrix_signatures = signatures
        self._matrix_path = path

        cells = data.get("cells") or {}
        candidates = data.get("candidates") or {}
        active_count = consumed_count = candidate_count = 0
        for key, label in self.matrix_labels.items():
            cell = cells.get(key)
            selected = key == self._selected_key
            changed = self._changed_until.get(key, 0.0) > now
            border = SELECTED_COLOR if selected else (UPDATED_COLOR if changed else EMPTY_COLOR)
            if not isinstance(cell, dict):
                label.configure(
                    text="", background=EMPTY_COLOR,
                    highlightbackground=border,
                )
                continue
            kind = str(cell.get("kind", "?"))
            consumed = bool(cell.get("consumed"))
            active_count += int(not consumed)
            consumed_count += int(consumed)
            text = KIND_NAMES.get(kind, kind)
            if consumed:
                text = "✓" + text
            label.configure(
                text=text,
                background=CONSUMED_COLOR if consumed else KIND_COLORS.get(kind, "#dee2e6"),
                highlightbackground=border,
            )
        for by_kind in candidates.values():
            if isinstance(by_kind, dict):
                candidate_count += len(by_kind)
        self.matrix_stats_var.set(
            f"已记录 {len(cells)}/45    有效 {active_count}    "
            f"已取走 {consumed_count}    保留候选 {candidate_count}"
        )
        try:
            stamp = dt.datetime.fromtimestamp(float(data.get("updated_at"))).strftime(
                "%H:%M:%S")
        except (TypeError, ValueError, OSError):
            stamp = "未知"
        self.matrix_status_var.set(f"{path.parent.name}    更新于 {stamp}")
        if self._selected_key:
            self.matrix_detail_var.set(format_cell_details(data, self._selected_key))
        self._schedule(self._poll_matrix, 1000)

    def _select_cell(self, key: str) -> None:
        self._selected_key = key
        self.matrix_detail_var.set(format_cell_details(self._matrix_document, key))
        for cell_key, label in self.matrix_labels.items():
            label.configure(
                highlightbackground=(SELECTED_COLOR if cell_key == key else EMPTY_COLOR))

    def _poll_summary(self) -> None:
        path: Path | None = None
        if self._matrix_path:
            same_run = self._matrix_path.parent / "summary.json"
            if same_run.is_file():
                path = same_run
        if path is None:
            path = self._newest_artifact("summary.json")
        if path is None:
            self._replace_text(self.summary_text, "（暂无比赛摘要）\n")
            self._schedule(self._poll_summary, 1500)
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self._schedule(self._poll_summary, 1500)
            return
        if not isinstance(data, dict):
            self._schedule(self._poll_summary, 1500)
            return

        lines = [
            f"运行：{data.get('run_prefix', '?')}",
            f"状态：{data.get('reason', '?')}",
            (
                f"进度：交付 {data.get('delivered', 0)} / "
                f"失败 {data.get('failed', 0)} / "
                f"待办 {data.get('pending', 0)} / "
                f"总计 {data.get('count', 0)}"
            ),
        ]
        elapsed = data.get("elapsed_s")
        if isinstance(elapsed, (int, float)):
            lines.append(
                f"耗时：{elapsed:.1f}s / 目标 {data.get('target_time_s', '?')}s / "
                f"目标剩余 {data.get('remaining_to_target_s', '?')}s"
            )
        orders = data.get("orders") or []
        if orders:
            lines.append("\n订单：")
            for order in orders:
                if not isinstance(order, dict):
                    continue
                lines.append(
                    f"  #{int(order.get('source_index', 0)) + 1} "
                    f"{order.get('id', '?')}  {order.get('kind', '?')}  "
                    f"{order.get('status', '?')}  尝试={order.get('attempts', 0)}"
                )
                if order.get("errors"):
                    lines.append(f"      错误：{order['errors']}")
        timings = data.get("order_timings") or []
        if timings:
            lines.append("\n订单耗时：")
            records = timings.values() if isinstance(timings, dict) else timings
            for timing in records:
                if isinstance(timing, dict):
                    lines.append(
                        f"  {timing.get('order_id', '?')}: "
                        f"elapsed={timing.get('elapsed_s', '?')}s, "
                        f"active={timing.get('active_elapsed_s', '?')}s"
                    )
        text = "\n".join(lines) + "\n"
        if self.summary_text.get("1.0", "end").strip() != text.strip():
            self._replace_text(self.summary_text, text)
        self._schedule(self._poll_summary, 1500)

    def _on_close(self) -> None:
        for job in self._poll_jobs.values():
            try:
                self.root.after_cancel(job)
            except tk.TclError:
                pass
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    LauncherApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
