#!/usr/bin/env python3
"""超市分拣 - competition_runner 一键启动 GUI（supermarket-nav-baseline 适配版）。

一键启动官方仿真 Server 与正式比赛入口 ``competition_runner.py``（与
``scripts/run_baseline.sh`` 同一入口）：

    Server 按 SUPERMARKET_TASKS 下发比赛任务（默认随机 5 个固定槽位
    product_XXX，可手动改为具体商品类别或 body 名）；
    Runner 订阅 /supermarket_sorting/task，按记忆矩阵选单并派发单件
    worker（内存矩阵直达、动态改道默认开启），结束后把 memory_matrix.json
    与 summary.json 写到 logs/competition_runner/<run_prefix>/ 下，
    界面实时显示 Server / Runner 日志、记忆矩阵和比赛摘要。

挂载的是当前仓库目录（容器内 /workspace/baseline），固定使用官方 final
镜像。Runner 的启动参数与 scripts/run_baseline.sh 保持一致，产物目录改为
仓库内 logs/competition_runner，方便宿主机 GUI 直接轮询。

依赖：宿主机 python3 + tkinter + docker。
"""

from __future__ import annotations

import datetime
import json
import os
import random
import subprocess
import tkinter as tk
from tkinter import messagebox, ttk


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
SERVER_NAME = "supermarket_runner_gui_server"
CLIENT_NAME = "supermarket_runner_gui_client"
ROS_DOMAIN_ID = "99"
RMW = "rmw_cyclonedds_cpp"
TORCH_CACHE = "/root/.cache/torch_extensions/cu128"

OFFICIAL_PREFIX = (
    "crpi-1pzq998p9m7w0auy.cn-hangzhou.personal.cr.aliyuncs.com/"
    "challengecup/supermarket_sorting_final")

# 固定使用官方 final 镜像（server 镜像自带完整代码，无需挂载仓库）
SERVER_IMAGE = f"{OFFICIAL_PREFIX}:server"
CLIENT_IMAGE = f"{OFFICIAL_PREFIX}:client"

KINDS = [
    "kele", "maidong", "heweidao", "shupian", "zhijin",
    "kouxiangtang", "sanmingzhi", "pingguo", "chengzi",
]
# 记忆矩阵：每种货物的展示用短名与颜色
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

# Runner 产物目录（容器内路径，挂载到宿主机 REPO_ROOT/logs/competition_runner）
RUNTIME_DIR_CONTAINER = "/workspace/baseline/logs/competition_runner"
RUNTIME_DIR_HOST = os.path.join(REPO_ROOT, "logs", "competition_runner")
WEIGHTS_CONTAINER = (
    "/workspace/baseline/examples/supermarket_sorting/"
    "perception/checkpoints/best.pt")

DEFAULT_ORDERS_COUNT = 5


def generate_tasks(count: int, seed: int | None) -> list[str]:
    """随机选择 count 个不重复的固定槽位 body（product_001..product_045）。"""
    rng = random.Random(seed)
    return [
        f"product_{index:03d}"
        for index in sorted(rng.sample(range(1, 46), max(1, min(count, 5))))
    ]


def run_cmd(args, **kwargs):
    """Run a command, returning (returncode, stdout, stderr)."""
    proc = subprocess.run(
        args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        **kwargs)
    return proc.returncode, proc.stdout, proc.stderr


class LauncherApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("超市分拣 - competition_runner 启动器 (baseline)")
        self.root.geometry("1000x760")
        self.root.minsize(840, 600)

        self._last_log = {SERVER_NAME: "", CLIENT_NAME: ""}
        self._poll_job = None
        self._matrix_job = None
        self._summary_job = None

        self._build_controls()
        self._build_logs()
        self._regenerate_tasks()
        self._poll_status()
        self._poll_memory_matrix()
        self._poll_summary()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---------------- UI ----------------
    def _build_controls(self):
        top = ttk.LabelFrame(self.root, text="启动配置", padding=8)
        top.pack(fill="x", padx=10, pady=(10, 4))

        self.orders_count_var = tk.IntVar(value=DEFAULT_ORDERS_COUNT)
        self.seed_var = tk.StringVar(value="")
        self.tasks_var = tk.StringVar(value="")
        self.cycles_var = tk.IntVar(value=2)
        self.attempts_var = tk.IntVar(value=2)
        self.confirmations_var = tk.IntVar(value=3)
        self.memory_conf_var = tk.DoubleVar(value=0.90)
        self.grab_policy_var = tk.StringVar(value="nearest")
        self.inference_hz_var = tk.DoubleVar(value=12.0)
        self.device_var = tk.StringVar(value="cpu")
        self.order_timeout_var = tk.IntVar(value=300)
        self.match_timeout_var = tk.IntVar(value=3600)
        self.target_time_var = tk.IntVar(value=400)

        self.obstacle_var = tk.BooleanVar(value=True)
        self.window_var = tk.BooleanVar(value=True)
        self.show_yolo_var = tk.BooleanVar(value=False)
        self.record_everywhere_var = tk.BooleanVar(value=True)
        self.perception_always_on_var = tk.BooleanVar(value=True)
        self.dynamic_direct_var = tk.BooleanVar(value=True)
        self.skip_recheck_var = tk.BooleanVar(value=False)
        self.wrist_center_var = tk.BooleanVar(value=False)

        row1 = ttk.Frame(top)
        row1.pack(fill="x")
        ttk.Label(row1, text="订单数量:").pack(side="left")
        ttk.Spinbox(
            row1, from_=1, to=5, textvariable=self.orders_count_var,
            width=4, command=self._regenerate_tasks).pack(
            side="left", padx=(4, 12))

        ttk.Label(row1, text="随机种子(留空=随机):").pack(side="left")
        self.seed_entry = ttk.Entry(row1, textvariable=self.seed_var, width=8)
        self.seed_entry.pack(side="left", padx=(4, 12))
        self.seed_entry.bind("<Return>", lambda _e: self._regenerate_tasks())

        ttk.Label(row1, text="任务列表(逗号分隔):").pack(side="left")
        self.tasks_entry = ttk.Entry(row1, textvariable=self.tasks_var, width=44)
        self.tasks_entry.pack(side="left", padx=(4, 8))
        ttk.Button(
            row1, text="随机生成", width=8,
            command=self._regenerate_tasks).pack(side="left")

        row2 = ttk.Frame(top)
        row2.pack(fill="x", pady=(6, 0))
        ttk.Label(row2, text="扫描轮数:").pack(side="left")
        ttk.Spinbox(
            row2, from_=1, to=10, textvariable=self.cycles_var,
            width=4).pack(side="left", padx=(4, 12))
        ttk.Label(row2, text="最大尝试:").pack(side="left")
        ttk.Spinbox(
            row2, from_=1, to=5, textvariable=self.attempts_var,
            width=4).pack(side="left", padx=(4, 12))
        ttk.Label(row2, text="记忆确认帧:").pack(side="left")
        ttk.Spinbox(
            row2, from_=1, to=10, textvariable=self.confirmations_var,
            width=4).pack(side="left", padx=(4, 12))
        ttk.Label(row2, text="记忆置信度:").pack(side="left")
        ttk.Spinbox(
            row2, from_=0.5, to=1.0, increment=0.05,
            textvariable=self.memory_conf_var,
            width=5).pack(side="left", padx=(4, 12))
        ttk.Label(row2, text="抓取策略:").pack(side="left")
        ttk.Combobox(
            row2, textvariable=self.grab_policy_var,
            values=("nearest", "sequence"), state="readonly",
            width=9).pack(side="left", padx=(4, 12))
        ttk.Label(row2, text="推理Hz:").pack(side="left")
        ttk.Spinbox(
            row2, from_=1, to=30, textvariable=self.inference_hz_var,
            width=4).pack(side="left", padx=(4, 12))
        ttk.Label(row2, text="设备:").pack(side="left")
        ttk.Combobox(
            row2, textvariable=self.device_var,
            values=("auto", "cpu", "cuda"), state="readonly",
            width=5).pack(side="left", padx=(4, 0))

        row3 = ttk.Frame(top)
        row3.pack(fill="x", pady=(6, 0))
        ttk.Label(row3, text="单订单超时(s):").pack(side="left")
        ttk.Spinbox(
            row3, from_=0, to=3600, increment=30,
            textvariable=self.order_timeout_var,
            width=6).pack(side="left", padx=(4, 12))
        ttk.Label(row3, text="比赛超时(s):").pack(side="left")
        ttk.Spinbox(
            row3, from_=60, to=7200, increment=60,
            textvariable=self.match_timeout_var,
            width=6).pack(side="left", padx=(4, 12))
        ttk.Label(row3, text="目标用时(s):").pack(side="left")
        ttk.Spinbox(
            row3, from_=60, to=3600, increment=30,
            textvariable=self.target_time_var,
            width=6).pack(side="left", padx=(4, 16))

        ttk.Checkbutton(
            row3, text="随机障碍物",
            variable=self.obstacle_var).pack(side="left", padx=(0, 8))
        ttk.Checkbutton(
            row3, text="显示仿真窗口",
            variable=self.window_var).pack(side="left", padx=(0, 8))
        ttk.Checkbutton(
            row3, text="显示YOLO窗口",
            variable=self.show_yolo_var).pack(side="left")

        row4 = ttk.Frame(top)
        row4.pack(fill="x", pady=(6, 0))
        ttk.Checkbutton(
            row4, text="全程录入矩阵(record-everywhere)",
            variable=self.record_everywhere_var).pack(side="left", padx=(0, 10))
        ttk.Checkbutton(
            row4, text="感知常开(perception-always-on)",
            variable=self.perception_always_on_var).pack(side="left", padx=(0, 10))
        ttk.Checkbutton(
            row4, text="矩阵直达/动态改道(dynamic-direct)",
            variable=self.dynamic_direct_var).pack(side="left", padx=(0, 10))
        ttk.Checkbutton(
            row4, text="跳过抓前复核(ArUco+类别,默认开启;勾选=关闭)",
            variable=self.skip_recheck_var).pack(side="left", padx=(0, 10))
        ttk.Checkbutton(
            row4, text="腕部对中(wrist-center)",
            variable=self.wrist_center_var).pack(side="left", padx=(0, 10))

        btn_row = ttk.Frame(top)
        btn_row.pack(fill="x", pady=(8, 0))
        ttk.Button(
            btn_row, text="启动 Server + Runner",
            command=self.start_all).pack(side="left")
        ttk.Button(
            btn_row, text="停止全部",
            command=self.stop_all).pack(side="left", padx=8)
        ttk.Button(
            btn_row, text="停止 Server",
            command=lambda: self.stop_container(SERVER_NAME)).pack(side="left")
        ttk.Button(
            btn_row, text="停止 Runner",
            command=lambda: self.stop_container(CLIENT_NAME)).pack(
            side="left", padx=8)

        self.status_var = tk.StringVar(value="Server: 未启动     Runner: 未启动")
        ttk.Label(
            top, textvariable=self.status_var,
            foreground="#0a6c0a").pack(anchor="w", pady=(6, 0))
        ttk.Label(
            top,
            text="提示：任务列表默认随机 5 个固定槽位(product_XXX)，也可手写"
                 "product_001..045 或商品类别(kele 等，类别会展开为该类全部"
                 "槽位，可能超过 5 单)。Runner 产物(矩阵/摘要)写入 "
                 "logs/competition_runner/<run_prefix>/。本 GUI 与 "
                 "scripts/run_baseline.sh 使用同一正式入口。",
            foreground="#666666").pack(anchor="w")

    def _build_logs(self):
        tabs = ttk.Notebook(self.root)
        tabs.pack(fill="both", expand=True, padx=10, pady=(4, 10))

        self.server_text = self._make_log_tab(tabs, "Server 日志")
        self.client_text = self._make_log_tab(tabs, "Runner 日志")
        self._build_memory_matrix_tab(tabs)
        self._build_summary_tab(tabs)

    def _build_memory_matrix_tab(self, tabs):
        """3 层 × 5 货架 × 3 列的记忆矩阵可视化面板（复用 555 版 GUI 布局）。"""
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
            ttk.Label(
                legend, text=KIND_SHORT[kind]).pack(side="left")
        consumed = tk.Label(
            legend, text="  ", bg=MATRIX_CONSUMED_COLOR, width=2)
        consumed.pack(side="left", padx=(16, 1))
        ttk.Label(legend, text="已取走").pack(side="left")

        tabs.add(frame, text="记忆矩阵")

    def _build_summary_tab(self, tabs):
        """展示最新 summary.json（比赛进度与耗时）。"""
        frame = ttk.Frame(tabs)
        self.summary_text = tk.Text(
            frame, wrap="word", state="disabled",
            font=("DejaVu Sans Mono", 9))
        scroll = ttk.Scrollbar(
            frame, orient="vertical", command=self.summary_text.yview)
        self.summary_text.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.summary_text.pack(side="left", fill="both", expand=True)
        tabs.add(frame, text="比赛摘要")

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

    # ---------------- helpers ----------------
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

    def _regenerate_tasks(self, *_args):
        try:
            count = self.orders_count_var.get()
        except tk.TclError:
            count = DEFAULT_ORDERS_COUNT
        self.tasks_var.set(
            ",".join(generate_tasks(max(1, count), self._seed_value())))

    def _task_tokens(self) -> list[str]:
        return [
            token.strip()
            for token in self.tasks_var.get().split(",")
            if token.strip()]

    # ---------------- docker args ----------------
    def _server_args(self) -> list[str]:
        show_window = self.window_var.get()
        seed = self._seed_value()
        args = [
            "docker", "run", "--rm", "-d",
            "--name", SERVER_NAME,
            "--runtime=nvidia", "--network", "host", "--ipc", "host",
            "-e", f"ROS_DOMAIN_ID={ROS_DOMAIN_ID}",
            "-e", f"RMW_IMPLEMENTATION={RMW}",
        ]
        if show_window:
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
            "-e",
            f"SUPERMARKET_RANDOMIZE_OBSTACLES="
            f"{1 if self.obstacle_var.get() else 0}",
            "-e", f"SUPERMARKET_TASKS={','.join(self._task_tokens())}",
        ]
        if seed is not None:
            args += ["-e", f"SUPERMARKET_SEED={seed}"]
        args += [
            "-e", f"TORCH_EXTENSIONS_DIR={TORCH_CACHE}",
        ]
        args += [
            "-v", "supermarket_sorting_cache:/root/.cache",
            SERVER_IMAGE,
            "bash", "-lc",
            "cd /workspace/supermarket_sorting_task && "
            "source /opt/ros/humble/setup.bash && "
            "mkdir -p logs && "
            "python3 examples/supermarket_sorting/"
            "supermarket_sorting_server.py 2>&1 | "
            "tee logs/gui_server_$(date +%H%M%S).log",
        ]
        return args

    def _runner_args(self) -> list[str]:
        runner_flags = []
        if self.record_everywhere_var.get():
            runner_flags.append("--record-everywhere")
        if self.perception_always_on_var.get():
            runner_flags.append("--perception-always-on")
        if self.dynamic_direct_var.get():
            runner_flags.append("--dynamic-direct")
        if self.skip_recheck_var.get():
            runner_flags.append("--no-close-recheck")
        if self.show_yolo_var.get():
            runner_flags.append("--show")
        if self.wrist_center_var.get():
            runner_flags.append("--wrist-center")
        args = [
            "docker", "run", "--rm", "-d",
            "--name", CLIENT_NAME,
            "--runtime=nvidia", "--network", "host", "--ipc", "host",
            "-e", f"ROS_DOMAIN_ID={ROS_DOMAIN_ID}",
            "-e", f"RMW_IMPLEMENTATION={RMW}",
            "-e", "YOLO_CONFIG_DIR=/tmp/Ultralytics",
            "-e", "SUPERMARKET_PATH_MEMORY=1",
            "-e",
            "SUPERMARKET_PATH_MEMORY_FILE="
            "/root/.cache/supermarket_path_memory.json",
            # 官方 client 镜像缺 discoverse，指向挂载的仓库
            "-e", "PYTHONPATH=/workspace/baseline",
        ]
        if self.show_yolo_var.get():
            subprocess.run(["xhost", "+local:docker"],
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
            args += [
                "-e", f"DISPLAY={os.environ.get('DISPLAY', '')}",
                "-v", "/tmp/.X11-unix:/tmp/.X11-unix:rw",
            ]
        args += [
            "-e", f"TORCH_EXTENSIONS_DIR={TORCH_CACHE}",
            "-v", f"{REPO_ROOT}:/workspace/baseline",
            "-v", "supermarket_sorting_cache:/root/.cache",
            CLIENT_IMAGE,
            "bash", "-lc",
            "cd /workspace/baseline && "
            "source /opt/ros/humble/setup.bash && "
            "mkdir -p logs/competition_runner && "
            "python3 examples/supermarket_sorting/competition_runner.py "
            f"--weights {WEIGHTS_CONTAINER} "
            f"--max-scan-cycles {int(self.cycles_var.get())} "
            f"--max-attempts {int(self.attempts_var.get())} "
            f"--memory-confirmations {int(self.confirmations_var.get())} "
            f"--memory-confidence-threshold "
            f"{float(self.memory_conf_var.get()):.2f} "
            f"--grab-policy {self.grab_policy_var.get()} "
            f"--inference-hz {float(self.inference_hz_var.get()):.0f} "
            f"--device {self.device_var.get()} "
            f"--order-timeout {int(self.order_timeout_var.get())} "
            f"--match-timeout {int(self.match_timeout_var.get())} "
            f"--target-time {int(self.target_time_var.get())} "
            f"--runtime-dir {RUNTIME_DIR_CONTAINER} "
            + (" ".join(runner_flags) + " " if runner_flags else "")
            + "2>&1 | tee logs/gui_runner_$(date +%H%M%S).log",
        ]
        return args

    # ---------------- actions ----------------
    def start_all(self):
        if not self._check_image(SERVER_IMAGE) or \
                not self._check_image(CLIENT_IMAGE):
            messagebox.showerror(
                "镜像缺失",
                f"请先拉取官方镜像:\n  {SERVER_IMAGE}\n  {CLIENT_IMAGE}")
            return
        tasks = self._task_tokens()
        if not tasks:
            messagebox.showerror(
                "任务为空", "请先用“随机生成”或手动填写任务列表。")
            return
        if len(tasks) > 5:
            messagebox.showwarning(
                "任务超过 5 单",
                "Runner 只有 5 个交付槽位，超过 5 单会在第 6 单停止；"
                "建议使用随机生成的 5 个 product_XXX 或手填具体 body 名。")
        # 清理同名残留容器（包括已退出但未删除的容器）
        for name in (CLIENT_NAME, SERVER_NAME):
            run_cmd(["docker", "rm", "-f", name])

        self._append_log(self.server_text, "正在启动 Server ...\n")
        code, _, err = run_cmd(self._server_args())
        if code != 0:
            messagebox.showerror("Server 启动失败", err.strip())
            return
        self._append_log(self.server_text, "Server 容器已创建，等待仿真启动...\n")

        self._append_log(
            self.client_text,
            f"正在启动 Runner（任务: {', '.join(tasks)}）...\n")
        code, _, err = run_cmd(self._runner_args())
        if code != 0:
            messagebox.showerror("Runner 启动失败", err.strip())
            return
        self._append_log(self.client_text, "Runner 容器已创建...\n")
        self._last_log = {SERVER_NAME: "", CLIENT_NAME: ""}

    def stop_container(self, name: str):
        run_cmd(["docker", "rm", "-f", name])
        self._last_log[name] = ""
        widget = self.server_text if name == SERVER_NAME else self.client_text
        self._append_log(widget, "已停止。\n")

    def stop_all(self):
        for name in (CLIENT_NAME, SERVER_NAME):
            run_cmd(["docker", "rm", "-f", name])
        self._last_log = {SERVER_NAME: "", CLIENT_NAME: ""}
        self._append_log(self.server_text, "Server 已停止。\n")
        self._append_log(self.client_text, "Runner 已停止。\n")

    # ---------------- polling ----------------
    def _poll_status(self):
        server_on = self._docker_running(SERVER_NAME)
        client_on = self._docker_running(CLIENT_NAME)
        self.status_var.set(
            f"Server: {'运行中' if server_on else '未启动'}     "
            f"Runner: {'运行中' if client_on else '未启动'}")
        self._poll_logs()
        self._poll_job = self.root.after(1500, self._poll_status)

    def _poll_logs(self):
        for name, widget in ((SERVER_NAME, self.server_text),
                             (CLIENT_NAME, self.client_text)):
            code, out, _ = run_cmd(["docker", "logs", "--tail", "400", name])
            if code != 0:
                continue
            if out != self._last_log[name]:
                self._last_log[name] = out
                self._append_log(widget, out if out else "(暂无输出)\n")

    @staticmethod
    def _newest_run_artifact(filename: str) -> str | None:
        """返回 run 目录下最新文件的路径（按 mtime）。"""
        root = RUNTIME_DIR_HOST
        if not os.path.isdir(root):
            return None
        matches = [
            os.path.join(run_dir, filename)
            for run_dir, _dirs, files in os.walk(root)
            if filename in files
        ]
        if not matches:
            return None
        return max(matches, key=os.path.getmtime)

    def _poll_memory_matrix(self):
        """轮询 logs/competition_runner/*/memory_matrix.json 并刷新网格。"""
        path = self._newest_run_artifact("memory_matrix.json")
        if path is None:
            self.matrix_status.config(text="暂无数据")
            self._matrix_job = self.root.after(1000, self._poll_memory_matrix)
            return
        try:
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, ValueError):
            self.matrix_status.config(text="读取失败")
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
        approx = data.get("approx", {})
        approx_cols = data.get("approx_cols", {})
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
            cols_map = approx_cols.get(key, {})
            if cols_map:
                for kind in kinds:
                    column = str(cols_map.get(kind, ""))
                    if column not in ("1", "2", "3"):
                        continue
                    label = self.matrix_labels[level][
                        col_base + int(column) - 1]
                    if str(label.cget("text")):
                        continue
                    label.config(
                        text="≈" + KIND_SHORT.get(kind, kind),
                        bg=KIND_COLORS.get(kind, "#DEE2E6"))
            else:
                for i in range(3):
                    label = self.matrix_labels[level][col_base + i]
                    if str(label.cget("text")):
                        continue
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

    def _poll_summary(self):
        """轮询最新 summary.json 并刷新比赛摘要面板。"""
        path = self._newest_run_artifact("summary.json")
        if path is None:
            self._replace_text(self.summary_text, "(暂无摘要)\n")
            self._summary_job = self.root.after(1500, self._poll_summary)
            return
        try:
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, ValueError):
            self._summary_job = self.root.after(1500, self._poll_summary)
            return
        lines = []
        lines.append(f"run_prefix : {data.get('run_prefix', '?')}")
        lines.append(f"reason     : {data.get('reason', '?')}")
        lines.append(
            f"进度       : {data.get('delivered', 0)} 交付 / "
            f"{data.get('failed', 0)} 失败 / "
            f"{data.get('pending', 0)} 待办 / 共 {data.get('count', 0)}")
        elapsed = data.get("elapsed_s")
        target = data.get("target_time_s")
        if isinstance(elapsed, (int, float)):
            remaining = data.get("remaining_to_target_s")
            elapsed_text = f"{elapsed:.1f}s / 目标 {target:.1f}s"
            if isinstance(remaining, (int, float)):
                elapsed_text += f" (剩余 {remaining:.1f}s)"
            lines.append(f"耗时       : {elapsed_text}")
        orders = data.get("orders", [])
        if orders:
            lines.append("订单       :")
            for order in orders:
                errors = (
                    f" 错误={order.get('errors')}"
                    if order.get("errors") else "")
                lines.append(
                    f"  #{order.get('source_index', 0) + 1} "
                    f"{order.get('id', '?')} "
                    f"kind={order.get('kind', '?')} "
                    f"status={order.get('status', '?')} "
                    f"attempts={order.get('attempts', 0)}"
                    f"{errors}")
        timings = data.get("order_timings") or {}
        if timings:
            lines.append("订单耗时(s) :")
            timing_items = (
                timings.items() if isinstance(timings, dict)
                else ((timing.get("order_id", "?"), timing)
                      for timing in timings))
            for order_id, timing in sorted(timing_items):
                lines.append(
                    f"  {order_id}: elapsed={timing.get('elapsed_s')} "
                    f"active={timing.get('active_elapsed_s')} "
                    f"attempts={timing.get('attempts')}")
        text = "\n".join(lines) + "\n"
        current = self.summary_text.get("1.0", "end")
        if current.strip() != text.strip():
            self._replace_text(self.summary_text, text)
        self._summary_job = self.root.after(1500, self._poll_summary)

    @staticmethod
    def _replace_text(widget: tk.Text, text: str):
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("end", text)
        widget.see("end")
        widget.configure(state="disabled")

    def _append_log(self, widget: tk.Text, text: str):
        self._replace_text(widget, text)

    def _on_close(self):
        if self._poll_job is not None:
            self.root.after_cancel(self._poll_job)
        if self._matrix_job is not None:
            self.root.after_cancel(self._matrix_job)
        if self._summary_job is not None:
            self.root.after_cancel(self._summary_job)
        self.root.destroy()


def main():
    root = tk.Tk()
    LauncherApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
