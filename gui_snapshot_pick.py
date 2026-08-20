#!/usr/bin/env python3
"""超市分拣 - 正式行走录入 + 记忆矩阵直达抓取启动 GUI。

一键启动仿真 Server 与快照优先客户端 ``snapshot_pick_client.py``：
客户端先按正式行走流程逐架走到货架前标准站点（E→D→C→B→A）录入全部商品，
写入 3×15 记忆矩阵（logs/memory_matrix.json），之后每单查矩阵直达对应
货架/层做局部定位抓取，避免整排货架扫描；每单执行「抓货区抓取 → 终点直接
扔货 → 返回抓货区 → 下一单」，界面实时显示 Server / Client 两个容器的
日志与记忆矩阵可视化。

挂载的是当前仓库目录，固定使用官方 final 镜像。

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
SERVER_NAME = "supermarket_gui_server"
CLIENT_NAME = "supermarket_gui_client"
RECORDER_NAME = "supermarket_gui_recorder"
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
DEFAULT_ORDERS_COUNT = 5


def generate_orders(count: int, seed: int | None) -> list[str]:
    """随机生成 count 个货物订单（允许重复，与客户端算法保持一致）。

    使用排序后的类别列表作为抽样总体，保证相同 seed 在 GUI 和客户端
    中生成完全一致的订单。
    """
    rng = random.Random(seed)
    return rng.choices(sorted(KINDS), k=count)


def run_cmd(args, **kwargs):
    """Run a command, returning (returncode, stdout, stderr)."""
    proc = subprocess.run(
        args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        **kwargs)
    return proc.returncode, proc.stdout, proc.stderr


class LauncherApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("超市分拣 - 正式行走录入+记忆直达抓取")
        self.root.geometry("960x700")
        self.root.minsize(800, 560)

        self._last_log = {SERVER_NAME: "", CLIENT_NAME: ""}
        self._poll_job = None
        self._recorder_started = False
        self._recorder_output = None

        self._build_controls()
        self._build_logs()
        self._regenerate_orders()
        self._poll_status()
        self._poll_memory_matrix()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---------------- UI ----------------
    def _build_controls(self):
        top = ttk.LabelFrame(self.root, text="启动配置", padding=8)
        top.pack(fill="x", padx=10, pady=(10, 4))

        self.orders_count_var = tk.IntVar(value=DEFAULT_ORDERS_COUNT)
        self.seed_var = tk.StringVar(value="")
        self.cycles_var = tk.IntVar(value=2)
        self.obstacle_var = tk.BooleanVar(value=True)
        self.window_var = tk.BooleanVar(value=True)
        self.show_yolo_var = tk.BooleanVar(value=False)
        self.record_var = tk.BooleanVar(value=True)
        self.res_var = tk.StringVar(value="640x480")
        self.fps_var = tk.StringVar(value="24")
        # 默认"余光"模式：不专门行走录入，订单抓取过程自动填记忆矩阵；
        # 需要完整矩阵时再勾选行走录入。
        self.snapshot_first_var = tk.BooleanVar(value=False)
        self.snapshot_passes_var = tk.IntVar(value=1)

        row1 = ttk.Frame(top)
        row1.pack(fill="x")
        ttk.Label(row1, text="订单数量:").pack(side="left")
        ttk.Spinbox(
            row1, from_=1, to=15, textvariable=self.orders_count_var,
            width=4, command=self._regenerate_orders).pack(
            side="left", padx=(4, 16))

        ttk.Label(row1, text="随机种子(留空=随机):").pack(side="left")
        self.seed_entry = ttk.Entry(row1, textvariable=self.seed_var, width=8)
        self.seed_entry.pack(side="left", padx=(4, 16))
        self.seed_entry.bind("<Return>", lambda _e: self._regenerate_orders())

        ttk.Label(row1, text="扫描轮数:").pack(side="left")
        ttk.Spinbox(
            row1, from_=1, to=10, textvariable=self.cycles_var,
            width=4).pack(side="left", padx=(4, 16))

        row2 = ttk.Frame(top)
        row2.pack(fill="x", pady=(6, 0))
        ttk.Checkbutton(
            row2, text="随机障碍物",
            variable=self.obstacle_var).pack(side="left", padx=(0, 8))
        ttk.Checkbutton(
            row2, text="显示仿真窗口",
            variable=self.window_var).pack(side="left", padx=(0, 8))
        ttk.Checkbutton(
            row2, text="显示YOLO窗口(仅本地镜像)",
            variable=self.show_yolo_var).pack(side="left")

        row2b = ttk.Frame(top)
        row2b.pack(fill="x", pady=(6, 0))
        ttk.Checkbutton(
            row2b, text="先正式行走录入(记忆矩阵)",
            variable=self.snapshot_first_var).pack(side="left", padx=(0, 8))
        ttk.Label(row2b, text="录入趟数:").pack(side="left")
        ttk.Spinbox(
            row2b, from_=1, to=5, textvariable=self.snapshot_passes_var,
            width=4).pack(side="left", padx=(4, 16))
        ttk.Checkbutton(
            row2b, text="自动录像",
            variable=self.record_var).pack(side="left", padx=(0, 16))
        ttk.Label(row2b, text="分辨率:").pack(side="left")
        ttk.Combobox(
            row2b, textvariable=self.res_var,
            values=("640x480", "320x240"), state="readonly",
            width=8).pack(side="left", padx=(4, 16))
        ttk.Label(row2b, text="帧率:").pack(side="left")
        ttk.Combobox(
            row2b, textvariable=self.fps_var,
            values=("24", "15", "10"), state="readonly",
            width=5).pack(side="left", padx=(4, 0))

        row3 = ttk.Frame(top)
        row3.pack(fill="x", pady=(6, 0))
        ttk.Label(row3, text="订单列表(可编辑):").pack(side="left", anchor="n")
        list_frame = ttk.Frame(row3)
        list_frame.pack(side="left", padx=(6, 10))
        self.orders_list = tk.Listbox(
            list_frame, height=5, width=42, exportselection=False)
        self.orders_list.pack(side="left", fill="y")
        list_scroll = ttk.Scrollbar(
            list_frame, orient="vertical", command=self.orders_list.yview)
        self.orders_list.configure(yscrollcommand=list_scroll.set)
        list_scroll.pack(side="left", fill="y")

        edit_col = ttk.Frame(row3)
        edit_col.pack(side="left", fill="y")
        ttk.Label(edit_col, text="商品类别:").pack(anchor="w")
        self.kind_edit_var = tk.StringVar(value=KINDS[0])
        kind_combo = ttk.Combobox(
            edit_col, textvariable=self.kind_edit_var,
            values=KINDS, state="readonly", width=10)
        kind_combo.pack(anchor="w", pady=(2, 4))
        edit_buttons = ttk.Frame(edit_col)
        edit_buttons.pack(anchor="w")
        ttk.Button(edit_buttons, text="添加", width=6,
                   command=self._add_order).pack(side="left")
        ttk.Button(edit_buttons, text="替换选中", width=8,
                   command=self._replace_order).pack(side="left", padx=2)
        ttk.Button(edit_buttons, text="删除选中", width=8,
                   command=self._remove_order).pack(side="left")
        ttk.Button(edit_col, text="清空", width=8,
                   command=self._clear_orders).pack(anchor="w", pady=(3, 0))
        ttk.Button(edit_col, text="随机生成", width=8,
                   command=self._regenerate_orders).pack(anchor="w", pady=(3, 0))

        btn_row = ttk.Frame(top)
        btn_row.pack(fill="x", pady=(8, 0))
        ttk.Button(
            btn_row, text="启动 Server + Client",
            command=self.start_all).pack(side="left")
        ttk.Button(
            btn_row, text="停止全部",
            command=self.stop_all).pack(side="left", padx=8)
        ttk.Button(
            btn_row, text="停止 Server",
            command=lambda: self.stop_container(SERVER_NAME)).pack(side="left")
        ttk.Button(
            btn_row, text="停止 Client",
            command=lambda: self.stop_container(CLIENT_NAME)).pack(
            side="left", padx=8)

        self.status_var = tk.StringVar(value="Server: 未启动     Client: 未启动")
        ttk.Label(
            top, textvariable=self.status_var,
            foreground="#0a6c0a").pack(anchor="w", pady=(6, 0))
        ttk.Label(
            top,
            text="提示：默认不专门扫描，订单抓取过程中 tracker 持续用 YOLO"
                 "余光录入当前货架（首单可能仍需扫描，多单后矩阵逐步变全，"
                 "后续订单直达）；勾选行走录入则开局先逐架录一遍完整矩阵；"
                 "每单执行「抓货区抓取 → 终点直接扔货 → 返回抓货区」；"
                 "订单可手动增删改；每次运行自动录像到 logs/。",
            foreground="#666666").pack(anchor="w")

    def _build_logs(self):
        tabs = ttk.Notebook(self.root)
        tabs.pack(fill="both", expand=True, padx=10, pady=(4, 10))

        self.server_text = self._make_log_tab(tabs, "Server 日志")
        self.client_text = self._make_log_tab(tabs, "Client 日志")
        self._build_memory_matrix_tab(tabs)

    def _build_memory_matrix_tab(self, tabs):
        """3 层 × 5 货架 × 3 列的记忆矩阵可视化面板。"""
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

    def _regenerate_orders(self, *_args):
        try:
            count = self.orders_count_var.get()
        except tk.TclError:
            count = DEFAULT_ORDERS_COUNT
        seed = self._seed_value()
        self.orders = generate_orders(max(1, count), seed)
        self._refresh_orders_list()

    def _refresh_orders_list(self):
        self.orders_list.delete(0, "end")
        for order in self.orders:
            self.orders_list.insert("end", order)

    def _add_order(self):
        kind = self.kind_edit_var.get()
        if kind in KINDS:
            self.orders.append(kind)
            self._refresh_orders_list()

    def _replace_order(self):
        selection = self.orders_list.curselection()
        kind = self.kind_edit_var.get()
        if not selection or kind not in KINDS:
            return
        self.orders[selection[0]] = kind
        self._refresh_orders_list()
        self.orders_list.selection_set(selection[0])

    def _remove_order(self):
        selection = self.orders_list.curselection()
        if not selection:
            return
        del self.orders[selection[0]]
        self._refresh_orders_list()

    def _clear_orders(self):
        self.orders = []
        self._refresh_orders_list()

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

    def _client_args(self) -> list[str]:
        orders = ",".join(self.orders)
        snapshot_args = ""
        if self.snapshot_first_var.get():
            snapshot_args = (
                " --record-first "
                f"--record-passes {int(self.snapshot_passes_var.get())}")
        args = [
            "docker", "run", "--rm", "-d",
            "--name", CLIENT_NAME,
            "--runtime=nvidia", "--network", "host", "--ipc", "host",
            "-e", f"ROS_DOMAIN_ID={ROS_DOMAIN_ID}",
            "-e", f"RMW_IMPLEMENTATION={RMW}",
            "-e", "YOLO_CONFIG_DIR=/tmp/Ultralytics",
            # 官方 client 镜像缺 discoverse，指向挂载的仓库
            "-e", "PYTHONPATH=/workspace/supermarket_sorting_task",
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
            "-v", f"{REPO_ROOT}:/workspace/supermarket_sorting_task",
            "-v", "supermarket_sorting_cache:/root/.cache",
            CLIENT_IMAGE,
            "bash", "-lc",
            "cd /workspace/supermarket_sorting_task && "
            "source /opt/ros/humble/setup.bash && "
            "python3 examples/supermarket_sorting/"
            "snapshot_pick_client.py "
            f"--orders {orders} "
            f"--max-scan-cycles {int(self.cycles_var.get())}"
            + snapshot_args
            + (" --show" if self.show_yolo_var.get() else "")
            + " 2>&1 | tee logs/gui_client_snapshot_pick_"
            + f"$(date +%H%M%S).log",
        ]
        return args

    # ---------------- 运行录像（即插即用，独立容器） ----------------
    def _recorder_args(self) -> list[str]:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output = f"/workspace/supermarket_sorting_task/logs/run_{stamp}.mp4"
        self._recorder_output = f"{REPO_ROOT}/logs/run_{stamp}.mp4"
        args = [
            "docker", "run", "--rm", "-d",
            "--name", RECORDER_NAME,
            "--network", "host", "--ipc", "host",
            "-e", f"ROS_DOMAIN_ID={ROS_DOMAIN_ID}",
            "-e", f"RMW_IMPLEMENTATION={RMW}",
            "-e", "PYTHONPATH=/workspace/supermarket_sorting_task",
            "-v", f"{REPO_ROOT}:/workspace/supermarket_sorting_task",
            CLIENT_IMAGE,
            "python3", "/workspace/supermarket_sorting_task/record_run.py",
            "--output", output,
            "--fps", str(self.fps_var.get()),
        ]
        try:
            width, height = (
                int(part) for part in self.res_var.get().split("x", 1))
        except (ValueError, TypeError):
            width, height = 640, 480
        if width > 0 and height > 0 and (width, height) != (640, 480):
            args += ["--width", str(width), "--height", str(height)]
        return args

    def _start_recorder(self):
        if not self.record_var.get():
            self._append_log(self.client_text, "录像已关闭（本次不生成视频）。\n")
            return
        code, _, err = run_cmd(self._recorder_args())
        if code == 0:
            self._recorder_started = True
            self._append_log(
                self.client_text,
                f"录像已启动: {self._recorder_output}\n")
        else:
            self._append_log(
                self.client_text,
                f"录像启动失败（不影响运行）: {err.strip()}\n")

    def _stop_recorder(self):
        if not self._recorder_started:
            return
        self._recorder_started = False
        # docker stop 发 SIGTERM，录像脚本收尾保存视频后再退出
        run_cmd(["docker", "stop", "-t", "8", RECORDER_NAME])
        run_cmd(["docker", "rm", "-f", RECORDER_NAME])
        self._append_log(
            self.client_text,
            f"录像已保存: {self._recorder_output}\n")

    # ---------------- actions ----------------
    def start_all(self):
        if not self._check_image(SERVER_IMAGE) or \
                not self._check_image(CLIENT_IMAGE):
            messagebox.showerror(
                "镜像缺失",
                f"请先拉取官方镜像:\n  {SERVER_IMAGE}\n  {CLIENT_IMAGE}")
            return
        if not self.orders:
            messagebox.showerror(
                "订单为空", "请先用“随机生成”或“添加”生成订单列表。")
            return
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
            f"正在启动 Client（订单: {', '.join(self.orders)}）...\n")
        code, _, err = run_cmd(self._client_args())
        if code != 0:
            messagebox.showerror("Client 启动失败", err.strip())
            return
        self._append_log(self.client_text, "Client 容器已创建...\n")
        self._start_recorder()
        self._last_log = {SERVER_NAME: "", CLIENT_NAME: ""}

    def stop_container(self, name: str):
        run_cmd(["docker", "rm", "-f", name])
        self._last_log[name] = ""
        if name == CLIENT_NAME:
            self._stop_recorder()
        widget = self.server_text if name == SERVER_NAME else self.client_text
        self._append_log(widget, "已停止。\n")

    def stop_all(self):
        for name in (CLIENT_NAME, SERVER_NAME):
            run_cmd(["docker", "rm", "-f", name])
        self._stop_recorder()
        self._last_log = {SERVER_NAME: "", CLIENT_NAME: ""}
        self._append_log(self.server_text, "Server 已停止。\n")
        self._append_log(self.client_text, "Client 已停止。\n")

    # ---------------- polling ----------------
    def _poll_status(self):
        server_on = self._docker_running(SERVER_NAME)
        client_on = self._docker_running(CLIENT_NAME)
        self.status_var.set(
            f"Server: {'运行中' if server_on else '未启动'}     "
            f"Client: {'运行中' if client_on else '未启动'}")
        if self._recorder_started:
            if not self._docker_running(RECORDER_NAME):
                # 录像进程自行退出（如 server 停止导致无帧超时）
                self._recorder_started = False
                self._append_log(
                    self.client_text, "录像进程已结束（视频已落盘）。\n")
            elif not client_on:
                # 正常运行完毕或 client 被停止 → 收尾保存录像
                self._append_log(
                    self.client_text, "Client 已退出，正在保存录像...\n")
                self._stop_recorder()
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

    def _poll_memory_matrix(self):
        """轮询 logs/memory_matrix.json 并刷新 3×15 网格。"""
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
        # YOLO-only 近似记录铺到网格：无精确记录的格子显示 "≈种类"，
        # 新版带 approx_cols 时按站点坐标放到具体列。
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
                        continue  # 已有精确记录，不覆盖
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

    def _append_log(self, widget: tk.Text, text: str):
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("end", text)
        widget.see("end")
        widget.configure(state="disabled")

    def _on_close(self):
        self._stop_recorder()
        if self._poll_job is not None:
            self.root.after_cancel(self._poll_job)
        if getattr(self, "_matrix_job", None) is not None:
            self.root.after_cancel(self._matrix_job)
        self.root.destroy()


def main():
    root = tk.Tk()
    LauncherApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
