#!/usr/bin/env python3
"""超市分拣 - 连续多单抓取启动 GUI（1111 版，与主目录界面完全一致）。

一键启动仿真 Server 与连续订单客户端 ``continuous_goods_client.py``：
客户端先随机生成一批不重复的货物订单（默认 5 单），每单执行「抓货区抓取 →
终点直接扔货 → 返回抓货区 → 下一单」，界面实时显示 Server / Client 两个
容器的日志。

与主目录 gui_launcher_continuous.py 完全一致，仅去掉了「镜像组」选择栏，
固定使用官方 final 镜像（server 镜像自带代码，无需挂载）。
挂载的是本目录（1111/supermarket-nav-baseline）。

依赖：宿主机 python3 + tkinter + docker。
"""

from __future__ import annotations

import datetime
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

# 镜像候选：完整官方名优先；队友机器上可能是同一个镜像但本地 tag 不同。
# 也可用环境变量 SUPERMARKET_SERVER_IMAGE / SUPERMARKET_CLIENT_IMAGE 强制指定。
SERVER_IMAGE_CANDIDATES = [
    os.environ.get("SUPERMARKET_SERVER_IMAGE", "").strip(),
    f"{OFFICIAL_PREFIX}:server",
    "supermarket_sorting_final:server",
    "challengecup/supermarket_sorting_final:server",
    "supermarket_sorting_task:sm120",
]
CLIENT_IMAGE_CANDIDATES = [
    os.environ.get("SUPERMARKET_CLIENT_IMAGE", "").strip(),
    f"{OFFICIAL_PREFIX}:client",
    "supermarket_sorting_final:client",
    "challengecup/supermarket_sorting_final:client",
    "supermarket_sorting_task:sm120-yolo11",
]

KINDS = [
    "kele", "maidong", "heweidao", "shupian", "zhijin",
    "kouxiangtang", "sanmingzhi", "pingguo", "chengzi",
]
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
        self.root.title("超市分拣 - 连续多单抓取启动器")
        self.root.geometry("960x700")
        self.root.minsize(800, 560)

        self._last_log = {SERVER_NAME: "", CLIENT_NAME: ""}
        self._poll_job = None
        self._recorder_started = False
        self._recorder_output = None
        self.server_image = None
        self.client_image = None

        self._build_controls()
        self._build_logs()
        self._regenerate_orders()
        self._poll_status()

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
            text="提示：客户端每单执行「抓货区抓取 → 终点直接扔货 → 返回抓货区」"
                 "，送完一单后自动返回并继续下一单；订单允许重复，可手动"
                 "增删改；每次运行自动录像到 logs/（无论正常结束还是停止）。",
            foreground="#666666").pack(anchor="w")

    def _build_logs(self):
        tabs = ttk.Notebook(self.root)
        tabs.pack(fill="both", expand=True, padx=10, pady=(4, 10))

        self.server_text = self._make_log_tab(tabs, "Server 日志")
        self.client_text = self._make_log_tab(tabs, "Client 日志")

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

    @staticmethod
    def _resolve_image(candidates: list[str]) -> str | None:
        """返回本机已存在的第一个候选镜像名（兼容不同的本地 tag）。"""
        for name in candidates:
            name = name.strip()
            if name and LauncherApp._check_image(name):
                return name
        return None

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
            "--gpus", "all", "--network", "host", "--ipc", "host",
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
            self.server_image,
            "bash", "-lc",
            "cd /workspace/supermarket_sorting_task && "
            "source /opt/ros/humble/setup.bash && "
            "python3 examples/supermarket_sorting/"
            "supermarket_sorting_server.py",
        ]
        return args

    def _client_args(self) -> list[str]:
        orders = ",".join(self.orders)
        args = [
            "docker", "run", "--rm", "-d",
            "--name", CLIENT_NAME,
            "--gpus", "all", "--network", "host", "--ipc", "host",
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
            self.client_image,
            "bash", "-lc",
            "cd /workspace/supermarket_sorting_task && "
            "source /opt/ros/humble/setup.bash && "
            "python3 examples/supermarket_sorting/"
            "continuous_goods_client.py "
            f"--orders {orders} "
            f"--max-scan-cycles {int(self.cycles_var.get())} "
            "--weights /workspace/supermarket_sorting_task/examples/"
            "supermarket_sorting/perception/checkpoints/best.pt"
            + (" --show" if self.show_yolo_var.get() else ""),
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
            self.client_image,
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
        self.server_image = self._resolve_image(SERVER_IMAGE_CANDIDATES)
        self.client_image = self._resolve_image(CLIENT_IMAGE_CANDIDATES)
        if not self.server_image or not self.client_image:
            messagebox.showerror(
                "镜像缺失",
                "未找到可用的 Server/Client 镜像。\n"
                "请先拉取官方镜像：\n"
                f"  {SERVER_IMAGE_CANDIDATES[1]}\n"
                f"  {CLIENT_IMAGE_CANDIDATES[1]}\n"
                "或用环境变量指定本机镜像名：\n"
                "  SUPERMARKET_SERVER_IMAGE=<名> "
                "SUPERMARKET_CLIENT_IMAGE=<名>")
            return
        self._append_log(
            self.server_text,
            f"使用镜像: server={self.server_image} "
            f"client={self.client_image}\n")
        if not self.orders:
            messagebox.showerror(
                "订单为空", "请先用“随机生成”或“添加”生成订单列表。")
            return
        # 清理同名残留容器
        for name in (CLIENT_NAME, SERVER_NAME):
            if self._docker_running(name):
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
        self.root.destroy()


def main():
    root = tk.Tk()
    LauncherApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
