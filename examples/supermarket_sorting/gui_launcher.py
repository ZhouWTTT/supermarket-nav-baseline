#!/usr/bin/env python3
"""超市分拣启动 GUI（整合导航 + 抓取版）。

基于官方镜像（server/client）一键启动仿真 Server 与整合客户端
``integrated_nav_pick_place.py``（导航到货架 → YOLO+ArUco 定位 → 抓取 →
导航穿越障碍区 → 配送台放置），界面实时显示两个容器的日志。

相对旧版 GUI 的差异：
  * 默认使用官方 final 镜像（server 镜像自带代码，无需挂载仓库）；
  * 服务器强制开启激光（SUPERMARKET_ENABLE_LIDAR=1），导航避障依赖它；
  * 客户端运行整合脚本，并自动加上 PYTHONPATH（官方 client 镜像缺少
    discoverse 包）；
  * 官方 client 镜像的 OpenCV 无窗口支持，YOLO 窗口选项只在本地镜像生效，
    脚本会自动降级不崩溃。

依赖：宿主机 python3 + tkinter + docker。
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tkinter as tk
from tkinter import messagebox, ttk


REPO_ROOT = str(Path(__file__).resolve().parents[2])
SERVER_NAME = "supermarket_gui_server"
CLIENT_NAME = "supermarket_gui_client"
ROS_DOMAIN_ID = "99"
RMW = "rmw_cyclonedds_cpp"
TORCH_CACHE = "/root/.cache/torch_extensions/cu128"

OFFICIAL_PREFIX = (
    "crpi-1pzq998p9m7w0auy.cn-hangzhou.personal.cr.aliyuncs.com/"
    "challengecup/supermarket_sorting_final")

# 镜像组：官方 final 镜像 / 本地自建镜像
IMAGE_SETS = {
    "官方镜像 (推荐)": {
        "server": f"{OFFICIAL_PREFIX}:server",
        "client": f"{OFFICIAL_PREFIX}:client",
        # 官方 server 镜像内置完整代码，无需挂载仓库
        "mount_server": False,
    },
    "本地镜像": {
        "server": "supermarket_sorting_task:sm120",
        "client": "supermarket_sorting_task:sm120-yolo11",
        "mount_server": True,
    },
}

# Five distinct source bodies are selected before the Server anonymises them.
# The organizer controls this value during formal evaluation; this default is
# only a reproducible local five-order task instead of the 45-item stress mode.
DEV_TASKS = os.environ.get(
    "SUPERMARKET_GUI_TASKS",
    "product_001,product_005,product_015,product_026,product_040")


def run_cmd(args, **kwargs):
    """Run a command, returning (returncode, stdout, stderr)."""
    proc = subprocess.run(
        args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        **kwargs)
    return proc.returncode, proc.stdout, proc.stderr


class LauncherApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("超市分拣 - 导航+抓取启动器")
        self.root.geometry("920x660")
        self.root.minsize(780, 540)

        self._last_log = {SERVER_NAME: "", CLIENT_NAME: ""}
        self._poll_job = None

        self._build_controls()
        self._build_logs()
        self._poll_status()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---------------- UI ----------------
    def _build_controls(self):
        top = ttk.LabelFrame(self.root, text="启动配置", padding=8)
        top.pack(fill="x", padx=10, pady=(10, 4))

        self.image_set_var = tk.StringVar(value=next(iter(IMAGE_SETS)))
        self.seed_var = tk.StringVar(value="11")
        self.cycles_var = tk.IntVar(value=2)
        self.obstacle_var = tk.BooleanVar(value=True)
        self.window_var = tk.BooleanVar(value=True)
        self.show_yolo_var = tk.BooleanVar(value=False)

        row1 = ttk.Frame(top)
        row1.pack(fill="x")
        ttk.Label(row1, text="镜像组:").pack(side="left")
        image_box = ttk.Combobox(
            row1, textvariable=self.image_set_var,
            values=list(IMAGE_SETS), state="readonly", width=14)
        image_box.pack(side="left", padx=(4, 16))

        ttk.Label(row1, text="随机种子(留空=随机):").pack(side="left")
        ttk.Entry(row1, textvariable=self.seed_var, width=8).pack(
            side="left", padx=(4, 16))

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
            text="提示：官方 server 镜像自带代码（无需挂载仓库）；官方 client "
                 "镜像的 OpenCV 无窗口，YOLO 窗口会自动跳过；"
                 "客户端从任务 Topic 接收五单，完成整局后自动退出。",
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

    def _image_set(self) -> dict:
        return IMAGE_SETS[self.image_set_var.get()]

    # ---------------- docker args ----------------
    def _server_args(self, image_set: dict) -> list[str]:
        show_window = self.window_var.get()
        seed = self.seed_var.get().strip()
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
            # 激光必须开启：导航避障依赖 /scan
            "-e", "SUPERMARKET_ENABLE_RENDER=1",
            "-e", "SUPERMARKET_ENABLE_LIDAR=1",
            "-e", "SUPERMARKET_USE_GS=1",
            "-e", "SUPERMARKET_RANDOMIZE=1",
            "-e", f"SUPERMARKET_TASKS={DEV_TASKS}",
            "-e",
            f"SUPERMARKET_RANDOMIZE_OBSTACLES="
            f"{1 if self.obstacle_var.get() else 0}",
        ]
        if seed:
            args += ["-e", f"SUPERMARKET_SEED={seed}"]
        args += [
            "-e", f"TORCH_EXTENSIONS_DIR={TORCH_CACHE}",
        ]
        if image_set["mount_server"]:
            args += [
                "-v", f"{REPO_ROOT}:/workspace/supermarket_sorting_task",
            ]
        args += [
            "-v", "supermarket_sorting_cache:/root/.cache",
            image_set["server"],
            "bash", "-lc",
            "cd /workspace/supermarket_sorting_task && "
            "source /opt/ros/humble/setup.bash && "
            "python3 examples/supermarket_sorting/"
            "supermarket_sorting_server.py",
        ]
        return args

    def _client_args(self, image_set: dict) -> list[str]:
        args = [
            "docker", "run", "--rm", "-d",
            "--name", CLIENT_NAME,
            "--gpus", "all", "--network", "host", "--ipc", "host",
            "-e", f"ROS_DOMAIN_ID={ROS_DOMAIN_ID}",
            "-e", f"RMW_IMPLEMENTATION={RMW}",
            "-e", "YOLO_CONFIG_DIR=/tmp/Ultralytics",
            # 官方 client 镜像缺 discoverse，指向挂载的仓库
            "-e", "PYTHONPATH=/workspace/baseline",
            "-e", f"SUPERMARKET_MAX_SCAN_CYCLES={int(self.cycles_var.get())}",
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
            "-v", f"{REPO_ROOT}:/workspace/baseline:ro",
            "-v", "supermarket_sorting_cache:/root/.cache",
            image_set["client"],
            "bash", "-lc",
            "cd /workspace/baseline && "
            + ("SUPERMARKET_SHOW=1 " if self.show_yolo_var.get() else "")
            + "./scripts/run_baseline.sh",
        ]
        return args

    # ---------------- actions ----------------
    def start_all(self):
        image_set = self._image_set()
        if not self._check_image(image_set["server"]) or \
                not self._check_image(image_set["client"]):
            messagebox.showerror(
                "镜像缺失",
                f"请先拉取/构建镜像:\n  {image_set['server']}\n"
                f"  {image_set['client']}")
            return
        # 清理同名残留容器
        for name in (CLIENT_NAME, SERVER_NAME):
            if self._docker_running(name):
                run_cmd(["docker", "rm", "-f", name])
        try:
            client_args = self._client_args(image_set)
        except ValueError as exc:
            messagebox.showerror("参数错误", str(exc))
            return

        self._append_log(self.server_text, "正在启动 Server ...\n")
        code, _, err = run_cmd(self._server_args(image_set))
        if code != 0:
            messagebox.showerror("Server 启动失败", err.strip())
            return
        self._append_log(self.server_text, "Server 容器已创建，等待仿真启动...\n")

        self._append_log(self.client_text, "正在启动 Client ...\n")
        code, _, err = run_cmd(client_args)
        if code != 0:
            messagebox.showerror("Client 启动失败", err.strip())
            return
        self._append_log(self.client_text, "Client 容器已创建...\n")
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
        self._append_log(self.client_text, "Client 已停止。\n")

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
        if self._poll_job is not None:
            self.root.after_cancel(self._poll_job)
        self.root.destroy()


def main():
    root = tk.Tk()
    LauncherApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
