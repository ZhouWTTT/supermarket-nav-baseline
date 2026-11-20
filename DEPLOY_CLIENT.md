# 超市分拣客户端 — 部署与运行说明

本文档面向拿到 `supermarket_sorting_client_ujs.tar` 的部署人员，按顺序照做即可把
客户端跑起来。客户端只负责“接收主办方 Server 下发的订单 → 抓取 → 配送 → 返航到
起点”，不含 Server、不含图形界面。

## 1. 交付物

| 文件 | 说明 |
| --- | --- |
| `supermarket_sorting_client_ujs.tar` | 客户端镜像，导入后标签为 `supermarket_sorting_client_ujs:latest` |
| `DEPLOY_CLIENT.md` | 本说明文档 |

镜像与归档信息（用于校验传输完整性）：

- 归档大小：约 12 GB
- 归档 SHA256：`3cd5f10705e8a6dbf643d69d7364a49e4c88f19a497fb3b00ea35798c74d2e99`
- 导入后镜像 ID：`sha256:817ceadd997b49df17c48e534871aedda4c0d9ea4d4122946d976ba1b3b26476`
- 导入后镜像大小：约 24.5 GB（解包后占用磁盘）
- 构建时间：2026-09-13 19:28 (+08:00)

校验示例：

```bash
sha256sum supermarket_sorting_client_ujs.tar
# 应输出上面那串 SHA256
```

## 2. 镜像内容

- 基础镜像：官方 client 镜像（ROS 2 Humble / Python 3.10 / CUDA 12.8）
- 项目代码位置：`/workspace/baseline`（工作目录，即本项目仓库内容）
- 默认入口：ENTRYPOINT 负责 `source /opt/ros/humble/setup.bash`，CMD 为
  `./scripts/run_baseline.sh`，所以直接 `docker run` 镜像即可开始运行
- 已内置 YOLO 权重：`examples/supermarket_sorting/perception/checkpoints/best.pt`
- 镜像内默认环境变量：
  `ROS_DOMAIN_ID=99`、`RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`、
  `PYTHONPATH=/workspace/baseline`、`YOLO_CONFIG_DIR=/tmp/Ultralytics`
- 不含：仿真 Server（由主办方提供）、任何 GUI / 显示窗口

## 3. 运行环境要求

- Linux（推荐 Ubuntu 22.04 及以上），Docker 24+
- NVIDIA GPU 及驱动，且已安装 `nvidia-container-toolkit`（支持 `--gpus all`）
- 与主办方 Server 位于同一台机器或同一网段，能够互通 DDS 组播
- 磁盘：镜像解包约 25 GB，建议预留 ≥ 40 GB

环境自检：

```bash
docker --version
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi
```

第二条能打印出 GPU 信息，说明 GPU 直通可用。

## 4. 导入镜像

```bash
docker load -i supermarket_sorting_client_ujs.tar
docker images | grep supermarket_sorting_client_ujs
```

看到 `supermarket_sorting_client_ujs  latest  ...  24.5GB` 即导入成功。

## 5. 运行客户端

### 5.1 与主办方 Server 对接（推荐）

Server 已在本机/同网段运行后，直接启动客户端：

```bash
docker run --rm -it --gpus all --network host --ipc host supermarket_sorting_client_ujs:latest
```

镜像自带的 ENTRYPOINT/CMD 会直接执行 `./scripts/run_baseline.sh`，无需再写命令。

如需指定 ROS 域或增加参数，用 `-e` 覆盖即可：

```bash
docker run --rm -it --gpus all --network host --ipc host -e ROS_DOMAIN_ID=99 -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp supermarket_sorting_client_ujs:latest
```

### 5.2 后台运行并看日志

```bash
docker run -d --name supermarket_sorting_client --gpus all --network host --ipc host supermarket_sorting_client_ujs:latest
docker logs -f supermarket_sorting_client
```

### 5.3 挂载源码目录运行（开发调试用）

用宿主机代码覆盖镜像内代码，改完即时生效：

```bash
docker run --rm -it --gpus all --network host --ipc host -v /home/wxj/888:/workspace/baseline:ro supermarket_sorting_client_ujs:latest
```

## 6. 运行结果与产物

运行期间的关键日志（正常流程）：

```
competition runner ready; waiting for transient task on /supermarket_sorting/task
accepted task run=run_xxxx count=5 kinds=[...]
starting order id=... kind=...
[flow] PLACE COMPLETE — ... delivered; table cleared ...
[return-start] final delivery complete; returning to start goal=(1.920,-3.170,90deg)   ← 最后一单
[return-supervisor] completion checkpoint accepted; independent return phase started    ← 返航独立计时
[return-start] reached start pose base=(...)
match finished reason=... delivered=5/5 failed=0 ...
[order-final-summary] count=5
[order-final] {"order_id":...,"kind":...,"status":"delivered",...}
```

运行产物默认写在**容器内** `/tmp/supermarket_competition/<run_prefix>/`
（`memory_matrix.json`、`summary.json`、`worker_*.json`、运行日志）。若要落盘到宿主机：

```bash
-v /home/wxj/888/logs/competition_runner:/tmp/supermarket_competition
```

另外每个进程的原始日志会写到容器内 `/workspace/baseline/logs/`
（如 `competition_runner_<时间>.log`、`persistent_perception_<时间>.log`、`worker_<时间>.log`）。
镜像内该目录可写；若用 `-v ...:/workspace/baseline:ro` 的方式挂载源码目录，日志会留在容器可写层。

## 7. 客户端可配置环境变量

在 `docker run` 里用 `-e 名称=值` 传入，全部可选：

| 变量 | 默认值 | 含义 |
| --- | --- | --- |
| `ROS_DOMAIN_ID` | `99` | 必须与 Server 一致 |
| `RMW_IMPLEMENTATION` | `rmw_cyclonedds_cpp` | 必须与 Server 一致 |
| `SUPERMARKET_BASELINE_WEIGHTS` | 内置 `best.pt` | 自定义 YOLO 权重路径 |
| `SUPERMARKET_DEVICE` | `cuda` | `cpu` 可强制 CPU 推理 |
| `SUPERMARKET_YOLO_CONFIDENCE` | `0.90` | YOLO 置信度下限 |
| `SUPERMARKET_INFERENCE_HZ` | `8` | 推理频率上限 |
| `SUPERMARKET_MAX_SCAN_CYCLES` | `2` | 每单货架扫描轮数上限 |
| `SUPERMARKET_MAX_ATTEMPTS` | `2` | 每单尝试次数上限 |
| `SUPERMARKET_MEMORY_CONFIRMATIONS` | `3` | 记忆矩阵复用前需要的确认次数 |
| `SUPERMARKET_MEMORY_CONFIDENCE` | `0.90` | 记忆矩阵置信度阈值 |
| `SUPERMARKET_GRAB_POLICY` | `nearest` | 订单选择策略 |
| `SUPERMARKET_ORDER_TIMEOUT` | `300` | 单订单超时（秒），0 关闭 |
| `SUPERMARKET_RETURN_TIMEOUT` | `180` | **最后一单交付后返航到起点的独立超时（秒）**，0 关闭 |
| `SUPERMARKET_MATCH_TIMEOUT` | `3600` | 整场软超时（秒） |
| `SUPERMARKET_TARGET_TIME` | `400` | 性能目标时间，仅用于统计 |
| `SUPERMARKET_RECORD_EVERYWHERE` | `1` | 全程记录记忆矩阵 |
| `SUPERMARKET_DYNAMIC_DIRECT` | `1` | 记忆矩阵直达抓取 |
| `SUPERMARKET_PERCEPTION_ALWAYS_ON` | `1` | 常开感知 |
| `SUPERMARKET_CLOSE_RECHECK` | `1` | 抓取前近距复核 |
| `SUPERMARKET_PATH_MEMORY` | `1` | 路径记忆 |
| `SUPERMARKET_PATH_MEMORY_FILE` | `/root/.cache/supermarket_path_memory.json` | 路径记忆落盘位置 |

示例：强制 CPU、放宽超时、把运行日志落到宿主机。

```bash
docker run --rm -it --gpus all --network host --ipc host -e SUPERMARKET_DEVICE=cpu -e SUPERMARKET_ORDER_TIMEOUT=600 -e SUPERMARKET_RETURN_TIMEOUT=300 -v /home/wxj/888/logs/competition_runner:/tmp/supermarket_competition supermarket_sorting_client_ujs:latest
```

## 8. 与 Server 的对接要点

1. 客户端与 Server 必须使用**同一 `ROS_DOMAIN_ID`** 和**同一 RMW**，并共用 host 网络
   （`--network host`）。
2. **先启动 Server 并等它预热完成**（能持续发布 `/joint_states`、`/slamware_ros_sdk_server_node/odom`、
   `/scan`），再启动客户端。否则 worker 的 10 秒反馈看门狗会判定
   `fatal-match: robot feedback unavailable` 并结束本场。
3. 客户端会等待 `/supermarket_sorting/task` 上的瞬时任务消息；Server 未下发任务时会一直
   打印 `waiting for transient task`。
4. 连通性自检（容器内执行）：

```bash
docker exec -it <client容器名或ID> bash -lc 'source /opt/ros/humble/setup.bash && ros2 topic list'
```

应能看到 `/supermarket_sorting/task`、`/joint_states`、`/head_camera/color/image_raw` 等话题。

## 9. 常见问题

| 现象 | 原因与处理 |
| --- | --- |
| `ModuleNotFoundError: No module named 'rclpy'` | 在宿主机直接跑了 `python3`。本客户端必须在容器内运行；镜像 ENTRYPOINT 已自动 source ROS |
| `docker: 'docker run' requires at least 1 argument` / `--name：未找到命令` | 多行命令粘贴时行尾 `\` 丢失。请把命令写成**单行**，或存成脚本后 `bash run.sh` |
| `docker: invalid reference format` | 命令里用了空变量作为镜像名（如 `"$SERVER"` 未赋值）。直接用完整镜像名 |
| 提示无 CUDA / 找不到设备 | 加 `--gpus all`；并确认已装 `nvidia-container-toolkit`（见第 3 节自检） |
| 一直 `waiting for transient task` | Server 未启动、`ROS_DOMAIN_ID` 不一致、或未使用 `--network host` |
| `fatal-match: robot feedback unavailable` | Server 还没预热完就启动了客户端。等 `/joint_states` 有数据后再启动客户端 |
| 看不到仿真画面 | 正常现象：客户端是无头的。仿真窗口由 Server 侧显示 |
| 想复现同一局 | 由 Server 的 `SUPERMARKET_SEED` / `SUPERMARKET_OBSTACLE_SEED` / `SUPERMARKET_TASKS` 决定，客户端侧无法指定 |

## 10. 停止与卸载

```bash
docker stop supermarket_sorting_client            # 停止后台容器
docker rm -f supermarket_sorting_client           # 删除容器
docker rmi supermarket_sorting_client_ujs:latest  # 卸载镜像
```
