# 超市分拣项目工作交接

> 生成时间：2026-08-15（Asia/Shanghai）
> 工作目录：`/home/zwt/baseline_dev`
> 用途：让新的 Codex 对话先阅读本文，再继续导航、识别、抓取、放置和多订单性能优化。

## 新对话开始时先做什么

1. 完整阅读本文。
2. 运行只读命令 `git status -sb` 和 `git log -5 --oneline`，确认用户是否在本文生成后继续改过代码。
3. 不要重置、覆盖或清理用户的新修改。
4. 用户没有明确要求时，不提交、不推送，也不启动仿真测试。

建议用户在新对话中的第一句话直接使用：

```text
请先阅读 /home/zwt/baseline_dev/CODEX_HANDOFF.md，然后检查当前 git 状态，继续这个项目的工作。
```

## 用户明确约束

- 正式比赛的 Server 镜像与 Client 隔离。Client 方案不得依赖或修改本地
  `examples/supermarket_sorting/supermarket_sorting_server.py`。
- `examples/supermarket_sorting/referee.py` 是废弃文件，不要分析或修改。
- 用户认为 `perception/kele_detect.py` 是废弃的独立代码，不应作为优化对象。
  但当前正式链路仍通过 `KeleDetectNode` 导入它；若正式依赖出现问题，需要先向
  用户说明这个事实，不要悄悄删除。
- 不允许一次抓取多件商品。正式流程必须保持“一次 worker / 一次行程 / 一件商品”。
- GUI 多单调试只保留 `gui_snapshot_pick.py -> snapshot_pick_client.py`；正式比赛
  入口仍是 `scripts/run_baseline.sh -> competition_runner.py`，不要混用两种入口。
- 用户之前明确要求不进行测试。默认只做代码审查、日志分析、AST/语法和
  `git diff --check` 等静态检查；需要启动仿真或正式流程时，等用户明确要求。
- 优化速度时必须同时考虑鲁棒性，不能仅提高上限而绕开激光急停、轨迹预测、
  桌面安全或抓取稳定性。
- 用户通常希望修改后先保留在本地；只有明确要求时才 commit/push。
- 不在文档、提交或日志中记录聊天里出现过的 sudo 密码或其他凭据。

## 当前 Git 基线

本文生成前的状态：

- 分支：`master`
- HEAD：`10b1957b49bb973e9ef21ee3f5cdac86ac196656`
- 提交说明：`导航、抓取、放置优化`
- `master` 与 `origin/master` 同步。
- 工作区在创建本文前是干净的；本文自身是新建但尚未提交的交接文件。

近期关键提交：

| 提交 | 内容 |
|---|---|
| `10b1957` | 导航脱困、抓取顺序、夹持保持、持续感知、计时和流程收尾等综合优化 |
| `f09ab31` | 五个桌面放置槽位，按从内到外的顺序使用 |
| `c7602a2` | 导航速度调整和 runner 日志问题修复 |
| `e67e28f` | 合并 wxj v2 的扫描顺序及识别/抓取逻辑 |
| `aaac78c` | 合并路径记忆增强 |

当前可见分支包括 `master`、`temp`、`agent/competition-client-refactor`，以及若干
远程连续多单、路径记忆分支。`temp` 中较短的速度轨迹预测和障碍减速距离已经
整合过，不要再次整分支覆盖。

## 正式代码结构

| 文件 | 职责 |
|---|---|
| `scripts/run_baseline.sh` | 正式 Client 入口，设置参数并启动 runner |
| `competition_runner.py` | 接收 `/supermarket_sorting/task`，管理多订单、单件 worker、重试、槽位、计时与汇总 |
| `persistent_perception.py` | 跨订单复用 YOLO/ArUco，非扫描阶段暂停推理 |
| `integrated_nav_pick_place.py` | 单件完整外层流程：抓取、后退、恢复高度、送货导航、放置、离桌 |
| `yolo_aruco_shelf_pick.py` | 扫描、关联、机械臂 IK 和货架抓取状态机 |
| `supermarket_navigation.py` | 代价地图、A*、Pure Pursuit、激光安全、脱困和路径记忆接入 |
| `path_memory.py` | 重复路线缓存 |
| `competition_task.py` | 任务 JSON、订单状态和检测/ArUco 关联 |
| `run_log.py` | 将 stdout/stderr 和 ROS 日志落盘 |

正式调用关系：

```text
run_baseline.sh
  -> competition_runner.py
       -> persistent_perception.py（全场复用，只发布感知结果）
       -> integrated_nav_pick_place.py（每次只处理一件）
            -> yolo_aruco_shelf_pick.py（货架抓取）
            -> supermarket_navigation.py（导航）
            -> 桌面槽位放置
```

## 已完成的主要修改

### 1. 导航和狭窄通道脱困

- 导航器最大线速度为 `0.90 m/s`，最大角速度为 `2.5 rad/s`。
- 抓取父控制器的速度输出限制也已对齐到 `±0.90 m/s`、`±2.50 rad/s`，不会再被
  旧的 `0.4 m/s`、`1.4 rad/s` 二次截断。
- 近目标开放区域速度上限为 `0.35 m/s`。
- 动态障碍膨胀半径为 `0.18 m`；静态障碍膨胀为 `0.50 m`，没有另加软安全区。
- 恢复了车体宽度走廊式激光判定：正常前方走廊半宽 `0.24 m`，无条件硬停走廊
  半宽 `0.21 m`。
- 激光硬停距离 `0.32 m`，减速距离 `0.55 m`。
- 桌子整体保护半径 `0.55 m`，旋转保护半径按用户要求为 `0.50 m`。
- 修复了动态膨胀层反复自膨胀问题：原始激光点与膨胀结果分层保存。
- 修复 `arc_blocked` 后只停在原地的问题，加入安全直线后退恢复。
- 后退仍无法改变路径侧别时，会规划带侧向 waypoint 的脱困路径；最多尝试两次，
  避免原地“后退—重规划—回到同一路径”的死循环。
- 对重复旋转、无路径、轨迹阻塞等停止原因，清除速度滤波器残留，避免急停后仍
  向前滑行。
- 路径记忆命中后如果触发恢复，会使该缓存失效且不再把这条问题路线保存成成功
  路线。
- 送货桌面的不同槽位只允许复用目标偏差不超过 `0.08 m` 的附近缓存，防止把旧
  槽位路线错误套到新槽位。

路径记忆默认文件位于 Client 容器：

```text
/root/.cache/supermarket_path_memory.json
```

正式命令挂载了 `supermarket_sorting_cache:/root/.cache`，因此容器重启后仍可保留。

### 2. 抓取后的运动顺序

原问题：中层或下层夹住商品后先上升再后缩，会把商品顶到上方货架板。

当前顺序：

```text
中层/下层：夹紧 -> 保持抓取高度水平后缩 -> TCP 脱离货架 -> DONE
           -> 车体安全直线后退，同时恢复升降柱 -> 确认 0.006 m -> 才允许旋转导航

顶层：夹紧 -> 原有小幅抬升 -> 后缩
```

- 普通单臂商品、中层球形商品、双臂纸巾均已改为先同高度后缩。
- 中层苹果/橙子连原来的 `10 mm` 试抬也取消，夹持稳定后直接后缩。
- 顶层上方没有货架板，所以保留原来的抬升清障动作。
- 每次抓取完成后的运输升降柱目标固定为 `0.006 m`；没有达到反馈容差前不进入
  送货旋转。

### 3. 运输夹爪保持

夹爪接口是位置命令，不是直接的力矩/夹持力命令；数值越小越闭合。

当前运输目标：

| 商品类型 | 抓取阶段目标 | 退出货架后的运输目标 |
|---|---:|---:|
| 苹果、橙子 | `0.08` | `0.06` |
| 三明治 | `0.16` | `0.12` |
| 其他普通单臂商品 | `0.00` | `0.00` |
| 双臂纸巾 | `0.00` | `0.00` |

- 苹果、橙子最初曾调到 `0.04`，用户认为太激进，最终改为 `0.06`。
- 从后退、恢复高度、导航旋转，到桌面定位和下降完成，代码每个控制周期都会重新
  保持运输夹爪命令。
- 只有放置状态机完成位置/高度校验并进入合法释放阶段，夹爪保持才停止。
- 如果仍然脱手，先查看日志中的 `grip_command` 和 `measured_grip`；不要继续盲目
  减小位置命令。真实力矩上限由 Server/执行器控制，Client 不能直接突破。

### 4. 桌面放置

五个确定性槽位，按从桌内侧到外侧使用：

```python
(-2.20, -3.45)  # 1，最深，内左
(-1.94, -3.43)  # 2，内中
(-1.68, -3.41)  # 3，内右
(-2.07, -3.29)  # 4，外左
(-1.81, -3.27)  # 5，最外/最近
```

- 三个订单直接使用前 3 个槽位，不会越界；五个订单使用全部槽位。
- 槽位仅在成功送达后消耗，失败重试仍使用原槽位，不会留下空洞。
- 最终靠桌速度为 `0.12 m/s`，并根据槽位深度提前停止，不再让所有槽位固定前进
  相同的 `0.20 m`。
- 离开桌子的后退速度为 `0.30 m/s`，抓取后的直线后退也是 `0.30 m/s`。
- 放置完成后，两条机械臂和两个夹爪都恢复到固定初始姿态，同时安全后退离桌。

已经静态审查过释放顺序，不存在“刚进入 place 就主动松爪”的分支。单臂流程为：

```text
靠近桌子
-> 求解并到达分配槽位上方
-> 校验 TCP 位于桌面且命中自己的槽位
-> 保持夹紧并垂直下降
-> 再次校验桌面范围、槽位 XY、释放 Z
-> stage 2 才发送 GRIP_OPEN
-> 回收双臂并倒车离桌
```

双臂纸巾流程同样在桌面范围、槽位 XY 和释放高度全部通过后才打开。

### 5. 多订单、持续感知和计时

- runner 会处理任务消息中的全部订单，目前桌面槽位容量为 5。
- 每个 worker 最终只能选择一个当前可见的 pending 商品，仍符合单次只能取一件
  的规则；候选类别只是允许扫描时选更合适的一件，不会一次抓多件。
- YOLO/ArUco 感知进程跨订单复用，扫描状态才启用，导航和机械臂运动期间暂停，
  避免重复加载模型并减少与 Server GPU 渲染竞争。
- runner 会利用已确认的跨订单 inventory 给下一个 worker 提供扫描 X/Z 提示。
- 增加每次尝试、每个订单、抓取状态、外层流程和行驶距离计时。
- 汇总写入 `/tmp/supermarket_competition/<run_prefix>/summary.json`。
- `--target-time 400` 只用于汇总是否达标，不会在 400 秒时强制停车。
- 修复 runner 结束时在 timer callback 内调用 `rclpy.shutdown()` 导致进程残留的问题。
- 修复同一 rclpy logger 调用点动态切换 info/error 引发
  `ValueError: Logger severity cannot be changed between calls.` 的问题。

## 最近一次已知运行结果

历史三单正式运行：

- run prefix：`run_69e92cbd1d30`
- 结果：`3/3 delivered`，`0 failed`，每单一次尝试。
- 总耗时：`785.177 s`，没有达到 `400 s` 目标。
- 各单约：
  - `heweidao`：`251.0 s`
  - `sanmingzhi`：`285.4 s`
  - `chengzi`：`242.6 s`
- 第二单导航在约 `(-1.07, 0.60)` 处经历约 60 秒激光停止/后退/侧向脱困，激光
  距离约 `0.30~0.32 m`。
- 开阔区域指令能达到 `0.90 m/s`，但当时里程计实测通常只有约
  `0.12~0.15 m/s`，Server 仿真实时率可能是主要耗时瓶颈。

重要：这次三单运行发生在最终 `10b1957` 的部分修复生效之前。它证明旧工作树能
完成三单，但不能视为当前 HEAD 的完整回归验证。当前抓取后缩顺序、夹爪 `0.06`
保持、runner 干净退出和最新脱困逻辑仍需要下一次用户授权的正式运行验证。

## 当前仍需关注的问题

1. **狭窄通道稳定性**：曾出现明明可直走或可后退，却反复旋转/停车。当前已经
   加入直线后退和侧向 waypoint，但需要用新日志确认是否真正跳出循环。
2. **商品运输脱手**：当前先以较温和的球形商品 `0.06` 运输夹持验证。若仍掉落，
   应区分“夹爪命令被覆盖”“实测夹爪张开”“机械臂加速度造成滑落”，不要直接
   继续加力。
3. **放置开始掉落**：代码释放顺序正确。若现象仍存在，优先检查商品是否在快速
   伸臂过程中已经物理滑落，并考虑只降低载货伸臂加速度，而不是降低全局导航速度。
4. **400 秒目标**：旧三单已用 785 秒，五单按旧平均时间会远超 400 秒。持续感知、
   扫描提示和缩短低速交接已加入，但 Server 实时率可能限制收益。必须先取得当前
   HEAD 的分阶段计时再决定继续优化哪里。
5. **路径记忆**：缓存并非每条路线都能命中；必须从日志同时确认 `cache_hit: true`
   和 `cached_path_active: true`。触发恢复的缓存现在会主动失效，这是预期行为。

## 正式启动方式

### Server（组织方镜像，Client 不得依赖本地 Server 源码）

三单使用 `TASK_COUNT=3`，五单只需改成 `TASK_COUNT=5`，Client 无需改代码：

```bash
echo "${DISPLAY}"
TASK_COUNT=3
TASKS="$(shuf -i 1-45 -n "${TASK_COUNT}" |
  awk '{printf "%sproduct_%03d", sep, $1; sep=","} END {print ""}')"

sudo docker rm -f supermarket_sorting_server 2>/dev/null || true
xhost +local:docker

sudo docker run --rm -d \
  --runtime=nvidia \
  --network host \
  --ipc host \
  --name supermarket_sorting_server \
  -e "DISPLAY=${DISPLAY}" \
  -e MUJOCO_GL=glfw \
  -e SUPERMARKET_HEADLESS=0 \
  -e SUPERMARKET_ENABLE_RENDER=1 \
  -e SUPERMARKET_USE_GS=1 \
  -e "SUPERMARKET_TASKS=${TASKS}" \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v supermarket_sorting_cache:/root/.cache \
  supermarket_sorting:server \
  bash -lc '
    source /opt/ros/humble/setup.bash &&
    cd /workspace/supermarket_sorting_task &&
    python3 examples/supermarket_sorting/supermarket_sorting_server.py
  '
```

### 正式 Client

```bash
sudo docker rm -f supermarket_sorting_client 2>/dev/null || true

sudo docker run --rm -d \
  --runtime=nvidia \
  --network host \
  --ipc host \
  --name supermarket_sorting_client \
  -e ROS_DOMAIN_ID=99 \
  -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
  -e SUPERMARKET_ORDER_TIMEOUT=0 \
  -e SUPERMARKET_MATCH_TIMEOUT=3600 \
  -e SUPERMARKET_TARGET_TIME=400 \
  -e YOLO_CONFIG_DIR=/tmp/Ultralytics \
  -v /home/zwt/baseline_dev:/workspace/baseline:ro \
  -v supermarket_sorting_cache:/root/.cache \
  supermarket_sorting:client \
  bash -lc '
    cd /workspace/baseline &&
    ./scripts/run_baseline.sh
  '
```

当前启动入口没有改变，仍是 `./scripts/run_baseline.sh`。新增参数都已有默认值：

- `SUPERMARKET_INFERENCE_HZ`，默认 `12`
- `SUPERMARKET_TARGET_TIME`，默认 `400`
- `SUPERMARKET_MAX_SCAN_CYCLES`，默认 `2`
- `SUPERMARKET_MAX_ATTEMPTS`，默认 `2`
- `SUPERMARKET_INVENTORY_CONFIRMATIONS`，默认 `3`
- `SUPERMARKET_ORDER_TIMEOUT`，默认 `0`（关闭单订单超时）
- `SUPERMARKET_MATCH_TIMEOUT`，脚本默认 `570`，上面的正式命令覆盖为 `3600`

## 日志查看

```bash
sudo docker logs --tail 300 supermarket_sorting_client
sudo docker logs -f supermarket_sorting_client

sudo docker exec supermarket_sorting_client \
  bash -lc 'find /tmp/supermarket_competition -name summary.json -printf "%T@ %p\n" | sort -nr | head'
```

只读挂载下，`run_log.py` 如果不能写仓库 `logs/`，会退回 `/tmp`。检查关键行为：

```bash
sudo docker logs supermarket_sorting_client 2>&1 | \
  grep -E 'middle-sphere-retreat|retreat_at_grasp_height|grip-hold|low pose verified|lateral_escape_replan|path_memory_runtime|order-timing|match finished'
```

新对话分析日志时，应先给出时间线：当前订单、`flow_phase`、抓取 `state`、
`stop_reason`、激光前/后间距、实际 `v/w`、夹爪目标/反馈，再判断根因。不要只凭一条
“机器人不动”日志改参数。

## 下一步建议

1. 用户授权运行时，先以当前 `master@10b1957` 做一次三单正式流程，确认没有功能
   回归，再尝试五单。
2. 重点验证日志顺序：同高度后缩发生在升降柱恢复之前；旋转前 slide 已到
   `0.006 m`；导航到桌期间夹爪保持为苹果/橙子 `0.06`；只有 low pose verified
   后才出现释放。
3. 若再次在狭窄通道卡住，先确认是否出现 `lateral_escape_replan`，以及侧向路径是否
   被普通重规划立即覆盖。
4. 使用新的 `summary.json` 分阶段耗时确定 400 秒瓶颈，优先处理扫描、抓取或仿真
   实时率中占比最大的部分，避免继续无依据地提高全局速度。
5. 未经用户明确要求，不修改废弃文件、不提交、不推送。
