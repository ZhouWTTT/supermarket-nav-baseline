# 超市分拣比赛代码健壮性与性能审查报告

> 审查对象：`/home/zwt/baseline_dev`（DG-202606 超市分拣任务 Client 正式链路）
> 审查范围：`scripts/run_baseline.sh → competition_runner.py → persistent_perception.py →
> integrated_nav_pick_place.py → yolo_aruco_shelf_pick.py / supermarket_navigation.py /
> memory_matrix.py / path_memory.py / competition_task.py / run_log.py`，以及
> discoverse 框架接口、`arm_kdl.py`/`mmk2_kdl.py`、perception 后端。
> 方法：全库静态阅读 + 行号级证据核对 + 5 路并行专项深审（导航 / 感知 /
> 记忆系统 / discoverse 框架 / 抓取状态机，均已合并；未运行仿真，未修改任何代码）。
> 日期：2026-08（与 CODEX_HANDOFF.md 基线一致，HEAD 假设为 `10b1957` 之后的工作区）

---

## 1. 总体结论

正式链路是**一个设计意图清晰、防御性意识很强、但复杂度已明显超载**的单体式
状态机系统。健壮性整体处于"**每一条已知失败路径都有人处理过，但未知路径缺乏
统一兜底**"的状态：

- **优点**：抓取/放置/导航各阶段都有超时、反馈门限与可诊断日志；结果文件原子写、
  进程级隔离（runner/worker/perception 三者分离）、激光陈旧自动停车、急停直发
  `/cmd_vel`、`rclpy.shutdown` 位置等工程细节都处理得比一般比赛代码好；
- **主要风险**：① 传感器（激光/里程计/关节）长期失活时多个阶段**没有硬超时**
  （GO_SCAN/ALIGN/REVISIT/中层 LIFT/restore_height/激光等待），与
  `SUPERMARKET_ORDER_TIMEOUT=0`（官方命令）、`--match-timeout 3600` 组合后，
  单点故障可把整场比赛挂死一小时；② 大量常量表在多个文件间**重复定义且已漂移**
  （`PRODUCT_CENTER_ABOVE_MARKER_M` 5/9 类不一致、`TOP_SHELF_Z_M` 与
  `LEVEL_Z_L3_MIN` 差 1 cm 等），是静默错误的高发源；③ 50 Hz 控制回调与
  12 Hz 感知回调、0.5 s 文件写、A* 重规划在同一进程内**耦合调度**，控制节拍
  存在抖动风险；④ 导航恢复链（后退→侧向脱困→重规划）已有计数上限，但预算
  耗尽后没有"放弃"信号、坏缓存路径可能被回写强化，且存在 `_rotate_with_unstick`
  无限重置与 `scan_unlocked_*` 字典竞态两个已坐实的循环/竞态隐患。

**综合评价（10 分制）**：健壮性 7/10，运行时鲁棒性 6.5/10，代码结构 6/10，
性能 6.5/10，可维护性 6/10。**代码可以完成 3 单，但离"5 单 400 秒 + 任意
故障不挂死"的正式目标还有明显距离。**

---

## 2. 架构与数据流

```text
scripts/run_baseline.sh
  └─ competition_runner.py（ROS 节点，0.2 s tick；MultiThreadedExecutor×2 线程）
       ├─ persistent_perception.py（常驻子进程：YOLO+ArUco，12 Hz 限频，
       │    /supermarket_sorting/perception_enable 门控推理）
       ├─ 每单一个 worker 子进程 integrated_nav_pick_place.py
       │    ├─ yolo_aruco_shelf_pick.py（抓取 FSM：GO_SCAN→SCAN→…→DONE/ABORT）
       │    ├─ supermarket_navigation.py（代价地图+A*+纯跟踪+激光安全+脱困）
       │    └─ 桌面放置子状态机（place_stage 0..5）
       ├─ memory_matrix.py MemoryMatrixTracker（同进程第二个节点，0.5 s 写盘）
       └─ path_memory.py（导航路线缓存，JSON 文件）
```

进程间通信全部走 ROS2 话题 + JSON 文件（`memory_matrix.json`、worker 结果文件、
`*.ready` 哨兵），无共享内存。这个"进程隔离 + 原子写"的架构本身是稳健的，
代价是：**没有任何跨进程的故障心跳**（worker 死了 runner 靠 `poll()` 发现；
perception 死了靠 ready 文件 + 30 s 超时发现——均不是实时心跳）。

---

## 3. 健壮性问题清单（按严重度排序）

### 🔴 P0-1 传感器长期失活没有硬超时，可挂死整场比赛

**证据**：
- `integrated_nav_pick_place.py:989-996`（drive_to 激光陈旧分支）、`:744-750`
  （`_route_leg_tick` 激光陈旧分支）都只 `return False`，**没有任何累计超时**；
- `integrated_nav_pick_place.py:2680-2694`（tick 顶部）在 odom/joints/laser
  陈旧时直发零速并 return——流相位 tick 全部停摆，属于"安全但无限等待"；
- `_restore_height_tick`（:1101-1123）在 `TRANSIT_SLIDE_TIMEOUT_S=8s` 后**只记
  警告、继续等待**，没有失败出口；
- `competition_runner.py:809` `--order-timeout` 默认 `0.0`（禁用），
  CODEX_HANDOFF 记录的官方命令 `--match-timeout 3600`（:373）。

**影响**：Server 重启、话题中断、关节控制器卡死等任何单一故障都会让当单 worker
无限等待，runner 不会杀掉它，整场比赛直到 3600 s 才被 match-timeout 收尾。
**建议**：
1. 给激光/里程计/关节陈旧各加硬超时（如 10~15 s），超时后按该相位可恢复的方式
   失败（导航相位→重试/换路；抓取相位→ABORT 并让 runner 重试）；
2. 正式命令启用 `--order-timeout 240`（同时把 worker 的 SIGTERM→SIGKILL 窗口
   从 3 s 提到 5 s，避免误杀正常收尾）；
3. `_restore_height_tick` 超时后不再干等，改为"以当前实测高度继续"并告警。

### 🔴 P0-2 关键常量表多处重复定义且已漂移（静默错误源）

**证据**（`PRODUCT_CENTER_ABOVE_MARKER_M` 两份定义数值不一致）：

| kind | competition_task.py:34 | yolo_aruco_shelf_pick.py:85 | 差 |
|---|---|---|---|
| heweidao | 0.0355 | 0.00 | **0.0355** |
| shupian | 0.054 | 0.040 | **0.014** |
| maidong | 0.104 | 0.034 | **0.070** |
| kele | 0.0715 | 0.0315 | **0.040** |
| kouxiangtang | 0.030 | 0.020 | **0.010** |

- 正式链路只用 `yolo_aruco_shelf_pick` 的表；`competition_task` 的表仅被
  `associate_detection_marker()`（:188）使用，而该函数**当前只有单测调用**
  （`tests/test_competition_task.py:98` 用 kele=0.0715 恰好通过）——测试在给
  死路径做回归，产生虚假安全感；
  > ✅ 2026-08 已处理：死路径单测（`test_inventory_association_uses_measured_geometry`、
  > `test_scheduler_optimizes_by_grasp_cost_not_source_order`）已从
  > `tests/test_competition_task.py` 删除，`test_failure_retries_then_moves_on`/
  > `test_success_and_summary` 已改为不依赖 `next_order`；`associate_detection_marker`
  > 与 `next_order` 现为零引用死代码，是否一并删除待用户确认；`competition_task.py`
  > 的 `PRODUCT_CENTER_ABOVE_MARKER_M` 表仍与正式链路不一致（单一来源合并未做）。
- `PRODUCT_HALF_HEIGHT_M` 在 `yolo_aruco_shelf_pick.py:102` 与
  `integrated_nav_pick_place.py:151` 各一份（当前数值恰好一致）；
- `GRASP_COST`（competition_task.py:22）与 runner 共享，无重复，但
  `DELIVERY_PLACE_SLOTS_XY`（integrated:136）与 CODEX_HANDOFF 中记录的槽位表
  一致，无第二份副本（良好）。

**建议**：以 `competition_task.py`（无 ROS 依赖、宿主机可测）为单一来源，
`yolo_aruco_shelf_pick.py`/`integrated_nav_pick_place.py` 改为 `import`；
短期至少加一致性断言测试（两个模块的 dict 相等）。

### 🟠 P1-1 50 Hz 控制回调与重计算耦合，控制节拍抖动

**证据**：
- `ShelfPickController.create_timer(0.02, self.tick)`（yolo_aruco_shelf_pick.py:993）
  与子类 tick 内 `nav.update` → `compute_velocity` 全同步执行；
- `compute_velocity` 每 0.4 s 跑一次 A*（`_replan_interval`，
  supermarket_navigation.py:796），失败时 `_try_plan_with_fallback` 再跑一次
  lidar-only 规划（:1160-1170），A* 是纯 Python heap 实现（:536-633），
  15,000 格地图上最坏可达几十 ms；
- 每个控制 tick 都执行 `_motion_is_free` 弧线预测（:1090），其内部逐采样点调用
  `raw_dynamic_clearance_world`（:496-502），对全部原始动态障碍点做 O(N) 距离
  计算（每次 9 步 × N 点 × numpy 分配）；
- yolo_cb（:1024-1107）在 ROS 回调里做解码+关联+聚簇+补拍累积（持 self.lock），
  单帧最坏 O(框数×码数)。

**影响**：同节点回调被 MutuallyExclusiveCallbackGroup 串行化，A*/感知回调慢时
`tick` 被推迟，`/cmd_vel` 发布出现抖动（50 Hz 变成 20~30 Hz），高速行驶中急停
响应延迟增大。实测 Server 实时率只有 0.12~0.15 m/s 时影响被掩盖，提速后放大。
**建议**：
1. 把 A* 重规划移出控制回调（在导航器内部用"上次规划结果 + 增量校验"，或放到
   独立线程/独立节点，控制线程只消费最新路径）；
2. `raw_dynamic_clearance_world` 改为在 `_rebuild_dynamic` 时用
   `scipy.ndimage.distance_transform_edt` 预计算一张动态距离场（每帧 1 次 O(网格)
   C 实现），查询变 O(1)，同时消除每 tick 的 numpy 小分配；
3. yolo_cb 只做入队，关联计算移到 tick 里按帧节流执行（或保持现状但确认
   单帧耗时 < 10 ms——当前是未知数，建议加计时日志）。

### 🟠 P1-2 导航恢复链存在"无进展振荡"窗口（需实跑确认）

**证据**：`_maybe_start_reverse_recovery`（supermarket_navigation.py:1269-1348）
在 `_reverse_recovery_attempts >= 2` 且 `_lateral_escape_attempts >= 2` 后返回
False，此后进入"每 0.4 s 重规划 → 停 → 再重规划"的纯等待；`recovery_exhausted()`
（:2011）被 integrated 的 `_route_leg_tick` 用作 35 s 停滞判失败的条件
（integrated:805-809），**但只有 `no_path`/`stuck_no_path` 才触发失败判定**——
如果 stop_reason 是 `lidar_stop`/`arc_blocked` 且停滞 35 s，`_route_leg_tick`
不会判失败（:797-809 的 `no_path` 判定不覆盖），会一直等到
`ROUTE_LEG_HARD_TIMEOUT_S=150s` 才硬超时。150 s 单腿耗时可观。
**建议**：停滞判失败条件放宽到"停滞 && (no_path || recovery_exhausted ||
stop_reason in {lidar_stop, arc_blocked})"；或把硬超时从 150 s 降到 60 s。

### 🟡 P2-1 runner 与 tracker 共享 executor，`_task_cb` 中旧 worker 结果被静默丢弃

**证据**：`competition_runner.py:124-129`（新 run_prefix 到达时 `current_order=None`
+ 请求停止旧 worker）；`_finish_worker`（:527-622）中 `_resolve_worker_order`
（:624-645）在 `dispatch_order is None` 时返回 `(None, False)`，旧 worker 的
结果（哪怕是 delivered）被**静默丢弃且不记账**——与 CODEX_HANDOFF"跨 run 竞态"
待办一致。同 run 内无此问题（同节点回调串行）。
**影响**：仅发生在比赛中途 server 重启换 run_prefix 的场景，会丢失已完成交付的
计数；正常比赛（单 run）不受影响。建议：结果文件命名带 run_prefix，收尾时校验
不一致则丢弃并告警（与 CODEX_HANDOFF 待办一致）。

### 🟡 P2-2 worker 在 `--show` 模式与正式模式行为不一致

**证据**：integrated main()（:3001-3034）在 viewer 存在时把 executor 移到后台
线程，主线程 `viewer.show()` 轮询；正式模式 `executor.spin()` 主线程。
两种模式下 `tick` 内 `rclpy.shutdown()`（:2749）的调用上下文不同
（后台线程 vs 主线程），`ExternalShutdownException` 的抛出位置也不同。
runner 已经在注释中（:855-862）明确"不要在 timer 回调里调 rclpy.shutdown"，
但 worker 的 tick 仍这么做——目前能跑通（三单实证），属于**已知可用的
反模式**，一旦 executor 线程数/节点数变化可能复现 runner 当年的挂死。
**建议**：worker 也改成 `finished` 标志 + 主循环退出，与 runner 一致。

### 🟡 P2-3 viewer 跨线程读控制器状态

**证据**：`MainThreadResultViewer`（yolo_aruco_shelf_pick.py:5320-5407）的
`aruco_cb`/`image_cb` 在 viewer 节点线程执行，读取 `self.controller.target_marker_id`
`/state/grasp_profile_name()` 等（:5370-5394）；`show()` 主线程同样读取。
控制器状态由另一节点的 tick 线程写。CPython GIL 下属性读是原子的，但
`grasp_profile_name()` 等复合计算可能读到不一致快照——仅影响显示，不影响控制。
**建议**：低优先，给 viewer 加一个 `threading.Lock` 或只读快照。

### 🟡 P2-4 深度单位三套写法（CODEX_HANDOFF 已列，本次复核确认）

**证据**：`perception/kele_detect.py` 的 `patch_depth_m` 无条件 ×1e-3（假定 mm）；
`foreground_depth_estimate` 用启发式判定；`perception/backends.py:42`
`_safe_depth_m` 与 `aruco_detect.py` 的 `depth > 20.0` 判定各自为政。
**影响**：官方深度流单位/格式变化会静默错 1000 倍，导致放置高度错 1 m 级别。
**建议**：在相机信息回调处统一一次单位判定并缓存。

---

## 4. 死锁与卡死风险专项分析

### 4.1 线程锁死锁：未发现经典死锁

全链路锁清单：`yolo_aruco_shelf_pick.py:755 self.lock`（控制器回调内）、
`memory_matrix.py:318/620`（RLock）、`kele_detect.py:253/258`（两把锁）、
`MainThreadResultViewer.frame_lock`。经核查：
- 所有 `with lock:` 都是**单锁短临界区**，没有嵌套持锁再取第二把锁的路径，
  不存在锁序反转；
- 控制器节点默认 MutuallyExclusiveCallbackGroup，`self.lock` 实际不会被争用
  （同节点回调串行）——**锁是冗余的，但没有害处**；
- 跨节点共享仅有 viewer 读控制器（见 P2-3，GIL 下安全）。

**结论：没有死锁（deadlock），但存在多处"活锁/无限等待"（见 P0-1、P1-2）。**

### 4.2 状态机卡死点核查（逐个状态）

| 状态/相位 | 退出条件 | 卡死风险 |
|---|---|---|
| GO_SCAN（transit） | drive_to 返回 True；激光陈旧时无限等待 | 🔴 激光停 → P0-1 |
| restore_height | slide 反馈到容差；超时只告警 | 🔴 P0-1 |
| SCAN→REVISIT→回退 | `revisit_total_rounds < REVISIT_MAX_ROUNDS_PER_SCAN(4)`、`REVISIT_MAX_ROUNDS_PER_MARKER=1`、`max_scan_cycles`（runner 传 2） | 🟢 有上限 |
| DEPLOY | 收敛/软门限/硬超时（8 s）+ 基座前移重试（`GENERIC_DEPLOY_RETRY_MAX=2`） | 🟢 有上限 |
| CLOSE（generic） | `GENERIC_CLOSE_DWELL_S=6s` 或稳定窗 | 🟢 有上限 |
| LIFT/RETREAT | 超时后 TCP-clear 判定或 ABORT | 🟢 有上限 |
| place stage 0..4 | 各阶段硬超时（10~15 s）→ RuntimeError → worker 退出 | 🟢 有上限 |
| `_route_leg_tick` | 硬超时 150 s | 🟡 过长（P1-2） |
| 旋转看门狗 | `_rotation_loop_limit=1.25π` + 重规划 hold | 🟢 有上限 |

### 4.3 进程级"死锁"：`rclpy.shutdown` 反模式（见 P2-2）

---

## 5. 逻辑重复清单

| 重复项 | 位置 | 一致性 | 风险 |
|---|---|---|---|
| `PRODUCT_CENTER_ABOVE_MARKER_M` | competition_task.py:34 / yolo:85 | ❌ 漂移 | P0-2 |
| `PRODUCT_HALF_HEIGHT_M` | yolo:102 / integrated:151 | ✅ 一致 | 低 |
| `GRASP_COST` | competition_task:22（唯一） | ✅ | — |
| `SHELF_SCAN_X` / `SCAN_X` | memory_matrix:52 / yolo:184 | ✅ 一致（A..E 与站序 x） | 低 |
| `SHELF_CENTERS_X` / `SCAN_X` | supermarket_navigation:78（E=1.805）/ yolo:184（E=1.80） | ❌ 5 mm 偏差 | 中（M2） |
| `LEVEL_MARKER_Z` / `SHELF_SURFACE_Z_M` | memory_matrix:57 / yolo:540 | ⚠️ 1 mm 偏差（0.500/0.852/1.190 vs 0.499/0.851/1.189） | 中（M2） |
| `TOP_SHELF_Z_M` / `LEVEL_Z_L3_MIN` | yolo:302（1.10）/ memory_matrix:66（1.09） | ❌ 1 cm 偏差 | 中（W6） |
| `SLOT_BY_MARKER` 读 JSON | memory_matrix:127 / yolo `fixed_layout_by_marker` | 两处独立加载 | 中 |
| `wrap_to_pi`/`angdist` | supermarket_navigation:96 / yolo:627 / path_memory:13 | 三份实现 | 低 |
| `marker_below_yolo` / `associate_detection_marker` | yolo:647 / competition_task:188 | 相似算法两份 | 中（死路径） |
| 目标/候选排序 `GRASP_COST+attempts+source_index` | competition_task:127 `next_order` / runner `_candidate_kinds_for`:647 | 两份排序键略不同 | 低（runner 为权威） |
| 深度单位判定 | kele_detect / backends / aruco_detect | 三套 | P2-4 |

> 补充：`yolo_aruco_shelf_pick.py:1658`（west_column 判定）与 :2767 附近
> （middle_tissue_column_x）在正式链路读取 `retail_competition_layout.json`，
> 属 CODEX_HANDOFF 记录的"灰色使用"，建议改 `shelf_for_scan_x`/`SHELF_SCAN_X`
> 常量判定后删除 JSON 依赖（正式链路仅保留诊断开关使用）。

---

## 6. 运行时鲁棒性缺口汇总

1. **无跨进程心跳**：perception 30 s ready 超时（competition_runner.py:376-392）
   是启动期兜底，运行期死亡靠"worker 发布 0 条检测"间接暴露；
2. **worker 结果一致性**：delivered 判定只看 return_code==0 + 结果文件
   （:532-536）；若 worker 交付后、写文件前被杀，runner 记失败并重试同类商品，
   可能重复取货/空取（有 excluded_markers 兜底，二次失败即标记 failed）；
3. **run_log.py 的 fd tee**：`_tee_fd`（run_log.py:41-78）用管道+守护线程泵
   stdout/stderr；若日志文件句柄失效，泵线程吞异常继续——但**守护线程在进程
   exit 时可能丢尾部日志**，且两个 fd 各一个线程，高日志量下（本代码日志极多）
   有背压风险（管道写满会阻塞调用线程）；
4. **`_write_result`/`atomic_write_json` 都是 tmp+rename**：✅ 正确；
5. **worker 日志量**：`[place-joints]` 快照、每秒状态日志、每 2 s 导航日志——
   正式 run 的日志体积可观，且 run_log 泵线程参与拷贝，属次要性能项；
6. **`joint_cb` 无关节名过滤**：`self.joints` 字典整体重建，若某关节缺失，
   `initialize_commands` 会用 0.0 兜底——缺失关节被静默当 0 处理（
   yolo:1923-1935），建议对关键关节缺失告警。

---

## 7. 性能瓶颈与优化方案（按收益排序）

> 背景：三单 785 s（含 60 s 导航卡顿），目标 5 单 400 s。Server 仿真实时率
> 目前是最大外因（里程计实测 0.12~0.15 m/s），但客户端仍有多处可优化。

### 7.1 时间预算结构（从现有计时可得）

- 每单约 242~285 s：扫描/定位（含跨站导航）、抓取（含各种 dwell）、
  送货导航、放置（含 2 s 释放 dwell + 1 s 撤退 dwell + 0.35 s 基座稳定 +
  0.3 s XY 稳定 + 0.3 s 释放位稳定）。
- 显眼的固定开销（每单必付）：`PLACE_BASE_SETTLE_S=0.35`、
  `PLACE_XY_REFINE_SETTLE_S=0.30`、`PLACE_RELEASE_POSE_SETTLE_S=0.30`、
  `place_release_dwell_s=2.0`、`place_retreat_dwell_s=1.0`（合计 ~4 s/单，
  5 单 ~20 s）；抓取侧 `DUAL_TISSUE_CLAMP_DWELL_S=4.0`、
  `GENERIC_CLOSE_DWELL_S=6.0`（上限）、`LOWER_LIFT_DWELL_S=1.2`、
  `DUAL_TISSUE_LIFT_DWELL_S=2.5` 等。

### 7.2 优化方案清单

| # | 方案 | 预期收益 | 风险/前提 |
|---|---|---|---|
| S1 | **并行化感知与导航**：把 YOLO 关联/聚簇从回调移到 tick 节流执行，A* 重规划移出控制回调（或降 `_replan_interval` 0.4→0.6 s 并只在"路径失效/停滞"时强制重规划） | 控制节拍稳定，急停响应可预测 | 需实测 A*/关联耗时 |
| S2 | **动态距离场缓存**：`_rebuild_dynamic` 时用 `distance_transform_edt` 预计算动态层距离场，`raw_dynamic_clearance_world` O(N)→O(1) | 每 tick 省 ~9×N 距离计算与 numpy 分配 | 无行为变化，纯性能 |
| S3 | **减少固定 dwell**：`place_release_dwell_s 2.0→1.0`、`place_retreat_dwell_s 1.0→0.5`、`PLACE_BASE_SETTLE_S 0.35→0.2`（保留反馈门限，dwell 只是上限保险） | 每单省 ~2.5 s，5 单 ~12 s | 低风险（反馈门限仍在） |
| S4 | **扫描/抓取提速**：`SCAN_DWELL_S 0.6→0.45`、`REVISIT_FIRST_POSE_DWELL_S 2.5→1.5`（全扫 8 位姿 ×5 站，dwell 每减 0.15 s 约省 6 s/全扫）；`GENERIC_CLOSE_DWELL_S 6.0` 依赖稳定窗已基本不吃满，确认即可 | 每全扫省 5~10 s | 需确认关联样本数门槛仍能满足（DEPTH_TARGET_MIN_SAMPLES=5） |
| S5 | **导航速度**：开阔区指令 0.90 m/s 已到；瓶颈是 Server 实时率——**不要在客户端继续加码**；可做的是减少"低速交接"：`NAV_TRANSIT_GATE_M 0.10` 已实现，确认 ALIGN 段 `NAV_ALIGN_LINEAR_MIN_MPS=0.10` 是否吃满 | 视 Server 而定 | CODEX_HANDOFF 明确反对无依据提速 |
| S6 | **内存/启动**：`persistent_perception.py` 通过 import integrated_nav_pick_place 加载整套控制栈（MMK2FK/KDL/导航器），只为复用检测节点——建议抽公共检测依赖，去掉 MuJoCo shim 等无关导入 | 启动快、内存低、崩溃面小 | 中低风险，需重构 import |
| S7 | **日志降噪**：`[place-joints]`/`[grasp-snapshot]` 等 JSON 快照保留（诊断价值高），但 `last_status_log 1.0 s` 与 `[nav]` 3 s 日志可合并；`run_log` 泵线程写盘频率与 flush 策略检查 | 降低 IO 与日志体积 | 低 |
| S8 | **memory_matrix 写盘 0.5 s 全量 JSON**：改脏标记增量（已有 `_dirty`）或降频到 1 s | 省 runner 进程 IO | 低 |
| S9 | **多订单编排**：`place_slot` 按"已交付数"分配（runner:214-215）导致 5 单时槽位固定；若允许"就近槽位+距离排序"可省绕行，但**会改变已验证的放置逻辑，风险高，建议保持现状** | — | 不推荐 |
| S10 | **A* 路径复用**：路径记忆已覆盖 delivery trunk；可扩展到"货架站间转移"常用腿（E↔A 全扫），减少 0.4 s 周期 A* 的重复计算 | 每单省若干 10~30 ms 级重规划 | 已有基础（path_memory），低风险 |

### 7.3 需要实测后才能定的点

- 各相位真实耗时（`summary.json` 的 `flow_phase_elapsed_s`）中，扫描 vs 导航
  vs 放置的占比——决定 S3/S4 的优先级；
- 每单 `pick_state_elapsed_s` 中 SCAN/REVISIT/DEPLOY 的占比；
- A* 与 `try_association_locked` 的单次耗时（建议加一次性计时日志）。

---

## 8. 客观评价

### 8.1 做得好的地方

1. **进程隔离**：runner/worker/perception 分离，单 worker 崩溃不影响 runner
   继续调度；worker 结果文件原子写 + `_resolve_worker_order` 严格校验；
2. **安全优先**：激光陈旧/里程计陈旧立即直发零速；放置前多重几何校验
   （桌内、槽位、商品底部高度）才允许松爪；急停绕过平滑直接发布；
3. **失败可恢复**：`excluded_markers`/`excluded_slot_keys`/memory 消费机制
   让重试不会反复抓同一位置；扫描有 revisit 补拍、部署有基座前移重试、
   导航有反向脱困+侧向脱困+路径记忆失效；
4. **诊断完备**：几乎每个状态转换、超时、门限都有日志；`summary.json`、
   `timing_snapshot()` 让性能分析有数据基础；
5. **纯逻辑可测**：competition_task/memory_matrix/path_memory/navigation
   均无 ROS 依赖，测试覆盖不错（约 50 个单测）。

### 8.2 主要不足

1. **复杂度失控**：`yolo_aruco_shelf_pick.py` 5523 行 + `integrated_nav_pick_place.py`
   3107 行 + `supermarket_navigation.py` 2018 行，单类 `ShelfPickController` 承载
   感知关联、抓取规划、双臂控制、状态机——任何改动都难以回归验证；
2. **常量漂移已成事实**（P0-2），说明缺少"单一事实来源"纪律；
3. **超时哲学不统一**：有的阶段 8 s 硬超时，有的无限等待，有的 150 s——
   没有统一的"每相位最大耗时"预算表；
4. **测试覆盖缺口**：runner 编排、worker 主循环、感知节点、放置状态机
   均无自动化测试，全靠实跑日志；
5. **文档与代码的漂移**：CODEX_HANDOFF 记录的"放置 IK refs 顺序问题"在
   当前代码已修复（`refs = [measured, pregrasp, compact]`，integrated:1519），
   但连续性门限与 FK 路径校验仍未实现——文档待办与实际进度需要对齐。

---

## 9. 整改优先级建议

**第一批（安全/防挂死，改动小）**：
1. 传感器陈旧硬超时（P0-1）：10~15 s 无激光/无里程计/无关节 → 按相位失败；
   同时补 GO_SCAN/ALIGN/REVISIT/中层 LIFT/restore_height 的状态驻留超时
   （W1）与 `_rotate_with_unstick` 总时长上限（W2）；
2. 正式命令启用 `--order-timeout 240`；
3. 导航恢复预算耗尽置 `recovery_exhausted` 信号（F1）、`_route_leg_tick`
   停滞判失败条件放宽 + 硬超时 150→60 s（P1-2）、force 重规划不再回写坏缓存
   （F22）、`update()` 入口 NaN 位姿校验（F7）；
4. `scan_unlocked_*` 的 tick 侧读写纳入 `self.lock`（W3，防字典竞态）；
5. `PRODUCT_CENTER_ABOVE_MARKER_M` 单一来源 + 一致性测试（P0-2）；
6. `PathMemory._save` 加异常处理（M1，防磁盘故障击穿 50 Hz 控制循环）。

**第二批（性能，改动小、可 A/B 验证）**：
7. 动态距离场缓存（S2/F16）；原地旋转碰撞校验（F9）；
8. dwell 参数下调（S3/S4）并用 summary.json 分阶段计时验证；
9. 控制回调解耦 A*（S1/F15）或至少加计时日志确认瓶颈。

**第三批（结构，改动大）**：
10. perception 依赖瘦身（S6）；YOLO 加载失败退避/降级（V2）；
11. worker 退出改为 finished 标志（P2-2）；
12. 状态机拆分（抓取 FSM / 放置 FSM / 导航监督器），常量表统一模块
    （K1/K3/M2——货架几何、层高、运动学常量全部收敛单一来源）；
13. 运动学常量以 mjcf 为唯一真值，删除硬编码多份拷贝（K1）。

---

## 10. 导航模块深审补充（supermarket_navigation.py，行号以该文件为准）

> 与第 3~7 节不重复的独立发现（由专项深审产出，已与第 3 节交叉核对）。

### 10.1 高优先级

- **F7 NaN 位姿无防护**：`world_to_grid` 对 NaN 直接 `math.floor` 抛 ValueError
  （:214-215），`update_from_scan` 每帧调用（:340）；`_update_rotation_watchdog`
  中 `angdist(last_yaw, NaN)` 让 `_rotation_accum` 永久 NaN（:1256-1259），
  看门狗从此失效。**建议**：`update()` 入口校验 base_x/y/yaw 有限性，无效即停车。
- **F22 坏缓存回写强化**：`update()` 的缓存失效集合只含 reverse/lateral/rotation
  （:1916-1921），但 `arc_blocked`/`lidar_stop` 触发的 force 重规划（:1073-1080,
  :1102-1103）会**静默替换缓存路径**且不失效，到达后 `_save_successful_path`
  （:1855-1877）把绕障后的路线以 cached source 写回 → 坏路由被永久强化，
  与 CODEX_HANDOFF"路径记忆命中后触发恢复应失效"的意图不符。建议：阻塞触发的
  force 替换一律 `invalidate_active_cached_path` 或置 `_suppress_path_memory_save`。
- **F9 原地旋转不做碰撞校验**：`_motion_is_free` 在 linear==0 时直接 True
  （:1678-1679），`arc_blocked` 要求 v>0（:1090）；窄走廊原地转向只受桌子保护
  （:1114-1118），可能扫到侧墙/货架。建议：w≠0 且 v==0 时按外接圆校验。
- **F1 恢复预算耗尽后无放弃信号**：`_maybe_start_reverse_recovery` 预算耗尽
  返回 False（:1310-1314）后，`lidar_stop` 仍每 0.35 s force 重规划（:1073-1080）
  ——机器人永久停在原地约 3 Hz 全量 A* 循环。外层 `_route_leg_tick` 的
  35 s 停滞判定又只认 no_path 类原因（integrated:797-809），实际只能等 150 s
  硬超时。建议：预算耗尽时置 `stop_reason="recovery_exhausted"` 并停止重规划，
  同时让外层停滞判定覆盖该原因。

### 10.2 中低优先级

- **F2 rotation_loop 恢复无次数上限**：`_rotation_recoveries` 只增不减（:917），
  建议设上限并与 F1 共用放弃出口；
- **F15 无条件周期重规划**：路径有效时也每 0.4 s 全量 A*（:934-936），失败叠加
  fallback 第二次 A* + 两次 `_rebuild_dynamic`（:1161-1170）；建议仅扫描更新/
  路径失效/阻塞时重规划（与第 7 节 S1 一致）；
- **F16 每 tick 热路径**：前/后向 clearance 每 tick 全量遍历 ranges；`_motion_is_free`
  每步调 `raw_dynamic_clearance_world`（:496-502，O(N) numpy 分配）；
  `_closest_index`/`_lookahead_point` 每 tick O(路径长)（:1485-1545）——与 S2
  （距离场缓存）一致；
- **F8 模块内无激光新鲜度检查**：laser None/空 → inf（:1564-1565, :1588-1589），
  独立使用会全速前进；目前仅靠集成层 `_laser_stale` 兜底（见 P0-1）；
- **F23 reached 判据偏离请求目标**：`_install_path` 用 path[-1] 作 nav_goal
  （:1197-1200），缓存 nearby 命中时端点偏差可达 0.20 m，`_nearest_free`
  （radius=12 格）同样会挪动目标——机器人可能在距请求目标约 0.3 m 处上报到达；
- **F25 缓存路径恢复不做代价地图校验**：`_try_restore_cached_path`（:1812-1853）
  恢复后不查 `_path_valid`，锁定模式下新障碍仅靠急停兜底；
- **F5 路径记忆同步落盘**：`PathMemory._save`（path_memory.py:58-74）在到达/失效
  帧同步写盘，偶发 50 Hz 抖动；建议异步或退出时保存；
- **F11/F20 死代码与小问题**：`_pause_depth` 从未置 True（:179, :409）；
  `_last_logged_reason` 未使用（:821, :865）；path_memory.py 3 处
  `except Exception` 吞异常无日志（:43, :260, :382）；
- **F12/F13 重复**：`wrap_to_pi` 三份实现（nav:96 / path_memory:13 /
  yolo:627）；`_front_clearance`/`_rear_clearance` 同构、`_straight_translation_is_free`
  与 `_motion_is_free` escaping 逻辑重复；`_try_restore_cached_path` 与
  `remembered_path_available` 的 offset 判定逐行重复；
- **F17** `update_from_scan` 用 Python set 收集 clear_cells（:342, :379-386），
  长射线单帧数万次 add，可接受但可优化。

## 11. discoverse 框架侧深审补充（Server/框架；正式 Client 不依赖本地 Server）

> 以下问题多数位于 Server 侧（组织方镜像），Client 不能修改；但 FK/IK 常量的
> 多份拷贝与 NaN 静默发散路径直接影响 Client 的抓取/放置精度，需关注。

### 11.1 与 Client 直接相关

- **K1 三套手臂运动学并存且常量漂移**：`mmk2_fk.py`（MuJoCo FK）、
  `mmk2_fik.py`/`mmk2_ik.py`（AirbotPlayIK 解析解，硬编码 a1=0.1172/a3=0.27009/
  a4=0.29015/a6=0.23645）、客户端 `mmk2_kdl.py`/`arm_kdl.py`（独立 DH）。
  挂载常量不一致：`mmk2_fik.py:9-12`（footprint→chest x=0.02371/z=1.311）vs
  `mmk2_kdl.py:50`（dx=0.033942/dz=1.406）；`arm_kdl.py:164`（d0=0.1117/d6=0.2466）
  vs `airbot_play_ik.py:4-8`（a1=0.1172/a6=0.23645）。仓库自认存在
  "右臂 TCP 比指令偏东约 1cm"（yolo:601-604 的补偿表）——根因很可能在此。
  **建议**：以 mjcf 为唯一真值（`mmk2_ik.py:25-41` 已有从 mjcf 生成 npz 的示范），
  客户端只导入数据。
- **K2 `properFK` 缺失**：`mmk2_fik.py:81,99` 调用 `self.arm_ik.properFK(q)`，
  但 `airbot_play_ik.py` 无该方法——`get_3dposition_wrt_arm_base/footprint`
  一旦被调用即 AttributeError；`properIK` 在 ref_q=None 时返回解列表
  （airbot_play_ik.py:66-73），`mmk2_fik.py:108` 未处理。当前客户端未走该路径，
  属潜伏缺陷。
- **K3 客户端耦合服务端常量 + 运行时猴子补丁**：yolo:2935/3039/3216 直接引用
  `MMK2FIK.TMat_chest2*_base`；integrated:100-122 monkeypatch `MMK2FK.__init__`
  改写运行时 XML（写 /tmp，只读部署会失败）。升级脆弱。
- **K4 NaN 静默发散链**：action NaN → `mmk2_base.py:193` `np.clip` 放行 NaN →
  `mj_step` 产出 NaN 状态 → 观测 NaN → 客户端 PID（controllor.py:13-19 无 NaN
  防护、`PIDarray.output` 无输出限幅 :41-48）永不平移。全程无报错。
  **建议**：ctrl 写入前 `np.nan_to_num`/有限性检查；PID 入口拒 NaN。
- **K5 传感器索引硬编码**：mmk2_base.py:100-150 按 sensordata 固定偏移切片，
  xml 增删传感器即静默错位；建议按名称解析（utils/__init__.py:49-65）。

### 11.2 Server/框架侧（仅供了解）

- **S1 `mmk2_ros2.py` 缺失**：supermarket_sorting_server.py:35
  `from mmk2_ros2 import MMK2ROS2`，当前 checkout 无此文件——Server 无法从
  本仓库启动，且**真正的线程同步/死锁风险藏在该文件中，无法审查**（审查盲区）。
- **S2 Server 三线程无锁**：spin 线程（rclpy 回调）+ 24 Hz 发布线程 +
  主线程 `mj_step` 之间无锁（supermarket_sorting_server.py:439-450）；
  `mj_data` 非线程安全，并发读写是未定义行为；spin 线程异常退出后主循环继续
  step → 机器人"失聪"的静默挂死。
- **S3 渲染在控制回路内同步**（simulator.py:727-740, 737-738）：慢渲染直接拉低
  控制频率；'R' 键 reset 会从 render 内触发 render 重入（:601, :689-694, :511）。
- **S4 类属性当实例状态**：statemachine.py:2-5 与 simulator.py:44-72 大量可变
  状态是类属性，多实例互相污染；SimpleStateMachine 默认 `max_state_cnt=-1`
  导致状态永不推进（:5, :8），使用方忘记配置即静默卡死。
- **S5 裸 except**：mmk2_ik.py:15 吞一切异常后 `np.savez`（:19）写仓库目录，
  只读部署下在 try 外抛未处理异常；simulator.py:520-521 吞渲染异常致 obs 静默
  过期。

---

## 12. 感知子系统深审补充（persistent_perception / kele_detect / aruco_detect / backends）

> 行号以对应文件为准。与第 3~7 节不重复的独立发现。

### 12.1 确认无问题（专项核对结论）

- **无死锁**：无持锁阻塞调用（rgb_cb 仅在开头短暂持 `_active_camera_lock`
  kele_detect.py:425-428；`_depth_lock` 只在拷贝 deque 时持有 :454-455）；
  无嵌套锁、无锁内 sleep/IO；默认 QoS 队列满时丢消息而非阻塞；
- **无内存/GPU 泄漏**：deque maxlen 有界，每帧临时数组可回收；YOLO 主路径零
  多余拷贝（`imgmsg_to_cv2` 视图，pub_res_img 关闭时 `vis = rgb` :472）；
- **消息缺失不崩溃**：K/None/base 缺失早退（kele:429-442、aruco:211-260）；
- **QoS 匹配**：发布/订阅全部默认 QoS 且一致（含消费者 yolo:990-992）。

### 12.2 高优先级

- **V1 `PRODUCT_CENTER_ABOVE_MARKER_M` 双份数值冲突**（与 P0-2 同一问题，
  从感知侧确认影响面）：competition_task.py:207 用于检测-货架关联、
  yolo:675/745/1222 用于抓取 Z——同语义不同值，至少一处错误，需单一来源；
- **V2 YOLO 加载失败 → 无限崩溃-重启循环**：构造抛 RuntimeError
  （kele_detect.py:287-290）→ 进程崩溃 → runner 每 1 s 无退避重启
  （competition_runner.py:400, 366-434）；checkpoint 损坏时循环不终止。
  **建议**：runner 增加重启次数上限/指数退避，或加载失败降级为本地感知并告警。

### 12.3 中优先级

- **V3 rgb_cb 无 per-frame 异常隔离**（kele_detect.py:424-558）：rclpy 只记日志，
  该帧剩余发布被跳过；异常反复时节点"活着但全盲"且 12 Hz 刷日志，runner 无感知
  （ready 文件仍在）。建议 detect/foreground 分段 try/except + 连续失败计数降级；
- **V4 深度单位判断 4 处不一致**：kele foreground dtype/中值启发式（:143-144）vs
  `patch_depth_m` **无条件 ×1e-3**（:421）vs aruco uint16/值>20（aruco:182-183）
  vs supermarket_navigation encoding 判断（:121-126）。对 mono16(mm) 全正确，
  但 float 米制发布者场景 `patch_depth_m` 错 1000 倍、kele 内部自相矛盾；
- **V5 ArUco 无限频**：24 Hz 全速 detect+PnP（aruco_detect.py:256-277），与
  YOLO 12 Hz 上限不对称，CPU 设备上与 YOLO 争 CPU；建议同样限频；
- **V6 ready 文件先于 spin 写**（persistent_perception.py:101-109）：不代表订阅
  已生效；runner 0.2 s 轮询（:104, 374-375），最多损失 0.5 s 扫描时间，低风险；
- **V7 perception_enable 是状态型话题却用 VOLATILE**：runner 的 False 为
  best-effort（competition_runner.py:716-721），丢失则 persistent 保持 enabled
  到下一 worker 首帧 tick（integrated:2650）——trip 间隙短暂 GPU 争用。
  **建议**：TRANSIENT_LOCAL + depth=1，或 runner 周期补发 False。

### 12.4 低优先级

- **V8 逻辑重复**：`camera_world_tmat` 近全重复（kele:378-402 vs aruco:209-233）；
  `patch_depth_m`（kele:413-421）与 `backends._safe_depth_m`（backends.py:42-49）
  逐行重复；150 ms 同步界三处表达且单位不同（kele:81 ns / aruco:31 ns /
  yolo:154 ms）；ArUco 初始化重复（aruco:104-114 vs yolo_aruco_goods_detect.py:
  77-87）；`set_active_cameras`（kele:328-338）全仓无调用者——死代码；
- **V9 性能**：foreground 每检测 8 次 percentile+nonzero（kele:167-204），大 bbox
  每帧每目标 ~50 万 numpy 操作（建议 searchsorted 一次算 8 分位）；JSON 载荷偏重
  （front_candidates 8 组 {camera,world}、aruco 每 marker 带 9 浮点 camera_matrix）；
  空检测仍 12 Hz 发布空消息（kele:549-551、aruco:325-327）；ArUco 每帧
  drawFrameAxes/cv2_to_imgmsg（aruco:276-277, 331-338）。

---

## 13. 抓取状态机深审补充（yolo_aruco_shelf_pick.py，行号以该文件为准）

> 与第 3~7 节不重复的独立发现（专项深审产出）。

### 13.1 必改（卡死/竞态）

- **W1 四个状态可无限驻留**：
  - `STATE_GO_SCAN`（:4474-4515）`drive_to` 无超时，导航/相机就绪永不满足即永久等待；
  - `STATE_ALIGN`（:4556-4572）`drive_to` 无超时，基座达不到 2.5 cm 容差即卡死
    （与 P0-1 的激光陈旧问题叠加）；
  - `STATE_REVISIT`（:4538-4548）驻留计时以 `scan_camera_ready` 为前提，相机/关节
    不到位永不推进；**对比 STATE_RECHECK 有 `CLOSE_RECHECK_POSE_TIMEOUT_S` 兜底
    （:4607-4611），REVISIT 没有等价 pose 超时**；
  - `STATE_LIFT` 中层 generic（:5182-5192）只凭 `abs(slide-des_slide)<0.025` 收敛
    判定，**无超时分支**（顶层 :5155-5161、下层 :5169、球体 :5135-5139、双臂
    :5077/:5094 均有超时或 dwell）。
  - **建议**：drive_to 及上述状态加总超时（30~60 s）→ ABORT；REVISIT 复用
    RECHECK 的 pose 超时模式。
- **W2 `_rotate_with_unstick` 无限重置循环**：卡死计数超 `NAV_ROT_UNSTICK_MAX(3)`
  后仅重置计数并继续旋转（:2570-2579），注释称"由上层兜底"，而上层 drive_to /
  GO_SCAN / ALIGN 均无超时（:2485-2518）→ 物理无法转向时"旋转停滞 2.5 s→倒车
  →再旋转"永久循环。**建议**：加总旋转时长/连续失败上限，超限 ABORT。
- **W3 `scan_unlocked_*` 跨线程数据竞争**：yolo_cb 持 `self.lock` 写
  （:1083-1104, :1500-1505），tick **无锁**迭代/清空/替换（:2229-2238, :2325,
  :1952-1953, :2331-2333）；MultiThreadedExecutor 4 线程（:5463）下并发，
  迭代中 `setdefault` 可触发 "dictionary changed size during iteration"
  RuntimeError → tick 被 rclpy 吞掉。**建议**：tick 侧读写 `scan_unlocked_*`
  统一进 `self.lock`，或回调只写快照、决策读快照。

### 13.2 建议

- **W4 每帧只评估最高置信 detection**（:1456-1463）：高置信假阳性持续存在时该
  槽位永远无法定位，只能靠整站轮转耗尽。建议遍历全部候选并保留"可接受的最低
  置信兜底"；
- **W5 定位参数块重复**：`_commit_yolo_only_target`（:1367-1414）与
  `try_association_locked`（:1669-1710）的 grasp_arm/align/shelf_level/slide_grasp
  组装几乎逐行重复；球体层判定（:1586-1592 vs :1609-1617）、IK 前推重试
  （:2999-3011 vs :3105-3117）、列偏移 dict（:1318 vs :2442）、
  `_try_position_fallback`（:2393-2453）与 `_maybe_lock_yolo_only_target_locked`
  （:1259-1329）骨架重复——建议提取公共函数；
- **W6 常量跨文件不一致（除 P0-2 外新增）**：`TOP_SHELF_Z_M=1.10`（:302）vs
  memory_matrix `LEVEL_Z_L3_MIN=1.09`（:66）差 1 cm；`SHELF_SURFACE_Z_M`
  （:540, 1.189/0.851/0.499）vs memory_matrix `LEVEL_MARKER_Z`（:57,
  0.500/0.852/1.190）差 1 mm——层判定边界两套系统各差 1 cm/1 mm，随机深度抖动
  跨层时可能得出不同货架层结论；
- **W7 `solve_kdl_world` 的 `item[1:]`（:2633）对 solutions 元素结构异常无防护**
  （IndexError 未捕获）；其余 IK 失败路径均已确认有捕获（全部调用点
  False→ABORT/retry，无遗漏）；
- **W8 球体轨迹每 tick 解完整 KDL IK**（:3985, :4093 在 progress 推进时），
  50 Hz×数秒 ≈ 150~300 次 IK/单——球体是 CPU 最热点，可预插值关节轨迹；
- **W9 回调/tick 无顶层异常兜底**（:1024, :1245, :4466）：未捕获异常被 rclpy
  吞掉后该帧静默失效。建议回调/tick 内 try/except 记日志 + 连续失败计数。

### 13.3 确认无问题（专项核对结论）

- `self.lock` 使用一致，**无锁死锁路径**（无重入、无持锁阻塞）；
- 扫描/补拍循环有双层计数上限（`REVISIT_MAX_ROUNDS_PER_SCAN=4`、
  `REVISIT_MAX_ROUNDS_PER_MARKER=1`、`max_scan_cycles`），正常数据流可终止；
- 单臂/双臂推进阶段（ARM_FORWARD/POST_EXTEND/DUAL_SQUEEZE）均有时长+settle
  双条件硬兜底，不会因反馈缺失卡死；
- 深度 NaN/inf、marker 坐标缺失、joints 缺失均有守卫（inf 哨兵 + isfinite
  校验），无 NaN 传播路径；
- deque 类共享状态（yolo_frames/aruco_frames/marker_positions）的锁覆盖一致；
- 日志频率合理（1 s 摘要/状态、0.5 s base-hold、0.5 s close 诊断、1 s 拒绝限流）。

---

## 14. 记忆系统深审补充（memory_matrix.py + path_memory.py，行号以对应文件为准）

> 专项深审产出，与第 3~7 节不重复。

### 14.1 高优先级

- **M1 `PathMemory._save` 无异常处理且位于 50 Hz 控制循环内**（path_memory.py:58-74）：
  调用点 `_save_successful_path`（supermarket_navigation.py:1855-1877，到达帧）与
  `invalidate_path`（:1922/:1968，恢复帧）都在 `update()` 里；磁盘满/权限错误
  一次即击穿 worker 控制循环。同时 `_load`（:38-44）吞掉一切异常得空目录，
  下次 `_save` 全量覆盖**静默丢失全部历史**。**建议**：`_save` 加 try/except +
  日志，写失败保留内存态、延后重试；`_load` 失败时备份损坏文件。
- **M2 货架几何常量三份拷贝且数值不一致**：货架 x 中心
  `SHELF_SCAN_X`（memory_matrix:52-55，E=1.800）vs `SCAN_X`
  （yolo_aruco_shelf_pick:184，E=1.80）vs `SHELF_CENTERS_X`
  （supermarket_navigation:78，E=**1.805**）——**不一致**；层高
  `LEVEL_MARKER_Z`（0.500/0.852/1.190）vs `SHELF_SURFACE_Z_M`
  （0.499/0.851/1.189）差 1 mm；列偏移 `COLUMN_X_OFFSET` vs shelf_pick:1318
  内联 dict；`SCAN_STATION_Y` vs `SCAN_Y`。当前偏差均在容差内
  （`COLUMN_X_TOLERANCE_M=0.14`）不产生行为差异，但独立维护必然漂移。
  **建议**：收敛单一来源统一导出。

### 14.2 中优先级

- **M3 `consume_slot` 幂等性**（memory_matrix:538-571）：重复消费仍 `changed=True`
  → 置 dirty 并立即全量写；competition_runner.py:574-578 注释自称 "idempotent
  fallback" 不成立（行为无害，但每次重复消费都触发一次全量写盘）。
  建议 False→True 才算 changed；
- **M4 `tick_write` 持 `_tracker_lock` 做完整文件 IO**（:881-911, :913-920）：
  锁内 `to_json` 深拷贝 + `json.dumps` + 写盘 + rename，阻塞 runner 线程的
  `routing_snapshot`/`consume_slot`/`start_run`（0.2 s tick 可能被拖住），
  且耗时随 candidates 规模无界增长。建议锁内只拷快照、锁外序列化+写盘；
- **M5 consume/reset 绕过 0.5 s 写节流**（:837, :875 无条件立即全量写）：
  每订单约 2~3 次全量写；建议 consume 也走节流队列；
- **M6 已消费槽位同 run 内永久失效**（`record_at` :367-368 对 consumed cell
  直接 return False 且不更新 candidates）：裁判补货到同一物理列也不会再被记忆，
  属设计盲区（注释只覆盖"落在未消费列"的情形）；
- **M7 抓取即消费早于投递完成**（integrated:1050-1070）：中途掉落则记忆已抹除，
  重试只能全扫；runner 用 `failed_memory_slots` 在 hint 层弥补——有界可接受，
  但注意这是"掉落商品位置不可再用"与"该位置仍有货"语义冲突的根源之一；
- **M8 排除集合双轨维护**：runner `failed_memory_slots`（competition_runner.py:
  580-582/:742-743）vs worker `excluded_slot_keys`（--exclude-slot-key :259-265
  + worker 自行追加 :1894），可能发散，仅影响 hint 命中率。

### 14.3 低优先级

- **M9 "waiting" 临时文件残留**：competition_runner.py:87-88 创建
  `memory_matrix_waiting_{pid}.json`，`start_run`（:141）切走后从不删除；
- **M10 性能**：worker 在控制热路径（integrated:2699）以最高 4 Hz 全量读盘解析
  （memory_matrix:640-643 节流 0.25 s，:628 失败立即再读）；tracker 每次脏写
  全量重写且 candidates 不淘汰、文件持续变大；PathMemory 每次保存全目录重写、
  `_catalog` 无容量上限（仅 invalidations 截断 50 条）、`load_path` miss 时
  O(N) 线性扫描、每次保存写 forward+reverse 两条；
- **M11 接口风格不统一**：runner :672 直接调 `matrix.to_json()` 未走
  `_tracker_lock`（仅靠 `matrix._lock`，一致且无死锁）。

### 14.4 确认无问题（专项核对结论）

- 未发现文件部分写入可见（原子 rename + 解析兜底 `read_memory_document` :300-306）；
- 未发现读-改-写丢失更新（tracker 全量内存序列化后原子替换）；
- **未发现死锁**（单一锁序 `_tracker_lock → matrix._lock`，均 RLock，无反向获取）；
- `select_memory_hint` 只有一份实现（memory_matrix:202-273，runner/worker 共用）；
  候选提取有内存版（`primary_candidates_for` :507-526）与文件版
  （`primary_candidates_from_document` :276-297）两份平行实现——语义一致；
- JSON 损坏在 memory_matrix 侧全部良性降级为全扫；
- tracker 并发模型正确（独立节点 + 互斥回调组 + 锁桥接）；`_slot_acc`/`base_xy`/
  `_dirty` 均只在锁内读写，未发现内存状态竞态；
- hint 失败回退链完整（worker `memory_failed_hint_levels`/`memory_exhausted_shelves`
  排除 + `reliable_only=True` 重选 + 无候选恢复全扫）。

---

## 15. 2026-08-19 运行实证与已实施修复

### 15.1 运行实证（run_abac6b642eca，21:36 启动）

第一单（chengzi）从出生点 `(1.92,-3.17)` 出发时命中跨 run 保存的反向配送通道
缓存（`saved_at=09:37Z`，来自当天 17:37 的一次运行），于是**先导航到配送桌旁的
trunk exit 再绕回货架**——go_scan 阶段耗时 **215.96 s**（直线约 40 s）。随后
delivery 段 trunk 缓存腿又因当天随机箱子布局与缓存路径冲突，**150 s 硬超时后
fallback**；最终橙子在运输途中掉落（drop-monitor 判定 `measured_grip 0.549 ≤
0.55`，失败重试）。第二单 zhijin 同样命中同一缓存、同样绕路。

**根因**（均已实锤）：
1. `_scan_trunk_route_tick` 的"是否需要走 trunk"只按 y 坐标判定
   （`base_y >= 1.25` 视为已在货架侧），出生点 y=-3.17 被误判为"在桌边"；
2. trunk 查询起点写死为锚点，与机器人实际位置无关；缓存跨 run 持久化，
   而比赛每 run 随机 5 个箱子障碍——跨 run 缓存命中反而有害；
3. 停滞判失败只认 no_path/recovery_exhausted，`arc_blocked`/`lidar_stop` 卡死
   只能等 150 s 硬超时（P1-2 的实锤）。

### 15.2 已实施修复（`integrated_nav_pick_place.py`，2026-08-19，未提交）

1. **trunk 桌边门控**（`_scan_trunk_route_tick`，:859-892）：
   在 y 判据之后新增物理邻近判定——距 trunk exit 锚点 ≤ 0.60 m **或**
   距配送桌 keep-out ≤ 1.01 m 才允许走反向 trunk；否则直接
   `scan_direct_to_shelf`（实时 A* 去货架站）。
   验证：出生点 dist=3.94 m → 直接去货架 ✓；放置完成各槽位 dist=0.12~0.30 m
   → 仍走 trunk ✓；`return_to_west`/drop 恢复不受影响 ✓。
2. **停滞判失败放宽**（`_route_leg_tick`，:814-831）：`persistent_stop =
   no_path or recovery_exhausted or stop_reason is not None`——任何持续停止
   原因停滞 35 s 即判失败，不再只能等硬超时。
3. **`ROUTE_LEG_HARD_TIMEOUT_S` 150 → 60**：停滞放宽后的最终兜底；
   按最低实测仿真速率（~0.12 m/s）60 s 仍覆盖场内所有单腿长度。
4. **每次 client 进程清空路径记忆**（`competition_runner.py` 新增
   `clear_path_memory_file()`）：runner 退出（finally）与启动时各清一次，
   读取与导航器一致的 `SUPERMARKET_PATH_MEMORY_FILE`（默认
   `/root/.cache/supermarket_path_memory.json`），best-effort（OSError 仅告警）。
   原因：比赛每 run 随机 5 个箱子，跨 run 缓存会把上一场的路线套到新布局上
   （run_abac6b642eca 实证：216 s 绕路 + 150 s trunk 超时）；启动时兜底清
   覆盖 SIGKILL 等未走 finally 的退出路径。
5. **全路径机械臂/夹爪归位**（2026-08-19 第二批）：
   - `yolo_aruco_shelf_pick.py`：`STATE_ABORT` 分支提取
     `_abort_recovery_ready()` hook（默认保持原 settle 语义，standalone 行为
     不变）；
   - `integrated_nav_pick_place.py`：
     a. 覆盖 `_abort_recovery_ready()` —— 抓取失败（ABORT）时双臂/双夹爪/
        slide/head 归位到初始姿态（`_command_initial_arm_posture`），
        ready 或 8 s 超时（`ABORT_RECOVERY_TIMEOUT_S`）后才退出；
     b. tick 顶层 try/except + `fatal_recover` 相位 —— 放置失败等
        RuntimeError（rclpy 会吞掉回调异常，原来会导致 worker 停在半伸臂
        姿态直到 runner 超时杀掉）现在会归位后干净退出，错误写入结果文件
        （`_fatal_error`，8 s 恢复上限 `FATAL_RECOVERY_TIMEOUT_S`）；
     c. worker 启动时（GO_SCAN 状态）强制归位一次
        （`_startup_posture_recovered`）—— 覆盖 SIGKILL/异常退出残留的
        闭合夹爪/前伸手臂（server 只重置车体位姿、不重置关节，实测三个
        worker 启动时 slide 分别为 0.006/0.113/0.123 证实关节被继承）。
   - 订单完成、drop 成功/失败路径原本已有归位（`_clear_delivery_table_tick`
     完成条件、`_drop_recovery_tick`），未改动。
6. **取消三段式导航**（2026-08-19 第三批，`integrated_nav_pick_place.py`）：
   - `_scan_trunk_route_tick`：删除 trunk exit → trunk entry → shelf 三段及
     反向 trunk 缓存查询，GO_SCAN 去货架站与 `return_to_west`（第一单交付后
     回 shelf A 扫描，逻辑保留）统一改为**一段直达** `scan_direct_to_shelf`；
   - `_start_delivery_navigation`/`_nav_to_delivery_tick`：删除
     to_trunk_entry → trunk_forward（use_memory=True）→ to_slot 三段及
     direct fallback 链，改为**一段直达** `delivery_direct_to_slot`（保留
     slot_refine 的 yaw 校正）；
   - 效果：消灭锚点绕路、135° 左转对准 trunk exit、坏缓存复用（桌子北侧
     卡死根源）；任何腿失败 35~60 s 内直接失败（`_route_leg_tick` 放宽判定
     + fatal_recover 兜底）。
   - **路径记忆影响**：`use_memory=True` 调用点已全部消失——PathMemory
     （路线缓存）不再被查询/保存，等于停用（代码保留、无害）；
     `memory_matrix`（商品位置记忆）不受影响，scan hint 照常工作。

预期效果：新 worker/重试 worker 从出生点直接去货架扫描（消灭 216 s 绕路）；
卡死腿 35~60 s 内失败并 fallback，而不是 150 s；每场从零开始规划，不继承
上一场的箱子布局路线；任何失败/完成路径退出前机械臂与夹爪都回到初始姿态；
去货架/送货/回 A 全部一段直达（无锚点绕路、无 135° 对准 trunk exit）。
未改动的部分：`return_to_west` 流程本身（第一单交付后回 shelf A 做库存扫描）、
`memory_matrix` 商品位置记忆、放置/抓取状态机。

---

## 附录 A：本次审查未覆盖/未深入项

- `continuous_goods_client.py`、`gui_launcher*.py`、`referee.py`：非正式链路
  （CODEX_HANDOFF 明确废弃/独立），未深入；
- `supermarket_sorting_server.py`：组织方 Server，正式 Client 不依赖；仅从
  框架侧核对了线程模型（见 S1/S2，`mmk2_ros2.py` 缺失导致 Server 线程同步
  实现无法审查——审查盲区）；
- 仿真实跑验证（激光卡死复现、放置扭动复现、`scan_unlocked_*` 竞态复现）：
  需用户授权后按 CODEX_HANDOFF 第 4 节流程执行。

## 附录 B：主要文件行数统计

| 文件 | 行数 | 职责 |
|---|---|---|
| yolo_aruco_shelf_pick.py | 5523 | 抓取 FSM（父类） |
| integrated_nav_pick_place.py | 3107 | 单件外层流程 |
| supermarket_navigation.py | 2018 | 导航 |
| competition_runner.py | 878 | 多单编排 |
| memory_matrix.py | 949 | 货架记忆 |
| path_memory.py | 408 | 路线缓存 |
| arm_kdl.py | 617 | 机械臂解析 IK |
| mmk2_kdl.py | 364 | 双机械臂 IK |
| competition_task.py | 243 | 任务模型 |
| persistent_perception.py | 131 | 常驻感知 |
| run_log.py | 116 | 日志落盘 |
