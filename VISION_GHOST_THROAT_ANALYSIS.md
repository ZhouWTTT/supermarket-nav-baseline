# 货架区咽喉点 "视觉幽灵" 卡死问题分析

> 分析日期：2026-08-21
> 涉及模块：`supermarket_navigation.py`（导航控制器 / costmap / 弧线检测）、
> 感知视觉层（深度相机 → vision costmap）
> 现象：多订单测试中订单反复在货架区同一咽喉点 `arc_blocked` 卡死，
> 倒退/横向逃逸恢复全部用尽后 fatal，订单失败率 80%~100%。

---

## 1. 问题概述

在五订单（kele/heweidao/zhijin/pingguo/maidong）完整流程测试中，
**除第一个订单（从出发区走东侧通道）外，其余从桌子/走廊区返回货架区的订单，
几乎全部卡死在货架区 C/D 货架通道南口**：

| 测试场次 | 布局 | 卡死点 | 结果 |
|---|---|---|---|
| 场次 1（heading_alignment 修复前） | 随机障碍（attempts=18） | (0.60, 2.24) | delivered=1/5 |
| 场次 2（倒退 0.2m 修复后） | 随机障碍（attempts=18） | (0.53, 2.23) | delivered=0/5 |

两次测试服务端随机障碍布局不同，但卡死点几乎重合（0.5~0.6, 2.2~2.4），
且失败原因从 `heading_alignment` 演变为 `arc_blocked:recovery_exhausted`。

---

## 2. 卡死现象

### 2.1 失败订单画像（场次 2）

```
[ERROR] ... [fatal] phase=grab state=go_scan place_stage=0:
  RuntimeError: shelf transit direct leg failed: stalled:35.0s:arc_blocked:recovery_exhausted=1
```

卡死时的 route 日志：

```
[route:scan_direct_to_shelf] pos=(0.53,2.23) yaw=157deg dist=0.46m
  v=0.00 w=-0.00 elapsed=33.1s stalled=33.1s reached=False
```

- 位置完全静止（v=0, w=0），连转向都没有——恢复预算用尽后导航冻结输出
- 每次尝试都在同一坐标停滞 35s（route 层 `ROUTE_LEG_STALL_TIMEOUT_S`）后 fatal

### 2.2 恢复机制确实触发但无效

日志中出现过：
- `stop_reason=reverse_recovery_start`（倒退恢复启动）
- `stop_reason=lateral_escape_replan`（横向逃逸重规划）

恢复参数（当前值）：
- 倒退距离 `_reverse_recovery_distance_m = 0.20`（初始 0.12，测试中加大到 0.2，无效）
- 倒退次数上限 `_reverse_recovery_max_attempts = 2`
- 横向偏移 `_lateral_escape_offset_m = 0.38`，上限 2 次

---

## 3. 根因分析

### 3.1 关键证据：arc_blocked 时激光完全开阔

```
[route:scan_direct_to_shelf] stop_reason=arc_blocked lidar=2.89m rear=2.04m
```

弧线检测（`_motion_is_free`）判定前方 0.45m 弧线不可通行，但：
- 激光（2D lidar）前方 **2.89m 开阔**
- 卡点 0.45m 范围内**没有任何动态障碍箱**（对照运行时 XML 的 box 位置）
- 货架层板最近也在 y≥3.17（卡点 y=2.23 相差 0.94m）

→ **障碍不是物理/激光层，而是深度相机视觉层。**

### 3.2 视觉层障碍机制

`supermarket_navigation.py` 的视觉障碍投影逻辑：

```python
def update_from_depth_obstacle(self, distance, robot_x, robot_y, robot_yaw, ...):
    """深度相机中心光束障碍投影到 2D costmap；
    同一栅格连续两帧命中 → vision_raw = LETHAL"""
    ...
    if hits >= 2:   # require two consecutive frames
        self.vision_raw[gy, gx] = LETHAL
```

- 深度相机（RGB-D，渲染自 **3DGS 超市场景背景**）视场内的静态陈列物
  （货架端头/展柜等）会被投影为永久 `vision_raw=LETHAL`
- 该障碍进入 `_rebuild_dynamic()` 的膨胀层，参与 `_motion_is_free` 弧线预测
- **激光平面测不到它**（位置/材质超出 2D lidar 感知），因此
  `lidar_clearance` 保持开阔、减速逻辑不触发，但弧线检测被 LETHAL 栅格硬停

### 3.3 为什么"老是在这一点"——结构性死锁

1. 卡点 (0.5~0.6, 2.2~2.4) 是**货架区 C/D 通道南口**，
   从桌子/走廊区（西侧）返回东侧货架（E/D）的**唯一咽喉**；
2. 机器人从西侧接近时相机朝西，背景幽灵不在视场 → A* 规划出经过咽喉的路径
   （规划地图此刻"干净"）；
3. 到达咽喉转向（yaw≈150°）→ 相机朝东南 → **背景幽灵进入深度相机视场
   → vision LETHAL → arc_blocked**；
4. 倒退/横向逃逸 → 重规划 → 路径仍经过咽喉（A* 此时看不到幽灵）
   → 再次到点转向 → 幽灵再现 → 循环；
5. 恢复预算（倒退×2 + 横向×2）用尽 → `recovery_exhausted` → 冻结 → 35s → fatal。

**3DGS 背景是固定的（不参与随机化）**，因此幽灵点恒定存在；
只要路径规划选择经过该咽喉，就必然触发。这是"每次测试都卡同一点"的根本原因。

### 3.4 与既有问题的关系

| 问题 | 状态 | 关系 |
|---|---|---|
| 配送方向 arc_blocked（kele 单场） | 速度 1.05→0.90 后解决 | 激光层问题，与本次不同 |
| heading_alignment 走廊原地打转 | 加入 recoverable 集合后触发倒退 | 本次已不再是卡死原因 |
| **咽喉点视觉幽灵 arc_blocked** | **本次根因** | 规划地图与检测地图不一致 |

---

## 4. 修复建议

### 方案 1（推荐）：弧线检测对 vision 层降权

`_motion_is_free` 中视觉层 LETHAL 不作为硬停条件（仅激光/静态层硬停），
视觉障碍保留在 A* 规划与减速（`depth_clearance`）中。
- 优点：改动小、直接消除"幽灵硬停"；幽灵仍参与规划，绕行时路径更保守
- 风险：若幽灵位置确实有真实障碍，可能轻微蹭碰（仿真中可接受）

### 方案 2：咽喉点黑名单（配合方案 1 更稳）

同一区域 arc_blocked 停滞超过 N 秒 → 将咽喉栅格临时标记为不可通行
（写回 costmap 膨胀层），强制重规划真正绕行，避免"规划—检测不一致"死循环。

### 方案 3：规划与检测地图一致化

A* 规划与弧线检测使用同一障碍口径（如弧线检测也走 lidar-only fallback 地图），
从机制上消除不一致。

---

## 5. 附录

### 5.1 相关代码位置

| 逻辑 | 文件:行 |
|---|---|
| 弧线检测 | `supermarket_navigation.py` `_motion_is_free`（约 1676 行） |
| arc_blocked 判定 | `supermarket_navigation.py` 约 1090 行 |
| 视觉障碍投影 | `supermarket_navigation.py` `update_from_depth_obstacle`（约 400 行） |
| 倒退恢复触发集合 | `supermarket_navigation.py` `_maybe_start_reverse_recovery`（约 1278 行） |
| 倒退距离参数 | `supermarket_navigation.py` `_reverse_recovery_distance_m`（约 770 行） |
| route 层停滞超时 | `integrated_nav_pick_place.py` `ROUTE_LEG_STALL_TIMEOUT_S`（35s） |

### 5.2 关键日志片段

```
# 卡死时激光开阔但弧线判定阻挡
[route:scan_direct_to_shelf] stop_reason=arc_blocked lidar=2.89m rear=2.04m

# 静止 35s 后 fatal
[route:scan_direct_to_shelf] pos=(0.53,2.23) yaw=157deg dist=0.46m
  v=0.00 w=-0.00 elapsed=33.1s stalled=33.1s reached=False
[fatal] phase=grab state=go_scan place_stage=0:
  RuntimeError: shelf transit direct leg failed: stalled:35.0s:arc_blocked:recovery_exhausted=1

# 恢复机制确实触发
stop_reason=reverse_recovery_start lidar=0.46m rear=2.1...
stop_reason=lateral_escape_replan lidar=0.35m rear=2...
```
