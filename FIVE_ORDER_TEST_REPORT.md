# 五订单完整流程测试报告

> 测试时间：2026-08-21 16:56 ~ 17:17
> 运行标识：`run_08c7ba15aba9`
> 测试代码：master `ac0c0d2` + 未提交导航修改
> （heading_alignment 近障碍触发倒退恢复；倒退距离 0.2m）

---

## 1. 测试配置

| 项 | 值 |
|---|---|
| 任务 | 5 订单：kele / heweidao / zhijin / pingguo / maidong |
| 商品布局 | 随机化（`SUPERMARKET_RANDOMIZE=1`） |
| 障碍走廊 | 随机化（`SUPERMARKET_RANDOMIZE_OBSTACLES=1`，attempts=1, path=5.62m, detour=0.96m） |
| 最大尝试次数 | 每单 2 次 |
| 匹配超时 | 3600s |

## 2. 最终结果

```
match finished: reason=orders_terminal
delivered=5/5  failed=0  elapsed=1187s (19.8 分钟)
```

| 订单 | 商品 | 结果 | 抓取位置 | 尝试 |
|---|---|---|---|---|
| 1 | kele | ✅ delivered | E L2 | 1 |
| 2 | heweidao | ✅ delivered | A L2（marker=3） | 1 |
| 3 | pingguo | ✅ delivered | A L3 | 1 |
| 4 | zhijin | ✅ delivered | E L3 | 1 |
| 5 | maidong | ✅ delivered | B L1 | 1 |

**全部订单一次尝试成功，无 fatal、无丢货、无订单级失败。**

## 3. 测试过程中出现的问题（告警级，均未影响结果）

### 3.1 放置触桌检测触发（5/5 单，预期行为）

全部 5 单放置时都走了 **触桌检测就地松爪** 路径：

```
[place] slide blocked during overhead approach; goods already at table height — releasing in place
[place] slide physically blocked (goods touching table); stopping descent and releasing in place
tcp=[-2.194 -3.52x ...]（均在桌面正上方）
```

- 触发时机：放置 approach 阶段 slide 被桌面顶住（商品底部已触桌）
- 处理：就地松爪 → 垂直向上 → 水平撤离 → PLACE COMPLETE
- **说明**：半高表回退后 release_z 计算偏低，正常下降必然触桌；
  触桌检测兜底成为实际主路径，商品均安全落桌成功。
- **遗留问题**：放置仍依赖"触桌检测"而非"1cm 悬空松爪"的正常路径，
  建议后续修正商品半高表或改用实测接触检测，使正常路径生效。

### 3.2 激光反馈短暂陈旧（3 次）

```
stopping for stale robot feedback (odom_stale=False, joints_stale=False, laser_stale=True)
```

- 服务端/激光话题短暂卡顿，机器人短暂停车后自动恢复，无订单影响。
- 与之前的服务端 GPU 偶发卡顿同源。

### 3.3 zhijin 中列过滤生效（1 次）

```
[tissue-filter] ignoring non-middle-column tissue marker=5 slot=unknown; only the middle column is eligible
[tissue-filter] position fallback rejected side-column tissue slot=L1|A|3
[revisit] FAILED box(any-marker) kind=zhijin; resuming normal scan
```

- 扫描时先发现侧列纸巾（A L1/L2 侧列），按 v2 规则拒绝（zhijin 只认中列）
- 恢复正常扫描后从 E L3 中列抓到，delivered
- 行为符合预期，仅多花约 1 分钟扫描时间

### 3.4 其他正常告警

- `[generic-retreat] TCP clear of the shelf ... treating as removed`（2 次）：抓取收回正常判定

## 4. 未触发的问题（本场布局未复现）

| 问题 | 状态 |
|---|---|
| 咽喉点"视觉幽灵" arc_blocked 卡死 | 本场障碍布局（attempts=1）未触发；**问题仍存在**，见 `VISION_GHOST_THROAT_ANALYSIS.md` |
| 走廊导航卡死（arc_blocked） | 速度 0.90 回退后未再出现 |
| heading_alignment 走廊原地打转 | 加入倒退恢复集合后未再出现 |
| 运输丢货 | 本场 5/5 未丢货（grip 保持良好） |

## 5. 结论

1. 当前代码（放置触桌检测 + 导航恢复加固 + 速度 0.90）在**简单障碍布局**下
   五订单完整流程可 5/5 通过；
2. 放置阶段"压桌卡 30s"问题已解决（触桌即松爪）；
3. **剩余主要风险**：复杂障碍布局下咽喉点视觉幽灵卡死仍可能复现
   （前两场 1/5、0/5 均为该原因），修复方案见分析文档第 4 节。
