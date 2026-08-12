# 路径记忆实验说明

本实验在不改原始 baseline 主流程的前提下，为导航模块增加了轻量级路径记忆功能，用于复用货架区与送货区之间的已走通路径。

## 当前已实现功能

- 导航成功后自动保存路径
- 同时保存正向路径和反向路径
- 下一次导航时优先尝试复用缓存路径
- 支持近邻匹配，允许起点和终点存在小范围偏差
- 日志中可直接看到是否命中缓存、是否实际启用缓存路径

## 相关文件

- `examples/supermarket_sorting/path_memory.py`
- `examples/supermarket_sorting/supermarket_navigation.py`
- `examples/supermarket_sorting/supermarket_navigation_demo.py`

## 已验证结论

已经完成单轮实验验证，路线为：

`起点 -> 货架D -> 送货区 -> 货架D`

验证目标：

- 先学习 `货架D -> 送货区`
- 再在同一轮内，不提前跑过 `送货区 -> 货架D`
- 直接通过反向路径记忆完成返回

实验结果：

- 日志显示 `cache_hit=True`
- 日志显示 `cached_path_active=True`
- 说明 `送货区 -> 货架D` 已成功使用反向路径记忆

## 当前测试结果

对比最后一段 `送货区 -> 货架D`：

- 无记忆：`15.802 s`
- 有记忆：`14.540 s`

本次实验中，返回货架阶段提速 `1.262 s`。

## 日志检查方法

可通过以下日志确认是否命中路径记忆：

```bash
grep -E "path_memory_goal|path_memory_reached|cache_hit|cached_path_active|matched_key|nearby_match" nav_ab_with_memory.log
```

命中时应看到类似结果：

- `cache_hit: True`
- `cached_path_active: True`
- `reason: nearby_match`

## 说明

本实验当前主要用于验证“导航路径记忆”本身是否有效，不涉及抓取与放置动作优化。
