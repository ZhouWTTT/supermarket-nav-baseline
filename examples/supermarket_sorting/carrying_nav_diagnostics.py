"""Read-only evidence capture for carrying-navigation planning failures.

This module copies navigation state and writes diagnostic artifacts.  It does
not mutate the costmap, planner, controller, goal, or robot commands.
"""

from __future__ import annotations

from collections import deque
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import distance_transform_edt, maximum_filter


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _component_labels(free: np.ndarray) -> tuple[np.ndarray, int]:
    """Label 8-connected free-space components without changing *free*."""
    labels = np.zeros(free.shape, dtype=np.int32)
    component = 0
    height, width = free.shape
    neighbors = (
        (-1, -1), (0, -1), (1, -1),
        (-1, 0), (1, 0),
        (-1, 1), (0, 1), (1, 1),
    )
    for row, col in np.argwhere(free):
        row, col = int(row), int(col)
        if labels[row, col] != 0:
            continue
        component += 1
        labels[row, col] = component
        queue = deque([(row, col)])
        while queue:
            current_row, current_col = queue.popleft()
            for dcol, drow in neighbors:
                next_row = current_row + drow
                next_col = current_col + dcol
                if not (0 <= next_row < height and 0 <= next_col < width):
                    continue
                if (not free[next_row, next_col]
                        or labels[next_row, next_col] != 0):
                    continue
                if drow != 0 and dcol != 0:
                    # Match AStarPlanner: diagonal motion may not cut between
                    # two occupied orthogonal cells.
                    if (not free[current_row, next_col]
                            or not free[next_row, current_col]):
                        continue
                labels[next_row, next_col] = component
                queue.append((next_row, next_col))
    return labels, component


def _effective_cell(costmap, planner, world_xy) -> dict[str, Any]:
    world_x, world_y = map(float, world_xy)
    exact_col, exact_row = costmap.world_to_grid(world_x, world_y)
    exact_free = costmap.is_free_grid(exact_col, exact_row)
    if exact_free:
        effective_col, effective_row = exact_col, exact_row
        displacement = 0.0
    else:
        effective_col, effective_row = planner._nearest_free(
            exact_col, exact_row)
        if effective_col is None:
            return {
                "world": [world_x, world_y],
                "exact_grid": [exact_col, exact_row],
                "exact_free": False,
                "effective_grid": None,
                "effective_world": None,
                "nearest_free_displacement_m": None,
            }
        effective_world = costmap.grid_to_world(
            effective_col, effective_row)
        displacement = math.dist((world_x, world_y), effective_world)
    effective_world = costmap.grid_to_world(effective_col, effective_row)
    return {
        "world": [world_x, world_y],
        "exact_grid": [exact_col, exact_row],
        "exact_free": bool(exact_free),
        "effective_grid": [effective_col, effective_row],
        "effective_world": list(map(float, effective_world)),
        "nearest_free_displacement_m": round(float(displacement), 6),
    }


def build_failure_evidence(
        navigator, start_xy, goal_xy, *, table_bounds=None,
        lethal_cost: int = 100) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Return metadata and immutable layer copies for one failed plan."""
    costmap = navigator.costmap
    planner = navigator.planner

    static = np.array(costmap.static, copy=True)
    lidar = np.array(costmap.dynamic_raw, copy=True)
    vision = np.array(costmap.vision_raw, copy=True)
    lidar_inflated = np.zeros_like(lidar)
    inflated_hits = maximum_filter(
        (lidar == lethal_cost).astype(np.int8), footprint=costmap._disk)
    lidar_inflated[inflated_hits > 0] = lethal_cost
    inflated = np.array(costmap.dynamic, copy=True)
    master = np.array(costmap.master, copy=True)

    start = _effective_cell(costmap, planner, start_xy)
    goal = _effective_cell(costmap, planner, goal_xy)
    free = master < lethal_cost
    labels, component_count = _component_labels(free)

    def component_for(endpoint: dict[str, Any]) -> int | None:
        grid = endpoint.get("effective_grid")
        if grid is None:
            return None
        col, row = grid
        return int(labels[row, col]) or None

    start_component = component_for(start)
    goal_component = component_for(goal)
    goal_grid = goal.get("effective_grid")
    goal_obstacle_clearance = None
    if goal_grid is not None:
        col, row = goal_grid
        clearance_cells = distance_transform_edt(free)
        goal_obstacle_clearance = round(
            float(clearance_cells[row, col] * costmap.resolution), 6)

    goal_table_clearance = None
    if table_bounds is not None:
        xmin, ymin, xmax, ymax = map(float, table_bounds)
        goal_x, goal_y = map(float, goal_xy)
        dx = max(xmin - goal_x, 0.0, goal_x - xmax)
        dy = max(ymin - goal_y, 0.0, goal_y - ymax)
        goal_table_clearance = round(math.hypot(dx, dy), 6)

    controller = navigator.controller
    metadata = {
        "schema_version": 1,
        "diagnostic_kind": "carrying_navigation_failure",
        "read_only_capture": True,
        "map": {
            "origin": [float(costmap.origin_x), float(costmap.origin_y)],
            "resolution_m": float(costmap.resolution),
            "width": int(costmap.width),
            "height": int(costmap.height),
            "lethal_cost": int(lethal_cost),
        },
        "start": start,
        "goal": goal,
        "start_component": start_component,
        "goal_component": goal_component,
        "same_component": (
            start_component is not None
            and start_component == goal_component),
        "free_component_count": component_count,
        "goal_clearance": {
            "nearest_master_obstacle_m": goal_obstacle_clearance,
            "delivery_table_costmap_m": goal_table_clearance,
        },
        "obstacle_counts": {
            "static_lethal": int(np.count_nonzero(static == lethal_cost)),
            "lidar_raw_lethal": int(np.count_nonzero(lidar == lethal_cost)),
            "vision_raw_lethal": int(np.count_nonzero(vision == lethal_cost)),
            "lidar_inflated_lethal": int(np.count_nonzero(
                lidar_inflated == lethal_cost)),
            "inflated_lethal": int(np.count_nonzero(
                inflated == lethal_cost)),
            "master_lethal": int(np.count_nonzero(master == lethal_cost)),
        },
        "planner": {
            "stop_reason": controller.stop_reason,
            "last_plan_mode": controller._last_plan_mode,
            "full_failure": controller._last_plan_full_failure,
            "fallback_failure": controller._last_plan_fallback_failure,
        },
    }
    layers = {
        "static_costmap": static,
        "lidar_costmap": lidar,
        "inflated_costmap": inflated,
        "master_costmap": master,
        "lidar_inflated_costmap": lidar_inflated,
        "component_labels": labels,
    }
    return metadata, layers


def build_controller_trace(
        navigator, current_pose, goal_pose, cmd_vel, *, time_now: float,
        stuck_duration_s: float, payload_state=None) -> dict[str, Any]:
    """Return one read-only local-controller observation.

    ``path_index`` is the closest point on the controller's currently
    installed path.  The helper deliberately does not request a plan or call
    the controller update function, so trace collection cannot influence the
    navigation transaction.
    """
    controller = navigator.controller
    current_x, current_y, current_yaw = map(float, current_pose)
    goal_x, goal_y, goal_yaw = map(float, goal_pose)
    linear, angular = map(float, cmd_vel)
    distance_error = math.hypot(goal_x - current_x, goal_y - current_y)
    goal_yaw_error = math.atan2(
        math.sin(goal_yaw - current_yaw),
        math.cos(goal_yaw - current_yaw))

    path_index = None
    local_heading_error = None
    local_target = None
    path = controller.path
    if path:
        path_index = int(controller._closest_index(current_x, current_y))
        target_index = min(path_index + 1, len(path) - 1)
        local_target = list(map(float, path[target_index]))
        local_heading = math.atan2(
            local_target[1] - current_y,
            local_target[0] - current_x)
        local_heading_error = math.atan2(
            math.sin(local_heading - current_yaw),
            math.cos(local_heading - current_yaw))

    controller_stuck_duration = None
    if controller._last_progress_time is not None:
        controller_stuck_duration = max(
            0.0, float(time_now) - float(controller._last_progress_time))

    return {
        "time_s": round(float(time_now), 6),
        "current_pose": [current_x, current_y, current_yaw],
        "goal_pose": [goal_x, goal_y, goal_yaw],
        "distance_error_m": round(distance_error, 6),
        "goal_yaw_error_rad": round(goal_yaw_error, 6),
        "local_heading_error_rad": (
            None if local_heading_error is None
            else round(local_heading_error, 6)),
        "cmd_vel": {
            "linear_mps": round(linear, 6),
            "angular_radps": round(angular, 6),
        },
        "planned_path_exists": bool(path),
        "path_index": path_index,
        "path_point_count": len(path),
        "local_target": local_target,
        "stuck_duration_s": round(max(0.0, float(stuck_duration_s)), 6),
        "controller_stuck_duration_s": (
            None if controller_stuck_duration is None
            else round(controller_stuck_duration, 6)),
        "stop_reason": controller.stop_reason,
        "plan_mode": controller._last_plan_mode,
        "full_plan_failure": controller._last_plan_full_failure,
        "fallback_plan_failure": controller._last_plan_fallback_failure,
        "lidar_clearance_m": round(float(controller.lidar_clearance), 6),
        "rotation_accum_rad": round(float(controller._rotation_accum), 6),
        "rotation_recoveries": int(controller._rotation_recoveries),
        "payload": _json_value(payload_state),
    }


def save_failure_evidence(
        directory: str | Path, metadata: dict[str, Any],
        layers: dict[str, np.ndarray]) -> Path:
    """Persist one already-copied evidence bundle and return metadata path."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    artifacts = {}
    for name, array in layers.items():
        path = directory / f"{name}.npy"
        np.save(path, np.asarray(array), allow_pickle=False)
        artifacts[name] = path.name
    document = dict(metadata)
    document["artifacts"] = artifacts
    metadata_path = directory / "metadata.json"
    metadata_path.write_text(
        json.dumps(_json_value(document), ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8")
    return metadata_path
