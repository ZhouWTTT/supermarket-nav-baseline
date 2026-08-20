#!/usr/bin/env python3
"""超市分拣任务的 ROS2 server。

加载 retail_competition 场景，复用本仓库 examples/ros2/mmk2_ros2.py 的
MMK2ROS2 发布相机、里程计、关节状态等标准话题，供
supermarket_sorting_client.py 控制机器人完成抓取放置。
"""
import json
import math
import os
import secrets
import sys
import threading
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import rclpy
from rclpy._rclpy_pybind11 import RCLError
from rclpy.executors import ExternalShutdownException
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

# 可迁移:从脚本自身位置推导示例目录和仓库根目录
TASK_DIR = Path(__file__).resolve().parent
REPO_ROOT = TASK_DIR.parents[1]
ROS2_EXAMPLES_DIR = REPO_ROOT / "examples" / "ros2"
if str(ROS2_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(ROS2_EXAMPLES_DIR))

ASSETS_DIR = TASK_DIR / "models"
os.environ["DISCOVERSE_ASSETS_DIR"] = str(ASSETS_DIR)

from discoverse.robots_env.mmk2_base import MMK2Cfg
from mmk2_ros2 import MMK2ROS2
from obstacle_layout import (
    ROBOT_CLEARANCE_RADIUS,
    SAFE_CLEARANCE_RADIUS,
    generate_obstacle_layout,
)

SOURCE_XML = TASK_DIR / "mjcf" / "retail_competition.xml"
RUNTIME_XML = Path("/tmp/retail_competition_ros2_runtime.xml")
LAYOUT_JSON = TASK_DIR / "retail_competition_layout.json"
# Start just south of the picking aisle for fast perception/grasp experiments.
# Environment overrides retain the option to restore the original competition
# start without editing code: SUPERMARKET_START_X=1.92, START_Y=-3.17.
START_XY = np.array([
    float(os.getenv("SUPERMARKET_START_X", "1.80")),
    float(os.getenv("SUPERMARKET_START_Y", "1.55")),
], dtype=float)


def env_flag(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def positive_env_float(name, default):
    value = os.getenv(name)
    result = float(default if value is None else value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a positive finite number, got {value!r}")
    return result


def configure_display_camera(node):
    """Select the GLFW window viewpoint without changing published cameras."""
    requested = os.getenv("SUPERMARKET_DISPLAY_CAMERA", "free").strip().lower()
    # The overhead diagnostic view must show collision-only objects such as the
    # randomized boxes, floor markings and delivery table. They have no 3DGS
    # assets, so use MuJoCo only for the window while sensor cameras retain GS.
    node.force_mujoco_display = requested == "top"
    node.dual_renderer_display = requested == "top_dual"
    aliases = {
        "free": -1,
        "head": "head_cam",
        "operator": "diagnostic_operator",
        "follow": "diagnostic_follow",
        "top": "diagnostic_top",
        "top_gs": "diagnostic_top",
        "top_dual": "diagnostic_top",
        "left": "lft_handeye",
        "right": "rgt_handeye",
    }
    selected = aliases.get(requested, requested)
    if selected == -1:
        node.cam_id = -1
        print("[server] display camera: free")
        return

    if isinstance(selected, str) and selected.lstrip("-").isdigit():
        camera_id = int(selected)
    else:
        try:
            camera_id = node.camera_names.index(selected)
        except ValueError as exc:
            valid = "free, operator, follow, top, top_gs, top_dual, head, left, right, " + ", ".join(node.camera_names)
            raise ValueError(
                f"Unknown SUPERMARKET_DISPLAY_CAMERA={requested!r}; valid values: {valid}"
            ) from exc

    if not 0 <= camera_id < len(node.camera_names):
        raise ValueError(
            f"SUPERMARKET_DISPLAY_CAMERA id {camera_id} is outside "
            f"0..{len(node.camera_names) - 1}"
        )
    node.cam_id = camera_id
    print(f"[server] display camera: {node.camera_names[camera_id]} (id={camera_id})")


def local_robot_gs_model_dict():
    gs_model_dict = {}
    for name, path in MMK2Cfg.gs_model_dict.items():
        if path.startswith("mobile_chassis/mmk2/"):
            gs_model_dict[name] = path.replace("mobile_chassis/mmk2/", "mmk2/")
        elif path.startswith("manipulator/airbot_play/"):
            gs_model_dict[name] = path.replace("manipulator/airbot_play/", "airbot_play/")
        else:
            gs_model_dict[name] = path
    return gs_model_dict


def resolve_background_ply():
    """Background 3DGS model (drawn at world identity; no MuJoCo body).

    SUPERMARKET_BACKGROUND_PLY overrides the path (relative to models/3dgs/, or
    absolute) -- handy while tuning tools_align_background.py knobs. Otherwise use
    the aligned scan retail_background_fit.ply if it has been baked, else fall back
    to the tiny dummy_background.ply placeholder."""
    override = os.getenv("SUPERMARKET_BACKGROUND_PLY")
    if override:
        return override
    fit = ASSETS_DIR / "3dgs" / "shentoon" / "retail_background_fit.ply"
    if fit.exists():
        return "shentoon/retail_background_fit.ply"
    return "shentoon/dummy_background.ply"


# 货架每层台面高度(世界 z)。物体静止 z = 台面 + 该物体自身半高。
SHELF_SURFACE = {"L1": 0.499, "L2": 0.851, "L3": 1.189}
DEFAULT_TASKS = "all"


def write_runtime_xml(
        pos_overrides=None, obstacle_yaw_overrides=None, body_name_overrides=None):
    """Render the runtime MJCF. If pos_overrides is given, rewrite each named
    body's pos="x y z" so the whole body (collision geom + gs ply travel
    together) moves to its randomized shelf slot."""
    text = SOURCE_XML.read_text().replace("__REPO_ROOT__", str(TASK_DIR))
    if pos_overrides:
        import re
        for body_name, (x, y, z) in pos_overrides.items():
            pattern = re.compile(
                r'(<body name="' + re.escape(body_name) + r'"[^>]*?pos=")[^"]*(")'
            )
            text, n = pattern.subn(rf"\g<1>{x:.5f} {y:.5f} {z:.5f}\g<2>", text)
            if n != 1:
                raise RuntimeError(
                    f"randomize: expected exactly 1 body pos for {body_name}, got {n}")
    if obstacle_yaw_overrides:
        import re
        for body_name, yaw in obstacle_yaw_overrides.items():
            geom_name = f"{body_name}_collision"
            pattern = re.compile(
                r'(<geom[^>]*name="' + re.escape(geom_name) + r'"[^>]*?euler=")[^"]*(")'
            )
            text, n = pattern.subn(rf"\g<1>0 0 {yaw:.6f}\g<2>", text)
            if n != 1:
                raise RuntimeError(
                    f"randomize: expected exactly 1 obstacle yaw for {body_name}, got {n}")
    if body_name_overrides:
        for source_name, runtime_name in body_name_overrides.items():
            count = text.count(source_name)
            if count != 3:
                raise RuntimeError(
                    f"body rename: expected 3 XML references for {source_name}, got {count}")
            text = text.replace(source_name, runtime_name)
    RUNTIME_XML.write_text(text)
    return str(RUNTIME_XML)


def randomize_positions(layout, seed=None, anchored_body=None):
    """Shuffle which shelf slot each object body occupies.

    Each body keeps its own collision geom AND its own gs ply (they stay bound);
    only the body's world position moves to another slot.  The new z is the new
    shelf surface plus the body's intrinsic half-height (derived from its
    original z), so the object rests on the shelf instead of clipping/floating.
    By default every product body participates in the shuffle.  An anchored body
    can be supplied explicitly for debugging, but normal competition startup
    should leave it unset.

    Returns (new_layout, pos_overrides) where pos_overrides maps body name ->
    (x, y, z) for the runtime MJCF rewrite.
    """
    import random
    rng = random.Random(seed)
    # target slot positions (x, y, level) taken from the original layout
    slots = [(s["world_position"][0], s["world_position"][1], s["level"]) for s in layout]
    # each body's intrinsic half-height = original z - its original shelf surface
    half_h = [s["world_position"][2] - SHELF_SURFACE[s["level"]] for s in layout]

    anchored_i = None
    if anchored_body:
        for i, slot in enumerate(layout):
            if slot["body"] == anchored_body:
                anchored_i = i
                break
        if anchored_i is None:
            raise RuntimeError(f"randomize: anchored body not found: {anchored_body}")

    body_indices = list(range(len(layout)))
    slot_indices = list(range(len(layout)))
    if anchored_i is not None:
        body_indices.remove(anchored_i)
        slot_indices.remove(anchored_i)

    # Make the shuffle a derangement: if a debug anchor is set it stays put,
    # and every other product is forced into a different slot.
    rng.shuffle(slot_indices)
    fixed = [i for i, (body_i, slot_i) in enumerate(zip(body_indices, slot_indices))
             if body_i == slot_i]
    if len(fixed) == 1 and len(slot_indices) > 1:
        i = fixed[0]
        j = 0 if i != 0 else 1
        slot_indices[i], slot_indices[j] = slot_indices[j], slot_indices[i]
    elif len(fixed) > 1:
        fixed_slots = [slot_indices[i] for i in fixed]
        fixed_slots = fixed_slots[1:] + fixed_slots[:1]
        for i, slot_i in zip(fixed, fixed_slots):
            slot_indices[i] = slot_i
    order = {body_i: slot_i for body_i, slot_i in zip(body_indices, slot_indices)}
    if anchored_i is not None:
        order[anchored_i] = anchored_i

    new_layout, pos_overrides = [], {}
    for body_i in range(len(layout)):
        slot_i = order[body_i]
        x, y, level = slots[slot_i]
        z = SHELF_SURFACE[level] + half_h[body_i]
        body = layout[body_i]["body"]
        pos_overrides[body] = (x, y, z)
        ns = layout[body_i].copy()
        for key in ("shelf", "level", "column", "aruco_id"):
            ns[key] = layout[slot_i][key]
        ns["world_position"] = [x, y, z]
        new_layout.append(ns)
    return new_layout, pos_overrides


def anonymize_runtime_bodies(layout):
    """Give every product an opaque body name for this Server process.

    The cryptographic prefix is intentionally independent from SUPERMARKET_SEED:
    a fixed seed reproduces the physical layout, but never a previous run's
    product-body identifiers.
    """
    run_prefix = f"run_{secrets.token_hex(6)}"
    body_name_overrides = {}
    runtime_layout = []
    for index, slot in enumerate(layout, start=1):
        runtime_body = f"item_{run_prefix}_{index:02d}"
        body_name_overrides[slot["body"]] = runtime_body
        runtime_slot = slot.copy()
        runtime_slot["body"] = runtime_body
        runtime_layout.append(runtime_slot)
    return run_prefix, runtime_layout, body_name_overrides


def optional_seed(name):
    value = os.getenv(name)
    if value is None or not value.strip():
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def select_tasks(layout):
    """选择本局下发的任务目标(body 名)。

    默认下发全场商品；SUPERMARKET_TASKS 可使用逗号分隔的 object_kind
    或具体 body 名覆盖。
    """
    spec = os.getenv("SUPERMARKET_TASKS", DEFAULT_TASKS).strip()
    if not spec or spec.lower() in {"all", "*"}:
        return [slot["body"] for slot in layout]
    tokens = [t.strip() for t in spec.replace(";", ",").split(",") if t.strip()]
    bodies = {slot["body"] for slot in layout}
    kinds = {}
    for slot in layout:
        kinds.setdefault(slot["object_kind"], []).append(slot["body"])
    out = []
    unknown = []
    for tok in tokens:
        if tok in bodies:
            out.append(tok)
        elif tok in kinds:
            out.extend(kinds[tok])
        else:
            unknown.append(tok)
    if unknown:
        raise ValueError(
            "SUPERMARKET_TASKS contains unknown targets: " + ", ".join(unknown))
    seen, res = set(), []
    for b in out:
        if b not in seen:
            seen.add(b)
            res.append(b)
    return res


def build_config():
    cfg = MMK2Cfg()
    cfg.use_gaussian_renderer = env_flag("SUPERMARKET_USE_GS", True)
    # 商品外观保存在 3DGS PLY 中。请求 GS 时禁止静默退化为红色碰撞体，
    # 否则依赖缺失很容易被误认为贴图路径或材质绑定错误。
    cfg.require_gaussian_renderer = cfg.use_gaussian_renderer
    cfg.gs_render_sequential = env_flag("SUPERMARKET_GS_SEQUENTIAL", True)
    cfg.enable_render = env_flag("SUPERMARKET_ENABLE_RENDER", True)
    cfg.headless = env_flag("SUPERMARKET_HEADLESS", False)

    # 货架场景的 3DGS 绑定:保留 MMK2Cfg 默认的机器人 link 绑定,追加 background + 货架物体
    layout = json.loads(LAYOUT_JSON.read_text())

    # 随机摆放功能(默认开启,给选手用):整把物体(碰撞geom+3DGS一起)随机搬到别的货架格子
    pos_overrides = {}
    if env_flag("SUPERMARKET_RANDOMIZE", True):
        seed = optional_seed("SUPERMARKET_SEED")
        anchored_body = os.getenv("SUPERMARKET_ANCHORED_TARGET", "").strip() or None
        layout, pos_overrides = randomize_positions(layout, seed, anchored_body)
        anchored_msg = anchored_body if anchored_body else "none"
        print(f"[server] randomized object positions (seed={seed}, anchored={anchored_msg})")
    else:
        print("[server] fixed layout (SUPERMARKET_RANDOMIZE=0)")

    if env_flag("SUPERMARKET_RANDOMIZE_OBSTACLES", True):
        obstacle_seed = optional_seed("SUPERMARKET_OBSTACLE_SEED")
        if obstacle_seed is None:
            product_seed = optional_seed("SUPERMARKET_SEED")
            obstacle_seed = None if product_seed is None else product_seed + 1000003
        safe_layout = env_flag("SUPERMARKET_SAFE_OBSTACLE_LAYOUT", True)
        default_clearance = (
            SAFE_CLEARANCE_RADIUS if safe_layout else ROBOT_CLEARANCE_RADIUS)
        obstacle_clearance = positive_env_float(
            "SUPERMARKET_OBSTACLE_CLEARANCE", default_clearance)
        obstacle_layout = generate_obstacle_layout(
            obstacle_seed, clearance_radius=obstacle_clearance)
        pos_overrides.update(obstacle_layout.positions)
        print(
            f"[server] randomized corridor obstacles (seed={obstacle_seed}, "
            f"attempts={obstacle_layout.attempts}, "
            f"path={obstacle_layout.path_length:.2f} m, "
            f"detour={obstacle_layout.detour:.2f} m, "
            f"bands={obstacle_layout.occupied_bands}, "
            f"validated_clearance={obstacle_clearance:.2f} m, "
            f"safe_mode={int(safe_layout)})"
        )
    else:
        print("[server] fixed corridor obstacles (SUPERMARKET_RANDOMIZE_OBSTACLES=0)")

    selected_source_bodies = select_tasks(layout)
    source_kind_by_body = {slot["body"]: slot["object_kind"] for slot in layout}
    run_prefix, runtime_layout, body_name_overrides = anonymize_runtime_bodies(layout)

    cfg.mjcf_file_path = write_runtime_xml(
        pos_overrides,
        obstacle_layout.yaws if env_flag("SUPERMARKET_RANDOMIZE_OBSTACLES", True) else None,
        body_name_overrides,
    )

    cfg.obj_list = [slot["body"] for slot in runtime_layout]
    cfg.gs_model_dict = local_robot_gs_model_dict()
    cfg.gs_model_dict["background"] = resolve_background_ply()
    for slot in runtime_layout:
        cfg.gs_model_dict[slot["body"]] = slot["gs_ply"]

    cfg.obs_rgb_cam_id = [0, 1, 2]     # head / lft / rgt
    cfg.obs_depth_cam_id = [0]
    cfg.lidar_s2_sim = env_flag("SUPERMARKET_ENABLE_LIDAR", True)
    cfg.render_set = {"fps": 24, "width": 640, "height": 480}

    # 起始位姿:出发区,朝北(+Y)
    cfg.init_state["base_position"] = [float(START_XY[0]), float(START_XY[1]), 0.0]
    cfg.init_state["base_orientation"] = Rotation.from_euler("z", np.pi / 2.0).as_quat()[[3, 0, 1, 2]].tolist()

    cfg.task_run_prefix = run_prefix
    diagnostic_source = os.getenv(
        "SUPERMARKET_DIAGNOSTIC_TRACK_SOURCE", "").strip()
    cfg.diagnostic_track_body = (
        body_name_overrides.get(diagnostic_source)
        if diagnostic_source else None)
    cfg.task_targets = [
        {"id": body_name_overrides[source_body], "kind": source_kind_by_body[source_body]}
        for source_body in selected_source_bodies
    ]
    return cfg


class TaskMMK2ROS2(MMK2ROS2):
    """MMK2 ROS2 Server，只负责仿真、传感器、控制和任务下发。"""
    def __init__(self, config):
        super().__init__(config)
        task_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.task_publisher = self.create_publisher(
            String, "/supermarket_sorting/task", task_qos)
        self.task_message = json.dumps(
            {
                "schema_version": 1,
                "run_prefix": config.task_run_prefix,
                "count": len(config.task_targets),
                "targets": config.task_targets,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        self.task_publisher.publish(String(data=self.task_message))
        print(f"[server] task published: {self.task_message}")
        self.diagnostic_track_body = config.diagnostic_track_body
        self.diagnostic_track_last_time = -1.0


def spin_node(node):
    try:
        rclpy.spin(node)
    except (ExternalShutdownException, RCLError):
        pass


def main():
    rclpy.init()
    np.set_printoptions(precision=3, suppress=True, linewidth=500)

    exec_node = TaskMMK2ROS2(build_config())
    exec_node.reset()
    configure_display_camera(exec_node)

    spin_thread = threading.Thread(target=spin_node, args=(exec_node,), daemon=True)
    spin_thread.start()

    pubtopic_thread = threading.Thread(target=exec_node.thread_pubros2topic, args=(24,), daemon=True)
    pubtopic_thread.start()

    if exec_node.config.lidar_s2_sim:
        print("[server] lidar enabled: /slamware_ros_sdk_server_node/scan (12 Hz)")

    try:
        while rclpy.ok() and exec_node.running:
            exec_node.step(exec_node.target_control)
            if (exec_node.diagnostic_track_body
                    and exec_node.mj_data.time
                    >= exec_node.diagnostic_track_last_time + 0.25):
                body = exec_node.mj_data.body(
                    exec_node.diagnostic_track_body)
                print(
                    "[body-track] "
                    f"t={exec_node.mj_data.time:.2f} "
                    f"name={exec_node.diagnostic_track_body} "
                    f"xyz={np.round(body.xpos, 4)} "
                    f"quat={np.round(body.xquat, 4)}",
                    flush=True)
                target_body_id = exec_node.mj_model.body(
                    exec_node.diagnostic_track_body).id
                contact_pairs = set()
                for contact in exec_node.mj_data.contact:
                    geom1 = exec_node.mj_model.geom(contact.geom1)
                    geom2 = exec_node.mj_model.geom(contact.geom2)
                    body1 = exec_node.mj_model.body(geom1.bodyid).name
                    body2 = exec_node.mj_model.body(geom2.bodyid).name
                    target_contact = (
                        geom1.bodyid == target_body_id
                        or geom2.bodyid == target_body_id)
                    robot_scene_contact = (
                        (body1.startswith("rgt_")
                         and (body2.startswith("lft_")
                              or body2.startswith("shelf_")))
                        or (body2.startswith("rgt_")
                            and (body1.startswith("lft_")
                                 or body1.startswith("shelf_"))))
                    if target_contact or robot_scene_contact:
                        label1 = f"{body1}:{geom1.name or contact.geom1}"
                        label2 = f"{body2}:{geom2.name or contact.geom2}"
                        contact_pairs.add(
                            f"{label1}<->{label2} "
                            f"pos={np.round(contact.pos, 4)} "
                            f"dist={contact.dist:.5f} "
                            f"centres={np.round(exec_node.mj_data.geom_xpos[contact.geom1], 4)}/"
                            f"{np.round(exec_node.mj_data.geom_xpos[contact.geom2], 4)}")
                if contact_pairs:
                    print(
                        "[contact-track] "
                        f"t={exec_node.mj_data.time:.2f} "
                        f"pairs={sorted(contact_pairs)}",
                        flush=True)
                exec_node.diagnostic_track_last_time = exec_node.mj_data.time
    except KeyboardInterrupt:
        pass
    finally:
        exec_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
