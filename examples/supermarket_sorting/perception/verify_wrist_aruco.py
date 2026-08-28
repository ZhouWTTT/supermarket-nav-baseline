#!/usr/bin/env python3
"""Offline wrist-camera ArUco visibility verification.

Checks whether the MMK2 wrist cameras (lft_handeye / rgt_handeye, RGB only)
can see the shelf ArUco markers at the grasp poses the controller actually
computes.  It takes the per-order target slots from a recent competition
runner log, rebuilds the contact/pregrasp poses with the controller's own
KDL IK, renders the wrist camera views offline through MMK2FK, runs
cv2.aruco decode, and cross-checks with the perception pipeline's
FK + camera-intrinsics projection.

No ROS and no live simulator server are required.

Example::

    python3 perception/verify_wrist_aruco.py \
        --log ../../logs/competition_runner_20260827_111228.log
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import cv2
import numpy as np


HERE = Path(__file__).resolve().parent
MODULE_DIR = HERE.parent  # examples/supermarket_sorting
REPO_ROOT = HERE.parents[2]
XML_PATH = MODULE_DIR / "mjcf" / "retail_competition.xml"
LAYOUT_PATH = MODULE_DIR / "retail_competition_layout.json"

sys.path.insert(0, str(MODULE_DIR))
sys.path.insert(0, str(REPO_ROOT))

try:
    import integrated_nav_pick_place as flow
except ImportError as exc:  # Host without ROS/discoverse dependencies.
    flow = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None
pick = flow.pick if flow is not None else None

from discoverse.robots.mmk2.mmk2_fk import MMK2FK  # noqa: E402
from discoverse.utils import camera2k  # noqa: E402


WRIST_FOVY_DEG = 42.58
IMG_W, IMG_H = 640, 480
MARKER_SIZE_M = 0.03
# kele_detect applies this extra pitch on top of the left_cam/right_cam site
# pose to reproduce the MuJoCo camera mounting from arm_left/right.xml.
WRIST_CAMERA_MOUNT_PITCH_RAD = -0.5236

# Neutral arm poses seen in the runner logs while the other arm works.
NEUTRAL_ARM = {
    "l": np.array([0.0, -0.166, 0.032, 0.0, 1.571, 2.223]),
    "r": np.array([0.0, -0.166, 0.032, 0.0, -1.571, -2.223]),
}


def load_marker_world_positions(xml_path: Path) -> dict[int, np.ndarray]:
    """Marker world positions exactly as placed in the MuJoCo scene."""
    import xml.etree.ElementTree as ET

    root = ET.parse(str(xml_path)).getroot()
    worldbody = root.find("worldbody")
    markers: dict[int, np.ndarray] = {}
    for shelf in worldbody.findall("body"):
        name = str(shelf.get("name", ""))
        if not re.fullmatch(r"shelf_[A-E]", name):
            continue
        sx, sy, sz = (float(v) for v in str(shelf.get("pos")).split())
        for child in shelf.findall("body"):
            child_name = str(child.get("name", ""))
            if not re.fullmatch(r"aruco_[A-E]_L[123]_C[123]", child_name):
                continue
            cx, cy, cz = (float(v) for v in str(child.get("pos")).split())
            geom = child.find("geom")
            material = str(geom.get("material", "")) if geom is not None else ""
            marker_id = int(material.rsplit("_", 1)[-1])
            markers[marker_id] = np.array(
                [sx + cx, sy + cy, sz + cz], dtype=float)
    return markers


def load_slot_marker_map(layout_path: Path) -> dict[tuple, int]:
    result = {}
    for slot in json.loads(layout_path.read_text(encoding="utf-8")):
        result[(str(slot["shelf"]), str(slot["level"]),
                str(slot["column"])[-1])] = int(slot["aruco_id"])
    return result


def yaw_quat(yaw: float) -> list[float]:
    return [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]


def parse_runner_log(path: Path) -> list[dict]:
    """Extract per-order target slot + kind from a runner log."""
    lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
    starts = [i for i, line in enumerate(lines) if "starting order" in line]
    orders = []
    for idx, i0 in enumerate(starts):
        i1 = starts[idx + 1] if idx + 1 < len(starts) else len(lines)
        seg = lines[i0:i1]
        text = "\n".join(seg)
        kind_m = re.search(
            r"starting order id=\S+ kind=(\S+)", text)
        marker_m = re.search(r"\[localised\] source=\S+ marker=(\d+)", text)
        slot_m = re.search(
            r"slot=\('([^']+)', '([^']+)', '([^']+)'\)", text)
        slot = None
        if slot_m:
            slot = (slot_m.group(1), slot_m.group(2), slot_m.group(3))
        orders.append({
            "kind": kind_m.group(1) if kind_m else None,
            "slot": slot,
            "marker": int(marker_m.group(1)) if marker_m else None,
        })
    return orders


def grasp_poses_for_slot(slot, kind, kdl) -> list[tuple[str, dict]]:
    """Reconstruct contact/pregrasp poses with the controller's own IK."""
    shelf, level, column = slot
    level_name = {"L3": "top", "L2": "middle", "L1": "lower"}[level]
    surface_z = float(pick.SHELF_SURFACE_Z_M[level_name])
    half_height = float(pick.PRODUCT_HALF_HEIGHT_M.get(kind, 0.0))
    target_z = surface_z + half_height
    shelf_x = float(pick.SCAN_X[pick.ShelfPickController._scan_x_index_for_shelf(
        shelf)])
    target_x = shelf_x + {"1": -0.22, "2": 0.00, "3": 0.22}[column]
    target_y = pick.SHELF_PRODUCT_CENTER_Y_M
    target = np.array([target_x, target_y, target_z], dtype=float)

    arm = "l" if shelf == "A" else "r"
    if arm == "l":
        base_x = target_x + pick.ARM_LATERAL_BIAS_M
    else:
        base_x = target_x - pick.ARM_LATERAL_BIAS_M
    if level_name == "top":
        base_y = target_y - pick.TOP_GRASP_CENTER_DISTANCE_M
    else:
        base_y = pick.SCAN_Y
    base_yaw = pick.YAW_NORTH
    slide = float(np.clip(
        pick.SLIDE_REFERENCE_COMMAND
        - (target_z - pick.SLIDE_REFERENCE_Z_M),
        pick.SLIDE_MIN, pick.SLIDE_MAX))
    reference = NEUTRAL_ARM[arm]

    def solve(world):
        delta = np.asarray(world, dtype=float) - np.array(
            [base_x, base_y, 0.0])
        cosine, sine = math.cos(-base_yaw), math.sin(-base_yaw)
        footprint = np.array([
            cosine * delta[0] - sine * delta[1],
            sine * delta[0] + cosine * delta[1],
            delta[2],
        ])
        target_t = np.eye(4)
        target_t[:3, 3] = footprint
        ref_with_slide = np.concatenate(([slide], reference))
        if arm == "r":
            solutions = kdl.inverse_kinematics(
                T_right=target_t, target_height=slide,
                ref_pos=ref_with_slide)
        else:
            solutions = kdl.inverse_kinematics(
                T_left=target_t, target_height=slide,
                ref_pos=ref_with_slide)
        if not solutions:
            return None
        candidates = [np.asarray(item[1:], dtype=float) for item in solutions]
        return min(
            candidates,
            key=lambda q: float(np.max(np.abs(q - reference))))

    # Contact: wrist at the product centre (upright), plus a pregrasp 12 cm
    # behind, matching the sphere profile's backoff.
    contact = target.copy()
    contact[2] += pick.GRASP_TCP_Z_RAISE_M
    pregrasp = contact.copy()
    pregrasp[1] -= pick.SPHERE_PREGRASP_BACKOFF_M
    poses = []
    for name, world in (("pregrasp", pregrasp), ("contact", contact)):
        joints = solve(world)
        if joints is None:
            continue
        poses.append((name, {
            "arm": arm,
            "base": [base_x, base_y, base_yaw],
            "slide": slide,
            "joints": joints,
            "tcp_target": world,
        }))
    return poses


def detect_aruco(image_rgb: np.ndarray) -> dict:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(
            dictionary, cv2.aruco.DetectorParameters())
        corners, ids, _ = detector.detectMarkers(gray)
    else:  # pragma: no cover - older OpenCV fallback
        corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary)
    decoded = {}
    if ids is not None:
        for corner, marker_id in zip(corners, ids.flatten()):
            pts = np.asarray(corner[0], dtype=float)
            decoded[int(marker_id)] = {
                "center_px": tuple(np.round(pts.mean(axis=0), 1)),
                "size_px": round(
                    float(np.linalg.norm(pts[0] - pts[1])), 1),
                "corners": np.round(pts, 1).tolist(),
            }
    return decoded


def wrist_projection(fk, marker_world: np.ndarray,
                     camera_id: int) -> dict | None:
    """Project a marker with the perception pipeline's wrist camera model.

    Uses the FK left/right_cam site pose plus the -30 deg mounting pitch,
    exactly like kele_detect/aruco_detect.  This model was verified against
    offline renders: it projects the rendered product centre to the same
    wrist-image pixel within ~1 cm.
    """
    from scipy.spatial.transform import Rotation
    arm = "l" if camera_id == 1 else "r"
    pose_getter = (
        fk.get_left_camera_pose if arm == "l"
        else fk.get_right_camera_pose)
    pos, quat_wxyz = pose_getter()
    T = np.eye(4)
    T[:3, 3] = np.asarray(pos, dtype=float)
    T[:3, :3] = Rotation.from_quat(
        np.asarray(quat_wxyz)[[1, 2, 3, 0]]).as_matrix()
    T[:3, :3] = T[:3, :3] @ Rotation.from_euler(
        "x", WRIST_CAMERA_MOUNT_PITCH_RAD).as_matrix()
    pc = np.linalg.inv(T) @ np.r_[np.asarray(marker_world, float), 1.0]
    if pc[2] <= 1e-6:
        return None
    K = camera2k(math.radians(WRIST_FOVY_DEG), IMG_W, IMG_H)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u = fx * pc[0] / pc[2] + cx
    v = fy * pc[1] / pc[2] + cy
    return {
        "u": round(float(u), 1),
        "v": round(float(v), 1),
        "in_frame": bool(
            pc[2] > 0 and 0 <= u < IMG_W and 0 <= v < IMG_H),
        "depth_m": round(float(pc[2]), 3),
        "size_px_est": round(float(MARKER_SIZE_M * fx / pc[2]), 1),
    }


def render_wrist(fk: MMK2FK, renderer, rec: dict, camera_id: int) -> np.ndarray:
    base = rec["base"]
    fk.set_base_pose([float(base[0]), float(base[1]), 0.0],
                     yaw_quat(float(base[2])))
    fk.set_slide_joint(float(rec["slide"]))
    fk.set_head_joints([0.0, 0.0])
    arm = str(rec["arm"])
    joints = np.asarray(rec["joints"], dtype=float)
    if arm == "l":
        fk.set_left_arm_joints(joints)
        fk.set_right_arm_joints(NEUTRAL_ARM["r"])
    else:
        fk.set_left_arm_joints(NEUTRAL_ARM["l"])
        fk.set_right_arm_joints(joints)
    fk.forward_kinematics()
    renderer.update_scene(fk.mj_data, camera=camera_id)
    return renderer.render()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log", action="append", default=[],
        help="competition runner log(s); defaults to the newest one")
    parser.add_argument(
        "--out", default=Path("/tmp") / "wrist_aruco_verify")
    args = parser.parse_args()

    log_dir = REPO_ROOT / "logs"
    logs = args.log or []
    if not logs:
        candidates = sorted(log_dir.glob("competition_runner_2026*.log"))
        if candidates:
            logs = [str(candidates[-1])]
    if not logs:
        parser.error("no runner log found")
    if pick is None:
        parser.error(
            f"controller dependencies unavailable: {IMPORT_ERROR}")

    markers = load_marker_world_positions(XML_PATH)
    slot_markers = load_slot_marker_map(LAYOUT_PATH)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    fk = MMK2FK()
    import mujoco
    renderer = mujoco.Renderer(fk.mj_model, IMG_H, IMG_W)
    cam_id = {
        "l": mujoco.mj_name2id(
            fk.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, "lft_handeye"),
        "r": mujoco.mj_name2id(
            fk.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, "rgt_handeye"),
    }

    summary = []
    for log_path in logs:
        orders = parse_runner_log(Path(log_path))
        kdl = pick.MMK2Kdl()
        for order_index, order in enumerate(orders, start=1):
            if order["slot"] is None:
                continue
            expected = (
                order["marker"]
                if order["marker"] is not None
                else slot_markers.get(order["slot"]))
            poses = grasp_poses_for_slot(
                order["slot"], order["kind"], kdl)
            for pose_name, rec in poses:
                arm = str(rec["arm"])
                camera_id = cam_id[arm]
                image = render_wrist(fk, renderer, rec, camera_id)
                decoded = detect_aruco(image)
                proj = (
                    wrist_projection(fk, markers[expected], camera_id)
                    if expected is not None and expected in markers else None)
                expected_decoded = expected in decoded if expected else False
                # annotate
                vis = image.copy()
                if expected is not None and proj and proj["in_frame"]:
                    cv2.circle(
                        vis, (int(proj["u"]), int(proj["v"])), 8,
                        (0, 255, 0), 2)
                for marker_id, info in decoded.items():
                    cv2.putText(
                        vis, str(marker_id),
                        (int(info["center_px"][0]) - 10,
                         int(info["center_px"][1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255), 2)
                png = (
                    out_dir
                    / f"{Path(log_path).stem}_order{order_index}"
                    / f"{pose_name}_{arm}cam.png")
                png.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(png), vis[:, :, ::-1])
                row = {
                    "log": Path(log_path).name,
                    "order": order_index,
                    "kind": order["kind"],
                    "slot": order["slot"],
                    "stage": pose_name,
                    "arm": arm,
                    "expected_marker": expected,
                    "decoded_ids": sorted(decoded),
                    "expected_decoded": expected_decoded,
                    "projection": proj,
                    "image": str(png),
                }
                summary.append(row)
                print(
                    f"order{order_index} {order['kind']} "
                    f"slot={order['slot']} {pose_name:8s} arm={arm} "
                    f"expected={expected} decoded={sorted(decoded)} "
                    f"expected_decoded={expected_decoded} "
                    f"proj={proj}")

    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1),
        encoding="utf-8")

    # Aggregate: how often was the expected marker in the wrist FOV / decoded.
    with_proj = [r for r in summary if r["projection"] is not None]
    in_frame = [r for r in with_proj if r["projection"]["in_frame"]]
    decoded_expected = [r for r in summary if r["expected_decoded"]]
    print("\n==== summary ====")
    print(f"sampled wrist poses: {len(summary)}")
    print(f"expected marker projected: {len(with_proj)}")
    print(f"  ...inside wrist FOV: {len(in_frame)}")
    print(f"expected marker decoded by wrist ArUco: {len(decoded_expected)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
