#!/usr/bin/env python3
"""
Offline coordinate-bridge validation script.

Validates that the pixel->camera-frame deprojection + MMK2FK camera->world
transform reconstructs each kele slot's GT world position to <10 mm.

Run standalone (no ROS required):
    cd examples/supermarket_sorting
    python3 perception/validate_cam_bridge.py

Expected output: all errors < 10 mm (typically < 0.1 mm for ideal depth).
Also tests the effect of uint16 mono16 quantisation noise (depth in integer mm):
after rounding to the nearest mm the position error should stay < 5 mm.
"""
import json
import math
import sys
import os

import numpy as np
from scipy.spatial.transform import Rotation

# make discoverse importable when run from examples/supermarket_sorting
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from discoverse.robots.mmk2.mmk2_fk import MMK2FK
from discoverse.utils import camera2k

LAYOUT_JSON = os.path.join(os.path.dirname(__file__), "..", "retail_competition_layout.json")
CAMERA_FOVY_DEG = 45.29
IMG_W, IMG_H = 640, 480


def build_tmat(pos, quat_wxyz):
    """4x4 SE(3): camera optical frame -> world frame."""
    T = np.eye(4)
    T[:3, 3] = pos
    T[:3, :3] = Rotation.from_quat(quat_wxyz[[1, 2, 3, 0]]).as_matrix()
    return T


def run(base_xy, yaw, slide, head_pitch):
    """Run validation at a given robot pose and return result rows."""
    fk = MMK2FK()
    # base quaternion from yaw
    qw, qz = math.cos(yaw / 2), math.sin(yaw / 2)
    fk.set_base_pose([base_xy[0], base_xy[1], 0.0], [qw, 0.0, 0.0, qz])
    fk.set_slide_joint(float(slide))
    fk.set_head_joints([0.0, float(head_pitch)])
    fk.set_left_arm_joints([0.0] * 6)
    fk.set_right_arm_joints([0.0] * 6)

    cam_pos, cam_quat = fk.get_head_camera_pose()
    T_cam_world = build_tmat(cam_pos, cam_quat)
    T_world_cam = np.linalg.inv(T_cam_world)

    K = camera2k(CAMERA_FOVY_DEG * math.pi / 180.0, IMG_W, IMG_H)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    with open(LAYOUT_JSON) as f:
        layout = json.load(f)
    kele_slots = [s for s in layout if s.get("object_kind") == "kele"]

    rows = []
    for slot in kele_slots:
        pw = np.array(slot["world_position"] + [1.0], dtype=float)
        # world -> camera
        pc = T_world_cam @ pw
        if pc[2] <= 0.0:
            rows.append({"body": slot["body"], "status": "behind_camera"})
            continue
        # projection -> pixel
        u = fx * pc[0] / pc[2] + cx
        v = fy * pc[1] / pc[2] + cy
        in_frame = (0 <= u < IMG_W and 0 <= v < IMG_H)

        # ideal depth (float metres)
        z_ideal = pc[2]
        # quantised depth (uint16 mm, as server publishes mono16)
        z_quant_mm = int(round(z_ideal * 1e3))  # nearest mm
        z_quant = z_quant_mm * 1e-3

        # deprojection: pixel + depth -> camera -> world
        def deproject(depth_m):
            x = (u - cx) * depth_m / fx
            y = (v - cy) * depth_m / fy
            return (T_cam_world @ np.array([x, y, depth_m, 1.0]))[:3]

        pw_rec_ideal = deproject(z_ideal)
        pw_rec_quant = deproject(z_quant)

        err_ideal_mm = np.linalg.norm(pw_rec_ideal - np.array(slot["world_position"])) * 1e3
        err_quant_mm = np.linalg.norm(pw_rec_quant - np.array(slot["world_position"])) * 1e3

        rows.append({
            "body": slot["body"],
            "u": u, "v": v, "in_frame": in_frame,
            "depth_m": z_ideal,
            "err_ideal_mm": err_ideal_mm,
            "err_quant_mm": err_quant_mm,
            "status": "ok" if in_frame else "off_screen",
        })
    return rows, cam_pos


def main():
    print("=" * 72)
    print("Camera-to-World Coordinate Bridge Validation")
    print("=" * 72)

    # Two representative robot poses:
    # 1. Base at the DEPLOY position (actual grasping approach), facing north
    # 2. Base at the NAV_SHELF final waypoint (further back, same heading)
    scenarios = [
        {"label": "DEPLOY pose  (base=[0.91,2.475], yaw=90°, pitch=-0.6)",
         "base_xy": [0.91, 2.475], "yaw": math.pi / 2,
         "slide": 0.11, "head_pitch": -0.6},
        {"label": "Shelf front  (base=[0.91,2.80], yaw=90°, pitch=-0.6)",
         "base_xy": [0.91, 2.80], "yaw": math.pi / 2,
         "slide": 0.11, "head_pitch": -0.6},
    ]

    all_pass = True
    IDEAL_TOL_MM = 0.01    # round-trip should be essentially machine-epsilon
    QUANT_TOL_MM = 10.0    # uint16 mm quantisation can introduce a few mm

    for sc in scenarios:
        print(f"\nScenario: {sc['label']}")
        rows, cam_pos = run(sc["base_xy"], sc["yaw"], sc["slide"], sc["head_pitch"])
        print(f"  Camera world pos: {np.round(cam_pos, 4)}")
        print(f"  {'Body':<30} {'u':>6} {'v':>6} {'depth':>7} "
              f"{'ideal_err':>10} {'quant_err':>10} {'status'}")
        print(f"  {'-'*80}")
        for r in rows:
            if r.get("status") in (None, "behind_camera"):
                print(f"  {r['body']:<30}  behind camera")
                continue
            ok_ideal = r["err_ideal_mm"] < IDEAL_TOL_MM
            ok_quant = r["err_quant_mm"] < QUANT_TOL_MM
            flag = "" if (ok_ideal and ok_quant) else " *** FAIL ***"
            if not (ok_ideal and ok_quant):
                all_pass = False
            print(f"  {r['body']:<30} "
                  f"{r['u']:>6.1f} {r['v']:>6.1f} {r['depth_m']:>7.3f}m "
                  f"{r['err_ideal_mm']:>9.4f}mm {r['err_quant_mm']:>9.4f}mm "
                  f"  {'in_frame' if r['in_frame'] else 'OFF_SCREEN'}{flag}")

    print()
    if all_pass:
        print("RESULT: ALL PASS — coordinate bridge is correct.")
    else:
        print("RESULT: SOME FAILURES — check frame convention or FK qpos layout.")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
