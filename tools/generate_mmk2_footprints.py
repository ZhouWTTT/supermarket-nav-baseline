#!/usr/bin/env python3
"""Generate checked-in 2-D safety envelopes from public MMK2 MJCF geometry."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET


MARGIN_M = 0.04
ALLOWED_BODIES = {"agv_link", "lft_wheel_link", "rgt_wheel_link"}


def vec(text: str | None, length: int, default: float = 0.0) -> tuple[float, ...]:
    values = [] if text is None else [float(value) for value in text.split()]
    values.extend([default] * (length - len(values)))
    return tuple(values[:length])


def matmul(a, b):
    return tuple(
        tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3))
        for i in range(3)
    )


def matvec(a, v):
    return tuple(sum(a[i][k] * v[k] for k in range(3)) for i in range(3))


def euler_matrix(euler):
    rx, ry, rz = euler
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    mx = ((1, 0, 0), (0, cx, -sx), (0, sx, cx))
    my = ((cy, 0, sy), (0, 1, 0), (-sy, 0, cy))
    mz = ((cz, -sz, 0), (sz, cz, 0), (0, 0, 1))
    return matmul(matmul(mz, my), mx)


def quaternion_matrix(quat):
    w, x, y, z = quat
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = (value / norm for value in (w, x, y, z))
    return (
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    )


IDENTITY = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def rotation(element):
    if element.get("quat"):
        return quaternion_matrix(vec(element.get("quat"), 4))
    return euler_matrix(vec(element.get("euler"), 3))


def projected_extent(geom, world_rotation):
    shape = geom.get("type", "sphere" if geom.get("size") else "mesh")
    size = vec(geom.get("size"), 3)
    if shape == "box":
        return tuple(
            sum(abs(world_rotation[axis][column]) * size[column] for column in range(3))
            for axis in (0, 1)
        )
    if shape == "sphere":
        return (size[0], size[0])
    if shape == "cylinder":
        radius, half_length = size[0], size[1]
        axis = (world_rotation[0][2], world_rotation[1][2], world_rotation[2][2])
        extents = []
        for direction in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)):
            dot = sum(a * b for a, b in zip(axis, direction))
            extents.append(
                half_length * abs(dot)
                + radius * math.sqrt(max(0.0, 1.0 - dot * dot))
            )
        return tuple(extents)
    raise ValueError(f"unsupported collision primitive {shape!r}")


def chassis_bounds(mjcf_path: Path) -> tuple[float, float, float, float]:
    root = ET.parse(mjcf_path).getroot()
    points: list[tuple[float, float, float, float]] = []

    def visit(body, parent_position, parent_rotation):
        name = body.get("name", "")
        if name not in ALLOWED_BODIES:
            return
        local_position = vec(body.get("pos"), 3)
        body_position_delta = matvec(parent_rotation, local_position)
        body_position = tuple(
            parent_position[index] + body_position_delta[index] for index in range(3)
        )
        body_rotation = matmul(parent_rotation, rotation(body))
        for geom in body.findall("geom"):
            if geom.get("class") == "visual" or geom.get("mesh"):
                continue
            geom_rotation = matmul(body_rotation, rotation(geom))
            local = vec(geom.get("pos"), 3)
            delta = matvec(body_rotation, local)
            center = tuple(body_position[index] + delta[index] for index in range(3))
            ex, ey = projected_extent(geom, geom_rotation)
            points.append((center[0] - ex, center[0] + ex, center[1] - ey, center[1] + ey))
        for child in body.findall("body"):
            visit(child, body_position, body_rotation)

    agv = root.find("body[@name='agv_link']")
    if agv is None:
        raise ValueError("MMK2 MJCF does not contain agv_link")
    visit(agv, (0.0, 0.0, 0.0), IDENTITY)
    if not points:
        raise ValueError("MMK2 MJCF contained no supported collision primitives")
    return (
        min(item[0] for item in points),
        max(item[1] for item in points),
        min(item[2] for item in points),
        max(item[3] for item in points),
    )


def rectangle(bounds):
    xmin, xmax, ymin, ymax = bounds
    return [[round(xmax, 4), round(ymax, 4)], [round(xmax, 4), round(ymin, 4)],
            [round(xmin, 4), round(ymin, 4)], [round(xmin, 4), round(ymax, 4)]]


def expand(bounds, *, front=0.0, rear=0.0, side=0.0):
    xmin, xmax, ymin, ymax = bounds
    return xmin - rear, xmax + front, ymin - side, ymax + side


def profiles_for(raw_bounds):
    physical = expand(raw_bounds, front=MARGIN_M, rear=MARGIN_M, side=MARGIN_M)
    profiles = {
        "COMPACT_TRANSIT": physical,
        "LOADED_TRANSIT": expand(physical, front=0.03, rear=0.03, side=0.03),
        "SHELF_APPROACH": expand(physical, front=0.05, rear=0.02, side=0.04),
        "DELIVERY_APPROACH": expand(physical, front=0.05, rear=0.04, side=0.05),
        # Standard manipulation posture sweep measured from the checked-in
        # dual-arm kinematic model.  It is unioned with the generated chassis
        # bound so changing physical geometry can never shrink this profile.
        "MANIPULATION_EXTENDED": (
            min(physical[0], -0.50), max(physical[1], 0.50),
            min(physical[2], -0.55), max(physical[3], 0.55),
        ),
    }
    return {name: rectangle(bounds) for name, bounds in profiles.items()}


def write_svg(path: Path, profiles):
    colors = ["#2563eb", "#16a34a", "#f59e0b", "#dc2626", "#7c3aed"]
    scale, cx, cy = 300.0, 260.0, 220.0
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="760" height="460" viewBox="0 0 760 460">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<line x1="20" y1="{cy}" x2="520" y2="{cy}" stroke="#aaa"/>',
        f'<line x1="{cx}" y1="20" x2="{cx}" y2="440" stroke="#aaa"/>',
    ]
    for index, (name, points) in enumerate(profiles.items()):
        color = colors[index]
        svg_points = " ".join(f"{cx + x * scale:.1f},{cy - y * scale:.1f}" for x, y in points)
        lines.append(f'<polygon points="{svg_points}" fill="none" stroke="{color}" stroke-width="2"/>')
        lines.append(f'<text x="540" y="{45 + 30 * index}" fill="{color}" font-family="sans-serif" font-size="14">{name}</text>')
    lines.append('</svg>')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("mjcf", type=Path)
    parser.add_argument("json_output", type=Path)
    parser.add_argument("svg_output", type=Path)
    args = parser.parse_args(argv)
    raw = chassis_bounds(args.mjcf)
    profiles = profiles_for(raw)
    document = {
        "source": str(args.mjcf),
        "margin_m": MARGIN_M,
        "raw_chassis_bounds": [round(value, 6) for value in raw],
        "profiles": profiles,
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    write_svg(args.svg_output, profiles)


if __name__ == "__main__":
    main()
