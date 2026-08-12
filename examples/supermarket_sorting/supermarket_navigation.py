#!/usr/bin/env python3
"""
Autonomous navigation module for the Supermarket Sorting task.

Hybrid costmap + A* + pure-pursuit controller with laser-based obstacle
avoidance for the MMK2 differential-drive mobile base.

Usage (in a ROS2 control loop)::

    nav = SupermarketNavigator()
    nav.set_goal(x, y, yaw)
    while not reached:
        v, w, reached = nav.update(base_x, base_y, base_yaw, laser_msg)
        publish_cmd_vel(v, w)
"""

import math
import heapq
import os
import time
import numpy as np
from scipy.ndimage import maximum_filter

from path_memory import PathMemory

# ============================================================================
# Constants
# ============================================================================
FREE = 0
LETHAL = 100

# Key locations in world frame
START_POSE = (1.92, -3.17, math.pi / 2.0)
# Delivery approach — close enough for the arm to place items on the table.
# The costmap inflates the table north-edge (y=-3.16) by 0.50 m → y≈-2.66.
# Staying just north of that keeps the goal reachable by A* while maximising
# proximity to the table surface.
DELIVERY_APPROACH = (-1.80, -2.60, -math.pi / 2.0)

# Delivery-table geometry copied from ``mjcf/retail_competition.xml``.  The
# second rectangle is deliberately 3 cm larger on every side so navigation
# does not depend on mesh/contact tolerances.  The parked arms have a measured
# planar sweep radius of about 0.455 m; 0.55 m includes link thickness,
# controller tracking error and odometry noise.
DELIVERY_TABLE_XML_BOUNDS = (-2.42, -3.63, -1.46, -3.19)
DELIVERY_TABLE_COSTMAP_BOUNDS = (-2.45, -3.66, -1.43, -3.16)
WHOLE_BODY_KEEP_OUT_RADIUS = 0.55
TABLE_ROTATION_KEEP_OUT_RADIUS = 0.50
ROBOT_HALF_WIDTH_M = 0.20
ROBOT_NAVIGATION_MARGIN_M = 0.04
FRONT_CORRIDOR_HALF_WIDTH_M = (
    ROBOT_HALF_WIDTH_M + ROBOT_NAVIGATION_MARGIN_M)
# Only the physical chassis plus a small measurement allowance should cause
# an unconditional laser stop.  Returns in the remaining navigation-margin
# band are still represented in the dynamic costmap and checked by the motion
# trajectory predictor, so a shelf beside the chassis does not masquerade as
# an obstacle directly in front of it.
HARD_STOP_CORRIDOR_HALF_WIDTH_M = ROBOT_HALF_WIDTH_M + 0.01

# Shelf approach poses.  y=2.40 stays inside the picking zone while leaving a
# clear cross-aisle above the middle wall endpoint at y=1.70.
SHELF_APPROACH = {
    "A": (-1.735, 2.40, math.pi / 2.0),
    "B": (-0.850, 2.40, math.pi / 2.0),
    "C": (0.035, 2.40, math.pi / 2.0),
    "D": (0.920, 2.40, math.pi / 2.0),
    "E": (1.805, 2.40, math.pi / 2.0),
}

# Shelf centers (for obstacle rendering)
SHELF_CENTERS_X = [-1.735, -0.850, 0.035, 0.920, 1.805]
SHELF_Y_CENTER = 3.323
SHELF_HALF_W = 0.45  # shelf half-width + post
SHELF_HALF_D = 0.15  # shelf half-depth


def point_to_rect_clearance(x, y, bounds):
    """Euclidean distance from a point to an axis-aligned rectangle.

    ``bounds`` is ``(xmin, ymin, xmax, ymax)``.  A point inside the
    rectangle has zero clearance.
    """
    xmin, ymin, xmax, ymax = bounds
    dx = max(xmin - x, 0.0, x - xmax)
    dy = max(ymin - y, 0.0, y - ymax)
    return math.hypot(dx, dy)


def wrap_to_pi(a):
    """Wrap angle to [-pi, pi]."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def angdist(a, b):
    """Shortest signed angular distance from a to b."""
    return wrap_to_pi(b - a)


def depth_image_clearance(image_msg, x_fraction=(0.35, 0.65),
                          y_fraction=(0.38, 0.62), percentile=10.0):
    """Extract a robust forward clearance from a ROS depth ``Image``.

    Supports the documented millimetre ``16UC1`` stream and metre ``32FC1``
    streams without requiring cv_bridge.  Only the central image region
    is used so side shelves do not stop forward motion.
    """
    height = int(image_msg.height)
    width = int(image_msg.width)
    if height <= 0 or width <= 0:
        return None

    encoding = str(image_msg.encoding).lower()
    big_endian = bool(getattr(image_msg, 'is_bigendian', False))
    if encoding in ('16uc1', 'mono16'):
        dtype = np.dtype('>u2' if big_endian else '<u2')
        scale = 0.001
    elif encoding == '32fc1':
        dtype = np.dtype('>f4' if big_endian else '<f4')
        scale = 1.0
    else:
        return None

    row_bytes = int(image_msg.step) if int(image_msg.step) > 0 else width * dtype.itemsize
    row_items = row_bytes // dtype.itemsize
    if row_items < width:
        return None
    count = height * row_items
    try:
        raw = np.frombuffer(image_msg.data, dtype=dtype, count=count)
    except (TypeError, ValueError):
        return None
    if raw.size != count:
        return None
    depth = raw.reshape(height, row_items)[:, :width].astype(np.float32)
    depth *= scale

    x0, x1 = (int(width * x_fraction[0]), int(width * x_fraction[1]))
    y0, y1 = (int(height * y_fraction[0]), int(height * y_fraction[1]))
    roi = depth[max(0, y0):min(height, y1),
                max(0, x0):min(width, x1)]
    valid = roi[np.isfinite(roi) & (roi > 0.15) & (roi < 6.0)]
    if valid.size < 20:
        return None
    return float(np.percentile(valid, percentile))


# ============================================================================
# Costmap2D
# ============================================================================
class Costmap2D:
    """2-D occupancy-grid costmap with static and dynamic (laser) layers.

    World frame origin is at world (-2.5, -3.75); the grid covers 5.0 m × 7.5 m
    at 0.05 m/cell → 100 columns × 150 rows.
    """

    def __init__(self, origin_x=-2.5, origin_y=-3.75,
                 width_m=5.0, height_m=7.5, resolution=0.05,
                 inflation_radius_m=0.18, static_inflation_radius_m=0.50,
                 dynamic_ttl_scans=36, vision_ttl_scans=12,
                 laser_offset_x=0.09):
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.resolution = resolution
        self.width = int(width_m / resolution)   # columns
        self.height = int(height_m / resolution)  # rows
        self.inflation_radius_m = inflation_radius_m
        self.static_inflation_radius_m = static_inflation_radius_m
        self.dynamic_ttl_scans = int(dynamic_ttl_scans)
        self.vision_ttl_scans = int(vision_ttl_scans)
        self.laser_offset_x = float(laser_offset_x)
        self._pause_depth = False  # set True during pure rotation

        # Layers
        self.static = np.zeros((self.height, self.width), dtype=np.int8)
        # Keep raw laser hits separate from their inflated representation.
        # Inflating ``dynamic`` in-place on every scan makes obstacles grow by
        # one robot radius per update until the whole corridor is blocked.
        self.dynamic_raw = np.zeros((self.height, self.width), dtype=np.int8)
        self.dynamic_age = np.zeros((self.height, self.width), dtype=np.uint16)
        self.dynamic_misses = np.zeros((self.height, self.width), dtype=np.uint8)
        # RGB-D observations are kept separate because a low-mounted lidar ray
        # may pass under a tabletop and must not clear a camera-detected object.
        self.vision_raw = np.zeros((self.height, self.width), dtype=np.int8)
        self.vision_age = np.zeros((self.height, self.width), dtype=np.uint16)
        self.vision_hits = np.zeros((self.height, self.width), dtype=np.uint8)
        self.dynamic = np.zeros((self.height, self.width), dtype=np.int8)
        self.master = np.zeros((self.height, self.width), dtype=np.int8)
        self._raw_dynamic_points_world = np.empty((0, 2), dtype=float)

        # Pre-compute disk kernel for inflation
        r = int(math.ceil(inflation_radius_m / resolution))
        y, x = np.ogrid[-r:r+1, -r:r+1]
        self._disk = ((x*x + y*y) <= r*r).astype(np.int8)
        static_r = int(math.ceil(static_inflation_radius_m / resolution))
        y, x = np.ogrid[-static_r:static_r+1, -static_r:static_r+1]
        self._static_disk = ((x*x + y*y) <= static_r*static_r).astype(np.int8)

        self._build_static()
        self._inflate_and_build()

    # ---- coordinate transforms ----
    def world_to_grid(self, wx, wy):
        """Convert world (x,y) -> grid (col, row)."""
        # ``int`` truncates toward zero and incorrectly maps points just below
        # the map origin into cell zero.  floor preserves out-of-bounds status.
        gx = math.floor((wx - self.origin_x) / self.resolution)
        gy = math.floor((wy - self.origin_y) / self.resolution)
        return gx, gy

    def grid_to_world(self, gx, gy):
        """Convert grid (col, row) -> world (x,y) at cell centre."""
        wx = (gx + 0.5) * self.resolution + self.origin_x
        wy = (gy + 0.5) * self.resolution + self.origin_y
        return wx, wy

    def in_bounds(self, gx, gy):
        return 0 <= gx < self.width and 0 <= gy < self.height

    # ---- building ----
    def _fill_rect(self, grid, xmin, ymin, xmax, ymax, val):
        """Set all cells overlapping the world-aligned rectangle to *val*."""
        gx0, gy0 = self.world_to_grid(xmin, ymin)
        gx1, gy1 = self.world_to_grid(xmax, ymax)
        gx0, gx1 = max(0, gx0), min(self.width - 1, gx1)
        gy0, gy1 = max(0, gy0), min(self.height - 1, gy1)
        if gx0 <= gx1 and gy0 <= gy1:
            grid[gy0:gy1+1, gx0:gx1+1] = val

    def _build_static(self):
        """Mark immutable scene geometry in the static layer."""
        s = self.static

        # --- perimeter walls (inner edges at ±2.47 and ±3.72) ---
        self._fill_rect(s, -2.50, -3.78,  2.50, -3.72, LETHAL)  # south
        self._fill_rect(s, -2.50,  3.72,  2.50,  3.78, LETHAL)  # north
        self._fill_rect(s, -2.53, -3.75, -2.47,  3.75, LETHAL)  # west
        self._fill_rect(s,  2.47, -3.75,  2.53,  3.75, LETHAL)  # east

        # --- corridor right board (x≈0.53, y∈[-3.72, 1.70], half-w 0.03) ---
        self._fill_rect(s, 0.50, -3.72, 0.56, 1.70, LETHAL)

        # --- five shelves (back edge at y=3.473, front at y=3.173) ---
        for cx in SHELF_CENTERS_X:
            self._fill_rect(s,
                            cx - SHELF_HALF_W, SHELF_Y_CENTER - SHELF_HALF_D,
                            cx + SHELF_HALF_W, SHELF_Y_CENTER + SHELF_HALF_D + 0.02,
                            LETHAL)

        # --- delivery table (0.96×0.44 m at -1.94, -3.41) ---
        # table top + legs → block a generous bounding box
        self._fill_rect(s, *DELIVERY_TABLE_COSTMAP_BOUNDS, LETHAL)

    def _inflate_and_build(self):
        """Inflate static obstacles, build cost gradient, combine into master."""
        from scipy.ndimage import distance_transform_edt

        occupied = (self.static == LETHAL).astype(np.int8)
        inflated = maximum_filter(occupied, footprint=self._static_disk)
        self.static[inflated > 0] = LETHAL

        # Build cost gradient: higher cost near obstacles
        # Distance transform gives distance to nearest obstacle
        dist = distance_transform_edt(1 - occupied)  # distance to nearest obstacle
        max_penalty_dist = self.static_inflation_radius_m * 2.0
        max_penalty_cells = int(max_penalty_dist / self.resolution)
        # Gradient: penalty decreases linearly from LETHAL/2 at obstacle to 0 at max_penalty_dist
        self._cost_gradient = np.maximum(0,
            (LETHAL // 3) * (1.0 - dist / max_penalty_cells)).astype(np.int8)
        # Don't add gradient where obstacle already exists
        self._cost_gradient[occupied > 0] = 0

        self.rebuild_master()

    def gradient_cost(self, gx, gy):
        """Return additional cost for being near an obstacle (0 = far, >0 = near)."""
        if not self.in_bounds(gx, gy):
            return LETHAL
        return int(self._cost_gradient[gy, gx])

    def rebuild_master(self):
        """Combine static + dynamic → master (element-wise max)."""
        self.master = np.maximum(self.static, self.dynamic)

    def obstacle_counts(self):
        """Return (lidar_cells, vision_cells) — raw lethal cells per layer."""
        lidar = int(np.count_nonzero(self.dynamic_raw == LETHAL))
        vision = int(np.count_nonzero(self.vision_raw == LETHAL))
        return lidar, vision

    # ---- laser update ----
    def update_from_scan(self, scan_msg, robot_x, robot_y, robot_yaw):
        """Update dynamic layer from a fresh LaserScan message.

        Sub-sampled Bresenham ray-cast: marks endpoints LETHAL and clears the
        space along each ray as FREE.
        """
        ranges = scan_msg.ranges
        if not ranges:
            return
        angle_min = scan_msg.angle_min
        angle_inc = scan_msg.angle_increment
        rng_max = float(scan_msg.range_max)
        rng_min = scan_msg.range_min if hasattr(scan_msg, 'range_min') else 0.02

        n = len(ranges)
        step = 2                     # subsample every 2nd ray
        r_ignore = max(0.08, float(rng_min))

        # Age raw observations once per *new* scan.  Occluded stale hits are
        # forgotten after roughly three seconds at the documented 12 Hz rate.
        occupied = self.dynamic_raw == LETHAL
        self.dynamic_age[occupied] = np.minimum(
            self.dynamic_age[occupied] + 1,
            np.iinfo(self.dynamic_age.dtype).max)
        expired = occupied & (self.dynamic_age > self.dynamic_ttl_scans)
        self.dynamic_raw[expired] = FREE
        self.dynamic_age[expired] = 0
        self.dynamic_misses[expired] = 0

        vision_occupied = self.vision_raw == LETHAL
        self.vision_age[vision_occupied] = np.minimum(
            self.vision_age[vision_occupied] + 1,
            np.iinfo(self.vision_age.dtype).max)
        vision_expired = (
            vision_occupied & (self.vision_age > self.vision_ttl_scans))
        self.vision_raw[vision_expired] = FREE
        self.vision_age[vision_expired] = 0

        # The lidar site is 9 cm in front of base_link in the supplied MMK2.
        laser_x = robot_x + self.laser_offset_x * math.cos(robot_yaw)
        laser_y = robot_y + self.laser_offset_x * math.sin(robot_yaw)
        rgx, rgy = self.world_to_grid(laser_x, laser_y)
        hit_cells = []
        clear_cells = set()

        for i in range(0, n, step):
            r = float(ranges[i])
            if math.isnan(r) or r <= r_ignore:
                continue

            angle = angle_min + i * angle_inc
            beam_yaw = robot_yaw + angle
            has_hit = math.isfinite(r) and r < rng_max
            ray_range = r if has_hit else rng_max
            if not math.isfinite(ray_range) or ray_range <= 0.0:
                continue

            # Clear almost to the return.  The old 85% rule left a large stale
            # tail on every ray (e.g. 60 cm at a 4 m return).
            clear_r = max(0.0, ray_range - (2.0 * self.resolution if has_hit else 0.0))
            mx = laser_x + clear_r * math.cos(beam_yaw)
            my = laser_y + clear_r * math.sin(beam_yaw)
            mgx, mgy = self.world_to_grid(mx, my)
            self._bresenham_clear(rgx, rgy, mgx, mgy, clear_cells)

            if not has_hit:
                continue

            wx = laser_x + r * math.cos(beam_yaw)
            wy = laser_y + r * math.sin(beam_yaw)
            egx, egy = self.world_to_grid(wx, wy)

            # Static walls/shelves are already represented and must not become
            # long-lived dynamic observations.
            if self.in_bounds(egx, egy) and self.static[egy, egx] < LETHAL:
                hit_cells.append((egx, egy))

        # Require several consecutive scan-level misses before deleting a hit.
        # This prevents one noisy beam at a box edge from making the global
        # route alternate left/right on successive replans.
        for gx, gy in clear_cells:
            if self.dynamic_raw[gy, gx] == LETHAL:
                misses = min(255, int(self.dynamic_misses[gy, gx]) + 1)
                self.dynamic_misses[gy, gx] = misses
                if misses >= 3:
                    self.dynamic_raw[gy, gx] = FREE
                    self.dynamic_age[gy, gx] = 0
                    self.dynamic_misses[gy, gx] = 0

        # Mark after clearing every ray.  Otherwise a neighbouring ray that
        # traverses the same grid cell can erase an endpoint recorded earlier
        # in this scan merely because of beam iteration order.
        for egx, egy in hit_cells:
            self.dynamic_raw[egy, egx] = LETHAL
            self.dynamic_age[egy, egx] = 0
            self.dynamic_misses[egy, egx] = 0

        self._rebuild_dynamic()

    def update_from_depth_obstacle(self, distance, robot_x, robot_y, robot_yaw,
                                   camera_offset_x=0.10,
                                   min_distance=0.18,
                                   max_distance=1.5):
        """Project a central RGB-D obstacle measurement onto the 2-D costmap.

        Requires *two consecutive frames* hitting the same cell before marking
        it LETHAL, so transient noise (e.g. from arm motion during rotation)
        does not create false obstacles.  The scalar ``depth_clearance`` value
        is still used by the controller for local braking regardless.
        """
        if distance is None or self._pause_depth:
            return
        distance = float(distance)
        if (not math.isfinite(distance) or distance < min_distance or
                distance > max_distance):
            return

        camera_x = robot_x + camera_offset_x * math.cos(robot_yaw)
        camera_y = robot_y + camera_offset_x * math.sin(robot_yaw)
        wx = camera_x + distance * math.cos(robot_yaw)
        wy = camera_y + distance * math.sin(robot_yaw)
        gx, gy = self.world_to_grid(wx, wy)
        if self.in_bounds(gx, gy):
            hits = min(255, int(self.vision_hits[gy, gx]) + 1)
            self.vision_hits[gy, gx] = hits
            if hits >= 2:   # require two consecutive frames
                self.vision_raw[gy, gx] = LETHAL
                self.vision_age[gy, gx] = 0
            self._rebuild_dynamic()

    def _rebuild_dynamic(self):
        """Inflate the union of lidar and RGB-D raw obstacle observations."""
        raw_occupied = ((self.dynamic_raw == LETHAL) |
                        (self.vision_raw == LETHAL))
        rows, cols = np.nonzero(raw_occupied)
        if rows.size:
            self._raw_dynamic_points_world = np.column_stack((
                (cols.astype(float) + 0.5) * self.resolution + self.origin_x,
                (rows.astype(float) + 0.5) * self.resolution + self.origin_y,
            ))
        else:
            self._raw_dynamic_points_world = np.empty((0, 2), dtype=float)
        # Always inflate from raw hits, never from the previous inflated layer.
        occ = raw_occupied.astype(np.int8)
        inflated = maximum_filter(occ, footprint=self._disk)
        self.dynamic.fill(FREE)
        self.dynamic[inflated > 0] = LETHAL
        self.rebuild_master()

    def _bresenham_clear(self, x0, y0, x1, y1, clear_cells=None):
        """Collect or clear cells along the line (x0,y0)→(x1,y1)."""
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        cx, cy = x0, y0
        while True:
            if self.in_bounds(cx, cy):
                if clear_cells is None:
                    self.dynamic_raw[cy, cx] = FREE
                    self.dynamic_age[cy, cx] = 0
                    self.dynamic_misses[cy, cx] = 0
                else:
                    clear_cells.add((cx, cy))
            if cx == x1 and cy == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                if cx == x1:
                    break
                err += dy
                cx += sx
            if e2 <= dx:
                if cy == y1:
                    break
                err += dx
                cy += sy

    # ---- collision queries ----
    def is_free_grid(self, gx, gy):
        if not self.in_bounds(gx, gy):
            return False
        return self.master[gy, gx] < LETHAL

    def is_free_world(self, wx, wy):
        gx, gy = self.world_to_grid(wx, wy)
        return self.is_free_grid(gx, gy)

    def is_static_free_world(self, wx, wy):
        """Whether a world point is outside the inflated static layer."""
        gx, gy = self.world_to_grid(wx, wy)
        return (
            self.in_bounds(gx, gy)
            and self.static[gy, gx] < LETHAL
        )

    def raw_dynamic_clearance_world(self, wx, wy):
        """Distance to the nearest uninflated lidar/RGB-D obstacle hit."""
        if self._raw_dynamic_points_world.size == 0:
            return float('inf')
        delta = self._raw_dynamic_points_world - np.array(
            [float(wx), float(wy)], dtype=float)
        return float(np.sqrt(np.min(np.sum(delta * delta, axis=1))))

    def line_is_free(self, wx0, wy0, wx1, wy1):
        """Return whether the complete world-space segment is collision-free."""
        dx = wx1 - wx0
        dy = wy1 - wy0
        distance = math.hypot(dx, dy)
        # Sampling at half-cell spacing is a conservative "supercover" check.
        # Classic Bresenham can miss a blocked cell touched near a grid corner,
        # which previously let smoothed paths graze the middle wall and boxes.
        steps = max(1, int(math.ceil(distance / (0.5 * self.resolution))))
        for i in range(steps + 1):
            ratio = i / steps
            wx = wx0 + ratio * dx
            wy = wy0 + ratio * dy
            if not self.is_free_world(wx, wy):
                return False
        return True

    # Kept for callers of the original module.  Despite its historical name,
    # this method has always returned True for a free segment.
    def line_collision(self, wx0, wy0, wx1, wy1):
        return self.line_is_free(wx0, wy0, wx1, wy1)


# ============================================================================
# A* Planner
# ============================================================================
_SQRT2 = math.sqrt(2.0)
_NEIGHBORS_8 = [(-1, -1, _SQRT2), (0, -1, 1.0), (1, -1, _SQRT2),
                (-1,  0, 1.0),                  (1,  0, 1.0),
                (-1,  1, _SQRT2), (0,  1, 1.0), (1,  1, _SQRT2)]


class AStarPlanner:
    """A* global planner on a Costmap2D.

    Returns a path as a list of world (x,y) tuples or None if no path is found.
    """

    def __init__(self, costmap):
        self.cm = costmap
        self.failure_reason = None  # set on each plan() call

    def plan(self, start_wx, start_wy, goal_wx, goal_wy):
        """Run A* from world start to world goal.

        Returns list[(wx,wy)] or None.  On failure, ``self.failure_reason``
        is set to one of ``"start_blocked"``, ``"goal_blocked"``, or
        ``"disconnected"``.
        """
        self.failure_reason = None
        sgx, sgy = self.cm.world_to_grid(start_wx, start_wy)
        ggx, ggy = self.cm.world_to_grid(goal_wx, goal_wy)
        exact_start_free = self.cm.is_free_grid(sgx, sgy)
        exact_goal_free = self.cm.is_free_grid(ggx, ggy)

        if not exact_start_free:
            sgx, sgy = self._nearest_free(sgx, sgy)
            if sgx is None:
                self.failure_reason = "start_blocked"
                return None
        if not exact_goal_free:
            ggx, ggy = self._nearest_free(ggx, ggy)
            if ggx is None:
                self.failure_reason = "goal_blocked"
                return None

        open_set = []
        came_from = {}
        g_score = {(sgx, sgy): 0.0}
        f_score = {(sgx, sgy): self._heuristic(sgx, sgy, ggx, ggy)}

        # Tie-breaker: monotonic counter
        counter = 0
        heapq.heappush(open_set, (f_score[(sgx, sgy)], counter, sgx, sgy))
        closed = set()

        while open_set:
            _, _, cx, cy = heapq.heappop(open_set)
            if (cx, cy) in closed:
                continue
            if cx == ggx and cy == ggy:
                path = self._reconstruct(came_from, sgx, sgy, ggx, ggy)
                if exact_start_free:
                    exact_start = (float(start_wx), float(start_wy))
                    if (len(path) == 1 or self.cm.line_is_free(
                            exact_start[0], exact_start[1],
                            path[1][0], path[1][1])):
                        path[0] = exact_start
                    else:
                        # Preserve the safe start-cell centre when replacing it
                        # would make the first shortcut graze a blocked cell.
                        path.insert(0, exact_start)
                if exact_goal_free:
                    exact_goal = (float(goal_wx), float(goal_wy))
                    if (len(path) == 1 or self.cm.line_is_free(
                            path[-2][0], path[-2][1],
                            exact_goal[0], exact_goal[1])):
                        path[-1] = exact_goal
                    else:
                        path.append(exact_goal)
                return path

            closed.add((cx, cy))

            for dx, dy, base_cost in _NEIGHBORS_8:
                nx, ny = cx + dx, cy + dy
                if (nx, ny) in closed:
                    continue
                if not self.cm.is_free_grid(nx, ny):
                    continue

                # Penalise diagonal moves that cut corners through obstacles
                if dx != 0 and dy != 0:
                    if not self.cm.is_free_grid(cx + dx, cy) or \
                       not self.cm.is_free_grid(cx, cy + dy):
                        continue

                # Gradient cost: penalise being close to walls
                edge_cost = base_cost + self.cm.gradient_cost(nx, ny) * 0.015

                tent_g = g_score[(cx, cy)] + edge_cost
                if tent_g < g_score.get((nx, ny), float('inf')):
                    came_from[(nx, ny)] = (cx, cy)
                    g_score[(nx, ny)] = tent_g
                    counter += 1
                    f = tent_g + self._heuristic(nx, ny, ggx, ggy)
                    heapq.heappush(open_set, (f, counter, nx, ny))

        self.failure_reason = "disconnected"
        return None  # no path

    def _heuristic(self, gx, gy, ggx, ggy):
        # Octile distance exactly matches the lower-bound movement cost of an
        # 8-connected grid and expands fewer irrelevant cells than Euclidean.
        dx = abs(gx - ggx)
        dy = abs(gy - ggy)
        return dx + dy + (_SQRT2 - 2.0) * min(dx, dy)

    def _reconstruct(self, came_from, sgx, sgy, ggx, ggy):
        path = []
        cx, cy = ggx, ggy
        while (cx, cy) != (sgx, sgy):
            path.append(self.cm.grid_to_world(cx, cy))
            cx, cy = came_from[(cx, cy)]
        path.append(self.cm.grid_to_world(sgx, sgy))
        path.reverse()
        return self._smooth(path)

    def _smooth(self, path):
        """Remove only collinear grid points, preserving A* obstacle clearance.

        Aggressive line-of-sight shortcuts tend to run exactly along inflated
        obstacle boundaries.  A real pure-pursuit arc then cuts inside that
        shortcut and can wedge the chassis at a wall corner.  Direction-run
        compression retains the clearance chosen by A* while still reducing
        redundant points on long straight sections.
        """
        if len(path) <= 2:
            return path
        smoothed = [path[0]]
        for i in range(1, len(path) - 1):
            prev_dx = round((path[i][0] - path[i - 1][0]) /
                            self.cm.resolution)
            prev_dy = round((path[i][1] - path[i - 1][1]) /
                            self.cm.resolution)
            next_dx = round((path[i + 1][0] - path[i][0]) /
                            self.cm.resolution)
            next_dy = round((path[i + 1][1] - path[i][1]) /
                            self.cm.resolution)
            if (prev_dx, prev_dy) != (next_dx, next_dy):
                smoothed.append(path[i])
        smoothed.append(path[-1])
        return smoothed

    def _nearest_free(self, gx, gy, radius=12):
        """Search outward from (gx,gy) for the nearest free cell."""
        for r in range(1, radius + 1):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if abs(dx) != r and abs(dy) != r:
                        continue
                    nx, ny = gx + dx, gy + dy
                    if self.cm.is_free_grid(nx, ny):
                        return nx, ny
        return None, None


# ============================================================================
# Path-following Controller
# ============================================================================
class NavigationController:
    """Pure-pursuit path follower with laser emergency stop and replanning.

    Call ``compute_velocity`` at the control-loop rate.
    """

    def __init__(self, costmap, planner):
        self.cm = costmap
        self.planner = planner
        self.path = []
        self.goal_x = self.goal_y = None
        self.goal_yaw = None
        self.nav_goal_x = self.nav_goal_y = None

        # Velocity limits
        self.max_lin = 0.70
        self.max_ang = 2.5
        self.max_lin_acc = 1.2
        self.max_ang_acc = 5.0
        self.dt = 0.02
        self._last_update_time = None

        # Current smoothed velocities
        self.cur_lin = 0.0
        self.cur_ang = 0.0

        # Lookahead
        self.lookahead_base = 0.55
        self.lookahead_min = 0.22

        # Gains
        self.k_ang = 2.5
        self.k_lin = 1.0

        # Tolerances
        self.pos_tol = 0.10
        self.yaw_tol = 0.15

        # Obstacle safety.  The laser is 9 cm ahead of base_link and the base
        # front is about 21 cm ahead, so 0.32 m still leaves braking margin.
        self._blocked_timer = 0.0
        self._blocked_thresh = 0.35
        self._stop_dist = 0.32
        self._slow_dist = 0.55
        self._stop_arc = math.radians(38)
        self.front_corridor_half_width = FRONT_CORRIDOR_HALF_WIDTH_M
        self.hard_stop_corridor_half_width = (
            HARD_STOP_CORRIDOR_HALF_WIDTH_M)
        self._arc_blocked_timer = 0.0
        self._arc_replan_threshold = 0.35

        # Safe straight-reverse recovery.  Replanning from an unchanged pose
        # cannot resolve a path whose first segment needs more turning room,
        # so a persistent local block may move the base a short measured
        # distance backwards before forcing a new plan.
        self._reverse_recovery_phase = None
        self._reverse_recovery_blocked_time = 0.0
        self._reverse_recovery_block_anchor_x = None
        self._reverse_recovery_block_anchor_y = None
        self._reverse_recovery_start_x = None
        self._reverse_recovery_start_y = None
        self._reverse_recovery_start_yaw = None
        self._reverse_recovery_started_at = 0.0
        self._reverse_recovery_trigger_s = 1.0
        self._reverse_recovery_no_progress_m = 0.03
        self._reverse_recovery_distance_m = 0.12
        self._reverse_recovery_speed = 0.10
        self._reverse_recovery_timeout_s = 10.0
        # The lidar is about 0.09 m ahead of base_link while the chassis rear
        # is roughly 0.22 m behind it.  A 0.45 m laser clearance therefore
        # preserves about 0.14 m behind the physical rear face.
        self._reverse_recovery_rear_stop_m = 0.45
        self._reverse_recovery_yaw_gain = 2.0
        self._reverse_recovery_max_ang = 0.30
        self._reverse_recovery_attempts = 0
        self._reverse_recovery_max_attempts = 2
        self._reverse_recovery_cooldown_until = float('-inf')

        # Replanning/progress state
        self._last_replan_time = float('-inf')
        self._replan_interval = 0.40
        self._replan_hold_until = float('-inf')
        self._path_improvement_ratio = 0.02
        self._best_goal_dist = float('inf')
        self._last_progress_time = None
        self._stuck_timeout = 2.0

        # Rotation-loop watchdog.  A legitimate in-place alignment is at most
        # pi radians; substantially more without translation means successive
        # replans are changing the desired side of an obstacle.
        self._rotation_accum = 0.0
        self._rotation_anchor_x = None
        self._rotation_anchor_y = None
        self._last_base_yaw = None
        # A normal route may legitimately require a full 180-degree turn
        # after leaving the delivery table.  Only substantially more rotation
        # without translation is considered a loop.
        self._rotation_loop_limit = 1.25 * math.pi
        self._rotation_recoveries = 0

        # Per-sensor clearance and stop-reason diagnostics
        self.lidar_clearance = float('inf')
        self.rear_clearance = float('inf')
        self.depth_clearance_val = float('inf')
        self.stop_reason = None          # current stop reason (str or None)
        self._last_logged_reason = None  # avoid log spam
        # Snapshot of the most recent planning transaction.  Do not infer
        # fallback diagnostics from AStarPlanner.failure_reason: a second plan
        # call resets that shared field before the controller can log it.
        self._last_plan_mode = None
        self._last_plan_full_failure = None
        self._last_plan_fallback_failure = None
        self._last_plan_lidar_count = 0
        self._last_plan_vision_count = 0

    def set_goal(self, x, y, yaw=None):
        """Set a new navigation goal.  Clears the current path."""
        self.goal_x = float(x)
        self.goal_y = float(y)
        self.goal_yaw = yaw
        self.nav_goal_x = self.goal_x
        self.nav_goal_y = self.goal_y
        self.path = []
        self._blocked_timer = 0.0
        self._arc_blocked_timer = 0.0
        self._reverse_recovery_phase = None
        self._reverse_recovery_blocked_time = 0.0
        self._reverse_recovery_block_anchor_x = None
        self._reverse_recovery_block_anchor_y = None
        self._reverse_recovery_start_x = None
        self._reverse_recovery_start_y = None
        self._reverse_recovery_start_yaw = None
        self._reverse_recovery_started_at = 0.0
        self._reverse_recovery_attempts = 0
        self._reverse_recovery_cooldown_until = float('-inf')
        self._last_replan_time = float('-inf')
        self._replan_hold_until = float('-inf')
        self._best_goal_dist = float('inf')
        self._last_progress_time = None
        self._rotation_accum = 0.0
        self._rotation_anchor_x = None
        self._rotation_anchor_y = None
        self._last_base_yaw = None
        self._rotation_recoveries = 0
        self.stop_reason = None
        self._last_logged_reason = None
        self.lidar_clearance = float('inf')
        self.rear_clearance = float('inf')
        self.depth_clearance_val = float('inf')
        self._last_plan_mode = None
        self._last_plan_full_failure = None
        self._last_plan_fallback_failure = None
        self._last_plan_lidar_count = 0
        self._last_plan_vision_count = 0
        self.cur_lin = 0.0
        self.cur_ang = 0.0

    def compute_velocity(self, base_x, base_y, base_yaw,
                         laser_msg=None, depth_clearance=None, time_now=None):
        """Return (v_lin, v_ang, reached_goal)."""

        now = time.monotonic() if time_now is None else float(time_now)
        self.last_safety_stop = None
        self.stop_reason = None

        # Capture current sensor values before any planner branch can return.
        # Previously no_path/stuck_no_path logs displayed values left over from
        # the last successful control cycle.
        self.lidar_clearance = self._front_clearance(
            laser_msg, self.hard_stop_corridor_half_width)
        self.rear_clearance = self._rear_clearance(laser_msg)
        if (depth_clearance is not None and
                math.isfinite(float(depth_clearance))):
            self.depth_clearance_val = float(depth_clearance)
        else:
            self.depth_clearance_val = float('inf')

        if self._last_update_time is not None:
            self.dt = max(0.005, min(0.10, now - self._last_update_time))
        self._last_update_time = now

        if self.goal_x is None:
            return 0.0, 0.0, True

        if self._reverse_recovery_phase == "backup":
            return self._reverse_recovery_tick(
                base_x, base_y, base_yaw, laser_msg, now)

        rotation_loop = self._update_rotation_watchdog(
            base_x, base_y, base_yaw)
        if rotation_loop:
            new_path = self._try_plan_with_fallback(
                base_x, base_y, self.goal_x, self.goal_y)
            if new_path is not None:
                self._install_path(new_path)
            else:
                self.path = []
            self._rotation_recoveries += 1
            self._replan_hold_until = now + min(
                6.0, 2.5 + self._rotation_recoveries)
            self._last_replan_time = now
            self._rotation_accum = 0.0
            self._rotation_anchor_x = base_x
            self._rotation_anchor_y = base_y
            self.cur_lin = self.cur_ang = 0.0
            self.stop_reason = "rotation_loop"
            if self._maybe_start_reverse_recovery(
                    "rotation_loop", base_x, base_y, base_yaw,
                    laser_msg, now):
                self.stop_reason = "reverse_recovery_start"
            return 0.0, 0.0, False

        # Plan periodically so newly observed boxes trigger a prompt detour.
        need_replan = (not self.path or
                       (now >= self._replan_hold_until and
                        now - self._last_replan_time >= self._replan_interval))
        if need_replan:
            self._last_replan_time = now
            new_path = self._try_plan_with_fallback(
                base_x, base_y, self.goal_x, self.goal_y)
            if new_path is not None:
                self._consider_new_path(new_path, base_x, base_y)
            elif not self.path or not self._path_valid(base_x, base_y):
                self.path = []
                self.cur_lin = self.cur_ang = 0.0
                self.stop_reason = self._format_no_path_reason("no_path")
                return 0.0, 0.0, False

        if not self.path:
            self.stop_reason = self._format_no_path_reason("no_path")
            return 0.0, 0.0, False

        dx = self.nav_goal_x - base_x
        dy = self.nav_goal_y - base_y
        dist_to_goal = math.hypot(dx, dy)

        # The planner may move a temporarily occupied goal to the nearest free
        # cell.  Use that effective endpoint until a later replan can restore
        # the exact requested goal.
        if dist_to_goal < self.pos_tol:
            self.cur_lin = 0.0
            if self.goal_yaw is not None:
                yaw_error = angdist(base_yaw, self.goal_yaw)
                if abs(yaw_error) > self.yaw_tol:
                    # A final in-place turn sweeps the arms around the base.
                    # Keep it outside a conservative table exclusion radius;
                    # this check is independent of A* and remains effective if
                    # odometry/controller error puts the base off the path.
                    if not self._table_rotation_is_free(base_x, base_y):
                        self.cur_ang = 0.0
                        self.last_safety_stop = "delivery_table_keepout"
                        self.stop_reason = "table_keepout"
                        return 0.0, 0.0, False
                    w = self._ramp_ang(self.k_ang * yaw_error)
                    return 0.0, w, False
            self.cur_ang = 0.0
            return 0.0, 0.0, True

        # Detect lack of odometric progress and force a fresh plan.  This is a
        # replan-only recovery: no unobserved reverse motion.
        if dist_to_goal < self._best_goal_dist - 0.06:
            self._best_goal_dist = dist_to_goal
            self._last_progress_time = now
        elif self._last_progress_time is None:
            self._last_progress_time = now
        elif now - self._last_progress_time > self._stuck_timeout:
            new_path = self._try_plan_with_fallback(
                base_x, base_y, self.goal_x, self.goal_y)
            self._last_replan_time = now
            self._last_progress_time = now
            self._best_goal_dist = dist_to_goal
            if new_path is None:
                self.path = []
                self.cur_lin = self.cur_ang = 0.0
                self.stop_reason = self._format_no_path_reason(
                    "stuck_no_path")
                return 0.0, 0.0, False
            self._consider_new_path(new_path, base_x, base_y, force=True)

        lookahead = self._lookahead_dist(base_x, base_y, dist_to_goal)
        la_x, la_y = self._lookahead_point(base_x, base_y, lookahead)
        while (lookahead > self.cm.resolution and
               not self.cm.line_is_free(base_x, base_y, la_x, la_y)):
            # A lookahead that spans multiple sharp segments creates a chord
            # through the inside of the corner.  Shorten it until the target
            # is directly visible in the inflated costmap.
            lookahead *= 0.5
            la_x, la_y = self._lookahead_point(
                base_x, base_y, lookahead)

        ang_to_la = math.atan2(la_y - base_y, la_x - base_x)
        heading_err = angdist(base_yaw, ang_to_la)

        # Speed proportional to distance, with natural slowdown into turns
        # and near the goal.  In open space (clearance ≥ slow_dist) large
        # heading errors pause translation so the robot can align.  Near
        # obstacles we allow combined forward + rotation ("creep") so the
        # robot does not sweep its arms while rotating in place.
        v_des = min(self.max_lin, self.k_lin * dist_to_goal)
        if dist_to_goal < 0.80:
            v_des = min(v_des, 0.18)
        v_des *= max(0.0, math.cos(heading_err))
        corner_scale = max(0.20, 1.0 - abs(heading_err) / 0.65)
        v_des *= corner_scale
        w_des = max(-self.max_ang,
                    min(self.max_ang, self.k_ang * heading_err))

        # Composite clearance for scaling (the tighter sensor governs braking)
        composite = min(self.lidar_clearance, self.depth_clearance_val)

        # ── decide stop reason ──
        new_reason = None

        if abs(heading_err) > 0.65:
            if composite < self._slow_dist:
                v_des = min(v_des, 0.06)
            else:
                self.cur_lin = 0.0
                v_des = 0.0
            new_reason = "heading_alignment"

        # Preserve the controller's intent before obstacle scaling.  When
        # clearance is already inside stop_dist the scale below becomes zero;
        # testing the scaled velocity would then hide a real lidar/depth stop
        # and prevent persistent-block recovery from ever starting.
        forward_intent = v_des > 1e-6

        if v_des > 0.0 and composite < self._slow_dist:
            scale = (composite - self._stop_dist) / (
                self._slow_dist - self._stop_dist)
            v_des *= max(0.0, min(1.0, scale))

        v = self._ramp_lin(v_des)
        w = self._ramp_ang(w_des)
        if dist_to_goal < 0.80 and v > 0.18:
            self.cur_lin = 0.18
            v = 0.18

        # ── emergency stop (per-sensor) ──
        if forward_intent and self.lidar_clearance <= self._stop_dist:
            self.cur_lin = 0.0
            v = 0.0
            new_reason = "lidar_stop"
            self._blocked_timer += self.dt
            if self._blocked_timer >= self._blocked_thresh:
                new_path = self._try_plan_with_fallback(
                    base_x, base_y, self.goal_x, self.goal_y)
                self._last_replan_time = now
                if new_path is not None:
                    self._consider_new_path(
                        new_path, base_x, base_y, force=True)
                self._blocked_timer = 0.0
        elif forward_intent and self.depth_clearance_val <= self._stop_dist:
            self.cur_lin = 0.0
            v = 0.0
            new_reason = "depth_stop"
        else:
            self._blocked_timer = max(0.0,
                                      self._blocked_timer - 2.0 * self.dt)

        # ── motion-arc prediction ──
        arc_blocked = v > 0.0 and not self._motion_is_free(
            base_x, base_y, base_yaw, v, w)
        if arc_blocked:
            self.cur_lin = 0.0
            v = 0.0
            new_reason = "arc_blocked"
            self._arc_blocked_timer += self.dt
            if self._arc_blocked_timer >= self._arc_replan_threshold:
                new_path = self._try_plan_with_fallback(
                    base_x, base_y, self.goal_x, self.goal_y)
                self._last_replan_time = now
                if new_path is not None:
                    self._consider_new_path(
                        new_path, base_x, base_y, force=True)
                self._arc_blocked_timer = 0.0
            if (point_to_rect_clearance(
                    base_x, base_y, DELIVERY_TABLE_COSTMAP_BOUNDS)
                    <= WHOLE_BODY_KEEP_OUT_RADIUS + 0.12):
                new_reason = "table_keepout"
        else:
            self._arc_blocked_timer = max(
                0.0, self._arc_blocked_timer - 2.0 * self.dt)

        # ── table-rotation guard ──
        if (abs(w) > 1e-6 and
                not self._table_rotation_is_free(base_x, base_y)):
            self.cur_ang = 0.0
            w = 0.0
            new_reason = "table_keepout"

        # ── no-path guard ──
        if not self.path and v == 0.0 and w == 0.0:
            new_reason = new_reason or self._format_no_path_reason(
                "no_path")

        self.stop_reason = new_reason
        if self._maybe_start_reverse_recovery(
                new_reason, base_x, base_y, base_yaw, laser_msg, now):
            self.stop_reason = "reverse_recovery_start"
            return 0.0, 0.0, False

        return v, w, False

    # ---- helpers ----
    def _try_plan_with_fallback(self, bx, by, gx, gy):
        """Plan with full costmap; fall back to lidar-only on failure.

        If the vision layer alone causes A* to fail, the lidar-only path is
        used while depth remains active for local braking.
        """
        # Store a transaction-local snapshot before either A* call can reset
        # its public ``failure_reason`` field.
        self._last_plan_mode = None
        self._last_plan_full_failure = None
        self._last_plan_fallback_failure = None
        lidar_cnt, vis_cnt = self.cm.obstacle_counts()
        self._last_plan_lidar_count = lidar_cnt
        self._last_plan_vision_count = vis_cnt

        # ── full map: static + lidar + vision ──
        path = self.planner.plan(bx, by, gx, gy)
        failure = self.planner.failure_reason
        self._last_plan_full_failure = failure

        if path is not None:
            self._last_plan_mode = "full"
            return path

        # ── fallback: static + lidar only (exclude vision) ──
        if vis_cnt > 0:
            saved_vision = self.cm.vision_raw.copy()
            try:
                self.cm.vision_raw.fill(FREE)
                self.cm._rebuild_dynamic()
                path2 = self.planner.plan(bx, by, gx, gy)
                failure2 = self.planner.failure_reason
                self._last_plan_fallback_failure = failure2
            finally:
                self.cm.vision_raw = saved_vision
                self.cm._rebuild_dynamic()

            if path2 is not None:
                # vision-only obstacle was blocking — use lidar path
                self._last_plan_mode = "lidar_only"
                self.planner.failure_reason = None  # not a real failure
                return path2

        self._last_plan_mode = "failed"
        # Preserve the failure of the least restrictive attempted map for
        # compatibility, while detailed logging uses both captured values.
        self.planner.failure_reason = (
            self._last_plan_fallback_failure or failure or "unknown")
        return None

    def _format_no_path_reason(self, prefix):
        """Format stable diagnostics for the last planning transaction."""
        full = self._last_plan_full_failure or "unknown"
        details = [f"full={full}"]
        if self._last_plan_vision_count > 0:
            fallback = self._last_plan_fallback_failure or "not_run"
            details.append(f"lidar_only={fallback}")
        details.append(
            "obs:lidar=" + str(self._last_plan_lidar_count) +
            ",vis=" + str(self._last_plan_vision_count))
        return prefix + "(" + ";".join(details) + ")"

    def _install_path(self, path):
        self.path = path
        if path:
            self.nav_goal_x, self.nav_goal_y = path[-1]

    @staticmethod
    def _polyline_length(path):
        return sum(math.hypot(path[i + 1][0] - path[i][0],
                              path[i + 1][1] - path[i][1])
                   for i in range(len(path) - 1))

    def _remaining_path_length(self, bx, by):
        if not self.path:
            return float('inf')
        ci = self._closest_index(bx, by)
        length = math.hypot(self.path[ci][0] - bx,
                            self.path[ci][1] - by)
        length += self._polyline_length(self.path[ci:])
        return length

    def _consider_new_path(self, new_path, bx, by, force=False):
        """Install a replan only when required or materially better.

        Laser/depth boundary noise can make two equal-cost routes around a box
        alternate every planning cycle.  Keeping the still-valid current path
        prevents the desired heading from flipping left/right while retaining
        immediate replacement when an obstacle invalidates that path.
        """
        if not new_path:
            return False
        if force or not self.path or not self._path_valid(bx, by):
            self._install_path(new_path)
            return True

        old_length = self._remaining_path_length(bx, by)
        new_length = self._polyline_length(new_path)
        required_gain = max(0.03, old_length * self._path_improvement_ratio)
        if new_length + required_gain < old_length:
            self._install_path(new_path)
            return True
        return False

    def _update_rotation_watchdog(self, bx, by, byaw):
        """Return True after excessive rotation without translational progress."""
        if self._rotation_anchor_x is None:
            self._rotation_anchor_x = bx
            self._rotation_anchor_y = by
            self._last_base_yaw = byaw
            return False

        translation = math.hypot(
            bx - self._rotation_anchor_x,
            by - self._rotation_anchor_y)
        if translation > 0.10:
            self._rotation_anchor_x = bx
            self._rotation_anchor_y = by
            self._rotation_accum = 0.0
            self._rotation_recoveries = 0
            self._reverse_recovery_attempts = 0
        elif self._last_base_yaw is not None:
            self._rotation_accum += abs(angdist(
                self._last_base_yaw, byaw))
        self._last_base_yaw = byaw
        return self._rotation_accum > self._rotation_loop_limit

    @staticmethod
    def _rear_scan_available(laser_msg):
        """Whether the current scan observes a substantial rear hemisphere."""
        if laser_msg is None or not laser_msg.ranges:
            return False
        span = abs(float(laser_msg.angle_increment)) * len(laser_msg.ranges)
        return span >= 1.5 * math.pi

    def _maybe_start_reverse_recovery(
            self, reason, bx, by, byaw, laser_msg, now):
        """Enter measured straight backup after a persistent local block."""
        recoverable = {"lidar_stop", "arc_blocked", "rotation_loop"}
        if reason not in recoverable:
            self._reverse_recovery_blocked_time = max(
                0.0,
                self._reverse_recovery_blocked_time - 2.0 * self.dt)
            if self._reverse_recovery_blocked_time <= 0.0:
                self._reverse_recovery_block_anchor_x = None
                self._reverse_recovery_block_anchor_y = None
            return False

        if self._reverse_recovery_block_anchor_x is None:
            self._reverse_recovery_block_anchor_x = float(bx)
            self._reverse_recovery_block_anchor_y = float(by)
            self._reverse_recovery_blocked_time = 0.0

        blocked_translation = math.hypot(
            bx - self._reverse_recovery_block_anchor_x,
            by - self._reverse_recovery_block_anchor_y)
        if blocked_translation > self._reverse_recovery_no_progress_m:
            self._reverse_recovery_block_anchor_x = float(bx)
            self._reverse_recovery_block_anchor_y = float(by)
            self._reverse_recovery_blocked_time = 0.0
            return False

        self._reverse_recovery_blocked_time += self.dt
        if reason == "rotation_loop":
            # The rotation watchdog already proves prolonged lack of
            # translation, so it need not wait for a second one-second timer.
            self._reverse_recovery_blocked_time = max(
                self._reverse_recovery_blocked_time,
                self._reverse_recovery_trigger_s)

        if (self._reverse_recovery_blocked_time
                < self._reverse_recovery_trigger_s):
            return False
        if now < self._reverse_recovery_cooldown_until:
            return False
        if (self._reverse_recovery_attempts
                >= self._reverse_recovery_max_attempts):
            return False
        if not self._rear_scan_available(laser_msg):
            return False
        if self.rear_clearance <= self._reverse_recovery_rear_stop_m:
            return False

        signed_distance = -self._reverse_recovery_distance_m
        if not self._straight_translation_is_free(
                bx, by, byaw, signed_distance):
            return False

        self._reverse_recovery_phase = "backup"
        self._reverse_recovery_start_x = float(bx)
        self._reverse_recovery_start_y = float(by)
        self._reverse_recovery_start_yaw = float(byaw)
        self._reverse_recovery_started_at = float(now)
        self._reverse_recovery_attempts += 1
        self._reverse_recovery_blocked_time = 0.0
        self.cur_lin = 0.0
        self.cur_ang = 0.0
        return True

    def _finish_reverse_recovery(self, bx, by, byaw, now):
        """Stop recovery, reset watchdogs and plan from the changed pose."""
        self._reverse_recovery_phase = None
        self._reverse_recovery_start_x = None
        self._reverse_recovery_start_y = None
        self._reverse_recovery_start_yaw = None
        self._reverse_recovery_blocked_time = 0.0
        self._reverse_recovery_block_anchor_x = float(bx)
        self._reverse_recovery_block_anchor_y = float(by)
        self._reverse_recovery_cooldown_until = float(now) + 2.0
        self.cur_lin = 0.0
        self.cur_ang = 0.0
        self._blocked_timer = 0.0
        self._arc_blocked_timer = 0.0
        self._rotation_accum = 0.0
        self._rotation_anchor_x = float(bx)
        self._rotation_anchor_y = float(by)
        self._last_base_yaw = float(byaw)
        self._best_goal_dist = float('inf')
        self._last_progress_time = float(now)

        new_path = self._try_plan_with_fallback(
            bx, by, self.goal_x, self.goal_y)
        self._last_replan_time = float(now)
        if new_path is not None:
            self._install_path(new_path)
            self._replan_hold_until = float(now) + 2.0
        else:
            self.path = []

    def _reverse_recovery_tick(self, bx, by, byaw, laser_msg, now):
        """Execute a low-speed odometry-measured straight reverse."""
        if (self._reverse_recovery_start_x is None
                or self._reverse_recovery_start_y is None
                or self._reverse_recovery_start_yaw is None):
            self._finish_reverse_recovery(bx, by, byaw, now)
            self.stop_reason = "reverse_recovery_invalid"
            return 0.0, 0.0, False

        heading_x = math.cos(self._reverse_recovery_start_yaw)
        heading_y = math.sin(self._reverse_recovery_start_yaw)
        moved_back = (
            (self._reverse_recovery_start_x - bx) * heading_x
            + (self._reverse_recovery_start_y - by) * heading_y)
        moved_back = max(0.0, float(moved_back))
        elapsed = float(now) - self._reverse_recovery_started_at
        remaining = max(
            0.0, self._reverse_recovery_distance_m - moved_back)

        if moved_back >= self._reverse_recovery_distance_m:
            self._finish_reverse_recovery(bx, by, byaw, now)
            self.stop_reason = "reverse_recovery_complete"
            return 0.0, 0.0, False
        if elapsed >= self._reverse_recovery_timeout_s:
            self._finish_reverse_recovery(bx, by, byaw, now)
            self.stop_reason = "reverse_recovery_timeout"
            return 0.0, 0.0, False
        if not self._rear_scan_available(laser_msg):
            self._finish_reverse_recovery(bx, by, byaw, now)
            self.stop_reason = "reverse_recovery_no_rear_scan"
            return 0.0, 0.0, False
        if self.rear_clearance <= self._reverse_recovery_rear_stop_m:
            self._finish_reverse_recovery(bx, by, byaw, now)
            self.stop_reason = "reverse_recovery_rear_stop"
            return 0.0, 0.0, False
        if not self._straight_translation_is_free(
                bx, by, byaw, -remaining):
            self._finish_reverse_recovery(bx, by, byaw, now)
            self.stop_reason = "reverse_recovery_path_blocked"
            return 0.0, 0.0, False

        yaw_error = angdist(byaw, self._reverse_recovery_start_yaw)
        angular = max(
            -self._reverse_recovery_max_ang,
            min(self._reverse_recovery_max_ang,
                self._reverse_recovery_yaw_gain * yaw_error))
        if not self._motion_is_free(
                bx, by, byaw, -self._reverse_recovery_speed, angular):
            self._finish_reverse_recovery(bx, by, byaw, now)
            self.stop_reason = "reverse_recovery_arc_blocked"
            return 0.0, 0.0, False
        self.cur_lin = -self._reverse_recovery_speed
        self.cur_ang = angular
        self.stop_reason = "reverse_recovery"
        return -self._reverse_recovery_speed, angular, False

    def _closest_index(self, bx, by):
        best_i, best_d2 = 0, float('inf')
        for i, (px, py) in enumerate(self.path):
            d2 = (px - bx)**2 + (py - by)**2
            if d2 < best_d2:
                best_d2, best_i = d2, i
        return best_i

    def _lookahead_dist(self, bx, by, dist_to_goal):
        """Adaptive lookahead: shrink near the goal."""
        la = self.lookahead_base
        if dist_to_goal < 0.8:
            la = self.lookahead_min + \
                (la - self.lookahead_min) * (dist_to_goal / 0.8)
        return la

    def _lookahead_point(self, bx, by, lookahead_dist):
        """Return a point exactly ``lookahead_dist`` ahead on the polyline.

        The original implementation returned the *next vertex* once a segment
        exceeded the desired distance.  On a five-metre segment this silently
        turned a 0.6 m lookahead into five metres and made the robot cut the
        middle-wall corner.  Projecting onto the path and interpolating within
        each segment keeps the geometric lookahead bounded.
        """
        if len(self.path) == 1:
            return self.path[0]

        best_d2 = float('inf')
        best_segment = 0
        best_point = self.path[0]

        for i in range(len(self.path) - 1):
            ax, ay = self.path[i]
            bx_seg, by_seg = self.path[i + 1]
            vx, vy = bx_seg - ax, by_seg - ay
            length2 = vx * vx + vy * vy
            if length2 <= 1e-12:
                t = 0.0
            else:
                t = max(0.0, min(1.0,
                    ((bx - ax) * vx + (by - ay) * vy) / length2))
            px, py = ax + t * vx, ay + t * vy
            d2 = (px - bx) ** 2 + (py - by) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best_segment = i
                best_point = (px, py)

        remaining = lookahead_dist
        px, py = best_point
        for i in range(best_segment, len(self.path) - 1):
            nx, ny = self.path[i + 1]
            seg_len = math.hypot(nx - px, ny - py)
            if seg_len >= remaining and seg_len > 1e-12:
                ratio = remaining / seg_len
                return (px + ratio * (nx - px),
                        py + ratio * (ny - py))
            remaining -= seg_len
            px, py = nx, ny
        return self.path[-1]

    def _ramp_lin(self, des):
        """Acceleration-limited linear velocity."""
        delta = max(-self.max_lin_acc * self.dt,
                    min(self.max_lin_acc * self.dt, des - self.cur_lin))
        self.cur_lin += delta
        return self.cur_lin

    def _ramp_ang(self, des):
        """Acceleration-limited angular velocity."""
        des = max(-self.max_ang, min(self.max_ang, des))
        delta = max(-self.max_ang_acc * self.dt,
                    min(self.max_ang_acc * self.dt, des - self.cur_ang))
        self.cur_ang += delta
        return self.cur_ang

    def _front_clearance(self, laser_msg, corridor_half_width=None):
        """Longitudinal clearance inside the chassis-width front corridor."""
        if laser_msg is None or not laser_msg.ranges:
            return float('inf')
        if corridor_half_width is None:
            corridor_half_width = self.front_corridor_half_width
        angle = float(laser_msg.angle_min)
        angle_inc = float(laser_msg.angle_increment)
        range_min = max(0.02, float(getattr(laser_msg, 'range_min', 0.02)))
        range_max = float(getattr(laser_msg, 'range_max', float('inf')))
        clearance = float('inf')
        for r in laser_msg.ranges:
            beam_angle = wrap_to_pi(angle)
            r = float(r)
            if (math.isfinite(r) and range_min < r <= range_max
                    and abs(beam_angle) < math.pi / 2.0):
                forward = r * math.cos(beam_angle)
                lateral = r * math.sin(beam_angle)
                if (forward > 0.0 and
                        abs(lateral) <= corridor_half_width):
                    clearance = min(clearance, forward)
            angle += angle_inc
        return clearance

    def _rear_clearance(self, laser_msg):
        """Longitudinal clearance inside the chassis-width rear corridor."""
        if laser_msg is None or not laser_msg.ranges:
            return float('inf')
        angle = float(laser_msg.angle_min)
        angle_inc = float(laser_msg.angle_increment)
        range_min = max(0.02, float(getattr(
            laser_msg, 'range_min', 0.02)))
        range_max = float(getattr(
            laser_msg, 'range_max', float('inf')))
        clearance = float('inf')
        for r in laser_msg.ranges:
            beam_angle = wrap_to_pi(angle)
            r = float(r)
            if (math.isfinite(r) and range_min < r <= range_max
                    and abs(beam_angle) > math.pi / 2.0):
                longitudinal = -r * math.cos(beam_angle)
                lateral = r * math.sin(beam_angle)
                if (longitudinal > 0.0 and
                        abs(lateral) <= self.front_corridor_half_width):
                    clearance = min(clearance, longitudinal)
            angle += angle_inc
        return clearance

    def _check_front_blocked(self, laser_msg, base_yaw=None):
        """Compatibility helper for users of the original controller API."""
        return (self._front_clearance(
            laser_msg, self.hard_stop_corridor_half_width)
            <= self._stop_dist)

    def _straight_translation_is_free(
            self, bx, by, byaw, signed_distance):
        """Check a complete straight candidate, including inflation escape."""
        distance = abs(float(signed_distance))
        if distance <= 1e-6:
            return True
        direction = 1.0 if signed_distance > 0.0 else -1.0
        start_free = self.cm.is_free_world(bx, by)
        start_static_free = self.cm.is_static_free_world(bx, by)
        start_clearance = self.cm.raw_dynamic_clearance_world(bx, by)
        escaping_dynamic_inflation = (
            not start_free
            and start_static_free
            and math.isfinite(start_clearance)
            and start_clearance > 0.5 * self.cm.resolution
        )
        previous_clearance = start_clearance
        sample_step = 0.5 * self.cm.resolution
        steps = max(1, int(math.ceil(distance / sample_step)))
        for index in range(1, steps + 1):
            travelled = distance * index / steps
            x = bx + direction * travelled * math.cos(byaw)
            y = by + direction * travelled * math.sin(byaw)
            if not self.cm.is_static_free_world(x, y):
                return False
            clearance = self.cm.raw_dynamic_clearance_world(x, y)
            if (escaping_dynamic_inflation
                    and math.isfinite(previous_clearance)
                    and math.isfinite(clearance)
                    and clearance + 1e-4 < previous_clearance):
                # While escaping an inflated obstacle halo, require every
                # sample to move away from the raw hit.  If the complete
                # candidate already lies in free cells, millimetre-level scan
                # geometry changes must not veto an otherwise safe backup.
                return False
            if not self.cm.is_free_world(x, y):
                if (not escaping_dynamic_inflation
                        or not math.isfinite(clearance)
                        or clearance <= 0.5 * self.cm.resolution
                        or clearance + 1e-4 < previous_clearance):
                    return False
            previous_clearance = clearance
            if (point_to_rect_clearance(
                    x, y, DELIVERY_TABLE_COSTMAP_BOUNDS)
                    <= WHOLE_BODY_KEEP_OUT_RADIUS):
                return False
        if (escaping_dynamic_inflation
                and previous_clearance < start_clearance + 0.001):
            return False
        return True

    def _motion_is_free(self, bx, by, byaw, linear, angular,
                        horizon=0.45, sample_dt=0.05):
        """Check the near-future arc, allowing safe inflation-layer escape.

        If the current pose lies only in a dynamic obstacle's inflated halo,
        permit a trajectory that monotonically increases clearance from the
        raw obstacle hits.  Static cells and raw dynamic hit cells remain
        forbidden, so this recovery cannot drive through a real obstacle.
        At low speed the prediction horizon is shortened so a safe creep is
        not rejected because of an arc the robot will not reach soon.
        """
        if abs(linear) <= 1e-6:
            return True
        adaptive_horizon = horizon * min(1.0, abs(linear) / 0.35)
        adaptive_horizon = max(0.08, adaptive_horizon)
        x, y, yaw = bx, by, byaw
        start_free = self.cm.is_free_world(x, y)
        start_static_free = self.cm.is_static_free_world(x, y)
        start_clearance = self.cm.raw_dynamic_clearance_world(x, y)
        escaping_dynamic_inflation = (
            not start_free
            and start_static_free
            and math.isfinite(start_clearance)
            and start_clearance > 0.5 * self.cm.resolution
        )
        previous_clearance = start_clearance
        steps = max(1, int(math.ceil(adaptive_horizon / sample_dt)))
        dt = adaptive_horizon / steps
        for _ in range(steps):
            yaw = wrap_to_pi(yaw + angular * dt)
            x += linear * math.cos(yaw) * dt
            y += linear * math.sin(yaw) * dt
            if not self.cm.is_static_free_world(x, y):
                return False
            clearance = self.cm.raw_dynamic_clearance_world(x, y)
            if not self.cm.is_free_world(x, y):
                if (not escaping_dynamic_inflation
                        or not math.isfinite(clearance)
                        or clearance <= 0.5 * self.cm.resolution
                        or clearance + 1e-4 < previous_clearance):
                    return False
            previous_clearance = clearance
            # The table is fixed and known exactly.  This analytical guard is
            # intentionally independent of the raster costmap so a rounding,
            # path-following or small localisation error cannot command the
            # chassis into the delivery table.
            if (point_to_rect_clearance(
                    x, y, DELIVERY_TABLE_COSTMAP_BOUNDS)
                    <= WHOLE_BODY_KEEP_OUT_RADIUS):
                return False
        if escaping_dynamic_inflation:
            required_progress = min(
                0.005,
                max(0.001, 0.10 * abs(linear) * adaptive_horizon))
            if previous_clearance < start_clearance + required_progress:
                return False
        return True

    @staticmethod
    def _table_rotation_is_free(bx, by):
        """Whether rotating the parked whole body is safe by the table."""
        return (point_to_rect_clearance(
                    bx, by, DELIVERY_TABLE_COSTMAP_BOUNDS)
                > TABLE_ROTATION_KEEP_OUT_RADIUS)

    def _path_valid(self, bx, by):
        """Check whether the remaining path is collision-free."""
        ci = self._closest_index(bx, by)
        if not self.cm.line_is_free(bx, by,
                                    self.path[ci][0], self.path[ci][1]):
            return False
        for i in range(ci, len(self.path) - 1):
            if not self.cm.line_is_free(
                    self.path[i][0], self.path[i][1],
                    self.path[i+1][0], self.path[i+1][1]):
                return False
        return True


# ============================================================================
# Top-level Navigator
# ============================================================================
class SupermarketNavigator:
    """Convenience wrapper around Costmap2D + AStarPlanner + NavigationController.

    Usage::

        nav = SupermarketNavigator()
        nav.set_goal(x, y, yaw)
        # in control loop:
        v, w, reached = nav.update(base_x, base_y, base_yaw, laser_msg)
    """

    def __init__(self):
        self.costmap = Costmap2D()
        self.planner = AStarPlanner(self.costmap)
        self.controller = NavigationController(self.costmap, self.planner)
        self._reached = False
        self._laser_count = 0
        self._last_scan_msg = None
        self._last_depth_token = None
        self._goal = None
        self._goal_start_pose = None
        self._goal_path_snapshot = None
        self._cached_path_active = False
        self._cached_path_info = {"enabled": False, "cache_hit": False}
        self.path_memory = PathMemory(
            enabled=os.environ.get("SUPERMARKET_PATH_MEMORY", "0") == "1",
            storage_path=os.environ.get(
                "SUPERMARKET_PATH_MEMORY_FILE",
                "/root/.cache/supermarket_path_memory.json",
            ),
        )
        self._cached_path_info = {
            "enabled": self.path_memory.enabled,
            "cache_hit": False,
        }

    def set_goal(self, x, y, yaw=None):
        self.controller.set_goal(x, y, yaw)
        self._reached = False
        self._goal = (float(x), float(y), None if yaw is None else float(yaw))
        self._goal_start_pose = None
        self._goal_path_snapshot = None
        self._cached_path_active = False
        self._cached_path_info = {"enabled": self.path_memory.enabled, "cache_hit": False}

    def _try_restore_cached_path(self, base_x, base_y, base_yaw, now):
        if not self.path_memory.enabled or self._goal is None or self.controller.path:
            return
        path, info = self.path_memory.load_path(
            start_x=base_x,
            start_y=base_y,
            start_yaw=base_yaw,
            goal_x=self._goal[0],
            goal_y=self._goal[1],
            goal_yaw=self._goal[2],
        )
        self._cached_path_info = info
        if path:
            self.controller._install_path(path)
            self.controller._last_replan_time = float(now)
            self.controller._replan_hold_until = float(now) + 2.0
            self._cached_path_active = True

    def _save_successful_path(self, base_x, base_y, base_yaw):
        if (not self.path_memory.enabled or self._goal is None
                or not self.controller.path):
            return
        start_pose = self._goal_start_pose
        if start_pose is None:
            start_pose = (float(base_x), float(base_y), float(base_yaw))
        path_to_save = self._goal_path_snapshot or list(self.controller.path)
        if not path_to_save:
            return
        self.path_memory.save_path(
            start_x=start_pose[0],
            start_y=start_pose[1],
            start_yaw=start_pose[2],
            goal_x=self._goal[0],
            goal_y=self._goal[1],
            goal_yaw=self._goal[2],
            path=path_to_save,
            source="cached" if self._cached_path_active else "planner",
        )

    def update(self, base_x, base_y, base_yaw, laser_msg=None,
               depth_clearance=None, depth_token=None, time_now=None):
        """Call at control-loop rate (≈50 Hz).

        Returns (v_linear, v_angular, reached_goal).
        """
        # A ROS subscriber typically exposes the same message object for many
        # 50 Hz control ticks.  Process each actual 12 Hz scan exactly once.
        if laser_msg is not None and laser_msg is not self._last_scan_msg:
            self._last_scan_msg = laser_msg
            self._laser_count += 1
            self.costmap.update_from_scan(laser_msg,
                                          base_x, base_y, base_yaw)

        if (depth_clearance is not None and
                (depth_token is None or
                 depth_token != self._last_depth_token)):
            self._last_depth_token = depth_token
            self.costmap.update_from_depth_obstacle(
                depth_clearance, base_x, base_y, base_yaw)

        current_now = time.monotonic() if time_now is None else time_now
        if self._goal is not None and self._goal_start_pose is None:
            self._goal_start_pose = (
                float(base_x), float(base_y), float(base_yaw))
        self._try_restore_cached_path(base_x, base_y, base_yaw, current_now)

        v, w, reached = self.controller.compute_velocity(
            base_x, base_y, base_yaw,
            laser_msg=laser_msg,
            depth_clearance=depth_clearance,
            time_now=current_now)

        if self._goal is not None and self._goal_path_snapshot is None and self.controller.path:
            # Keep the first full path produced for this goal.  Near-goal
            # replans later in the run can replace controller.path with a
            # short terminal segment, which is not useful as a remembered route.
            self._goal_path_snapshot = list(self.controller.path)

        self._reached = reached
        if reached:
            self._save_successful_path(base_x, base_y, base_yaw)
        return v, w, reached

    @property
    def reached(self):
        return self._reached

    def current_path(self):
        """Return the current planned path (list of (x,y)) for visualisation."""
        return self.controller.path

    def get_costmap_grid(self):
        """Return the master costmap (for debugging/visualisation)."""
        return self.costmap.master

    def path_memory_status(self):
        status = self.path_memory.summary()
        status.update(self._cached_path_info)
        status["cached_path_active"] = self._cached_path_active
        return status
