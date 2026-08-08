"""Constraint-checked random layouts for the supermarket obstacle corridor."""

from dataclasses import dataclass
import heapq
import math
import random


OBSTACLE_BODIES = tuple(f"dynamic_obstacle_box_{index:02d}" for index in range(1, 6))

# Positions are in dynamic_obstacle_corridor's local frame. The candidates keep
# the physical boxes inside the corridor; the validator additionally checks the
# assigned box orientation, pairwise clearance, and robot traversability.
OBSTACLE_POSITIONS = (
    (-1.15, -1.95), (-0.45, -1.45), (0.35, -1.90),
    (-0.95, -0.55), (-0.10, -0.35), (0.85, -0.55),
    (-1.15, 0.55), (-0.35, 0.95), (0.55, 0.45),
    (-0.95, 1.70), (-0.05, 1.75), (0.85, 1.95),
    (-1.15, 2.35), (-0.35, 2.30), (0.55, 2.25),
)

# Approved bottom-to-top slalom skeletons. The first one is the sixth visual
# layout that was manually accepted during validation. Runtime jitter makes each
# round distinct while evaluate_obstacle_layout keeps every realization safe.
SLALOM_TEMPLATE_POSITIONS = (
    ((0.35, -1.90), (-0.95, -0.55), (0.55, 0.45), (-0.35, 0.95), (-1.15, 2.35)),
    ((0.75, -1.95), (-0.75, -0.30), (0.45, 0.35), (-0.55, 1.15), (-1.05, 2.35)),
    ((0.30, -1.75), (-0.85, -0.55), (0.70, 0.55), (-0.45, 1.05), (-1.00, 2.10)),
)
POSITION_JITTER_X = 0.10
POSITION_JITTER_Y = 0.08

# All obstacle bodies use the same 0.60 x 0.40 m box. Its yaw is randomized
# in the horizontal plane; Y-axis rotation is intentionally not used because it
# would tip a physical obstacle into the floor.
OBSTACLE_HALF_SIZE = (0.30, 0.20)
OBSTACLE_YAWS = (0.0, math.pi / 4.0, -math.pi / 4.0, math.pi / 2.0)

# MMK2's chassis collision boxes are about 0.44 x 0.40 m including the wheels.
# A circle covers the complete turning sweep, then adds 5 cm navigation margin.
ROBOT_HALF_LENGTH = 0.22
ROBOT_HALF_WIDTH = 0.20
ROBOT_SAFETY_MARGIN = 0.05
ROBOT_CLEARANCE_RADIUS = math.hypot(ROBOT_HALF_LENGTH, ROBOT_HALF_WIDTH) + ROBOT_SAFETY_MARGIN
SAFE_CLEARANCE_RADIUS = 0.40

# Corridor inner edges in its local frame. The west/south edges come from the
# competition perimeter walls; the east edge is corridor_right_board.
CORRIDOR_X_MIN = -1.51
CORRIDOR_X_MAX = 1.46
CORRIDOR_Y_MIN = -2.71
CORRIDOR_Y_MAX = 2.71
OBSTACLE_WALL_GAP = 0.05
OBSTACLE_PAIR_GAP = 0.05

# This is the nominal shelf-side corridor entry and the baseline delivery-table
# approach, both transformed into dynamic_obstacle_corridor's local frame.
ROUTE_START = (0.46, 2.30)
ROUTE_GOAL = (-0.92, -1.79)
GRID_RESOLUTION = 0.05
MIN_DETOUR_METERS = 0.30
MIN_LONGITUDINAL_BANDS = 5
MIN_LATERAL_SHIFT = 0.40
MIN_LONGITUDINAL_SEPARATION = 0.40
MAX_LAYOUT_ATTEMPTS = 5000


@dataclass(frozen=True)
class LayoutEvaluation:
    valid: bool
    reason: str
    path_length: float | None = None
    detour: float | None = None
    occupied_bands: int = 0


@dataclass(frozen=True)
class GeneratedLayout:
    positions: dict[str, tuple[float, float, float]]
    yaws: dict[str, float]
    attempts: int
    path_length: float
    detour: float
    occupied_bands: int


def _longitudinal_band(y):
    if y < -1.00:
        return 0
    if y < 0.10:
        return 1
    if y < 0.75:
        return 2
    if y < 1.40:
        return 3
    return 4


def _oriented_half_extents(yaw):
    half_x, half_y = OBSTACLE_HALF_SIZE
    cos_yaw = abs(math.cos(yaw))
    sin_yaw = abs(math.sin(yaw))
    return (
        cos_yaw * half_x + sin_yaw * half_y,
        sin_yaw * half_x + cos_yaw * half_y,
    )


def _physical_geometry_is_clear(selected):
    for x, y, yaw in selected:
        half_x, half_y = _oriented_half_extents(yaw)
        if x - half_x < CORRIDOR_X_MIN + OBSTACLE_WALL_GAP:
            return False, "obstacle too close to west wall"
        if x + half_x > CORRIDOR_X_MAX - OBSTACLE_WALL_GAP:
            return False, "obstacle too close to east wall"
        if y - half_y < CORRIDOR_Y_MIN + OBSTACLE_WALL_GAP:
            return False, "obstacle too close to south boundary"
        if y + half_y > CORRIDOR_Y_MAX - OBSTACLE_WALL_GAP:
            return False, "obstacle too close to north boundary"

    for index, (x, y, yaw) in enumerate(selected):
        half_x, half_y = _oriented_half_extents(yaw)
        for other_index in range(index):
            other_x, other_y, other_yaw = selected[other_index]
            other_half_x, other_half_y = _oriented_half_extents(other_yaw)
            separated_x = abs(x - other_x) >= half_x + other_half_x + OBSTACLE_PAIR_GAP
            separated_y = abs(y - other_y) >= half_y + other_half_y + OBSTACLE_PAIR_GAP
            if not (separated_x or separated_y):
                return False, "obstacles overlap or violate pairwise gap"
    return True, "ok"


def _grid_path_length(selected, clearance_radius=ROBOT_CLEARANCE_RADIUS):
    radius = float(clearance_radius)
    x_min = CORRIDOR_X_MIN + radius
    x_max = CORRIDOR_X_MAX - radius
    y_min = CORRIDOR_Y_MIN + radius
    y_max = CORRIDOR_Y_MAX - radius
    resolution = GRID_RESOLUTION
    nx = round((x_max - x_min) / resolution) + 1
    ny = round((y_max - y_min) / resolution) + 1

    def to_cell(point):
        return (
            round((point[0] - x_min) / resolution),
            round((point[1] - y_min) / resolution),
        )

    def to_point(cell):
        return x_min + cell[0] * resolution, y_min + cell[1] * resolution

    def in_bounds(cell):
        return 0 <= cell[0] < nx and 0 <= cell[1] < ny

    def blocked(cell):
        x, y = to_point(cell)
        for obstacle_x, obstacle_y, yaw in selected:
            half_x, half_y = _oriented_half_extents(yaw)
            if abs(x - obstacle_x) <= half_x + radius and abs(y - obstacle_y) <= half_y + radius:
                return True
        return False

    start = to_cell(ROUTE_START)
    goal = to_cell(ROUTE_GOAL)
    if not in_bounds(start) or not in_bounds(goal) or blocked(start) or blocked(goal):
        return None

    moves = (
        (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
        (1, 1, math.sqrt(2.0)), (1, -1, math.sqrt(2.0)),
        (-1, 1, math.sqrt(2.0)), (-1, -1, math.sqrt(2.0)),
    )
    distance = {start: 0.0}
    queue = [(0.0, start)]
    while queue:
        _, cell = heapq.heappop(queue)
        cell_distance = distance[cell]
        if cell == goal:
            return cell_distance
        for dx, dy, cost in moves:
            neighbor = cell[0] + dx, cell[1] + dy
            if not in_bounds(neighbor) or blocked(neighbor):
                continue
            # A circular robot cannot cut diagonally through two occupied corners.
            if dx and dy:
                if blocked((cell[0] + dx, cell[1])) or blocked((cell[0], cell[1] + dy)):
                    continue
            candidate_distance = cell_distance + cost * resolution
            if candidate_distance >= distance.get(neighbor, math.inf):
                continue
            distance[neighbor] = candidate_distance
            heuristic = math.dist(neighbor, goal) * resolution
            heapq.heappush(queue, (candidate_distance + heuristic, neighbor))
    return None


EMPTY_CORRIDOR_PATH_LENGTH = _grid_path_length(())


def evaluate_obstacle_layout(selected, clearance_radius=ROBOT_CLEARANCE_RADIUS):
    selected = tuple(selected)
    if len(selected) != len(OBSTACLE_BODIES):
        return LayoutEvaluation(False, "layout must contain exactly five obstacles")
    if any(len(candidate) != 3 for candidate in selected):
        return LayoutEvaluation(False, "each obstacle requires x, y, and yaw")
    if len({(x, y) for x, y, _ in selected}) != len(selected):
        return LayoutEvaluation(False, "obstacle positions must be unique")

    physical_clear, reason = _physical_geometry_is_clear(selected)
    if not physical_clear:
        return LayoutEvaluation(False, reason)

    bands = [_longitudinal_band(y) for _, y, _ in selected]
    occupied_bands = len(set(bands))
    if occupied_bands < MIN_LONGITUDINAL_BANDS:
        return LayoutEvaluation(False, "obstacles do not cover enough of the route",
                                occupied_bands=occupied_bands)

    slalom = sorted(selected, key=lambda candidate: _longitudinal_band(candidate[1]))
    if [_longitudinal_band(y) for _, y, _ in slalom] != list(range(MIN_LONGITUDINAL_BANDS)):
        return LayoutEvaluation(False, "obstacles must occupy one position per route band",
                                occupied_bands=occupied_bands)
    if any(abs(next_x - x) < MIN_LATERAL_SHIFT
           for (x, _, _), (next_x, _, _) in zip(slalom, slalom[1:])):
        return LayoutEvaluation(False, "adjacent route bands do not force a lateral detour",
                                occupied_bands=occupied_bands)
    if any(next_y - y < MIN_LONGITUDINAL_SEPARATION
           for (_, y, _), (_, next_y, _) in zip(slalom, slalom[1:])):
        return LayoutEvaluation(False, "adjacent route bands are too close together",
                                occupied_bands=occupied_bands)

    path_length = _grid_path_length(selected, clearance_radius)
    if path_length is None:
        return LayoutEvaluation(False, "no robot-sized route from shelf to delivery",
                                occupied_bands=occupied_bands)
    empty_path_length = _grid_path_length((), clearance_radius)
    detour = path_length - empty_path_length
    if detour < MIN_DETOUR_METERS:
        return LayoutEvaluation(False, "layout does not require meaningful avoidance",
                                path_length, detour, occupied_bands)
    return LayoutEvaluation(True, "ok", path_length, detour, occupied_bands)


def generate_obstacle_layout(seed=None, clearance_radius=ROBOT_CLEARANCE_RADIUS):
    rng = random.Random(seed)
    last_reason = "no candidates evaluated"
    for attempts in range(1, MAX_LAYOUT_ATTEMPTS + 1):
        template = rng.choice(SLALOM_TEMPLATE_POSITIONS)
        selected = tuple(
            (
                x + rng.uniform(-POSITION_JITTER_X, POSITION_JITTER_X),
                y + rng.uniform(-POSITION_JITTER_Y, POSITION_JITTER_Y),
                rng.choice(OBSTACLE_YAWS),
            )
            for x, y in template
        )
        evaluation = evaluate_obstacle_layout(selected, clearance_radius)
        if not evaluation.valid:
            last_reason = evaluation.reason
            continue
        positions = {
            body: (x, y, 0.25)
            for body, (x, y, _) in zip(OBSTACLE_BODIES, selected)
        }
        yaws = {
            body: yaw
            for body, (_, _, yaw) in zip(OBSTACLE_BODIES, selected)
        }
        return GeneratedLayout(
            positions=positions,
            yaws=yaws,
            attempts=attempts,
            path_length=evaluation.path_length,
            detour=evaluation.detour,
            occupied_bands=evaluation.occupied_bands,
        )
    raise RuntimeError(
        f"failed to generate a traversable obstacle layout after "
        f"{MAX_LAYOUT_ATTEMPTS} attempts (last rejection: {last_reason})"
    )
