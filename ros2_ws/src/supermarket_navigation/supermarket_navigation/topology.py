"""Small semantic route graph for the fixed competition arena."""

from __future__ import annotations

from dataclasses import dataclass, field
import heapq
import math
from typing import Iterable, Mapping


@dataclass(frozen=True)
class Point2:
    x: float
    y: float

    def distance(self, other: "Point2") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)


@dataclass(frozen=True)
class RouteNode:
    name: str
    point: Point2
    tags: frozenset[str] = frozenset()


@dataclass(frozen=True)
class RouteEdge:
    source: str
    target: str
    min_width_m: float
    bidirectional: bool = True


@dataclass
class ArenaTopology:
    nodes: dict[str, RouteNode]
    edges: tuple[RouteEdge, ...]
    failure_penalties: dict[tuple[str, str], float] = field(default_factory=dict)

    @classmethod
    def competition_default(cls) -> "ArenaTopology":
        # Coordinates are public fixed-scene approach poses already used by
        # the baseline.  Product locations never appear in this graph.
        points = {
            "start": (1.92, -3.17),
            "east_south": (1.55, -2.20),
            "east_north": (1.55, 1.45),
            "west_south": (-1.90, -2.35),
            "west_north": (-1.90, 1.45),
            "north_cross": (0.00, 2.05),
            "shelf_A": (-1.735, 2.40),
            "shelf_B": (-0.850, 2.40),
            "shelf_C": (0.035, 2.40),
            "shelf_D": (0.920, 2.40),
            "shelf_E": (1.805, 2.40),
            "delivery_north": (-1.80, -2.60),
            "delivery_east": (-1.35, -2.55),
        }
        nodes = {
            name: RouteNode(name, Point2(*xy), frozenset(name.split("_")))
            for name, xy in points.items()
        }
        edges = (
            RouteEdge("start", "east_south", 0.70),
            RouteEdge("east_south", "east_north", 0.62),
            RouteEdge("east_south", "delivery_east", 0.62),
            RouteEdge("delivery_east", "delivery_north", 0.58),
            RouteEdge("delivery_north", "west_south", 0.58),
            RouteEdge("west_south", "west_north", 0.62),
            RouteEdge("east_north", "north_cross", 0.70),
            RouteEdge("west_north", "north_cross", 0.70),
            # Shelf approaches are leaves, never transit shortcuts.  The
            # outer shelves are reached from their adjacent flank; forcing
            # every goal through north_cross makes D/E (and the symmetric
            # A/B egress) double back along the shelf face.
            RouteEdge("west_north", "shelf_A", 0.58),
            RouteEdge("west_north", "shelf_B", 0.58),
            RouteEdge("north_cross", "shelf_C", 0.58),
            RouteEdge("east_north", "shelf_D", 0.58),
            RouteEdge("east_north", "shelf_E", 0.58),
        )
        return cls(nodes=nodes, edges=edges)

    def mark_failed(self, a: str, b: str, penalty: float = 50.0) -> None:
        key = tuple(sorted((a, b)))
        self.failure_penalties[key] = max(
            float(penalty), self.failure_penalties.get(key, 0.0)
        )

    def nearest_node(self, point: Point2, allowed: Iterable[str] | None = None) -> str:
        candidates = self.nodes.keys() if allowed is None else allowed
        return min(candidates, key=lambda name: self.nodes[name].point.distance(point))

    def candidate_routes(
        self,
        start: Point2,
        goal: Point2,
        footprint_width_m: float,
        blocked_edges: Mapping[tuple[str, str], float] | None = None,
        limit: int = 3,
    ) -> list[tuple[str, tuple[RouteNode, ...], float]]:
        source = self.nearest_node(start)
        target = self.nearest_node(goal)
        blocked_edges = blocked_edges or {}
        paths = self._all_simple_paths(
            source, target, footprint_width_m, blocked_edges
        )
        routes: list[tuple[str, tuple[RouteNode, ...], float]] = []
        for rank, (cost, path) in enumerate(paths[: max(1, int(limit))], start=1):
            name = f"{source}->{target}#{rank}:" + ">".join(path)
            routes.append((name, tuple(self.nodes[n] for n in path), cost))
        return routes

    def _adjacency(
        self,
        footprint_width_m: float,
        extra_penalties: Mapping[tuple[str, str], float],
    ) -> dict[str, list[tuple[str, float]]]:
        adjacency: dict[str, list[tuple[str, float]]] = {
            name: [] for name in self.nodes
        }
        for edge in self.edges:
            if edge.min_width_m + 1e-9 < footprint_width_m:
                continue
            key = tuple(sorted((edge.source, edge.target)))
            weight = (
                self.nodes[edge.source].point.distance(self.nodes[edge.target].point)
                + self.failure_penalties.get(key, 0.0)
                + float(extra_penalties.get(key, 0.0))
            )
            adjacency[edge.source].append((edge.target, weight))
            if edge.bidirectional:
                adjacency[edge.target].append((edge.source, weight))
        return adjacency

    def _all_simple_paths(
        self,
        source: str,
        target: str,
        footprint_width_m: float,
        extra_penalties: Mapping[tuple[str, str], float],
    ) -> list[tuple[float, list[str]]]:
        """Enumerate this small arena graph without returning duplicate routes."""
        adjacency = self._adjacency(footprint_width_m, extra_penalties)
        found: list[tuple[float, list[str]]] = []

        def visit(node: str, cost: float, path: list[str], seen: set[str]) -> None:
            if node == target:
                found.append((cost, list(path)))
                return
            for neighbor, edge_cost in adjacency[node]:
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                path.append(neighbor)
                visit(neighbor, cost + edge_cost, path, seen)
                path.pop()
                seen.remove(neighbor)

        visit(source, 0.0, [source], {source})
        found.sort(key=lambda item: (item[0], tuple(item[1])))
        return found

    def obstacle_edge_penalties(
        self,
        obstacle_points: Iterable[Point2],
        footprint_width_m: float,
        penalty: float = 100.0,
    ) -> dict[tuple[str, str], float]:
        """Project sensor-confirmed points onto semantic route corridors."""
        threshold = 0.5 * float(footprint_width_m) + 0.12
        penalties: dict[tuple[str, str], float] = {}
        points = tuple(obstacle_points)
        for edge in self.edges:
            a = self.nodes[edge.source].point
            b = self.nodes[edge.target].point
            if any(_point_segment_distance(point, a, b) <= threshold for point in points):
                key = tuple(sorted((edge.source, edge.target)))
                penalties[key] = float(penalty)
        return penalties

    def mark_route_failed_near(self, route: Iterable[RouteNode], point: Point2) -> None:
        nodes = tuple(route)
        if len(nodes) < 2:
            return
        a, b = min(
            zip(nodes, nodes[1:]),
            key=lambda pair: _point_segment_distance(point, pair[0].point, pair[1].point),
        )
        self.mark_failed(a.name, b.name)

    def reset_failures(self) -> None:
        self.failure_penalties.clear()

    def _shortest_path(
        self,
        source: str,
        target: str,
        footprint_width_m: float,
        extra_penalties: Mapping[tuple[str, str], float],
    ) -> tuple[list[str], float]:
        adjacency = self._adjacency(footprint_width_m, extra_penalties)
        queue = [(0.0, source)]
        distance = {source: 0.0}
        previous: dict[str, str] = {}
        while queue:
            cost, node = heapq.heappop(queue)
            if cost != distance.get(node):
                continue
            if node == target:
                break
            for neighbor, edge_cost in adjacency[node]:
                candidate = cost + edge_cost
                if candidate < distance.get(neighbor, math.inf):
                    distance[neighbor] = candidate
                    previous[neighbor] = node
                    heapq.heappush(queue, (candidate, neighbor))
        if target not in distance:
            return [], math.inf
        path = [target]
        while path[-1] != source:
            path.append(previous[path[-1]])
        path.reverse()
        return path, distance[target]


def _point_segment_distance(point: Point2, start: Point2, end: Point2) -> float:
    dx = end.x - start.x
    dy = end.y - start.y
    length_sq = dx * dx + dy * dy
    if length_sq <= 1e-12:
        return point.distance(start)
    ratio = ((point.x - start.x) * dx + (point.y - start.y) * dy) / length_sq
    ratio = max(0.0, min(1.0, ratio))
    projection = Point2(start.x + ratio * dx, start.y + ratio * dy)
    return point.distance(projection)
