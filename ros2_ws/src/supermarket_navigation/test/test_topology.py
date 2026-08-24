from supermarket_navigation.topology import ArenaTopology, Point2
import random


START = Point2(1.92, -3.17)
SHELF_A = Point2(-1.735, 2.40)
SHELF_E = Point2(1.805, 2.40)


def route_names(route):
    return tuple(node.name for node in route[1])


def test_candidates_are_genuinely_distinct_paths():
    topology = ArenaTopology.competition_default()
    routes = topology.candidate_routes(START, SHELF_A, 0.48, limit=3)
    names = [route_names(route) for route in routes]
    assert len(routes) == 2
    assert len(names) == len(set(names))
    assert "east_north" in names[0]
    assert "west_north" in names[1]


def test_outer_shelf_uses_adjacent_flank_without_central_double_back():
    topology = ArenaTopology.competition_default()
    east_route = topology.candidate_routes(START, SHELF_E, 0.48, limit=1)[0]
    assert route_names(east_route) == (
        "start", "east_south", "east_north", "shelf_E"
    )
    assert "north_cross" not in route_names(east_route)


def test_shelf_nodes_are_leaves_not_route_cycle_connectors():
    topology = ArenaTopology.competition_default()
    adjacency = topology._adjacency(0.48, {})
    for shelf in ("shelf_A", "shelf_B", "shelf_C", "shelf_D", "shelf_E"):
        assert len(adjacency[shelf]) == 1


def test_confirmed_obstacle_changes_first_homotopy():
    topology = ArenaTopology.competition_default()
    penalties = topology.obstacle_edge_penalties([Point2(1.55, 0.0)], 0.48)
    routes = topology.candidate_routes(
        START, SHELF_A, 0.48, blocked_edges=penalties, limit=3
    )
    assert "west_north" in route_names(routes[0])


def test_extended_manipulator_profile_is_not_navigable():
    topology = ArenaTopology.competition_default()
    assert topology.candidate_routes(START, SHELF_A, 1.10, limit=3) == []


def test_recent_failed_edge_penalty_changes_next_route():
    topology = ArenaTopology.competition_default()
    first = topology.candidate_routes(START, SHELF_A, 0.48, limit=2)[0]
    topology.mark_route_failed_near(first[1], Point2(1.55, 0.0))
    next_first = topology.candidate_routes(START, SHELF_A, 0.48, limit=2)[0]
    assert "west_north" in route_names(next_first)
    topology.reset_failures()
    reset_first = topology.candidate_routes(START, SHELF_A, 0.48, limit=2)[0]
    assert "east_north" in route_names(reset_first)


def test_500_random_obstacle_sets_return_only_simple_bounded_routes():
    for seed in range(500):
        rng = random.Random(seed)
        topology = ArenaTopology.competition_default()
        obstacles = [
            Point2(rng.uniform(-2.2, 2.2), rng.uniform(-3.3, 2.8))
            for _ in range(5)
        ]
        penalties = topology.obstacle_edge_penalties(obstacles, 0.48)
        routes = topology.candidate_routes(
            START, SHELF_A, 0.48, blocked_edges=penalties, limit=3
        )
        assert 1 <= len(routes) <= 3
        for _name, nodes, _cost in routes:
            names = [node.name for node in nodes]
            assert len(names) == len(set(names))
