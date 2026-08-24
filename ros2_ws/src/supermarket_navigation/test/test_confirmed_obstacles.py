from supermarket_navigation.confirmed_obstacles import (
    ConfirmedObstacleGrid,
    StaticOccupancyMask,
)


def test_three_hits_inside_one_second_confirm_cell():
    grid = ConfirmedObstacleGrid(hits_required=3, hit_window_s=1.0)
    assert not grid.observe_hit(1.01, -0.01, 10.0)
    assert not grid.observe_hit(1.02, -0.02, 10.4)
    assert grid.observe_hit(1.03, -0.03, 10.8)
    assert grid.is_confirmed(1.04, -0.04)


def test_hit_window_expiry_restarts_evidence():
    grid = ConfirmedObstacleGrid(hits_required=3, hit_window_s=1.0)
    grid.observe_hit(0.0, 0.0, 0.0)
    grid.observe_hit(0.0, 0.0, 0.5)
    assert not grid.observe_hit(0.0, 0.0, 1.1)
    assert not grid.is_confirmed(0.0, 0.0)


def test_multiple_laser_beams_in_one_frame_count_as_one_hit():
    grid = ConfirmedObstacleGrid(hits_required=3, hit_window_s=1.0)
    for _ in range(20):
        assert not grid.observe_hit(0.0, 0.0, 0.0, observation_id="scan-1")
    assert not grid.observe_hit(0.0, 0.0, 0.2, observation_id="scan-2")
    assert grid.observe_hit(0.0, 0.0, 0.4, observation_id="scan-3")


def test_only_spatial_clusters_are_exported_to_topology():
    grid = ConfirmedObstacleGrid(hits_required=1, resolution_m=0.10)
    grid.observe_hit(0.01, 0.01, 0.0)
    assert grid.clustered_confirmed_points(minimum_cells=2) == []
    grid.observe_hit(0.11, 0.01, 0.1)
    assert len(grid.clustered_confirmed_points(minimum_cells=2)) == 2


def test_static_occupancy_mask_filters_known_map_geometry_only():
    mask = StaticOccupancyMask(
        width=3,
        height=2,
        resolution_m=1.0,
        origin_x=-1.0,
        origin_y=-1.0,
        origin_yaw=0.0,
        data=(0, 100, 0, 0, 0, 0),
    )
    assert mask.is_occupied(0.5, -0.5)
    assert not mask.is_occupied(-0.5, 0.5)


def test_five_explicit_free_rays_clear_confirmed_cell():
    grid = ConfirmedObstacleGrid(hits_required=3, free_rays_to_clear=5)
    for now in (0.0, 0.1, 0.2):
        grid.observe_hit(0.2, 0.2, now)
    for _ in range(4):
        assert not grid.observe_free(0.2, 0.2)
    assert grid.is_confirmed(0.2, 0.2)
    assert grid.observe_free(0.2, 0.2)
    assert not grid.is_confirmed(0.2, 0.2)


def test_run_prefix_change_atomically_clears_evidence():
    grid = ConfirmedObstacleGrid(hits_required=1)
    grid.reset_for_run("run_a")
    grid.observe_hit(1.0, 1.0, 0.0)
    assert grid.confirmed_points()
    assert not grid.reset_for_run("run_a")
    assert grid.confirmed_points()
    assert grid.reset_for_run("run_b")
    assert grid.confirmed_points() == []
