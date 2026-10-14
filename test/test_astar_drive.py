import math
from types import SimpleNamespace

from my_car_astar.astar_drive import (
    astar_path,
    front_obstacle_distance,
    grid_to_world,
    normalize_angle,
    select_lookahead_index,
    world_to_grid,
)


def test_front_obstacle_distance_uses_only_forward_sector():
    scan = SimpleNamespace(
        angle_min=-math.pi,
        angle_increment=math.pi / 2,
        range_min=0.12,
        range_max=3.5,
        ranges=[0.2, 1.2, 0.3, 0.8, 0.4],
    )

    assert front_obstacle_distance(scan, 0.4) == 0.3


def test_front_obstacle_distance_ignores_obstacle_already_in_map():
    scan = SimpleNamespace(
        angle_min=-0.1,
        angle_increment=0.1,
        range_min=0.12,
        range_max=3.5,
        ranges=[math.inf, 0.3, math.inf],
    )
    map_message = SimpleNamespace(
        info=SimpleNamespace(
            origin=SimpleNamespace(position=SimpleNamespace(x=-1.0, y=-1.0)),
            resolution=1.0,
            width=3,
            height=3,
        ),
        data=[0, 0, 0, 0, 100, 0, 0, 0, 0],
    )

    assert math.isinf(front_obstacle_distance(
        scan,
        0.2,
        (0.0, 0.0, 0.0),
        map_message,
    ))


def test_front_obstacle_distance_ignores_invalid_returns():
    scan = SimpleNamespace(
        angle_min=-0.1,
        angle_increment=0.1,
        range_min=0.12,
        range_max=3.5,
        ranges=[math.nan, 0.05, math.inf],
    )

    assert math.isinf(front_obstacle_distance(scan, 0.2))


def test_astar_finds_path_on_empty_map():
    grid = [0] * 25
    path = astar_path(grid, 5, 5, (0, 0), (4, 4))

    assert path[0] == (0, 0)
    assert path[-1] == (4, 4)


def test_astar_uses_gap_in_wall():
    grid = [0] * 25
    for y in range(5):
        if y != 2:
            grid[y * 5 + 2] = 100

    path = astar_path(grid, 5, 5, (0, 0), (4, 4))

    assert (2, 2) in path
    assert path[-1] == (4, 4)


def test_astar_rejects_unknown_and_blocked_goal():
    grid = [0] * 9
    grid[8] = -1
    assert astar_path(grid, 3, 3, (0, 0), (2, 2)) == []

    grid[8] = 100
    assert astar_path(grid, 3, 3, (0, 0), (2, 2)) == []


def test_astar_does_not_cut_blocked_corner():
    grid = [0] * 9
    grid[1] = 100
    grid[3] = 100

    assert astar_path(grid, 3, 3, (0, 0), (2, 2)) == []


def test_astar_keeps_clearance_around_obstacles():
    grid = [0] * 49
    grid[3 * 7 + 3] = 100

    path = astar_path(grid, 7, 7, (1, 3), (5, 3), clearance_cells=1)

    assert path
    assert all(
        math.hypot(cell[0] - 3, cell[1] - 3) > 1
        for cell in path
    )


def test_world_grid_conversion_uses_cell_centers():
    cell = world_to_grid((1.25, -0.25), (1.0, -1.0), 0.5)
    assert cell == (0, 1)
    assert grid_to_world(cell, (1.0, -1.0), 0.5) == (1.25, -0.25)


def test_normalize_angle():
    assert math.isclose(normalize_angle(3.0 * math.pi), math.pi)
    assert math.isclose(normalize_angle(-3.0 * math.pi), -math.pi)


def test_select_lookahead_point():
    path = [(0.0, 0.0), (0.2, 0.0), (0.6, 0.0), (1.0, 0.0)]
    index = select_lookahead_index(path, (0.0, 0.0), 0.5)
    assert index == 2


def test_select_lookahead_returns_goal_when_remaining_path_is_short():
    path = [(0.0, 0.0), (0.2, 0.0)]
    index = select_lookahead_index(path, (0.1, 0.0), 0.5)
    assert index == 1
