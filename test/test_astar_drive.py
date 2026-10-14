import math

from my_car_astar.astar_drive import (
    astar_path,
    grid_to_world,
    normalize_angle,
    select_lookahead_index,
    world_to_grid,
)


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
