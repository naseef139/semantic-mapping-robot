#!/usr/bin/env python3
"""Offline checks for frontier_core (no ROS).  python3 test/test_frontier_core.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
import frontier_core as fc  # noqa: E402

RES = 0.05


def corridor_with_pocket():
    """60x60 grid: free (0) everywhere, unknown (-1) in the right 10 columns (= frontier),
    a 5x5 free pocket at the robot ringed by inflation (70, drivable), walls (100) at the border."""
    g = np.zeros((60, 60), dtype=np.int16)
    g[:, 50:] = -1
    g[0, :] = g[-1, :] = g[:, 0] = 100
    g[8:19, 8:19] = 70          # inflation ring around the pocket
    g[11:16, 11:16] = 0         # 5x5 pocket = 25 free cells
    return g, (13, 13)


def test_pocket_is_not_a_dead_end():
    g, robot = corridor_with_pocket()
    strict = fc._bfs(g, robot, 50, 20)
    assert strict.sum() == 25, strict.sum()                 # old behaviour: pocket only
    reach = fc.reachable_mask(g, robot, 50)
    assert reach.sum() > 1000, reach.sum()                  # new: reaches the open area
    fr, _ = fc.find_frontiers(g, robot, RES, 50, 0.5, 0.3, 60)
    assert len(fr) >= 1, fr                                 # and finds the frontier


def test_open_area_unchanged():
    g, _ = corridor_with_pocket()
    robot = (30, 30)                                        # in the big free area
    assert (fc.reachable_mask(g, robot, 50) == fc._bfs(g, robot, 50, 20)).all()


def test_lethal_still_blocks():
    g, robot = corridor_with_pocket()
    g[8:19, 8:19] = 100                                     # ring is a real wall now
    g[11:16, 11:16] = 0
    assert fc.reachable_mask(g, robot, 50).sum() == 25      # relaxed search must not cross walls
    g[8:19, 8:19] = 99                                      # inscribed: not drivable either
    g[11:16, 11:16] = 0
    assert fc.reachable_mask(g, robot, 50).sum() == 25


if __name__ == '__main__':
    for name, f in list(globals().items()):
        if name.startswith('test_'):
            f()
            print('ok ', name)
