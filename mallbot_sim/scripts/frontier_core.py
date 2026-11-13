"""
Frontier detection on a Nav2 costmap (nav_msgs/OccupancyGrid values: -1 unknown, 0..100 cost).
Pure numpy + BFS, no ROS, so it can be unit-tested offline.

  reachable  = free cells (0 <= cost < free_thresh) connected to the robot
  frontier   = unknown cell with a reachable 4-neighbour
  clusters   = 8-connected groups of frontier cells, >= min_size metres long
  goal       = the reachable cell next to the cluster that is closest to the cluster centre
"""
from collections import deque

import numpy as np


def reachable_mask(grid, start, free_thresh=50, search_radius_cells=20):
    """BFS over free cells from the robot cell. If the robot cell itself is not free (inflated
    area near a wall), start from the nearest free cell within search_radius_cells."""
    h, w = grid.shape
    free = (grid >= 0) & (grid < free_thresh)
    r0, c0 = start
    if not (0 <= r0 < h and 0 <= c0 < w):
        return np.zeros_like(free)
    if not free[r0, c0]:
        best, bd = None, None
        rs, cs = np.nonzero(free[max(0, r0 - search_radius_cells):r0 + search_radius_cells + 1,
                                 max(0, c0 - search_radius_cells):c0 + search_radius_cells + 1])
        for r, c in zip(rs + max(0, r0 - search_radius_cells), cs + max(0, c0 - search_radius_cells)):
            d = (r - r0) ** 2 + (c - c0) ** 2
            if bd is None or d < bd:
                best, bd = (r, c), d
        if best is None:
            return np.zeros_like(free)
        r0, c0 = best
    seen = np.zeros_like(free)
    seen[r0, c0] = True
    q = deque([(r0, c0)])
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and free[rr, cc] and not seen[rr, cc]:
                seen[rr, cc] = True
                q.append((rr, cc))
    return seen


def frontier_mask(grid, reach):
    unknown = grid < 0
    nb = np.zeros_like(reach)
    nb[1:, :] |= reach[:-1, :]
    nb[:-1, :] |= reach[1:, :]
    nb[:, 1:] |= reach[:, :-1]
    nb[:, :-1] |= reach[:, 1:]
    return unknown & nb


def clusters(fmask):
    """8-connected components of frontier cells -> list of [(r, c), ...]."""
    h, w = fmask.shape
    seen = np.zeros_like(fmask)
    out = []
    for r0, c0 in zip(*np.nonzero(fmask)):
        if seen[r0, c0]:
            continue
        comp, q = [], deque([(r0, c0)])
        seen[r0, c0] = True
        while q:
            r, c = q.popleft()
            comp.append((r, c))
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w and fmask[rr, cc] and not seen[rr, cc]:
                        seen[rr, cc] = True
                        q.append((rr, cc))
        out.append(comp)
    return out


def find_frontiers(grid, robot_rc, resolution, free_thresh=50, min_size=0.5):
    """Returns a list of dicts {goal_rc, centroid_rc, size_m, n_cells}, unsorted."""
    reach = reachable_mask(grid, robot_rc, free_thresh)
    if reach.sum() < 10:
        return [], reach
    fr = []
    for comp in clusters(frontier_mask(grid, reach)):
        size_m = len(comp) * resolution
        if size_m < min_size:
            continue
        cr = sum(p[0] for p in comp) / len(comp)
        cc = sum(p[1] for p in comp) / len(comp)
        # reachable neighbours of the cluster -> the one closest to the centroid is the goal
        best, bd = None, None
        h, w = grid.shape
        for r, c in comp:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, c2 = r + dr, c + dc
                if 0 <= rr < h and 0 <= c2 < w and reach[rr, c2]:
                    d = (rr - cr) ** 2 + (c2 - cc) ** 2
                    if bd is None or d < bd:
                        best, bd = (rr, c2), d
        if best is not None:
            fr.append({"goal_rc": best, "centroid_rc": (cr, cc), "size_m": size_m, "n_cells": len(comp)})
    return fr, reach
