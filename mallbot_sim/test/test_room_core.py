#!/usr/bin/env python3
"""Offline test for room_core: a simulated mall explored with a 2D LiDAR.

The robot drives along the corridor, a ray-cast LiDAR (360 rays, 12 m) builds the
map step by step, and room_core.detect() runs every metre, as the node would.

Checks
  1. safety:   no corridor cell and no cell near the driven path is ever marked
  2. coverage: every doorway is closed (no passable route through it) whenever the
               robot is within 1.5 m of it; also reported: when it was first closed,
               how far away the robot was then, and updates in which it re-opened
Usage: python3 test_room_core.py [--plot out.png] [--no-signs]
"""
import argparse
import math
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts'))
import room_core as rc  # noqa: E402

RES = 0.05
OX, OY = -12.5, -7.5
W, H = int(25 / RES), int(15 / RES)
CORR = 2.25                    # corridor half-width (4.5 m wide)
SHOP_X = [-12.5, -7.5, -2.5, 2.5, 7.5, 12.5]
# doorway width per shop, north row then south row; 4.0 and 3.4 are open shop fronts
NORTH_DOORS = [1.2, 1.8, 2.2, 4.0, 1.5]
SOUTH_DOORS = [2.0, 1.0, 2.4, 1.6, 3.4]


def cell(x, y):
    return int((x - OX) / RES), int((y - OY) / RES)


def build_truth():
    occ = np.zeros((H, W), np.uint8)

    def box(x0, y0, x1, y1):
        c0, r0 = cell(min(x0, x1), min(y0, y1))
        c1, r1 = cell(max(x0, x1), max(y0, y1))
        occ[max(r0, 0):min(r1 + 1, H), max(c0, 0):min(c1 + 1, W)] = 1

    def disc(x, y, r):
        cv2.circle(occ, cell(x, y), int(r / RES), 1, -1)

    t = 0.15
    box(-12.5, -7.5, 12.5, -7.5 + t)          # outer walls
    box(-12.5, 7.5 - t, 12.5, 7.5)
    box(-12.5, -7.5, -12.5 + t, 7.5)
    box(12.5 - t, -7.5, 12.5, 7.5)
    doors, signs, shops = [], [], []
    for side, widths in ((1, NORTH_DOORS), (-1, SOUTH_DOORS)):
        yf = side * CORR
        for i in range(5):
            x0, x1 = SHOP_X[i], SHOP_X[i + 1]
            xc, w = (x0 + x1) / 2, widths[i]
            box(x0, yf, xc - w / 2, yf + side * t)          # shop front, left of door
            box(xc + w / 2, yf, x1, yf + side * t)          # right of door
            if i > 0:
                box(x0 - t / 2, yf, x0 + t / 2, side * 7.5)  # partition wall
            doors.append(((xc - w / 2, yf), (xc + w / 2, yf), w))
            signs.append((xc + np.random.uniform(-0.3, 0.3), yf + side * np.random.uniform(-0.3, 0.3)))
            shops.append((x0 + t, min(yf, side * 7.5) + (t if side < 0 else t),
                          x1 - t, max(yf, side * 7.5) - t, side, w))
            # furniture inside the shop
            box(x0 + 0.6, side * 6.6, x0 + 1.6, side * 7.2)
            disc(xc + 1.2, side * 4.6, 0.35)
    # corridor clutter: central bench, bench near a wall, plant, bin, two standing people
    box(-5.8, -0.25, -4.2, 0.25)
    box(5.2, -1.85, 6.8, -1.35)
    disc(0.0, 1.7, 0.3)
    disc(8.5, 0.6, 0.25)
    disc(-2.0, 1.3, 0.25)
    disc(3.0, -1.9, 0.25)
    return occ, doors, signs, shops


NOISE = {'on': False}
COUNTS = {}


def scan(truth, grid, x, y, n=360, rng=12.0):
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    steps = np.arange(0.1, rng, RES / 2)
    if NOISE['on']:   # SLAM pose error of a few cm / half a degree, applied to the whole scan
        ang = ang + np.random.normal(0, math.radians(0.5))
        x, y = x + np.random.normal(0, 0.03), y + np.random.normal(0, 0.03)
    xs = x + np.outer(np.cos(ang), steps)
    ys = y + np.outer(np.sin(ang), steps)
    cx = ((xs - OX) / RES).astype(int)
    cy = ((ys - OY) / RES).astype(int)
    inside = (cx >= 0) & (cx < W) & (cy >= 0) & (cy < H)
    cx = np.clip(cx, 0, W - 1)
    cy = np.clip(cy, 0, H - 1)
    hit = truth[cy, cx].astype(bool) & inside
    first = np.where(hit.any(1), hit.argmax(1), len(steps))
    if NOISE['on']:   # range noise (sigma 3 cm) and 5 % dropped returns
        first = np.clip(first + np.round(np.random.normal(0, 0.03 / (RES / 2), n)).astype(int), 0, len(steps))
        first[np.random.rand(n) < 0.05] = len(steps) + 1
    hits, misses = COUNTS['hit'], COUNTS['miss']
    for k in range(n):
        if first[k] > len(steps):
            continue
        f = first[k]
        misses[cy[k, :f], cx[k, :f]] += 1
        if f < len(steps):
            hits[cy[k, f], cx[k, f]] += 1
    # occupancy from hit/miss counts, as a SLAM back end (log-odds) would do
    tot = hits + misses
    ratio = np.divide(hits, tot, out=np.zeros_like(hits, dtype=float), where=tot > 0)
    grid[:] = -1
    grid[(tot > 0) & (ratio <= 0.25)] = 0
    grid[(hits >= 1) & (ratio > 0.25)] = 100


def path_points(glance=False):
    wp = [(-10, 0), (-6.5, -1.2), (-3.5, -1.2), (10, 0), (10, -0.9), (-11, -0.9), (-11, 0.9), (-10, 0)]
    if glance:
        # an explorer that walks up to every doorway mouth (0.9 m from the shop front),
        # the worst case: the robot is close to a door before it is detected.
        # Travels at y = -0.8, clear of the benches, bin, plant and people.
        yn, ys, yc = CORR - 0.9, -CORR + 0.9, -0.8
        wp = [(-10, 0), (-10, yc)]
        for xc in (-10, -5, 0, 5, 10):
            if xc == -5:   # the central bench sits at x -5.8..-4.2: go round its east end
                wp += [(-3.9, yc), (-3.9, yn), (xc, yn), (-3.9, yn), (-3.9, yc), (xc, ys), (-3.9, yc)]
            else:
                wp += [(xc, yc), (xc, yn), (xc, yc), (xc, ys), (xc, yc)]
        wp += [(-10, yc), (-10, 0)]
    pts = []
    for (x0, y0), (x1, y1) in zip(wp[:-1], wp[1:]):
        n = max(1, int(math.hypot(x1 - x0, y1 - y0) / 0.25))
        for i in range(n):
            pts.append((x0 + (x1 - x0) * i / n, y0 + (y1 - y0) * i / n))
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plot', default='')
    ap.add_argument('--no-signs', action='store_true')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--noise', action='store_true', help='LiDAR range noise, dropouts and SLAM pose error')
    ap.add_argument('--glance', action='store_true', help='drive up to every doorway mouth')
    a = ap.parse_args()
    NOISE['on'] = a.noise
    np.random.seed(a.seed)

    truth, doors, signs_true, shops = build_truth()
    grid = -np.ones((H, W), np.int16)
    COUNTS['hit'] = np.zeros((H, W), np.int32)
    COUNTS['miss'] = np.zeros((H, W), np.int32)
    g = rc.GridGeom(W, H, RES, OX, OY)
    P = rc.Params()
    pts = path_points(a.glance)

    # corridor interior: free corridor cells at least 0.35 m from any wall or object
    # (a virtual wall drawn exactly on a shop front line is not a corridor blockage)
    corridor = np.zeros((H, W), bool)
    c0, r0 = cell(-12.0, -CORR + 0.35)
    c1, r1 = cell(12.0, CORR - 0.35)
    corridor[r0:r1, c0:c1] = True
    near_obj = cv2.dilate(truth, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))).astype(bool)
    corridor &= ~near_obj
    path_mask = np.zeros((H, W), np.uint8)
    for x, y in pts:
        cv2.circle(path_mask, cell(x, y), int(0.25 / RES), 1, -1)

    memory, visited, seen_signs = [], [], []
    worst_corridor = 0
    worst_path = 0
    first_closed = {}
    reopened = {}
    near_open = {}
    travelled, last = 0.0, pts[0]
    lethal = np.zeros((H, W), np.uint8)
    marked = []
    times = []
    import time
    for k, (x, y) in enumerate(pts):
        scan(truth, grid, x, y)
        travelled += math.hypot(x - last[0], y - last[1])
        last = (x, y)
        visited.append((x, y))
        if not a.no_signs:
            for s in signs_true:
                if s not in seen_signs and math.hypot(s[0] - x, s[1] - y) < 6.0:
                    seen_signs.append(s)
        if k % 2 and k != len(pts) - 1:      # every 0.5 m (the node runs at 1 Hz, robot <= 0.34 m/s)
            continue
        t0 = time.time()
        lethal, marked, memory = rc.detect(grid, g, (x, y), pts[0], visited, seen_signs, P, memory)
        times.append(time.time() - t0)
        worst_corridor = max(worst_corridor, int((lethal.astype(bool) & corridor).sum()))
        worst_path = max(worst_path, int((lethal.astype(bool) & path_mask.astype(bool)).sum()))
        # a doorway is CLOSED when no 4-connected path of non-keepout, non-wall cells
        # crosses its opening from the corridor side to the shop side (a 5 cm leak counts);
        # once closed it must stay closed in every later update
        lb = lethal.astype(bool)
        for i, (p, q, w) in enumerate(doors):
            yw = p[1]
            c0_, r0_ = cell(p[0] - 0.3, min(yw - 0.6, yw + 0.6))
            c1_, r1_ = cell(q[0] + 0.3, max(yw - 0.6, yw + 0.6))
            win = ~(lb[r0_:r1_, c0_:c1_] | truth[r0_:r1_, c0_:c1_].astype(bool))
            _, lab = cv2.connectedComponents(win.astype(np.uint8), connectivity=4)
            top = set(lab[0][lab[0] > 0].tolist())
            bot = set(lab[-1][lab[-1] > 0].tolist())
            closed = not (top & bot)
            # the safety-critical event: robot within 1.5 m of a doorway that is still open
            if not closed and math.hypot(x - (p[0] + q[0]) / 2, y - yw) < 1.5:
                near_open[i] = near_open.get(i, 0) + 1
            if i in first_closed:
                if not closed:
                    reopened[i] = reopened.get(i, 0) + 1
            elif closed:
                kind = next((d.kind for d in marked
                             if math.hypot(d.mid[0] - (p[0] + q[0]) / 2, d.mid[1] - yw) < 1.5), '?')
                dist = math.hypot(x - (p[0] + q[0]) / 2, y - p[1])
                first_closed[i] = (round(travelled, 1), kind, w, round(dist, 2))

    print(f"updates: {len(times)}, mean {1000 * np.mean(times):.0f} ms, max {1000 * max(times):.0f} ms")
    print(f"corridor cells ever marked: {worst_corridor}   ({worst_corridor * RES * RES:.2f} m^2)")
    print(f"path cells ever marked:     {worst_path}")
    # closest the robot came to each doorway BEFORE it was closed
    print("shop  side  door_w  closed@m  kind   interior_seen  interior_marked  robot_dist_at_close  reopened  open_with_robot_near")
    ok = True
    for i, (x0, y0, x1, y1, side, w) in enumerate(shops):
        c0_, r0_ = cell(x0, min(y0, y1))
        c1_, r1_ = cell(x1, max(y0, y1))
        inter = np.zeros((H, W), bool)
        inter[r0_:r1_, c0_:c1_] = True
        inter &= truth == 0
        seen = inter & (grid == 0)
        mk = seen & lethal.astype(bool)
        frac = mk.sum() / max(1, seen.sum())
        st = first_closed.get(i)
        name = ('N' if side > 0 else 'S') + str(i % 5 + 1)
        print(f"{name:5s} {'N' if side > 0 else 'S':4s}  {w:5.1f}   {str(st[0]) if st else '-':>7s}   "
              f"{st[1] if st else '-':5s}  {seen.sum() * RES * RES:8.1f} m2   {100 * frac:5.1f} %"
              f"          {st[3] if st else '-'} m           {reopened.get(i, 0):3d}       {near_open.get(i, 0)}")
        if not st or near_open.get(i, 0) > 0:
            ok = False
    safe = worst_corridor == 0 and worst_path == 0
    print("SAFETY:", "PASS" if safe else "FAIL", "  COVERAGE:", "PASS" if ok else "FAIL")

    if a.plot:
        img = np.full((H, W, 3), 128, np.uint8)
        img[grid == 0] = (255, 255, 255)
        img[grid == 100] = (0, 0, 0)
        red = lethal.astype(bool)
        img[red & (grid == 0)] = (120, 120, 255)
        img[red & (grid < 0)] = (90, 90, 200)
        for d in marked:
            cv2.line(img, cell(*d.p), cell(*d.q), (0, 0, 220) if d.kind == 'door' else (200, 0, 200), 3)
        for (x, y) in pts:
            cv2.circle(img, cell(x, y), 1, (0, 160, 0), -1)
        for s in seen_signs:
            cv2.drawMarker(img, cell(*s), (255, 120, 0), cv2.MARKER_STAR, 14, 2)
        img = cv2.flip(img, 0)
        img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(a.plot, img)
    return 0 if (safe and ok) else 1


if __name__ == '__main__':
    sys.exit(main())
