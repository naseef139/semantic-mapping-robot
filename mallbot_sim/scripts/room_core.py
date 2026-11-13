#!/usr/bin/env python3
"""room_core.py - find doorways and the rooms/shops behind them in a SLAM map.

No ROS imports here, so the logic can be tested offline (see test_room_core.py).

Idea
----
A doorway is a gap between two *collinear* wall segments (the two jambs of one
wall). A corridor's two side walls are parallel but far apart, so they are never
paired; a bench in the corridor is not on the wall line, so it is never paired
with a wall either.

For every candidate doorway we draw a "virtual wall" across the gap, so the
robot can never drive through it. Then we follow the known free space behind it
(away from the robot). If that space is sealed off from the robot, its start
pose and everywhere it has already driven, it is a room and is filled in as
keepout too (up to `room_depth` metres from the doorway). Unknown cells never
count as a connection, so an unexplored room stays closed by its virtual wall.

Candidates must be holes in a long straight wall: solid, re-fitted wall
segments, collinear jambs, open gap, and wall on both sides of the gap. This is
what keeps benches, bins and people standing near a wall from producing lines.

Wide-open shop fronts (wider than `door_max`) are accepted only when a shop
*sign* detected by the camera lies on the gap. This is the semantic cue.

Grid convention (nav_msgs/OccupancyGrid): -1 unknown, 0..100 occupancy.
Index [row, col] = [y, x], world = origin + (col + 0.5, row + 0.5) * res.
"""
import math
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Params:
    occ_thresh: int = 65          # cell >= this is occupied
    free_thresh: int = 25         # 0 <= cell <= this is free
    door_min: float = 0.6         # m, narrower gaps are ignored (LiDAR holes)
    door_max: float = 2.5         # m, widest doorway accepted without a sign
    sign_door_max: float = 8.0    # m, widest shop front accepted when a sign is on it
    sign_on_line: float = 0.9     # m, max distance of a sign from the door line
    wide_jamb_min: float = 0.6    # m, both walls either side of a wide shop front
    min_segment: float = 0.5      # m, shortest wall segment used as a jamb
    min_solidity: float = 0.9     # fraction of a segment that must be occupied cells
    min_support: float = 1.5      # m of straight wall that must lie on the door line
    min_side_support: float = 0.6 # m of that wall needed on EACH side of the gap
    collinear_deg: float = 5.0    # deg, max angle between the two jamb segments
    collinear_off: float = 0.2    # m, max sideways offset between the jamb segments
    gap_max_occ: float = 0.1      # max fraction of occupied cells along the gap
    room_depth: float = 8.0       # m, how far behind a doorway the room is flooded
    seed_back: float = 0.35       # m, flood seeds start this far behind the line
    sliver_depth: float = 0.6     # m, wall closer than this behind the gap = sliver, reject
    protect_radius: float = 0.6   # m, never mark cells this close to the robot / start
    visited_radius: float = 0.3   # m, trajectory cells counted as public area
    wall_thickness: int = 2       # cells, thickness of the drawn virtual wall
    forget_after: int = 10        # updates a doorway is kept after it was last detected
    confirm_hits: int = 2         # detections needed before an unsealed doorway is closed
    keep_hits: int = 2            # detections after which a doorway is never forgotten


@dataclass
class Doorway:
    p: tuple                      # (x, y) jamb 1, metres
    q: tuple                      # (x, y) jamb 2
    width: float
    kind: str                     # 'door' (geometric) or 'sign' (wide shop front)
    hits: int = 1                 # times re-detected
    misses: int = 0               # consecutive updates in which it was not re-detected
    separates: bool = False       # True when the room behind it is sealed off and filled
    region_cells: int = 0
    region: object = field(default=None, repr=False)

    @property
    def mid(self):
        return ((self.p[0] + self.q[0]) / 2.0, (self.p[1] + self.q[1]) / 2.0)


class GridGeom:
    def __init__(self, width, height, res, ox, oy):
        self.w, self.h, self.res, self.ox, self.oy = width, height, res, ox, oy

    def to_cell(self, x, y):
        return int(math.floor((x - self.ox) / self.res)), int(math.floor((y - self.oy) / self.res))

    def to_world(self, cx, cy):
        return self.ox + (cx + 0.5) * self.res, self.oy + (cy + 0.5) * self.res

    def inside(self, cx, cy):
        return 0 <= cx < self.w and 0 <= cy < self.h


# ----------------------------------------------------------------- segments
def extract_segments(occ, g, P):
    """Wall line segments from the occupied mask with a probabilistic Hough transform."""
    cv2.setRNGSeed(12345)   # HoughLinesP samples randomly; fixed seed = repeatable output
    img = (occ.astype(np.uint8) * 255)
    img = cv2.dilate(img, np.ones((3, 3), np.uint8))
    lines = cv2.HoughLinesP(img, rho=1, theta=np.pi / 360,
                            threshold=max(5, int(0.3 / g.res)),
                            minLineLength=max(3, int(P.min_segment / g.res)),
                            maxLineGap=max(1, int(0.2 / g.res)))
    segs = []
    if lines is None:
        return segs
    solid = img > 0
    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
        # Hough also "finds" lines through scattered points (furniture, people, shadow
        # edges). A real wall is a solid run of occupied cells, so keep only those.
        n = max(2, int(math.hypot(x2 - x1, y2 - y1)))
        xs = np.linspace(x1, x2, n).round().astype(int)
        ys = np.linspace(y1, y2, n).round().astype(int)
        if solid[ys, xs].mean() < P.min_solidity:
            continue
        seg = _refine(occ, g, (x1, y1), (x2, y2))
        if seg is not None:
            segs.append(seg)
    return segs


def _refine(occ, g, c1, c2, band=2):
    """Re-fit a Hough segment to the occupied cells around it (least squares / PCA).

    Hough angles on a wall that is several cells thick can be off by ~10 degrees; over
    a few metres that tilt would wrongly line a wall up with a bench or a person."""
    x0, x1 = sorted((c1[0], c2[0]))
    y0, y1 = sorted((c1[1], c2[1]))
    x0, y0 = max(0, x0 - band), max(0, y0 - band)
    x1, y1 = min(g.w - 1, x1 + band), min(g.h - 1, y1 + band)
    win = np.zeros((y1 - y0 + 1, x1 - x0 + 1), np.uint8)
    cv2.line(win, (c1[0] - x0, c1[1] - y0), (c2[0] - x0, c2[1] - y0), 1, thickness=2 * band + 1)
    ys, xs = np.nonzero(win & occ[y0:y1 + 1, x0:x1 + 1])
    if len(xs) < 4:
        return None
    pts = np.stack([xs + x0, ys + y0], 1).astype(np.float64)
    for _ in range(3):
        # fit, then drop cells far from the line (a person or bin standing against the
        # wall) and fit again, so the line follows the wall face only
        c = pts.mean(0)
        _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
        d = vt[0]
        r = np.abs((pts - c) @ np.array([-d[1], d[0]]))
        keep = r <= 1.5
        if keep.all() or keep.sum() < 4:
            break
        pts = pts[keep]
    c = pts.mean(0)
    t = (pts - c) @ d
    # trim to the run that is actually continuous (no gap longer than 0.2 m along the line)
    ts = np.sort(t)
    gaps = np.where(np.diff(ts) > max(2.0, 0.2 / g.res))[0]
    if len(gaps):
        bounds = np.concatenate([[0], gaps + 1, [len(ts)]])
        k = int(np.argmax(np.diff(bounds)))
        lo, hi = ts[bounds[k]], ts[bounds[k + 1] - 1]
        t = t[(t >= lo) & (t <= hi)]
    a = c + d * t.min()
    b = c + d * t.max()
    return g.to_world(a[0], a[1]), g.to_world(b[0], b[1])


def _unit(a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    n = math.hypot(dx, dy)
    return (dx / n, dy / n) if n > 1e-9 else (1.0, 0.0)


def _sample_line(g, p, q, step=None):
    step = step or g.res * 0.5
    n = max(2, int(math.hypot(q[0] - p[0], q[1] - p[1]) / step))
    xs = np.linspace(p[0], q[0], n)
    ys = np.linspace(p[1], q[1], n)
    cx = np.floor((xs - g.ox) / g.res).astype(int)
    cy = np.floor((ys - g.oy) / g.res).astype(int)
    ok = (cx >= 0) & (cx < g.w) & (cy >= 0) & (cy < g.h)
    return cx[ok], cy[ok]


def candidate_gaps(segs, occ, g, P, signs):
    """Pairs of collinear segments with a door-sized gap between facing ends."""
    cands = []
    cos_tol = math.cos(math.radians(P.collinear_deg))
    k = 2 * max(1, int(round(0.15 / g.res))) + 1
    occ_band = cv2.dilate(occ.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)
    n = len(segs)
    dirs = [_unit(a, b) for a, b in segs]
    for i in range(n):
        a0, a1 = segs[i]
        da = dirs[i]
        for j in range(i + 1, n):
            db = dirs[j]
            if abs(da[0] * db[0] + da[1] * db[1]) < cos_tol:
                continue
            b0, b1 = segs[j]
            # sideways offset of b's endpoints from a's infinite line
            nx, ny = -da[1], da[0]
            off0 = (b0[0] - a0[0]) * nx + (b0[1] - a0[1]) * ny
            off1 = (b1[0] - a0[0]) * nx + (b1[1] - a0[1]) * ny
            if abs(off0) > P.collinear_off or abs(off1) > P.collinear_off:
                continue
            # positions along the line
            ta = sorted([0.0, (a1[0] - a0[0]) * da[0] + (a1[1] - a0[1]) * da[1]])
            tb0 = (b0[0] - a0[0]) * da[0] + (b0[1] - a0[1]) * da[1]
            tb1 = (b1[0] - a0[0]) * da[0] + (b1[1] - a0[1]) * da[1]
            tb = sorted([tb0, tb1])
            if tb[0] >= ta[1]:
                t_p, t_q = ta[1], tb[0]
            elif ta[0] >= tb[1]:
                t_p, t_q = tb[1], ta[0]
            else:
                continue                       # overlapping: same wall, no gap
            width = t_q - t_p
            if width < P.door_min:
                continue
            mo = (off0 + off1) / 4.0           # put the line midway between the two segments
            p = (a0[0] + da[0] * t_p + nx * mo, a0[1] + da[1] * t_p + ny * mo)
            q = (a0[0] + da[0] * t_q + nx * mo, a0[1] + da[1] * t_q + ny * mo)
            kind = 'door'
            if width > P.door_max:
                if width > P.sign_door_max or not _sign_on_gap(p, q, signs, P):
                    continue
                # a wide shop front must sit between two real stretches of wall
                la = math.hypot(a1[0] - a0[0], a1[1] - a0[1])
                lb = math.hypot(b1[0] - b0[0], b1[1] - b0[1])
                if min(la, lb) < P.wide_jamb_min:
                    continue
                kind = 'sign'
            # the gap itself must be open: no wall piece in it (checked in a +-0.15 m band,
            # so a line that runs along a wall face across several doors is rejected)
            f = min(0.45, 0.15 / width)
            cx, cy = _sample_line(g, _lerp(p, q, f), _lerp(p, q, 1 - f))
            if len(cx) == 0 or occ_band[cy, cx].mean() > P.gap_max_occ:
                continue
            cands.append(Doorway(p=p, q=q, width=width, kind=kind))
    out = []
    for c in _dedupe(cands):
        left, right = _wall_support(c, segs, P)
        # a doorway is a hole in a wall: there must be wall on BOTH sides of the gap,
        # on the same line. A person or a bin next to a wall gives only a short stub.
        if min(left, right) >= P.min_side_support and left + right >= P.min_support:
            out.append(c)
    return out


def _wall_support(d, segs, P, reach=6.0):
    """Metres of solid wall on the door's own line, (before p, after q), within `reach`.

    A real doorway is a hole in a long straight wall. Lines that merely join a person,
    a bin or a bench to a wall corner have little or no wall on one side."""
    u = _unit(d.p, d.q)
    nx, ny = -u[1], u[0]
    L = math.hypot(d.q[0] - d.p[0], d.q[1] - d.p[1])
    cos_tol = math.cos(math.radians(P.collinear_deg))
    left, right = [], []
    for a, b in segs:
        s = _unit(a, b)
        if abs(s[0] * u[0] + s[1] * u[1]) < cos_tol:
            continue
        oa = (a[0] - d.p[0]) * nx + (a[1] - d.p[1]) * ny
        ob = (b[0] - d.p[0]) * nx + (b[1] - d.p[1]) * ny
        if abs(oa) > P.collinear_off or abs(ob) > P.collinear_off:
            continue
        ta = (a[0] - d.p[0]) * u[0] + (a[1] - d.p[1]) * u[1]
        tb = (b[0] - d.p[0]) * u[0] + (b[1] - d.p[1]) * u[1]
        t0, t1 = min(ta, tb), max(ta, tb)
        if t1 > -reach and t0 < 0.15:
            left.append((max(t0, -reach), min(t1, 0.0)))
        if t0 < L + reach and t1 > L - 0.15:
            right.append((max(t0, L), min(t1, L + reach)))
    return _union_len(left), _union_len(right)


def _union_len(iv):
    iv = sorted((a, b) for a, b in iv if b > a)
    total, c0, c1 = 0.0, None, None
    for a, b in iv:
        if c1 is None or a > c1:
            if c1 is not None:
                total += c1 - c0
            c0, c1 = a, b
        else:
            c1 = max(c1, b)
    if c1 is not None:
        total += c1 - c0
    return total


def _lerp(p, q, t):
    return (p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t)


def _sign_on_gap(p, q, signs, P):
    d = _unit(p, q)
    L = math.hypot(q[0] - p[0], q[1] - p[1])
    for sx, sy in signs:
        t = (sx - p[0]) * d[0] + (sy - p[1]) * d[1]
        off = abs((sx - p[0]) * -d[1] + (sy - p[1]) * d[0])
        if -0.3 <= t <= L + 0.3 and off <= P.sign_on_line:
            return True
    return False


def _jambs_present(d, occ, g, probe=0.4):
    """True if there is still wall just beyond both ends of the doorway line."""
    u = _unit(d.p, d.q)
    k = 2 * max(1, int(round(0.1 / g.res))) + 1
    band = cv2.dilate(occ.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)
    for end, sgn in ((d.p, -1.0), (d.q, 1.0)):
        a = (end[0] + sgn * u[0] * 0.05, end[1] + sgn * u[1] * 0.05)
        b = (end[0] + sgn * u[0] * probe, end[1] + sgn * u[1] * probe)
        cx, cy = _sample_line(g, a, b)
        if len(cx) == 0 or band[cy, cx].mean() < 0.6:
            return False
    return True


def _same_door(a, b, tol=0.45):
    """Same opening if both jambs match (in either order) within tol."""
    def d(u, v):
        return math.hypot(u[0] - v[0], u[1] - v[1])
    return (d(a.p, b.p) < tol and d(a.q, b.q) < tol) or (d(a.p, b.q) < tol and d(a.q, b.p) < tol)


def _dedupe(cands):
    # Hough gives several near-identical segments per wall; keep the narrowest gap per opening
    cands = sorted(cands, key=lambda c: c.width)
    out = []
    for c in cands:
        if any(math.hypot(c.mid[0] - o.mid[0], c.mid[1] - o.mid[1]) < 0.5 and
               abs(abs(_dot(_unit(c.p, c.q), _unit(o.p, o.q))) - 1) < 0.02 for o in out):
            continue
        out.append(c)
    return out


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1]


# ------------------------------------------------------------- separation
def _draw_wall(img, g, d, value=1, extra=0.15):
    """Rasterise the virtual wall, extended a little past each jamb so it seals."""
    u = _unit(d.p, d.q)
    p = (d.p[0] - u[0] * extra, d.p[1] - u[1] * extra)
    q = (d.q[0] + u[0] * extra, d.q[1] + u[1] * extra)
    c1 = g.to_cell(*p)
    c2 = g.to_cell(*q)
    cv2.line(img, c1, c2, int(value), thickness=2)


def _far_side_normal(d, ref):
    """Unit normal of the door line pointing away from ref (robot / public side)."""
    u = _unit(d.p, d.q)
    n = (-u[1], u[0])
    m = d.mid
    if (ref[0] - m[0]) * n[0] + (ref[1] - m[1]) * n[1] > 0:
        n = (-n[0], -n[1])
    return n


def detect(grid, g, robot_xy, start_xy, visited_xy=(), signs=(), P=None, prev=None):
    """Main entry point.

    grid      int8/int16 HxW occupancy (-1 unknown)
    g         GridGeom
    robot_xy  current robot position (m)
    start_xy  start pose (m), always public
    visited_xy iterable of past robot positions (m), always public
    signs     iterable of (x, y) sign positions from the semantic mapper
    prev      the `memory` list returned by the previous call
    returns   (lethal_mask uint8 HxW, marked doorways, memory for the next call)
    """
    P = P or Params()
    occ = grid >= P.occ_thresh
    free = (grid >= 0) & (grid <= P.free_thresh)

    segs = extract_segments(occ, g, P)
    cands = candidate_gaps(segs, occ, g, P, list(signs))

    # keep earlier doorways that were not re-detected this time (Hough is not perfectly repeatable)
    if prev:
        for old in prev:
            match = next((c for c in cands if _same_door(c, old)), None)
            if match is not None:
                match.hits = old.hits + 1
                match.separates = old.separates
            elif ((old.hits >= P.keep_hits or old.separates or old.misses + 1 < P.forget_after)
                  and _jambs_present(old, occ, g)):
                # A doorway that was seen twice, or that sealed off a room, is kept for good:
                # from far away its jambs may be hidden behind people or furniture, so it
                # is not re-detected on every update. Any doorway is kept only while both
                # of its jamb walls are still in the current map.
                cands.append(Doorway(p=old.p, q=old.q, width=old.width, kind=old.kind,
                                     hits=old.hits, misses=old.misses + 1,
                                     separates=old.separates))

    # public area that must never be closed off: robot, start pose and the driven path
    public = np.zeros(grid.shape, np.uint8)
    for (x, y), r in [(robot_xy, P.protect_radius), (start_xy, P.protect_radius)]:
        cx, cy = g.to_cell(x, y)
        cv2.circle(public, (cx, cy), max(1, int(r / g.res)), 1, -1)
    rv = max(1, int(P.visited_radius / g.res))
    for x, y in visited_xy:
        cx, cy = g.to_cell(x, y)
        cv2.circle(public, (cx, cy), rv, 1, -1)

    # sliver test: a wall right behind the gap means it is not an opening into a room
    kept = []
    for d in cands:
        n = _far_side_normal(d, robot_xy)
        m = d.mid
        cx, cy = _sample_line(g, (m[0] + n[0] * 0.1, m[1] + n[1] * 0.1),
                              (m[0] + n[0] * P.sliver_depth, m[1] + n[1] * P.sliver_depth))
        if len(cx) and occ[cy, cx].any():
            continue
        kept.append(d)
    cands = kept

    rr = int(P.room_depth / g.res)
    public_b = public.astype(bool)

    def run(doors):
        # Rooms are followed through *known free* cells only. Unknown cells never count
        # as a connection, so an unexplored room cannot "leak" round the building into
        # the corridor; it is still sealed by its virtual wall.
        walls = np.zeros(grid.shape, np.uint8)
        for d in doors:
            _draw_wall(walls, g, d)
        passable = (free & (walls == 0)).astype(np.uint8)
        _, labels = cv2.connectedComponents(passable, connectivity=4)
        public_labels = set(np.unique(labels[public_b & (passable > 0)]).tolist()) - {0}
        results = []
        for d in doors:
            n = _far_side_normal(d, robot_xy)
            seed_labels = set()
            for t in np.linspace(0.15, 0.85, 7):
                x, y = _lerp(d.p, d.q, t)
                for back in (P.seed_back, P.seed_back + 0.3):
                    cx, cy = g.to_cell(x + n[0] * back, y + n[1] * back)
                    if g.inside(cx, cy) and labels[cy, cx] > 0:
                        seed_labels.add(int(labels[cy, cx]))
            hit = bool(seed_labels & public_labels)
            region = np.zeros(grid.shape, np.uint8)
            if seed_labels and not hit:
                region = np.isin(labels, list(seed_labels)).astype(np.uint8)
                # only mark the room itself, not anything far away that happens to connect
                near = np.zeros_like(region)
                cv2.circle(near, g.to_cell(*d.mid), rr, 1, -1)
                region &= near
            results.append((d, region, hit))
        return results

    # Every doorway gets a virtual wall across it, so the robot can never drive through,
    # unless the line would cross the robot, its start pose or its driven path.
    doors = []
    for d in cands:
        line = np.zeros(grid.shape, np.uint8)
        _draw_wall(line, g, d)
        if (line.astype(bool) & public_b).any():
            continue
        doors.append(d)

    # The room behind a doorway is filled in only if it is really closed off from the
    # public area by real walls and the virtual walls together.
    lethal = np.zeros(grid.shape, np.uint8)
    marked = []
    for d, region, hit in run(doors):
        d.separates = not hit
        if not hit:
            region = region & (public == 0) & (~occ).astype(np.uint8)
            d.region_cells = int(region.sum())
            d.region = region
            lethal |= region
        else:
            d.region_cells = 0
            d.region = None
            # a doorway whose room is not (yet) sealed off is closed only once it has
            # been seen in two updates, so a one-off line from map noise never appears
            if d.hits < P.confirm_hits:
                continue
        _draw_wall(lethal, g, d)
        marked.append(d)
    memory = cands
    lethal &= (public == 0)
    return lethal, marked, memory
