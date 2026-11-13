#!/usr/bin/env python3
"""
evaluate_notes - score the saved object notes against the simulator ground truth.

  ros2 run mallbot_perception evaluate_notes                      # defaults below
  ros2 run mallbot_perception evaluate_notes --notes ~/mallbot_notes/notes.json \
      --gt ~/mallbot_ws/src/mallbot_sim/config/mall_v2_ground_truth.csv --tol 1.0

IMPORTANT: notes are in the SLAM `map` frame, ground truth in the Gazebo world frame. They match
when the robot starts at the default spawn (x=-10, y=0, yaw=0): map origin = spawn pose.
Pass --spawn x y yaw if you spawned elsewhere.

Metrics (per class and overall):
  found / missed      ground-truth objects with a note of the same label within --tol metres
  false notes         notes that match no ground-truth object
  duplicates          extra notes on an object that already had one
  position error      mean / max distance note -> true centre (front-surface bias is expected)
  people stored       notes labelled with a dynamic class (must be 0)
"""
import argparse
import csv
import json
import math
import os

# ground-truth label -> label the detector uses
GT_TO_DET = {"dining table": "table", "sign": "shop sign"}


def load_gt(path):
    rows = []
    for r in csv.DictReader(open(path)):
        lab = r["label"]
        text = ""
        if lab.startswith("sign:"):
            lab, text = "sign", lab[5:]
        rows.append({"name": r["model"], "label": GT_TO_DET.get(lab, lab), "text": text,
                     "x": float(r["x"]), "y": float(r["y"])})
    return rows


def main():
    home = os.path.expanduser("~")
    ap = argparse.ArgumentParser()
    ap.add_argument("--notes", default=os.path.join(home, "mallbot_notes", "notes.json"))
    ap.add_argument("--gt", default=os.path.join(home, "mallbot_ws", "src", "mallbot_sim", "config",
                                                 "mall_v2_ground_truth.csv"))
    ap.add_argument("--tol", type=float, default=1.0, help="match distance (m)")
    ap.add_argument("--spawn", type=float, nargs=3, default=[-10.0, 0.0, 0.0], metavar=("X", "Y", "YAW"))
    ap.add_argument("--dynamic", default="person")
    ap.add_argument("--no-details", dest="details", action="store_false", help="hide the per-object lists")
    a = ap.parse_args()

    notes = json.load(open(a.notes))["notes"]
    gt = load_gt(a.gt)
    sx, sy, syaw = a.spawn
    c, s = math.cos(syaw), math.sin(syaw)
    for n in notes:                         # map frame -> world frame
        n["wx"] = sx + c * n["x"] - s * n["y"]
        n["wy"] = sy + s * n["x"] + c * n["y"]
    dynamic = set(a.dynamic.split(","))

    # greedy matching: closest (note, gt) pairs first, same label, within tol
    pairs = []
    for i, n in enumerate(notes):
        for j, g in enumerate(gt):
            if n["label"] == g["label"]:
                d = math.hypot(n["wx"] - g["x"], n["wy"] - g["y"])
                if d <= a.tol:
                    pairs.append((d, i, j))
    pairs.sort()
    note_used, gt_used, matches = set(), {}, []
    dup = set()
    for d, i, j in pairs:
        if i in note_used:
            continue
        if j in gt_used:
            dup.add(i)                      # a second note on the same real object
            note_used.add(i)
            continue
        note_used.add(i)
        gt_used[j] = i
        matches.append((d, i, j))
    false_notes = [i for i in range(len(notes)) if i not in note_used]

    labels = sorted({g["label"] for g in gt} | {n["label"] for n in notes})
    print(f"\nnotes: {len(notes)}   ground-truth objects: {len(gt)}   tolerance: {a.tol} m\n")
    print(f"{'class':14s} {'GT':>4s} {'found':>6s} {'missed':>7s} {'false':>6s} {'dup':>4s} {'err mean':>9s} {'err max':>8s}")
    tot = dict(gt=0, found=0, false=0, dup=0)
    errs_all = []
    for lab in labels:
        g_n = sum(1 for g in gt if g["label"] == lab)
        found = [(d, i, j) for d, i, j in matches if gt[j]["label"] == lab]
        fa = sum(1 for i in false_notes if notes[i]["label"] == lab)
        du = sum(1 for i in dup if notes[i]["label"] == lab)
        errs = [d for d, _, _ in found]
        errs_all += errs
        mark = "  <- dynamic, must be 0 notes" if lab in dynamic else ""
        print(f"{lab:14s} {g_n:4d} {len(found):6d} {g_n - len(found):7d} {fa:6d} {du:4d} "
              f"{(sum(errs) / len(errs)) if errs else float('nan'):9.2f} {max(errs) if errs else float('nan'):8.2f}{mark}")
        if lab not in dynamic:
            tot["gt"] += g_n
            tot["found"] += len(found)
        tot["false"] += fa
        tot["dup"] += du
    people = sum(1 for n in notes if n["label"] in dynamic)
    rec = tot["found"] / tot["gt"] if tot["gt"] else 0
    prec = len(matches) / len(notes) if notes else 0
    print("\nstatic objects found (recall): {}/{} = {:.0%}".format(tot["found"], tot["gt"], rec))
    print("notes that are correct (precision): {}/{} = {:.0%}".format(len(matches), len(notes), prec))
    print("duplicates: {}   false notes: {}   people stored: {}{}".format(
        tot["dup"], tot["false"], people, "  OK" if people == 0 else "  <- should be 0"))
    if errs_all:
        print("position error: mean {:.2f} m, max {:.2f} m".format(sum(errs_all) / len(errs_all), max(errs_all)))

    if a.details:
        def nearest(x, y, items, key_x, key_y):
            best = min(items, key=lambda it: math.hypot(it[key_x] - x, it[key_y] - y), default=None)
            return (best, math.hypot(best[key_x] - x, best[key_y] - y)) if best else (None, 0)
        print("\nfalse / duplicate notes -> nearest real object (wrong label? or too far?):")
        for i in sorted(set(false_notes) | dup):
            n = notes[i]
            g, d = nearest(n["wx"], n["wy"], gt, "x", "y")
            tag = "DUPLICATE" if i in dup else "false"
            print(f"  {tag:9s} {n['label']:13s} #{n['id']:<3d} ({n['sightings']:3d}x) at ({n['wx']:6.2f},{n['wy']:6.2f})"
                  f"  nearest real: {g['label']} {g['name']} {d:.2f} m")
        print("\nmissed objects -> nearest note of ANY label (none nearby = never seen / never detected):")
        for j, g in enumerate(gt):
            if j in gt_used or g["label"] in dynamic:
                continue
            n, d = nearest(g["x"], g["y"], notes, "wx", "wy")
            near = f"{n['label']} #{n['id']} at {d:.2f} m" if n and d < 2.0 else "-"
            print(f"  {g['label']:13s} {g['name']:22s} ({g['x']:6.2f},{g['y']:6.2f})  nearest note: {near}")


if __name__ == "__main__":
    main()
