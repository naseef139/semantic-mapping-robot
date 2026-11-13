"""Offline tests: geometry helpers + NoteBook on synthetic noisy detections from the mall_v2 ground truth."""
import csv, json, math, os, random, subprocess, sys, types
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mallbot_perception.geometry import backproject, depth_in_box, image_to_numpy, quat_to_mat, apply_tf
from mallbot_perception.notes import NoteBook

GT = os.path.join(os.path.dirname(__file__), "..", "..", "mallbot_sim", "config", "mall_v2_ground_truth.csv")


def test_backproject_centre_pixel():
    p = backproject(320, 200, 3.0, 500, 500, 320, 200)
    assert np.allclose(p, [0, 0, 3.0])
    p = backproject(420, 200, 2.0, 500, 500, 320, 200)        # 100 px right at 2 m -> 0.4 m right
    assert np.allclose(p, [0.4, 0, 2.0])


def test_optical_to_map_rotation():
    # camera_optical_frame -> camera_link rotation (rpy -pi/2, 0, -pi/2): optical z (forward) = link x
    from math import sin, cos, pi
    def q_from_rpy(r, p, y):
        cr, sr, cp, sp, cy, sy = cos(r/2), sin(r/2), cos(p/2), sin(p/2), cos(y/2), sin(y/2)
        return (sr*cp*cy - cr*sp*sy, cr*sp*cy + sr*cp*sy, cr*cp*sy - sr*sp*cy, cr*cp*cy + sr*sp*sy)
    R = quat_to_mat(*q_from_rpy(-pi/2, 0, -pi/2))
    assert np.allclose(R @ [0, 0, 1], [1, 0, 0], atol=1e-9)   # forward
    assert np.allclose(R @ [1, 0, 0], [0, -1, 0], atol=1e-9)  # optical right = link -y
    assert np.allclose(R @ [0, 1, 0], [0, 0, -1], atol=1e-9)  # optical down = link -z


def test_depth_in_box_ignores_inf_and_background():
    d = np.full((100, 100), np.inf, dtype=np.float32)
    d[30:70, 30:70] = 2.0                                      # object
    d[30:70, 30:35] = 9.0                                      # background leak at the edge
    assert abs(depth_in_box(d, 25, 25, 75, 75) - 2.0) < 1e-6
    assert depth_in_box(np.full((50, 50), np.inf, np.float32), 0, 0, 50, 50) is None


def test_image_to_numpy_rgb_and_depth():
    m = types.SimpleNamespace(height=2, width=3, step=9, encoding="rgb8", data=bytes(range(18)))
    assert image_to_numpy(m).shape == (2, 3, 3)
    dep = np.arange(6, dtype=np.float32).reshape(2, 3)
    m = types.SimpleNamespace(height=2, width=3, step=12, encoding="32FC1", data=dep.tobytes())
    assert np.allclose(image_to_numpy(m), dep)


def test_notebook_on_synthetic_mall(tmp_path=None):
    random.seed(1)
    rows = list(csv.DictReader(open(GT)))
    book = NoteBook(gate=0.8, class_gates={"cabinet": 0.35, "bookshelf": 0.6}, min_sightings=3, dynamic=["person"])
    LABEL = {"dining table": "table"}
    spawn = (-10.0, 0.0)
    t = 0.0
    for r in rows:
        lab = r["label"]
        lab = "shop sign" if lab.startswith("sign:") else LABEL.get(lab, lab)
        mx, my = float(r["x"]) - spawn[0], float(r["y"]) - spawn[1]   # world -> map
        k = random.choice([0, 1, 2, 4, 6, 8, 10])                     # times seen (0-2 = not confirmed)
        sig = 0.06 if lab == "cabinet" else 0.12
        for _ in range(k):
            t += 0.5
            book.observe(lab, mx + random.gauss(0, sig), my + random.gauss(0, sig), float(r["z"]), 0.5, t)
    for _ in range(25):                                                # one-off false positives
        book.observe(random.choice(["bench", "table", "counter"]), random.uniform(-2, 22), random.uniform(-7, 7), 0.4, 0.3, t)
    book.merge()
    people = [n for n in book.notes if n.label == "person"]
    assert not people, "people must never become notes"
    out = os.path.join("/tmp", "test_notes.json")
    n = book.to_json(out)
    assert n == len(book.confirmed())
    r = subprocess.run([sys.executable, "-m", "mallbot_perception.evaluate", "--notes", out, "--gt", GT, "--tol", "0.6"],
                       capture_output=True, text=True, cwd=os.path.join(os.path.dirname(__file__), ".."))
    print(r.stdout)
    assert "people stored: 0" in r.stdout
    assert "duplicates: 0" in r.stdout


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
