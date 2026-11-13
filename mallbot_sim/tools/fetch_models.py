#!/usr/bin/env python3
"""
Download the furniture models used by worlds/mall_v2.sdf from github.com/osrf/gazebo_models
into mallbot_sim/models/, and convert their Gazebo-Classic material scripts (ignored by
Ignition Fortress) into plain colours. Run once, then rebuild:

  python3 ~/mallbot_ws/src/mallbot_sim/tools/fetch_models.py
  cd ~/mallbot_ws && colcon build --symlink-install --packages-select mallbot_sim

Models (c) their original authors - see each model.config. Source: https://github.com/osrf/gazebo_models
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

MODELS = ["cabinet", "bookshelf", "table", "cafe_table", "first_2015_trash_can", "person_standing"]
REPO = "https://github.com/osrf/gazebo_models.git"
HERE = os.path.dirname(os.path.abspath(__file__))
DEST = os.path.normpath(os.path.join(HERE, "..", "models"))

# Gazebo Classic material name -> RGB
COLOURS = {"Wood": "0.55 0.37 0.22", "Grey": "0.5 0.5 0.5", "White": "0.9 0.9 0.9",
           "Black": "0.08 0.08 0.08", "DarkGrey": "0.3 0.3 0.3", "Red": "0.8 0.1 0.1",
           "Blue": "0.1 0.2 0.8", "Green": "0.1 0.6 0.1"}
SCRIPT = re.compile(r"<material>\s*<script>.*?<name>Gazebo/(\w+)</name>.*?</script>\s*</material>", re.S)


def patch(sdf_path):
    s = open(sdf_path).read()

    def repl(m):
        rgb = COLOURS.get(m.group(1), "0.7 0.7 0.7")
        return f"<material><ambient>{rgb} 1</ambient><diffuse>{rgb} 1</diffuse></material>"
    n = len(SCRIPT.findall(s))
    if n:
        open(sdf_path, "w").write(SCRIPT.sub(repl, s))
    return n


def main():
    os.makedirs(DEST, exist_ok=True)
    missing = [m for m in MODELS if not os.path.exists(os.path.join(DEST, m, "model.sdf"))]
    if not missing:
        print("all models already present in", DEST)
        return
    tmp = tempfile.mkdtemp(prefix="gzmodels_")
    try:
        run = lambda *a: subprocess.run(a, cwd=tmp, check=True)
        print("downloading:", ", ".join(missing))
        run("git", "clone", "--depth", "1", "--filter=blob:none", "--no-checkout", REPO, "gzm")
        repo = os.path.join(tmp, "gzm")
        subprocess.run(["git", "sparse-checkout", "init", "--no-cone"], cwd=repo, check=True)
        subprocess.run(["git", "sparse-checkout", "set"] + missing, cwd=repo, check=True)
        subprocess.run(["git", "checkout"], cwd=repo, check=True)
        for m in missing:
            shutil.copytree(os.path.join(repo, m), os.path.join(DEST, m), dirs_exist_ok=True)
            n = sum(patch(os.path.join(DEST, m, f)) for f in os.listdir(os.path.join(DEST, m))
                    if f.endswith(".sdf"))
            print(f"  {m:22s} ok" + (f"  ({n} classic materials -> colours)" if n else ""))
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        sys.exit(f"download failed ({e}). Is git installed and is GitHub reachable?")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("done ->", DEST, "\nnow rebuild: cd ~/mallbot_ws && colcon build --symlink-install --packages-select mallbot_sim")


if __name__ == "__main__":
    main()
