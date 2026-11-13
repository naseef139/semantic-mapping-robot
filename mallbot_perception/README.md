# mallbot_perception – object notes

YOLO-World detections + depth → 3D position in the SLAM `map` → one **note per real object**
(same label within a gate = same object, confirmed after 3 sightings). People are shown in red and **never stored**.

## Install once
```bash
pip3 install --user ultralytics "numpy<2"      # numpy<2 keeps ROS Humble's Python packages working
```
The first run downloads `yolov8s-worldv2.pt` and the CLIP text encoder (internet needed once).

## Run (with the simulation)
```bash
mlaunch mallbot_sim sim.launch.py                       # 1
mlaunch mallbot_sim nav.launch.py explore:=true         # 2
mlaunch mallbot_perception perception.launch.py         # 3   (device:=0 for an NVIDIA GPU)
```
RViz: **Object notes** (cubes + "label #id (sightings)"), **Detections** panel (boxes; people marked `[ignored]`).
Notes are saved to `~/mallbot_notes/notes.json` every 5 s and on Ctrl+C.

## Score against the ground truth
```bash
ros2 run mallbot_perception evaluate_notes             # found / missed / duplicates / position error / people stored
```

## Tune (`config/perception.yaml`)
`classes` (words YOLO-World looks for), `confidence`, `min_sightings`, `gate` / `class_gate_*`, `dynamic_classes`, `max_range`.
