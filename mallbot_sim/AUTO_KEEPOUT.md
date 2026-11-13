# Automatic keepout zones (no preset mask)

`auto_keepout.py` finds shop and room entrances in the SLAM map while the robot
explores, and closes them on the Nav2 keepout mask. Nothing is drawn by hand, so
it also works in an environment the robot has never seen.

## How it works
1. **Doorways from the LiDAR map.** A doorway is a 0.6–2.5 m gap between two
   collinear wall segments. The two jambs must lie on the same straight wall,
   with at least 0.6 m of wall on each side. Benches, bins and people standing
   near a wall do not qualify.
2. **Wide shop fronts from the camera.** An opening of up to 8 m is closed when
   the semantic mapper has placed a shop **sign** on it.
3. **Virtual wall.** A lethal line is drawn across every doorway, so the planner
   and controller treat it like a wall.
4. **Room fill.** If the known free space behind the line is sealed off from the
   robot, it is filled as keepout too, and the explorer drops the frontiers in it.
5. **Safety rules.** The node never marks the robot, its start pose or anywhere it
   has already driven. A doorway is never drawn across those places.

The mask goes out on `/keepout_filter_mask`, the topic the KeepoutFilter already
uses, so the planner, controller, explorer and RViz "Keepout zones" display all
work unchanged.

## Install (5 minutes)
Copy the files into `~/mallbot_ws/src/mallbot_sim/` (same folders as in this zip):

```
scripts/auto_keepout.py      scripts/room_core.py
launch/auto_keepout.launch.py
config/auto_keepout.yaml     config/keepout_blank.yaml   config/keepout_blank.pgm
test/test_room_core.py
```

In `mallbot_sim/CMakeLists.txt`, add the two scripts to the existing
`install(PROGRAMS ...)` list (next to `frontier_explorer.py` and `frontier_core.py`):

```cmake
  scripts/auto_keepout.py
  scripts/room_core.py
```

Then build:

```bash
cd ~/mallbot_ws/src/mallbot_sim && chmod +x scripts/auto_keepout.py scripts/room_core.py
cd ~/mallbot_ws && colcon build --symlink-install --packages-select mallbot_sim && source install/setup.bash
```

## Run
Same three terminals as before, plus the keepout node. Start the node **before**
exploration begins. `keepout_blank.yaml` replaces the old hand-made mall mask.

```bash
# T1  simulation + SLAM + RViz
ros2 launch mallbot_sim sim.launch.py
# T2  automatic keepout (start before navigation/exploration)
ros2 launch mallbot_sim auto_keepout.launch.py
# T3  navigation + exploration, with an empty starting mask
ros2 launch mallbot_sim nav.launch.py explore:=true keepout:=true \
  keepout_mask:=$(ros2 pkg prefix mallbot_sim)/share/mallbot_sim/config/keepout_blank.yaml
# T4  perception (provides the sign cue for wide shop fronts)
ros2 launch mallbot_perception perception.launch.py
```

In RViz, add a **MarkerArray** display on `/auto_keepout/doorways`. Red lines are
doorways and magenta lines are signed shop fronts; each carries its width as a label.
The existing "Keepout zones" map display shows the filled shops.

Log lines look like this:
`closed doorway at (4.2, -2.2), 1.6 m wide, room behind it marked`

Options: `door_max:=2.0` (stricter) and `use_signs:=false` (LiDAR only).
Every threshold is in `config/auto_keepout.yaml`.

## Offline test (no ROS needed)
```bash
cd ~/mallbot_ws/src/mallbot_sim/test
python3 test_room_core.py --glance --noise --plot result.png
```

The test builds a simulated 25 × 15 m mall with 10 shops. Their doorways are 1.0–2.4 m
wide, plus two open shop fronts of 3.4 m and 4.0 m. A central bench, a bench by the wall,
a plant, a bin and two people are added as clutter. A ray-cast LiDAR builds the map step
by step, with range noise, dropped returns and SLAM pose error. Results over 32 runs
(2 robot paths × clean or noisy maps × up to 15 seeds):

| Check | Result |
|---|---|
| Corridor cells ever marked as keepout | 0 in 32/32 runs |
| Cells on the robot's path ever marked | 0 in 32/32 runs |
| Every doorway closed before the robot came within 1.5 m | 32/32 runs |
| Robot distance when a doorway was first closed | ≥ 1.72 m (usually 2–6 m) |
| Doorway re-opened after closing | 0 times |
| Update time | ≈ 0.1 s (max 0.22 s) for a 25 × 15 m map |

## Limits (worth knowing)
- **Wide-open shop fronts need a sign.** Without the camera sign cue
  (`use_signs:=false`), openings wider than `door_max` stay open.
- **Narrow side passages look like doorways.** A side corridor narrower than
  `door_max` is closed as if it were a doorway. Lower `door_max` if your building
  has narrow passages.
- **Glass shop fronts are invisible to a 2D LiDAR.** Only the sign cue can close them.
- **One condition for closing a doorway in time:** the robot must see both jambs
  before it reaches the doorway. In the tests this happened at ≥ 1.7 m.
