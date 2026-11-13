#!/usr/bin/env python3
"""
Generate worlds/mall_v1.sdf - a simple single-floor mall for MallBot.

Layout (top view, metres):
  - Main corridor along x from -12 to 12, 4 m wide (y = -2 .. 2)
  - 4 shops on the north side, 4 on the south side, each 6 m wide x 5 m deep
  - Each shop has an open 3 m entrance facing the corridor and a coloured
    sign board above the entrance (text textures are added in a later step)
  - Static objects in the corridor: benches, plant pots, bins

Run:  python3 make_mall_world.py  (writes ../worlds/mall_v1.sdf)
"""
import os

WALL_H = 3.0     # wall height
WALL_T = 0.15    # wall thickness
X_MIN, X_MAX = -12.0, 12.0
Y_CORR = 2.0     # corridor half-width
Y_OUT = 7.0      # outer wall y
SHOP_W = 6.0
DOOR_W = 3.0
SIGN_Z = 2.5     # sign centre height

# Shop names, west -> east  (used later for sign textures / ground truth)
SHOPS_NORTH = ["PHARMACY", "BOOKSHOP", "CAFE", "SHOES"]
SHOPS_SOUTH = ["ELECTRONICS", "BAKERY", "TOYS", "FASHION"]
SIGN_COLOURS = [
    (0.10, 0.55, 0.25), (0.75, 0.45, 0.10), (0.45, 0.25, 0.10), (0.70, 0.10, 0.15),
    (0.10, 0.30, 0.70), (0.85, 0.65, 0.20), (0.80, 0.20, 0.60), (0.20, 0.20, 0.20),
]

parts = []


def box(name, x, y, z, sx, sy, sz, rgb=(0.85, 0.85, 0.82), static=True):
    r, g, b = rgb
    parts.append(f"""
    <model name="{name}">
      <static>{'true' if static else 'false'}</static>
      <pose>{x:.3f} {y:.3f} {z:.3f} 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry><box><size>{sx:.3f} {sy:.3f} {sz:.3f}</size></box></geometry>
        </collision>
        <visual name="visual">
          <geometry><box><size>{sx:.3f} {sy:.3f} {sz:.3f}</size></box></geometry>
          <material>
            <ambient>{r} {g} {b} 1</ambient>
            <diffuse>{r} {g} {b} 1</diffuse>
          </material>
        </visual>
      </link>
    </model>""")


def cylinder(name, x, y, z, radius, length, rgb):
    r, g, b = rgb
    parts.append(f"""
    <model name="{name}">
      <static>true</static>
      <pose>{x:.3f} {y:.3f} {z:.3f} 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry><cylinder><radius>{radius}</radius><length>{length}</length></cylinder></geometry>
        </collision>
        <visual name="visual">
          <geometry><cylinder><radius>{radius}</radius><length>{length}</length></cylinder></geometry>
          <material>
            <ambient>{r} {g} {b} 1</ambient>
            <diffuse>{r} {g} {b} 1</diffuse>
          </material>
        </visual>
      </link>
    </model>""")


def wall_x(name, x0, x1, y):
    """Wall running along x at fixed y."""
    if x1 - x0 < 1e-3:
        return
    box(name, (x0 + x1) / 2, y, WALL_H / 2, x1 - x0, WALL_T, WALL_H)


def wall_y(name, y0, y1, x):
    """Wall running along y at fixed x."""
    box(name, x, (y0 + y1) / 2, WALL_H / 2, WALL_T, y1 - y0, WALL_H)


# ---------- outer walls ----------
wall_x("outer_north", X_MIN, X_MAX, Y_OUT)
wall_x("outer_south", X_MIN, X_MAX, -Y_OUT)
wall_y("outer_west", -Y_OUT, Y_OUT, X_MIN)
wall_y("outer_east", -Y_OUT, Y_OUT, X_MAX)

# ---------- shops ----------
ground_truth = []
for side, names, sgn in (("n", SHOPS_NORTH, 1), ("s", SHOPS_SOUTH, -1)):
    for i, shop in enumerate(names):
        x0 = X_MIN + i * SHOP_W
        x1 = x0 + SHOP_W
        cx = (x0 + x1) / 2
        y_front = sgn * Y_CORR
        # front wall with a centred entrance
        wall_x(f"{side}{i}_front_l", x0, cx - DOOR_W / 2, y_front)
        wall_x(f"{side}{i}_front_r", cx + DOOR_W / 2, x1, y_front)
        # wall above the entrance (lintel)
        box(f"{side}{i}_lintel", cx, y_front, (2.2 + WALL_H) / 2, DOOR_W, WALL_T, WALL_H - 2.2)
        # dividing wall between shops (skip the last one: outer wall closes it)
        if i < len(names) - 1:
            y_a, y_b = sorted((sgn * Y_CORR, sgn * Y_OUT))
            wall_y(f"{side}{i}_divider", y_a, y_b, x1)
        # sign board on the corridor side, just in front of the lintel
        sign_y = y_front - sgn * (WALL_T / 2 + 0.03)
        colour = SIGN_COLOURS[(0 if side == "n" else 4) + i]
        box(f"sign_{shop.lower()}", cx, sign_y, SIGN_Z, 2.4, 0.05, 0.5, colour)
        ground_truth.append((f"sign_{shop.lower()}", shop, cx, sign_y, SIGN_Z))
        # something inside each shop: a counter
        box(f"{side}{i}_counter", cx, sgn * (Y_OUT - 1.2), 0.5, 2.0, 0.6, 1.0, (0.55, 0.40, 0.30))
        ground_truth.append((f"{side}{i}_counter", "counter", cx, sgn * (Y_OUT - 1.2), 0.5))

# ---------- static objects in the corridor ----------
for k, x in enumerate((-8.0, 1.0, 8.5)):
    box(f"bench_{k}", x, 1.3, 0.225, 1.6, 0.5, 0.45, (0.35, 0.25, 0.15))
    ground_truth.append((f"bench_{k}", "bench", x, 1.3, 0.225))
for k, x in enumerate((-4.5, 5.0)):
    cylinder(f"plant_{k}", x, -1.4, 0.35, 0.3, 0.7, (0.15, 0.45, 0.15))
    ground_truth.append((f"plant_{k}", "potted plant", x, -1.4, 0.35))
for k, x in enumerate((-1.5, 10.5)):
    cylinder(f"bin_{k}", x, -1.6, 0.4, 0.2, 0.8, (0.3, 0.3, 0.3))
    ground_truth.append((f"bin_{k}", "bin", x, -1.6, 0.4))

world = f"""<?xml version="1.0"?>
<!-- AUTO-GENERATED by tools/make_mall_world.py - edit the script, not this file -->
<sdf version="1.8">
  <world name="mall_v1">
    <physics name="1ms" type="ignored">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>

    <plugin filename="ignition-gazebo-physics-system" name="ignition::gazebo::systems::Physics"/>
    <plugin filename="ignition-gazebo-user-commands-system" name="ignition::gazebo::systems::UserCommands"/>
    <plugin filename="ignition-gazebo-scene-broadcaster-system" name="ignition::gazebo::systems::SceneBroadcaster"/>
    <plugin filename="ignition-gazebo-sensors-system" name="ignition::gazebo::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>
    <plugin filename="ignition-gazebo-imu-system" name="ignition::gazebo::systems::Imu"/>

    <scene>
      <ambient>0.6 0.6 0.6 1</ambient>
      <background>0.8 0.85 0.9 1</background>
      <grid>false</grid>
    </scene>

    <light type="directional" name="sun">
      <cast_shadows>false</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.9 0.9 0.9 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <direction>-0.3 0.2 -0.9</direction>
    </light>

    <model name="floor">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry><plane><normal>0 0 1</normal><size>30 20</size></plane></geometry>
        </collision>
        <visual name="visual">
          <geometry><plane><normal>0 0 1</normal><size>30 20</size></plane></geometry>
          <material>
            <ambient>0.75 0.73 0.70 1</ambient>
            <diffuse>0.75 0.73 0.70 1</diffuse>
          </material>
        </visual>
      </link>
    </model>
{''.join(parts)}
  </world>
</sdf>
"""

here = os.path.dirname(os.path.abspath(__file__))
out_world = os.path.join(here, "..", "worlds", "mall_v1.sdf")
with open(out_world, "w") as f:
    f.write(world)

# Ground truth for evaluation later (name, label, x, y, z)
out_gt = os.path.join(here, "..", "config", "mall_v1_ground_truth.csv")
with open(out_gt, "w") as f:
    f.write("model,label,x,y,z\n")
    for row in ground_truth:
        f.write("{},{},{:.3f},{:.3f},{:.3f}\n".format(*row))

print(f"wrote {os.path.normpath(out_world)} and {os.path.normpath(out_gt)}")
