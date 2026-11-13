#!/usr/bin/env python3
"""
Generate worlds/mall_v2.sdf: the same mall layout as v1, furnished for PERCEPTION testing.

  - shop signs printed with the shop name (texture on a quad mesh, readable from the corridor) -> OCR
  - real furniture models (osrf/gazebo_models, fetched by tools/fetch_models.py):
      cabinet, bookshelf, table, cafe_table, first_2015_trash_can, person_standing
  - benches and potted plants built from primitives (solid base: the 2D LiDAR at 0.23 m must see them)
  - 2 standing people in the corridor: COCO "person" -> must be FILTERED OUT of the object notes
Also writes:
  models/mall_signs/...           sign quad mesh + one PNG per shop
  config/mall_v2_ground_truth.csv true centre of every object and sign (for evaluation)

Run:  python3 tools/make_mall_v2.py
"""
import math
import os

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.normpath(os.path.join(HERE, ".."))

WALL_H, WALL_T = 3.0, 0.15
X_MIN, X_MAX = -12.0, 12.0
Y_CORR, Y_OUT = 2.0, 7.0
SHOP_W, DOOR_W = 6.0, 3.0
SIGN_Z, SIGN_W, SIGN_H = 2.5, 2.4, 0.5

SHOPS_NORTH = ["PHARMACY", "BOOKSHOP", "CAFE", "SHOES"]
SHOPS_SOUTH = ["ELECTRONICS", "BAKERY", "TOYS", "FASHION"]
SIGN_COLOURS = [
    (0.10, 0.55, 0.25), (0.75, 0.45, 0.10), (0.45, 0.25, 0.10), (0.70, 0.10, 0.15),
    (0.10, 0.30, 0.70), (0.85, 0.65, 0.20), (0.80, 0.20, 0.60), (0.20, 0.20, 0.20),
]

parts, gt = [], []


def mat(rgb):
    r, g, b = rgb
    return f"<material><ambient>{r} {g} {b} 1</ambient><diffuse>{r} {g} {b} 1</diffuse></material>"


def geom_box(sx, sy, sz):
    return f"<box><size>{sx:.3f} {sy:.3f} {sz:.3f}</size></box>"


def link_shape(name, pose, geom, rgb, collide=True):
    col = f'<collision name="{name}_c"><pose>{pose}</pose><geometry>{geom}</geometry></collision>' if collide else ""
    return col + f'<visual name="{name}_v"><pose>{pose}</pose><geometry>{geom}</geometry>{mat(rgb)}</visual>'


def model(name, x, y, z, yaw, body):
    parts.append(f"""
    <model name="{name}"><static>true</static><pose>{x:.3f} {y:.3f} {z:.3f} 0 0 {yaw:.4f}</pose>
      <link name="link">{body}</link></model>""")


def box(name, x, y, z, sx, sy, sz, rgb=(0.85, 0.85, 0.82)):
    model(name, x, y, z, 0, link_shape("b", "0 0 0 0 0 0", geom_box(sx, sy, sz), rgb))


def include(uri, name, x, y, yaw=0.0):
    parts.append(f"""
    <include><uri>model://{uri}</uri><name>{name}</name><static>true</static>
      <pose>{x:.3f} {y:.3f} 0 0 0 {yaw:.4f}</pose></include>""")


def add_gt(name, label, x, y, z, local=(0, 0), yaw=0.0):
    """Store the object's true centre: model origin + local centre offset rotated by yaw."""
    lx, ly = local
    cx = x + lx * math.cos(yaw) - ly * math.sin(yaw)
    cy = y + lx * math.sin(yaw) + ly * math.cos(yaw)
    gt.append((name, label, cx, cy, z))


def wall_x(name, x0, x1, y):
    if x1 - x0 > 1e-3:
        box(name, (x0 + x1) / 2, y, WALL_H / 2, x1 - x0, WALL_T, WALL_H)


def wall_y(name, y0, y1, x):
    box(name, x, (y0 + y1) / 2, WALL_H / 2, WALL_T, y1 - y0, WALL_H)


# ---------------- sign textures + quad mesh ----------------
SIGN_DIR = os.path.join(PKG, "models", "mall_signs")
os.makedirs(os.path.join(SIGN_DIR, "meshes"), exist_ok=True)
os.makedirs(os.path.join(SIGN_DIR, "materials", "textures"), exist_ok=True)
with open(os.path.join(SIGN_DIR, "model.config"), "w") as f:
    f.write('<?xml version="1.0"?>\n<model><name>mall_signs</name><version>1.0</version>'
            '<sdf version="1.8">model.sdf</sdf><description>Shop sign textures for MallBot</description></model>\n')
with open(os.path.join(SIGN_DIR, "model.sdf"), "w") as f:
    f.write('<?xml version="1.0"?>\n<sdf version="1.8"><model name="mall_signs"><static>true</static>'
            '<link name="l"/></model></sdf>\n')
# unit quad in the x-z plane, facing -y, UV (0,0) bottom-left: text reads correctly from -y
with open(os.path.join(SIGN_DIR, "meshes", "sign_quad.obj"), "w") as f:
    f.write("# unit quad, normal -y\n"
            "v -0.5 0 -0.5\nv 0.5 0 -0.5\nv 0.5 0 0.5\nv -0.5 0 0.5\n"
            "vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\nvn 0 -1 0\n"
            "f 1/1/1 2/2/1 3/3/1\nf 1/1/1 3/3/1 4/4/1\n")

FONT = None
for cand in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
             "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"):
    if os.path.exists(cand):
        FONT = cand
        break


def make_sign_png(text, rgb, path):
    W, H = 1200, 250                        # same 4.8:1 aspect as the 2.4 x 0.5 m board
    img = Image.new("RGB", (W, H), tuple(int(c * 255) for c in rgb))
    d = ImageDraw.Draw(img)
    size = 170
    while True:
        font = ImageFont.truetype(FONT, size) if FONT else ImageFont.load_default()
        bb = d.textbbox((0, 0), text, font=font)
        if bb[2] - bb[0] <= W * 0.86 or size < 40:
            break
        size -= 6
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    d.text(((W - tw) / 2 - bb[0], (H - th) / 2 - bb[1]), text, fill=(255, 255, 255), font=font)
    d.rectangle([6, 6, W - 7, H - 7], outline=(255, 255, 255), width=6)
    img.save(path)


def sign(shop, rgb, cx, y, facing_minus_y):
    tex = f"{shop.lower()}.png"
    make_sign_png(shop, rgb, os.path.join(SIGN_DIR, "materials", "textures", tex))
    yaw = 0.0 if facing_minus_y else math.pi
    body = (link_shape("board", "0 0 0 0 0 0", geom_box(SIGN_W, 0.04, SIGN_H), rgb)
            + f'<visual name="face"><pose>0 -0.021 0 0 0 0</pose><geometry><mesh>'
              f'<uri>model://mall_signs/meshes/sign_quad.obj</uri><scale>{SIGN_W} 1 {SIGN_H}</scale></mesh></geometry>'
              f'<material><ambient>1 1 1 1</ambient><diffuse>1 1 1 1</diffuse><pbr><metal>'
              f'<albedo_map>model://mall_signs/materials/textures/{tex}</albedo_map>'
              f'<metalness>0.0</metalness><roughness>0.9</roughness></metal></pbr></material></visual>')
    model(f"sign_{shop.lower()}", cx, y, SIGN_Z, yaw, body)
    gt.append((f"sign_{shop.lower()}", f"sign:{shop}", cx, y, SIGN_Z))


# ---------------- primitive furniture ----------------
def bench(name, x, y, yaw):
    wood, stone = (0.45, 0.30, 0.18), (0.55, 0.55, 0.55)
    body = (link_shape("base", "0 0 0.20 0 0 0", geom_box(1.5, 0.42, 0.40), stone)
            + link_shape("seat", "0 0 0.425 0 0 0", geom_box(1.7, 0.50, 0.05), wood)
            + link_shape("back", "0 0.22 0.68 -0.15 0 0", geom_box(1.7, 0.05, 0.42), wood))
    model(name, x, y, 0, yaw, body)
    add_gt(name, "bench", x, y, 0.4)


def potted_plant(name, x, y):
    pot, leaf = (0.72, 0.38, 0.22), (0.15, 0.50, 0.18)
    body = (link_shape("pot", "0 0 0.25 0 0 0", "<cylinder><radius>0.25</radius><length>0.5</length></cylinder>", pot)
            + link_shape("soil", "0 0 0.51 0 0 0", "<cylinder><radius>0.22</radius><length>0.02</length></cylinder>",
                         (0.25, 0.17, 0.10), collide=False))
    for i, (dx, dy, dz, r) in enumerate(((0, 0, 0.95, 0.38), (0.18, 0.1, 0.78, 0.25), (-0.17, -0.08, 0.80, 0.24),
                                         (0.05, -0.2, 0.75, 0.22), (-0.05, 0.2, 1.2, 0.22))):
        body += link_shape(f"leaf{i}", f"{dx} {dy} {dz} 0 0 0", f"<sphere><radius>{r}</radius></sphere>", leaf,
                           collide=False)
    model(name, x, y, 0, 0, body)
    add_gt(name, "potted plant", x, y, 0.6)


# ---------------- walls + shops ----------------
wall_x("outer_north", X_MIN, X_MAX, Y_OUT)
wall_x("outer_south", X_MIN, X_MAX, -Y_OUT)
wall_y("outer_west", -Y_OUT, Y_OUT, X_MIN)
wall_y("outer_east", -Y_OUT, Y_OUT, X_MAX)

HALF_PI = math.pi / 2
for side, names, sgn in (("n", SHOPS_NORTH, 1), ("s", SHOPS_SOUTH, -1)):
    for i, shop in enumerate(names):
        x0 = X_MIN + i * SHOP_W
        x1 = x0 + SHOP_W
        cx = (x0 + x1) / 2
        y_front = sgn * Y_CORR
        wall_x(f"{side}{i}_front_l", x0, cx - DOOR_W / 2, y_front)
        wall_x(f"{side}{i}_front_r", cx + DOOR_W / 2, x1, y_front)
        box(f"{side}{i}_lintel", cx, y_front, (2.2 + WALL_H) / 2, DOOR_W, WALL_T, WALL_H - 2.2)
        if i < len(names) - 1:
            y_a, y_b = sorted((sgn * Y_CORR, sgn * Y_OUT))
            wall_y(f"{side}{i}_divider", y_a, y_b, x1)
        sign(shop, SIGN_COLOURS[(0 if side == "n" else 4) + i], cx,
             y_front - sgn * (WALL_T / 2 + 0.02 + 0.001), facing_minus_y=(sgn > 0))
        # counter at the back of every shop
        cy = sgn * (Y_OUT - 1.2)
        box(f"{side}{i}_counter", cx, cy, 0.5, 2.0, 0.6, 1.0, (0.55, 0.40, 0.30))
        add_gt(f"{side}{i}_counter", "counter", cx, cy, 0.5)

        back = sgn * (Y_OUT - WALL_T / 2)              # inner face of the back wall
        shelf_yaw = 0.0 if sgn > 0 else math.pi         # bookshelf opening faces the room
        cab_yaw = HALF_PI if sgn > 0 else -HALF_PI      # cabinet back plate against the wall
        if shop in ("PHARMACY", "FASHION", "ELECTRONICS"):
            for k, x in enumerate((x0 + 0.7, x0 + 1.25, x1 - 1.25, x1 - 0.7)):
                y = back - sgn * 0.25
                include("cabinet", f"{side}{i}_cabinet_{k}", x, y, cab_yaw)
                add_gt(f"{side}{i}_cabinet_{k}", "cabinet", x, y, 0.51)
        if shop in ("BOOKSHOP", "SHOES", "TOYS"):
            for k, x in enumerate((cx - 1.6, cx, cx + 1.6)):
                y = back - sgn * 0.01
                include("bookshelf", f"{side}{i}_bookshelf_{k}", x, y, shelf_yaw)
                add_gt(f"{side}{i}_bookshelf_{k}", "bookshelf", x, y, 0.6, local=(0, -0.195), yaw=shelf_yaw)
        if shop in ("CAFE", "BAKERY"):
            for k, x in enumerate((cx - 1.3, cx + 1.3)):
                y = sgn * 3.9
                include("cafe_table", f"{side}{i}_cafe_table_{k}", x, y)
                add_gt(f"{side}{i}_cafe_table_{k}", "dining table", x, y, 0.38)
        if shop == "ELECTRONICS":
            y = sgn * 4.2
            include("table", f"{side}{i}_display_table", cx, y)
            add_gt(f"{side}{i}_display_table", "table", cx, y, 0.5)

# ---------------- corridor ----------------
for k, x in enumerate((-8.0, 1.0, 8.5)):
    bench(f"bench_{k}", x, 1.3, 0.0)                    # backrest towards the shop wall
for k, x in enumerate((-4.5, 5.0)):
    potted_plant(f"plant_{k}", x, -1.4)
for k, x in enumerate((-1.5, 10.5)):
    include("first_2015_trash_can", f"trash_can_{k}", x, -1.6)
    add_gt(f"trash_can_{k}", "trash can", x, -1.6, 0.37)
for k, (x, y, yaw) in enumerate(((-6.0, -0.7, 1.2), (3.5, 0.6, -2.0))):
    include("person_standing", f"person_{k}", x, y, yaw)
    add_gt(f"person_{k}", "person", x, y, 0.9)            # dynamic class: must NOT end up in the notes

world = f"""<?xml version="1.0"?>
<!-- AUTO-GENERATED by tools/make_mall_v2.py - edit the script, not this file.
     Needs the furniture models: run tools/fetch_models.py once. -->
<sdf version="1.8">
  <world name="mall_v2">
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
      <ambient>0.65 0.65 0.65 1</ambient>
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
    <light type="directional" name="fill">
      <cast_shadows>false</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.35 0.35 0.35 1</diffuse>
      <specular>0 0 0 1</specular>
      <direction>0.3 -0.2 -0.6</direction>
    </light>
    <model name="floor"><static>true</static><link name="link">
      <collision name="c"><geometry><plane><normal>0 0 1</normal><size>30 20</size></plane></geometry></collision>
      <visual name="v"><geometry><plane><normal>0 0 1</normal><size>30 20</size></plane></geometry>
        <material><ambient>0.75 0.73 0.70 1</ambient><diffuse>0.75 0.73 0.70 1</diffuse></material></visual>
    </link></model>
{''.join(parts)}
  </world>
</sdf>
"""
with open(os.path.join(PKG, "worlds", "mall_v2.sdf"), "w") as f:
    f.write(world)
with open(os.path.join(PKG, "config", "mall_v2_ground_truth.csv"), "w") as f:
    f.write("model,label,x,y,z\n")
    for row in gt:
        f.write("{},{},{:.3f},{:.3f},{:.3f}\n".format(*row))
print(f"mall_v2.sdf: {len(parts)} models, {len(gt)} ground-truth rows, sign font: {FONT}")
