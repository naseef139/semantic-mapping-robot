#!/usr/bin/env python3
"""
semantic_mapper - the object-note part of "Online Object-Level Semantic Mapping".

  /camera/image + /camera/depth_image + /camera/camera_info   (time-synchronised)
      -> YOLO-World (open vocabulary; or plain COCO YOLO) detections
      -> depth inside each box -> 3D point in camera_optical_frame
      -> TF to the SLAM `map` frame (at the image time stamp)
      -> NoteBook: same label within a gate = same object; confirmed after N sightings;
         dynamic classes (person) shown but NEVER stored
Outputs
  /semantic_notes     visualization_msgs/MarkerArray  (cubes + text labels in RViz, frame `map`)
  /detections/image   sensor_msgs/Image               (camera image with boxes, for RViz)
  <save_path>         JSON of confirmed notes, written every few seconds and on shutdown
"""
import os
import time

import numpy as np
import rclpy
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from .geometry import apply_tf, backproject, depth_in_box, image_to_numpy
from .notes import NoteBook

# RViz colours per label (anything else gets a hashed colour)
COLOURS = {
    "bench": (0.55, 0.35, 0.17), "potted plant": (0.15, 0.7, 0.25), "trash can": (0.4, 0.4, 0.4),
    "cabinet": (0.6, 0.6, 0.65), "bookshelf": (0.65, 0.4, 0.2), "table": (0.85, 0.65, 0.4),
    "dining table": (0.85, 0.65, 0.4), "counter": (0.7, 0.5, 0.35), "shop sign": (0.2, 0.5, 1.0),
    "chair": (0.9, 0.5, 0.1), "person": (1.0, 0.1, 0.1),
}


def colour(label):
    if label in COLOURS:
        return COLOURS[label]
    h = abs(hash(label)) % 360 / 360.0
    return (0.5 + 0.5 * np.cos(6.283 * h), 0.5 + 0.5 * np.cos(6.283 * (h + 0.33)), 0.5 + 0.5 * np.cos(6.283 * (h + 0.66)))


class SemanticMapper(Node):
    def __init__(self):
        super().__init__("semantic_mapper")
        P = self.declare_parameter
        self.model_name = P("model", "yolov8s-worldv2.pt").value
        self.classes = list(P("classes", ["person", "bench", "potted plant", "trash can", "cabinet",
                                          "bookshelf", "table", "counter", "shop sign"]).value)
        # synonyms: extra prompt words that map onto one note label ("prompt=label")
        aliases = list(P("aliases", [""]).value)
        self.alias = dict(a.split("=", 1) for a in aliases if "=" in a)
        for prompt in self.alias:                                 # make sure YOLO-World looks for them
            if prompt not in self.classes:
                self.classes.append(prompt)
        self.conf = float(P("confidence", 0.25).value)
        self.rate = float(P("max_rate_hz", 2.0).value)
        self.max_range = float(P("max_range", 6.0).value)
        self.map_frame = P("map_frame", "map").value
        self.device = P("device", "").value                      # "" = auto, "cpu", "0" (GPU)
        dyn = list(P("dynamic_classes", ["person"]).value)
        gate = float(P("gate", 0.8).value)
        cg_names = list(P("class_gate_names", ["cabinet", "bookshelf"]).value)
        cg_vals = list(P("class_gate_values", [0.35, 0.6]).value)
        min_s = int(P("min_sightings", 3).value)
        self.save_path = os.path.expanduser(P("save_path", "~/mallbot_notes/notes.json").value)
        save_every = float(P("save_every_s", 5.0).value)

        self.book = NoteBook(gate=gate, class_gates=dict(zip(cg_names, cg_vals)),
                             min_sightings=min_s, dynamic=dyn)
        self.dynamic_seen = []                                    # (x, y, z, label) this frame
        self.raw = {}            # label -> [detections, no-depth/too-far]  (diagnostics, logged every 30 s)
        self.model = self._load_model()

        self.tf_buffer = Buffer(cache_time=Duration(seconds=30))
        # spin_thread=True: TF arrives on its own thread, so lookup_transform(timeout) can actually
        # wait for the pose matching the image stamp (with the node's single thread it never could)
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)
        rgb = Subscriber(self, Image, "/camera/image", qos_profile=qos_profile_sensor_data)
        dep = Subscriber(self, Image, "/camera/depth_image", qos_profile=qos_profile_sensor_data)
        info = Subscriber(self, CameraInfo, "/camera/camera_info", qos_profile=qos_profile_sensor_data)
        self.sync = ApproximateTimeSynchronizer([rgb, dep, info], queue_size=5, slop=0.05)
        self.sync.registerCallback(self.on_frames)

        self.pub_markers = self.create_publisher(MarkerArray, "/semantic_notes", 10)
        self.pub_debug = self.create_publisher(Image, "/detections/image", 2)
        self.create_timer(1.0, self.publish_markers)
        self.create_timer(save_every, self.save)
        self.last_proc = 0.0
        self.frames = 0
        self.get_logger().info(f"model={self.model_name} classes={self.classes} conf={self.conf} "
                               f"dynamic={dyn} min_sightings={min_s} -> {self.save_path}")

    # ------------------------------------------------------------------ model
    def _load_model(self):
        try:
            if "world" in self.model_name.lower():
                from ultralytics import YOLOWorld
                m = YOLOWorld(self.model_name)
                m.set_classes(self.classes)                     # open-vocabulary prompt list
            else:
                from ultralytics import YOLO
                m = YOLO(self.model_name)                       # plain COCO model
        except ImportError:
            self.get_logger().fatal("ultralytics not installed:  pip3 install --user ultralytics 'numpy<2'")
            raise
        self.get_logger().info(f"loaded {self.model_name}")
        return m

    # ------------------------------------------------------------------ main loop
    def on_frames(self, img_msg, depth_msg, info_msg):
        now = time.monotonic()
        if now - self.last_proc < 1.0 / self.rate:
            return
        self.last_proc = now
        tf = self._lookup(img_msg)
        if tf is None:
            return
        rgb = image_to_numpy(img_msg)
        depth = image_to_numpy(depth_msg)
        if depth.shape[:2] != rgb.shape[:2]:
            self.get_logger().warn("RGB and depth sizes differ", throttle_duration_sec=10.0)
            return
        fx, fy, cx, cy = info_msg.k[0], info_msg.k[4], info_msg.k[2], info_msg.k[5]

        kw = {"conf": self.conf, "verbose": False}
        if self.device:
            kw["device"] = self.device
        res = self.model.predict(rgb[..., ::-1], **kw)[0]        # ultralytics wants BGR arrays
        names = res.names
        t = time.time()
        self.dynamic_seen = []
        drawn = []
        for b in res.boxes:
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
            label = names[int(b.cls[0])]
            label = self.alias.get(label, label)                 # synonym -> note label
            conf = float(b.conf[0])
            z = depth_in_box(depth, x1, y1, x2, y2, max_d=self.max_range)
            st = self.raw.setdefault(label, [0, 0])
            st[0] += 1
            if z is None:
                st[1] += 1
                drawn.append((x1, y1, x2, y2, label, conf, None, False))
                continue
            p_cam = backproject((x1 + x2) / 2, (y1 + y2) / 2, z, fx, fy, cx, cy)
            p_map = apply_tf(tf, p_cam)
            if self.book.is_dynamic(label):
                self.dynamic_seen.append((*p_map, label))
                drawn.append((x1, y1, x2, y2, label, conf, z, True))
            else:
                self.book.observe(label, *p_map, conf=conf, t=t)
                drawn.append((x1, y1, x2, y2, label, conf, z, False))
        self.frames += 1
        if self.frames % 10 == 0:
            self.book.merge()
        self._publish_debug(img_msg, rgb, drawn)

    def _lookup(self, img_msg):
        """map <- camera pose at the image time. Waits up to 0.3 s for TF to catch up; if the image
        is only a few ms newer than the newest TF, uses the newest TF (error < 1 cm at 0.3 m/s)."""
        frame, stamp = img_msg.header.frame_id, Time.from_msg(img_msg.header.stamp)
        try:
            return self.tf_buffer.lookup_transform(self.map_frame, frame, stamp, Duration(seconds=0.3))
        except Exception as e:
            err = e
        try:
            latest = self.tf_buffer.lookup_transform(self.map_frame, frame, Time())
            age = abs((stamp - Time.from_msg(latest.header.stamp)).nanoseconds) * 1e-9
            if age <= 0.03:
                return latest
            err = f"TF is {age:.3f} s away from the image time"
        except Exception as e2:
            err = e2                                             # no map yet / TF not ready
        self.get_logger().warn(f"no TF {self.map_frame}<-{frame}: {err}", throttle_duration_sec=5.0)
        return None

    # ------------------------------------------------------------------ outputs
    def _publish_debug(self, img_msg, rgb, drawn):
        if self.pub_debug.get_subscription_count() == 0:
            return
        try:
            import cv2
        except ImportError:
            return
        out = np.ascontiguousarray(rgb.copy())
        for x1, y1, x2, y2, label, conf, z, dyn in drawn:
            c = (255, 40, 40) if dyn else tuple(int(255 * v) for v in colour(label))
            cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), c, 2)
            txt = f"{label} {conf:.2f}" + (f" {z:.1f}m" if z else " no depth") + (" [ignored]" if dyn else "")
            cv2.putText(out, txt, (int(x1), max(12, int(y1) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)
        msg = Image()
        msg.header = img_msg.header
        msg.height, msg.width = out.shape[:2]
        msg.encoding, msg.step = "rgb8", out.shape[1] * 3
        msg.data = out.tobytes()
        self.pub_debug.publish(msg)

    def publish_markers(self):
        arr = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)
        stamp = self.get_clock().now().to_msg()
        mid = 0
        for n in self.book.notes:
            ok = n.n >= self.book.min_sightings
            r, g, b = colour(n.label)
            m = Marker()
            m.header.frame_id, m.header.stamp = self.map_frame, stamp
            m.ns, m.id = "notes" if ok else "candidates", mid
            m.type, m.action = Marker.CUBE, Marker.ADD
            m.pose.position.x, m.pose.position.y, m.pose.position.z = n.x, n.y, max(n.z, 0.15)
            m.pose.orientation.w = 1.0
            s = 0.35 if ok else 0.2
            m.scale.x = m.scale.y = m.scale.z = s
            m.color.r, m.color.g, m.color.b, m.color.a = r, g, b, (0.9 if ok else 0.35)
            arr.markers.append(m)
            mid += 1
            if ok:
                t = Marker()
                t.header = m.header
                t.ns, t.id = "labels", mid
                t.type, t.action = Marker.TEXT_VIEW_FACING, Marker.ADD
                t.pose.position.x, t.pose.position.y, t.pose.position.z = n.x, n.y, max(n.z, 0.15) + 0.45
                t.pose.orientation.w = 1.0
                t.scale.z = 0.28
                t.color.r = t.color.g = t.color.b = t.color.a = 1.0
                t.text = f"{n.label} #{n.id} ({n.n})" + (f" '{n.text}'" if n.text else "")
                arr.markers.append(t)
                mid += 1
        for (x, y, z, label) in self.dynamic_seen:           # people: shown red, never stored
            m = Marker()
            m.header.frame_id, m.header.stamp = self.map_frame, stamp
            m.ns, m.id = "dynamic_ignored", mid
            m.type, m.action = Marker.SPHERE, Marker.ADD
            m.pose.position.x, m.pose.position.y, m.pose.position.z = x, y, max(z, 0.3)
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.4
            m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.1, 0.1, 0.6
            m.lifetime = Duration(seconds=1.5).to_msg()
            arr.markers.append(m)
            mid += 1
        self.pub_markers.publish(arr)

    def save(self):
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        n = self.book.to_json(self.save_path, meta={"frame": self.map_frame, "model": self.model_name,
                                                    "saved": time.strftime("%Y-%m-%d %H:%M:%S")})
        raw = ", ".join(f"{k} {v[0]}" + (f" ({v[1]} no depth/too far)" if v[1] else "")
                        for k, v in sorted(self.raw.items(), key=lambda kv: -kv[1][0]))
        self.get_logger().info(f"{n} confirmed notes ({len(self.book.notes)} candidates) -> {self.save_path}"
                               f" | raw detections so far: {raw or 'none'}", throttle_duration_sec=30.0)


def main():
    rclpy.init()
    node = SemanticMapper()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.book.merge()
        node.save()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
