#!/usr/bin/env python3
"""auto_keepout.py - builds the Nav2 keepout mask online, with no preset map.

While the robot explores, this node watches the SLAM map and
  * finds doorways (gaps between two collinear wall segments, 0.6-2.5 m wide),
  * finds wide-open shop fronts (up to 8 m) when a shop SIGN from the semantic
    mapper sits on the gap,
  * draws a virtual wall across each one and fills the room/shop behind it,
and publishes the result on /keepout_filter_mask, the topic the Nav2 KeepoutFilter
already listens to. The planner, controller and frontier explorer then treat every
shop and room as forbidden, exactly as with the old hand-made mask.

Never marked: the robot itself, its start pose and anywhere it has already driven.

Topics
  sub  /map                      nav_msgs/OccupancyGrid   (slam_toolbox)
  sub  /semantic_notes           visualization_msgs/MarkerArray  (sign positions, optional)
  pub  /keepout_filter_mask      nav_msgs/OccupancyGrid   (100 = keepout, -1 = no effect)
  pub  /auto_keepout/doorways    visualization_msgs/MarkerArray  (RViz)
"""
import dataclasses
import math

import numpy as np
import rclpy
from geometry_msgs.msg import Point
from nav_msgs.msg import OccupancyGrid
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from std_msgs.msg import ColorRGBA
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

import room_core as rc

GREEN = '\033[92m'
YELLOW = '\033[93m'
RESET = '\033[0m'


class AutoKeepout(Node):
    def __init__(self):
        super().__init__('auto_keepout')
        self.declare_parameter('map_topic', '/map')
        self.declare_parameter('mask_topic', '/keepout_filter_mask')
        self.declare_parameter('sign_topic', '/semantic_notes')
        self.declare_parameter('sign_keyword', 'sign')
        self.declare_parameter('global_frame', 'map')
        self.declare_parameter('robot_frame', 'base_link')
        self.declare_parameter('update_rate', 1.0)
        # every field of room_core.Params is also a ROS parameter (door_max, ...)
        self.P = rc.Params()
        for f in dataclasses.fields(rc.Params):
            self.declare_parameter(f.name, getattr(self.P, f.name))
            setattr(self.P, f.name, type(getattr(self.P, f.name))(self.get_parameter(f.name).value))

        gp = self.get_parameter
        self.global_frame = gp('global_frame').value
        self.robot_frame = gp('robot_frame').value
        self.keyword = str(gp('sign_keyword').value).lower()

        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.map_msg = None
        self.create_subscription(OccupancyGrid, gp('map_topic').value, self._on_map, latched)
        self.create_subscription(MarkerArray, gp('sign_topic').value, self._on_notes, 10)
        self.mask_pub = self.create_publisher(OccupancyGrid, gp('mask_topic').value, latched)
        self.viz_pub = self.create_publisher(MarkerArray, '/auto_keepout/doorways', latched)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)

        self.start_xy = None
        self.visited = []
        self.signs = []
        self.memory = []
        self.known = []          # doorways already announced in the log
        self.create_timer(1.0 / float(gp('update_rate').value), self._update)
        self.get_logger().info(
            f'auto keepout: doorways {self.P.door_min:.1f}-{self.P.door_max:.1f} m, '
            f'signed shop fronts up to {self.P.sign_door_max:.1f} m')

    # ------------------------------------------------------------- inputs
    def _on_map(self, msg):
        self.map_msg = msg

    def _on_notes(self, msg):
        for m in msg.markers:
            label = (m.text or '') + ' ' + (m.ns or '')
            if self.keyword not in label.lower():
                continue
            if m.header.frame_id and m.header.frame_id != self.global_frame:
                continue
            x, y = m.pose.position.x, m.pose.position.y
            if all(math.hypot(x - sx, y - sy) > 0.5 for sx, sy in self.signs):
                self.signs.append((x, y))
                self.get_logger().info(f'sign cue at ({x:.1f}, {y:.1f})')

    def _robot_xy(self):
        try:
            t = self.tf_buffer.lookup_transform(self.global_frame, self.robot_frame, Time(),
                                                timeout=Duration(seconds=0.1))
        except TransformException:
            return None
        return t.transform.translation.x, t.transform.translation.y

    # ------------------------------------------------------------- update
    def _update(self):
        if self.map_msg is None:
            return
        xy = self._robot_xy()
        if xy is None:
            return
        if self.start_xy is None:
            self.start_xy = xy
        if not self.visited or math.hypot(xy[0] - self.visited[-1][0], xy[1] - self.visited[-1][1]) > 0.2:
            self.visited.append(xy)

        m = self.map_msg
        info = m.info
        if info.width == 0 or info.height == 0:
            return
        grid = np.asarray(m.data, dtype=np.int16).reshape(info.height, info.width)
        g = rc.GridGeom(info.width, info.height, info.resolution,
                        info.origin.position.x, info.origin.position.y)

        lethal, doors, self.memory = rc.detect(grid, g, xy, self.start_xy, self.visited,
                                               self.signs, self.P, self.memory)

        out = OccupancyGrid()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = m.header.frame_id or self.global_frame
        out.info = info
        mask = np.full(grid.shape, -1, np.int8)      # -1 = unknown = mask has no effect
        mask[lethal.astype(bool)] = 100              # 100 = lethal = keepout
        out.data = mask.ravel().tolist()
        self.mask_pub.publish(out)
        self._publish_markers(doors, out.header)

        for d in doors:
            if not any(rc._same_door(d, k) for k in self.known):
                self.known.append(d)
                colour = GREEN if d.kind == 'door' else YELLOW
                self.get_logger().info(
                    f'{colour}closed {"doorway" if d.kind == "door" else "signed shop front"} '
                    f'at ({d.mid[0]:.1f}, {d.mid[1]:.1f}), {d.width:.1f} m wide'
                    f'{", room behind it marked" if d.separates else ""}{RESET}')

    def _publish_markers(self, doors, header):
        arr = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)
        lines = Marker()
        lines.header = header
        lines.ns = 'virtual_walls'
        lines.id = 0
        lines.type = Marker.LINE_LIST
        lines.scale.x = 0.08
        lines.pose.orientation.w = 1.0
        for i, d in enumerate(doors):
            c = ColorRGBA(r=0.9, g=0.1, b=0.1, a=1.0) if d.kind == 'door' else \
                ColorRGBA(r=0.8, g=0.1, b=0.8, a=1.0)
            lines.points += [Point(x=d.p[0], y=d.p[1], z=0.05), Point(x=d.q[0], y=d.q[1], z=0.05)]
            lines.colors += [c, c]
            t = Marker()
            t.header = header
            t.ns = 'doorway_labels'
            t.id = i + 1
            t.type = Marker.TEXT_VIEW_FACING
            t.pose.position.x, t.pose.position.y, t.pose.position.z = d.mid[0], d.mid[1], 0.4
            t.pose.orientation.w = 1.0
            t.scale.z = 0.25
            t.color = c
            t.text = f'{"door" if d.kind == "door" else "shop front"} {d.width:.1f} m'
            arr.markers.append(t)
        arr.markers.append(lines)
        self.viz_pub.publish(arr)


def main():
    rclpy.init()
    node = AutoKeepout()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
