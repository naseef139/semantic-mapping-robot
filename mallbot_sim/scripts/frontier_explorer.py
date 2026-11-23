#!/usr/bin/env python3
"""
frontier_explorer - MallBot's own exploration node (replaces explore_lite).

Behaviour (one goal at a time, short steps, never interrupts a running goal):
  1. wait until the global costmap and the robot pose are available and the robot sits in free space
  2. find frontiers (unknown cells next to free space the robot can reach) on /global_costmap/costmap
     -> keepout zones are lethal in that costmap, so frontiers inside shops never appear
  3. choose the NEAREST frontier (not blacklisted)
  4. if it is further than `max_step` (3 m), ask Nav2's planner for a path and drive only the
     first `max_step` metres of it (an intermediate goal); otherwise drive to the frontier itself
  5. send ONE NavigateToPose goal and WAIT for its result (succeeded / aborted / cancelled / timeout)
     - no re-sending, no pre-empting while the robot is moving
     - aborted / timed-out targets are blacklisted so they are not tried again
  6. repeat; when no frontier has been found for `done_after_s` seconds -> exploration complete,
     optionally drive back to the start pose
Markers on /explore/frontiers: small spheres = candidate frontiers, big = current target,
arrow = current (short) goal, red = blacklisted.
"""
import math
import os
import sys

import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, Twist
from nav2_msgs.action import ComputePathToPose, NavigateToPose, Spin
from nav_msgs.msg import OccupancyGrid
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from frontier_core import find_frontiers  # noqa: E402

STATUS = {GoalStatus.STATUS_SUCCEEDED: "succeeded", GoalStatus.STATUS_ABORTED: "aborted",
          GoalStatus.STATUS_CANCELED: "cancelled"}


def yaw_quat(yaw):
    return 0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)


class FrontierExplorer(Node):
    def __init__(self):
        super().__init__("frontier_explorer")
        P = self.declare_parameter
        self.max_step = float(P("max_step", 3.0).value)               # m per goal
        self.min_size = float(P("min_frontier_size", 0.5).value)      # m
        self.free_thresh = int(P("free_thresh", 50).value)            # costmap value (0..100)
        self.goal_timeout = float(P("goal_timeout", 60.0).value)      # s per goal
        self.blacklist_r = float(P("blacklist_radius", 0.7).value)    # m
        self.done_after = float(P("done_after_s", 15.0).value)        # s without frontiers -> done
        self.return_home = bool(P("return_to_start", True).value)
        self.map_frame = P("map_frame", "map").value
        self.base_frame = P("robot_base_frame", "base_link").value
        rate = float(P("rate_hz", 1.0).value)
        P("look_around", False)                                        # read at run time (ros2 param set)
        self.stuck_time = float(P("stuck_time_s", 15.0).value)        # s without getting closer -> give up goal
        self.near_enough = float(P("near_enough", 0.35).value)        # m: close enough counts as reached
        self.near_hit = False
        self.stuck_dist = float(P("stuck_dist", 0.25).value)          # m that counts as progress
        self.plan_timeout = float(P("plan_timeout_s", 10.0).value)    # s to wait for the planner
        self.clear = float(P("goal_clearance", 0.3).value)            # m of clear space around a goal
        self.safe_cost = int(P("safe_cost", 60).value)                # costmap value that counts as "not clear"      
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.VOLATILE, history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(OccupancyGrid, P("costmap_topic", "/global_costmap/costmap").value,
                                 self.on_costmap, qos)
        self.pub_mk = self.create_publisher(MarkerArray, "/explore/frontiers", 5)
        self.pub_cmd = self.create_publisher(Twist, "/cmd_vel", 10)
        self.esc_dir, self.esc_until = 1.0, 0.0        
        self.nav = ActionClient(self, NavigateToPose, "navigate_to_pose")
        self.plan = ActionClient(self, ComputePathToPose, "compute_path_to_pose")
        self.spin = ActionClient(self, Spin, "spin")        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)

        self.grid = None
        self.info = None
        self.state = "WAIT"          # WAIT, IDLE, PLANNING, DRIVING, RETURNING, DONE
        self.blacklist = []          # [(x, y)]
        self.target = None           # frontier we are heading for (x, y)
        self.goal = None             # current short goal (x, y, yaw)
        self.goal_handle = None
        self.goal_sent_at = 0.0
        self.no_frontier_since = None
        self.start_pose = None
        self.n_goals = self.n_ok = self.n_fail = 0
        self.candidates = []
        self.goal_is_frontier = False   # True if the current goal is the frontier itself (not a short step)
        self.visits = {}             # frontier reached but still there -> count; 2 = blacklist
        self.prog_xy, self.prog_t = None, 0.0   # last position that counted as progress
        self.plan_sent_at = self.look_started = 0.0
        self.n_looks = 0        
        self.create_timer(1.0 / rate, self.tick)
        self.get_logger().info(f"frontier explorer: max_step {self.max_step} m, goal timeout {self.goal_timeout} s, "
                               f"one goal at a time (no pre-emption)")

    # ------------------------------------------------------------------ inputs
    def on_costmap(self, msg):
        self.info = msg.info
        self.grid = np.array(msg.data, dtype=np.int16).reshape(msg.info.height, msg.info.width)

    def robot_xy(self):
        try:
            t = self.tf_buffer.lookup_transform(self.map_frame, self.base_frame, Time())
        except Exception:
            return None
        q = t.transform.rotation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        return t.transform.translation.x, t.transform.translation.y, yaw

    def to_rc(self, x, y):
        i = self.info
        return int((y - i.origin.position.y) / i.resolution), int((x - i.origin.position.x) / i.resolution)

    def to_xy(self, r, c):
        i = self.info
        return (i.origin.position.x + (c + 0.5) * i.resolution, i.origin.position.y + (r + 0.5) * i.resolution)

    def blacklisted(self, x, y):
        return any(math.hypot(x - bx, y - by) < self.blacklist_r for bx, by in self.blacklist)

    def now(self):
        """Seconds on the node clock = SIM time (use_sim_time). All timeouts use it, so the explorer
        behaves the same at any real-time factor (wall time made a 15 s wait ~4 sim-s at RTF 0.28, run24)."""
        return self.get_clock().now().nanoseconds * 1e-9

    # ------------------------------------------------------------------ main loop
    def tick(self):
        now = self.now()
        if self.state == "DONE":
            return
        if self.state == "ESCAPING":
            pose = self.robot_xy()
            t = Twist()
            if now < self.esc_until and pose and self.trapped(pose):
                t.linear.x = 0.1 * self.esc_dir             # slowly, straight, no turning
                self.pub_cmd.publish(t)
            else:
                self.pub_cmd.publish(t)                     # stop
                self.state = "IDLE"
            return            
        if self.state == "LOOKING":
            if now - self.look_started > 30.0:
                self.state = "IDLE"
            return
        if self.state == "PLANNING":
            if now - self.plan_sent_at > self.plan_timeout:
                self.fail_target(f"planner gave no answer in {self.plan_timeout:.0f} s")
            return
        if self.state in ("DRIVING", "RETURNING"):
            if self.state == "DRIVING" and self.goal_handle:
                pose = self.robot_xy()
                d_goal = math.hypot(pose[0] - self.goal[0], pose[1] - self.goal[1]) if pose else None
                if d_goal is not None and (self.prog_xy is None or d_goal < self.prog_xy - self.stuck_dist):
                    self.prog_xy, self.prog_t = d_goal, now            # best distance to goal so far: progress
                if d_goal is not None and d_goal < self.near_enough:
                    self.get_logger().info(f"goal #{self.n_goals} close enough ({d_goal:.2f} m) -> next")
                    self.near_hit = True
                    self.goal_handle.cancel_goal_async()
                    self.goal_handle = None
                    return
                why = None
                if now - self.goal_sent_at > self.goal_timeout:
                    why = f"goal timed out after {self.goal_timeout:.0f} s"
                elif now - self.prog_t > self.stuck_time:
                    why = f"no progress for {self.stuck_time:.0f} s (stuck)"
                if why:
                    self.get_logger().warn(f"{why} -> cancel + blacklist target")
                    self.goal_handle.cancel_goal_async()
                    self.goal_handle = None      # result callback will still arrive (cancelled)
            return
        pose = self.robot_xy()
        if pose is None or self.grid is None:
            if self.state == "WAIT":
                self.get_logger().info("waiting for costmap + robot pose ...", throttle_duration_sec=5.0)
            return
        if self.start_pose is None:
            self.start_pose = pose
        rx, ry, _ = pose
        fr, reach = find_frontiers(self.grid, self.to_rc(rx, ry), self.info.resolution,
                                   self.free_thresh, self.min_size, self.clear, self.safe_cost)
        if self.state == "WAIT":
            if reach.sum() < 50:
                self.get_logger().info("waiting for the costmap around the robot to fill ...", throttle_duration_sec=5.0)
                return
            self.state = "IDLE"
            self.get_logger().info("costmap ready, exploring")
        cands = []
        for f in fr:
            gx, gy = self.to_xy(*f["goal_rc"])
            if self.blacklisted(gx, gy):
                continue
            cands.append((math.hypot(gx - rx, gy - ry), gx, gy, f))
        cands.sort(key=lambda c: c[0])                     # NEAREST first
        self.candidates = [(c[1], c[2]) for c in cands]
        if not cands:
            now = self.now()
            self.no_frontier_since = self.no_frontier_since or now
            left = self.done_after - (now - self.no_frontier_since)
            if left > 0:
                self.get_logger().info(f"no frontier right now, re-checking ({left:.0f} s left)", throttle_duration_sec=5.0)
                self.publish_markers()
                return
            self.get_logger().info(f"EXPLORATION COMPLETE: {self.n_goals} goals, {self.n_ok} reached, "
                                   f"{self.n_fail} failed, {len(self.blacklist)} blacklisted")
            if self.return_home and self.start_pose:
                self.state = "RETURNING"
                self.send_goal(*self.start_pose, returning=True)
            else:
                self.state = "DONE"
            return
        self.no_frontier_since = None
        dist, gx, gy, f = cands[0]
        cx, cy = self.to_xy(*f["centroid_rc"])
        self.target = (gx, gy)
        face = math.atan2(cy - ry, cx - rx)                 # look towards the unknown area
        if dist <= self.max_step:
            self.goal_is_frontier = True
            self.send_goal(gx, gy, face)
        else:
            self.goal_is_frontier = False
            self.request_path(gx, gy, face)
        self.publish_markers()

    # ------------------------------------------------------------------ planning a short step
    def request_path(self, gx, gy, yaw):
        if not self.plan.wait_for_server(timeout_sec=0.5):
            self.get_logger().warn("planner action not available yet", throttle_duration_sec=5.0)
            return
        g = ComputePathToPose.Goal()
        g.goal = self.make_pose(gx, gy, yaw)
        g.use_start = False
        self.state = "PLANNING"
        self.plan_sent_at = self.now()        
        fut = self.plan.send_goal_async(g)
        fut.add_done_callback(self.on_plan_accepted)

    def on_plan_accepted(self, fut):
        gh = fut.result()
        if not gh.accepted:
            self.fail_target("planner rejected the request")
            return
        gh.get_result_async().add_done_callback(self.on_plan_result)

    def on_plan_result(self, fut):
        res = fut.result()
        if self.state != "PLANNING":
            return                               # answer came too late, target already given up        
        poses = res.result.path.poses if res.status == GoalStatus.STATUS_SUCCEEDED else []
        if len(poses) < 2:
            self.fail_target("no path to the frontier")
            return
        # walk along the path until max_step metres, use that pose as the short goal
        acc, pick = 0.0, poses[-1]
        for a, b in zip(poses[:-1], poses[1:]):
            acc += math.hypot(b.pose.position.x - a.pose.position.x, b.pose.position.y - a.pose.position.y)
            if acc >= self.max_step:
                pick = b
                break
        px, py = pick.pose.position.x, pick.pose.position.y
        q = pick.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        self.state = "IDLE"
        self.send_goal(px, py, yaw)

    # ------------------------------------------------------------------ driving
    def make_pose(self, x, y, yaw):
        p = PoseStamped()
        p.header.frame_id = self.map_frame
        p.header.stamp = self.get_clock().now().to_msg()
        p.pose.position.x, p.pose.position.y = x, y
        p.pose.orientation.x, p.pose.orientation.y, p.pose.orientation.z, p.pose.orientation.w = yaw_quat(yaw)
        return p

    def send_goal(self, x, y, yaw, returning=False):
        if not self.nav.wait_for_server(timeout_sec=0.5):
            self.get_logger().warn("navigate_to_pose not available yet", throttle_duration_sec=5.0)
            if not returning:
                self.state = "IDLE"
            return
        g = NavigateToPose.Goal()
        g.pose = self.make_pose(x, y, yaw)
        self.goal = (x, y, yaw)
        self.n_goals += 0 if returning else 1
        self.state = "RETURNING" if returning else "DRIVING"
        self.goal_sent_at = self.now()
        self.prog_xy, self.prog_t = None, self.goal_sent_at        
        r = self.robot_xy()
        d = math.hypot(x - r[0], y - r[1]) if r else float("nan")
        self.get_logger().info(("returning to start" if returning else f"goal #{self.n_goals}") +
                               f": ({x:.2f}, {y:.2f}), {d:.1f} m away")
        self.publish_markers()
        self.nav.send_goal_async(g).add_done_callback(self.on_goal_accepted)

    def on_goal_accepted(self, fut):
        gh = fut.result()
        if not gh.accepted:
            if self.state == "RETURNING":
                self.state = "DONE"
            else:
                self.fail_target("goal rejected")
            return
        self.goal_handle = gh
        gh.get_result_async().add_done_callback(self.on_goal_result)

    def on_goal_result(self, fut):
        status = fut.result().status
        if self.near_hit:                        # we cancelled it ourselves because it was close enough
            status, self.near_hit = GoalStatus.STATUS_SUCCEEDED, False        
        self.goal_handle = None
        txt = STATUS.get(status, str(status))
        if self.state == "RETURNING":
            self.get_logger().info(f"back at the start ({txt}). Done.")
            self.state = "DONE"
            return
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.n_ok += 1
            self.get_logger().info(f"goal #{self.n_goals} {txt}")
            self.state = "IDLE"
            if self.goal_is_frontier and self.target:
                # reached the frontier itself; if it is still a frontier after 2 visits it can't be
                # seen from here (e.g. unknown pocket behind furniture) -> stop coming back
                key = (round(self.target[0] / 0.5), round(self.target[1] / 0.5))
                self.visits[key] = self.visits.get(key, 0) + 1
                if self.visits[key] >= 2:
                    self.blacklist.append(self.target)
                    self.get_logger().info("frontier still unknown after 2 visits -> blacklisted")
            if self.get_parameter("look_around").value:
                self.look_around()
        else:
            self.fail_target(f"goal #{self.n_goals} {txt}")

    def cost_at(self, x, y):
        r, c = self.to_rc(x, y)
        h, w = self.grid.shape
        return int(self.grid[r, c]) if 0 <= r < h and 0 <= c < w else 100

    # Nav2 footprint in base_link (= nav.launch.py FOOTPRINT): the tail reaches 0.28 m behind the
    # front axle, so the centre can be free while the tail sits on a wall (Smac then refuses to plan:
    # "Starting point in lethal space").
    FP_X = (-0.28, 0.10)
    FP_Y = (-0.17, 0.17)

    def footprint_cost(self, x, y, th):
        """Highest costmap value under the robot footprint at pose (x, y, th)."""
        c, s = math.cos(th), math.sin(th)
        step = self.info.resolution
        worst = 0
        for fx in np.arange(self.FP_X[0], self.FP_X[1] + 1e-6, step):
            for fy in np.arange(self.FP_Y[0], self.FP_Y[1] + 1e-6, step):
                worst = max(worst, self.cost_at(x + c * fx - s * fy, y + s * fx + c * fy))
        return worst

    def trapped(self, pose):
        """Centre in the no-go band, or any part of the footprint on an inscribed/lethal cell."""
        return self.cost_at(pose[0], pose[1]) >= 90 or self.footprint_cost(*pose) >= 99

    def start_escape(self):
        """If the robot stands in the no-go band (e.g. squeezed behind a bench) or its footprint
        touches a wall (tail swung into a door jamb), drive straight out (no turning = no tail swing)."""
        pose = self.robot_xy()
        if pose is None or self.grid is None or not self.trapped(pose):
            return
        x, y, th = pose
        def worst(sign):     # footprint cost after moving 0.1-0.3 m straight forward (+1) / back (-1)
            return min(self.footprint_cost(x + sign * d * math.cos(th), y + sign * d * math.sin(th), th)
                       for d in (0.1, 0.2, 0.3))
        fwd, back = worst(1), worst(-1)
        if min(fwd, back) >= 100:
            self.get_logger().warn("robot footprint on a lethal cell, but no free straight move -> no escape")
            return
        self.esc_dir = 1.0 if fwd <= back else -1.0
        self.esc_until = self.now() + 6.0
        self.state = "ESCAPING"
        self.get_logger().warn("robot is inside the no-go band -> escaping " +
                               ("forward" if self.esc_dir > 0 else "backward"))

    def look_around(self):
        """Yamauchi's 360-degree sweep, for the camera: one full turn on the spot (Nav2 Spin)."""
        if not self.spin.wait_for_server(timeout_sec=0.5):
            self.get_logger().warn("spin action not available, skipping look-around")
            return
        g = Spin.Goal()
        g.target_yaw = 2 * math.pi
        g.time_allowance = Duration(seconds=20).to_msg()
        self.state = "LOOKING"
        self.look_started = self.now()
        self.n_looks += 1
        self.get_logger().info(f"look-around #{self.n_looks}")
        self.spin.send_goal_async(g).add_done_callback(self.on_spin_accepted)

    def on_spin_accepted(self, fut):
        gh = fut.result()
        if not gh.accepted:
            self.state = "IDLE"
            return
        gh.get_result_async().add_done_callback(lambda f: setattr(self, "state", "IDLE"))
        
    def fail_target(self, why):
        self.n_fail += 1
        if self.target:
            self.blacklist.append(self.target)
        self.get_logger().warn(f"{why} -> target blacklisted ({len(self.blacklist)} so far)")
        self.state = "IDLE"
        self.start_escape()

    # ------------------------------------------------------------------ RViz
    def publish_markers(self):
        arr = MarkerArray()
        clr = Marker()
        clr.action = Marker.DELETEALL
        arr.markers.append(clr)
        stamp = self.get_clock().now().to_msg()

        def sphere(i, x, y, s, rgb, a=0.9, ns="frontiers"):
            m = Marker()
            m.header.frame_id, m.header.stamp, m.ns, m.id = self.map_frame, stamp, ns, i
            m.type, m.action = Marker.SPHERE, Marker.ADD
            m.pose.position.x, m.pose.position.y, m.pose.position.z = x, y, 0.1
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = s
            m.color.r, m.color.g, m.color.b, m.color.a = (*rgb, a)
            return m
        i = 0
        for x, y in self.candidates:
            arr.markers.append(sphere(i, x, y, 0.15, (0.2, 0.6, 1.0), 0.7)); i += 1
        for x, y in self.blacklist:
            arr.markers.append(sphere(i, x, y, 0.2, (1.0, 0.1, 0.1), 0.8, "blacklisted")); i += 1
        if self.target:
            arr.markers.append(sphere(i, *self.target, 0.35, (0.1, 0.9, 0.3), 0.9, "target")); i += 1
        if self.goal:
            m = Marker()
            m.header.frame_id, m.header.stamp, m.ns, m.id = self.map_frame, stamp, "goal", i
            m.type, m.action = Marker.ARROW, Marker.ADD
            m.pose = self.make_pose(*self.goal).pose
            m.pose.position.z = 0.15
            m.scale.x, m.scale.y, m.scale.z = 0.5, 0.08, 0.08
            m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.8, 0.0, 1.0
            arr.markers.append(m)
        self.pub_mk.publish(arr)


def main():
    rclpy.init()
    node = FrontierExplorer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
