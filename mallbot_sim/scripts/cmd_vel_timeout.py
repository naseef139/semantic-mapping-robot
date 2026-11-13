#!/usr/bin/env python3
"""
cmd_vel_timeout - safety watchdog between any velocity source and the drive.

  /cmd_vel  (teleop, Nav2, ...)  ->  [this node]  ->  /cmd_vel_safe  ->  drive

- Forwards the latest /cmd_vel at a fixed rate while commands keep arriving.
- If no /cmd_vel arrives for `timeout` seconds, it sends zero velocity, so the
  robot stops (e.g. teleop key released, Nav2 crashed, Wi-Fi link lost).
Uses a monotonic wall clock for the timeout, so it still works if sim time stalls.
The real robot's ESP32 firmware should apply the same rule.
"""
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node


class CmdVelTimeout(Node):
    def __init__(self):
        super().__init__('cmd_vel_timeout')
        self.timeout = float(self.declare_parameter('timeout', 0.7).value)
        rate = float(self.declare_parameter('rate', 20.0).value)
        self.last_cmd = Twist()
        self.last_time = 0.0          # monotonic seconds of the last /cmd_vel
        self.stopped = True
        self.pub = self.create_publisher(Twist, 'cmd_vel_safe', 10)
        self.create_subscription(Twist, 'cmd_vel', self.on_cmd, 10)
        self.create_timer(1.0 / rate, self.tick)
        self.get_logger().info(f'cmd_vel -> cmd_vel_safe, timeout {self.timeout:.2f} s')

    def on_cmd(self, msg: Twist):
        self.last_cmd = msg
        self.last_time = time.monotonic()

    def tick(self):
        if time.monotonic() - self.last_time <= self.timeout:
            self.pub.publish(self.last_cmd)
            self.stopped = False
        else:
            self.pub.publish(Twist())   # zero velocity
            if not self.stopped:
                self.get_logger().info('cmd_vel timeout -> stop')
                self.stopped = True


def main():
    rclpy.init()
    node = CmdVelTimeout()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
