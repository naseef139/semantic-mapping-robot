#!/usr/bin/env python3
"""
colorlog - colour ROS 2 launch output so each node is easy to spot.

  - every node/process name gets its own fixed colour (same colour every run)
  - [WARN] lines yellow, [ERROR]/[FATAL] lines red and bold
  - optional: --only / --hide to filter by name (comma-separated substrings)

Use through the `mlaunch` shell function (see README):
  mlaunch mallbot_sim sim.launch.py
  mlaunch mallbot_sim nav.launch.py explore:=true
  MLAUNCH_HIDE=controller_server,velocity_smoother mlaunch mallbot_sim nav.launch.py
  MLAUNCH_ONLY=explore,bt_navigator mlaunch mallbot_sim nav.launch.py explore:=true
"""
import argparse
import re
import signal
import sys
import zlib

RESET, BOLD = "\033[0m", "\033[1m"
# readable on dark and light terminals
PALETTE = [39, 208, 41, 170, 220, 45, 202, 141, 118, 207, 33, 214, 49, 165, 190, 75]
# fixed colours for the nodes you will look at most
FIXED = {
    "slam_toolbox": 41, "sync_slam_toolbox_node": 41, "async_slam_toolbox_node": 41,
    "explore": 207, "explore_node": 207,
    "controller_server": 39, "planner_server": 45, "bt_navigator": 33, "behavior_server": 75,
    "cmd_vel_timeout": 220, "parameter_bridge": 141, "ign gazebo": 208, "ign": 208,
    "rviz2": 118, "robot_state_publisher": 190, "create": 214,
    "semantic_mapper": 51,
    "frontier_explorer": 207,
}
PREFIX = re.compile(r"^(\[(?P<proc>[^\]]+?)(?:-\d+)?\])")
SEV = re.compile(r"\[(DEBUG|INFO|WARN|WARNING|ERROR|FATAL|Wrn|Err|Dbg|Msg)\]")   # ROS 2 + Gazebo tags
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def colour_of(name):
    for key, c in FIXED.items():
        if name == key or name.startswith(key):
            return c
    return PALETTE[zlib.crc32(name.encode()) % len(PALETTE)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="show only names containing any of these (comma list)")
    ap.add_argument("--hide", default="", help="hide names containing any of these (comma list)")
    a = ap.parse_args()
    only = [s for s in a.only.split(",") if s]
    hide = [s for s in a.hide.split(",") if s]
    # Ctrl+C goes to the whole pipeline: ignore it here and keep printing until launch has shut down
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    for raw in sys.stdin:
        line = ANSI.sub("", raw.rstrip("\n"))
        m = PREFIX.match(line)
        if m and m.group("proc") in ("DEBUG", "INFO", "WARN", "WARNING", "ERROR", "FATAL"):
            m = None                       # e.g. "[INFO] [launch]: ..." from ros2 launch itself
        name = m.group("proc") if m else "launch"
        if only and not any(o in name for o in only):
            continue
        if hide and any(h in name for h in hide):
            continue
        sev = SEV.search(line)
        sev = sev.group(1) if sev else ""
        body = line[m.end():] if m else line
        c = colour_of(name)
        head = f"\033[38;5;{c}m{BOLD}{m.group(1) if m else ''}{RESET}"
        if sev in ("ERROR", "FATAL", "Err"):
            body = f"\033[91m{BOLD}{body}{RESET}"
        elif sev in ("WARN", "WARNING", "Wrn"):
            body = f"\033[93m{body}{RESET}"
        elif not m:
            body = f"\033[38;5;{c}m{body}{RESET}"
        sys.stdout.write(head + body + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        pass
