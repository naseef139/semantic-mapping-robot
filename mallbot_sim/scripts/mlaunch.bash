# mlaunch: `ros2 launch` with coloured, per-node logs.   Add to ~/.bashrc (once):
#   source ~/mallbot_ws/install/mallbot_sim/share/mallbot_sim/mlaunch.bash
# Use exactly like ros2 launch:
#   mlaunch mallbot_sim sim.launch.py
#   mlaunch mallbot_sim nav.launch.py explore:=true
# Filter (comma-separated parts of node names):
#   MLAUNCH_HIDE=controller_server,velocity_smoother mlaunch mallbot_sim nav.launch.py
#   MLAUNCH_ONLY=explore,bt_navigator,planner mlaunch mallbot_sim nav.launch.py explore:=true
mlaunch() {
  local filt=()
  [ -n "$MLAUNCH_ONLY" ] && filt+=(--only "$MLAUNCH_ONLY")
  [ -n "$MLAUNCH_HIDE" ] && filt+=(--hide "$MLAUNCH_HIDE")
  RCUTILS_LOGGING_BUFFERED_STREAM=0 PYTHONUNBUFFERED=1 ros2 launch "$@" 2>&1 \
    | python3 -u "$(ros2 pkg prefix mallbot_sim)/lib/mallbot_sim/colorlog.py" "${filt[@]}"
}
