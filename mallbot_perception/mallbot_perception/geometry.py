"""
Camera geometry helpers (numpy only, no cv_bridge -> no numpy-version clash with ROS Humble).

Pinhole back-projection in the camera OPTICAL frame (z forward, x right, y down):
    X = (u - cx) * Z / fx,   Y = (v - cy) * Z / fy
"""
import numpy as np


def image_to_numpy(msg):
    """sensor_msgs/Image -> numpy (rgb8/bgr8 -> HxWx3 RGB uint8, 32FC1 -> HxW float32 metres,
    16UC1 -> HxW float32 metres)."""
    h, w, step = msg.height, msg.width, msg.step
    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    enc = msg.encoding.lower()
    if enc in ("rgb8", "bgr8"):
        img = buf.reshape(h, step)[:, : w * 3].reshape(h, w, 3)
        return img[..., ::-1].copy() if enc == "bgr8" else img.copy()
    if enc == "32fc1":
        return np.frombuffer(bytes(msg.data), dtype=np.float32).reshape(h, step // 4)[:, :w].copy()
    if enc in ("16uc1", "mono16"):
        return np.frombuffer(bytes(msg.data), dtype=np.uint16).reshape(h, step // 2)[:, :w].astype(np.float32) / 1000.0
    raise ValueError(f"unsupported encoding {msg.encoding}")


def depth_in_box(depth, x1, y1, x2, y2, shrink=0.3, min_d=0.2, max_d=8.0, pct=40.0, min_px=15):
    """Robust depth of the object inside a box: use the inner part of the box (less background)
    and a low-ish percentile (front surface). Returns metres or None."""
    h, w = depth.shape
    bw, bh = x2 - x1, y2 - y1
    xa, xb = int(x1 + shrink * bw), int(x2 - shrink * bw)
    ya, yb = int(y1 + shrink * bh), int(y2 - shrink * bh)
    xa, ya = max(0, xa), max(0, ya)
    xb, yb = min(w, max(xb, xa + 1)), min(h, max(yb, ya + 1))
    patch = depth[ya:yb, xa:xb]
    valid = patch[np.isfinite(patch) & (patch > min_d) & (patch < max_d)]
    if valid.size < min_px:
        return None
    return float(np.percentile(valid, pct))


def backproject(u, v, z, fx, fy, cx, cy):
    return np.array([(u - cx) * z / fx, (v - cy) * z / fy, z])


def quat_to_mat(x, y, z, w):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def apply_tf(tf, p):
    """geometry_msgs/TransformStamped (target <- source) applied to a 3-vector."""
    t, q = tf.transform.translation, tf.transform.rotation
    return quat_to_mat(q.x, q.y, q.z, q.w) @ p + np.array([t.x, t.y, t.z])
