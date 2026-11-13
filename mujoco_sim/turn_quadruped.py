"""
Make the 4-leg robot walk forward, turn in place, and walk on a curve.

Run (from the project folder):
    mujoco_env/bin/mjpython mujoco_sim/turn_quadruped.py          # viewer
    mujoco_env/bin/python   mujoco_sim/turn_quadruped.py --test   # headless check

Builds entirely on walk_quadruped.gait_targets(t, forward, turn) - the crawl
gait itself is NOT changed; only the (forward, turn) command fed into it
changes over time here. walk_quadruped.py's own --test (straight walk) still
passes exactly as before.

Turning mechanism: left-side legs (FL, RL) and right-side legs (RR, FR) get
opposite stride commands (SIDE_SIGN in walk_quadruped.py), like a tank-steer
skid turn. Swing timing never changes, so the "3 feet always down" stability
guarantee holds through every maneuver below.

Demo schedule (seconds):
   0-4   walk forward
   4-9   turn in place (one direction)
   9-13  walk forward again
  13-18  turn in place (the other direction)
  18-24  walk forward while turning (an arc)
"""

import os
import sys
import time

import numpy as np
import mujoco

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import walk_quadruped as W  # noqa: E402  (reuses load()/gait_targets() as-is)


def command(t):
    """(forward, turn) demo schedule. See module docstring for the timeline."""
    if t < 4:
        return 1.0, 0.0
    if t < 9:
        return 0.0, 1.0
    if t < 13:
        return 1.0, 0.0
    if t < 18:
        return 0.0, -1.0
    return 1.0, 0.4


def heading_deg(data, trunk):
    """World-frame heading (degrees) of the trunk's local +Z (forward) axis,
    projected onto the horizontal (world XY) plane. Not unwrapped."""
    R = data.xmat[trunk].reshape(3, 3)
    fwd = R[:, 2]                      # local +Z (forward) in world coords
    return np.degrees(np.arctan2(fwd[1], fwd[0]))


def headless_test(seconds=24.0):
    model, data = W.load()
    trunk = model.body("trunk").id
    start_pos = data.xpos[trunk].copy()
    heading = []
    boundaries = [0, 4, 9, 13, 18, seconds]
    print(f"{'t (s)':>6} {'x (cm)':>8} {'y (cm)':>8} {'heading (deg)':>13} "
          f"{'tilt (deg)':>10} {'max torque':>10}")
    max_tilt = 0.0
    next_report = 1.0
    while data.time < seconds:
        fwd, turn = command(data.time)
        data.ctrl[:] = W.gait_targets(data.time, forward=fwd, turn=turn)
        mujoco.mj_step(model, data)
        heading.append((data.time, heading_deg(data, trunk)))
        up = data.xmat[trunk].reshape(3, 3)[:, 1]
        tilt = np.degrees(np.arccos(np.clip(up[2], -1, 1)))
        max_tilt = max(max_tilt, tilt)
        if data.time >= next_report:
            p = data.xpos[trunk] - start_pos
            print(f"{data.time:6.1f} {p[0]*100:8.2f} {p[1]*100:8.2f} "
                  f"{heading_deg(data, trunk):13.1f} {tilt:10.2f} "
                  f"{np.abs(data.actuator_force).max():10.3f}")
            next_report += 1.0

    # unwrap heading so turns > 180 deg read correctly
    ts = np.array([h[0] for h in heading])
    hs = np.unwrap(np.radians([h[1] for h in heading]))
    hs = np.degrees(hs)

    def heading_at(t_target):
        i = np.searchsorted(ts, t_target)
        return hs[min(i, len(hs) - 1)]

    def pos_at(t_target):
        i = np.searchsorted(ts, t_target)
        return i

    print("\n--- segment summary ---")
    labels = ["walk forward", "turn (A)", "walk forward", "turn (B)", "walk + turn (arc)"]
    for lo, hi, label in zip(boundaries[:-1], boundaries[1:], labels):
        dh = heading_at(min(hi, seconds)) - heading_at(lo)
        print(f"  [{lo:>4.0f}-{hi:<4.0f}] {label:20s} heading change {dh:+7.1f} deg "
              f"({dh / max(hi - lo, 1e-9):+5.1f} deg/s)")

    net = hs[-1] - hs[0]
    final = data.xpos[trunk] - start_pos
    print(f"\nnet heading change over {seconds:.0f} s: {net:+.1f} deg")
    print(f"final displacement: ({final[0]*100:+.1f}, {final[1]*100:+.1f}) cm, "
          f"max tilt {max_tilt:.1f} deg, max servo torque "
          f"{np.abs(data.actuator_force).max():.3f} N*m (SG90 stall "
          f"{model.actuator_forcerange[0, 1]:.3f})")


def view():
    import mujoco.viewer
    model, data = W.load()
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.trackbodyid = model.body("trunk").id
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.distance, viewer.cam.elevation, viewer.cam.azimuth = 0.6, -25, 135
        while viewer.is_running():
            t0 = time.time()
            fwd, turn = command(data.time % 30.0)   # loop the demo every 30 s
            data.ctrl[:] = W.gait_targets(data.time, forward=fwd, turn=turn)
            mujoco.mj_step(model, data)
            viewer.sync()
            dt = model.opt.timestep - (time.time() - t0)
            if dt > 0:
                time.sleep(dt)


if __name__ == "__main__":
    headless_test() if "--test" in sys.argv else view()
