"""
Make the 4-leg robot walk (crawl / creep gait) in MuJoCo.

Run (from the project folder):
    mujoco_env/bin/mjpython mujoco_sim/walk_quadruped.py          # viewer
    mujoco_env/bin/python   mujoco_sim/walk_quadruped.py --test   # headless check

Gait (open-loop crawl, joint space, same angle convention as the FK):
  - Only ONE leg is in the air at a time; the other three always form a
    support tripod, so the body never has to balance (statically stable).
    (A trot - two legs at a time - was tried first: with these splayed legs
    the body rolled over the diagonal, so the crawl is used instead.)
  - Swing order: RL -> FL -> RR -> FR (standard creep sequence).
  - Stance (3/4 of the cycle): foot on the ground, the coxa sweeps the foot
    backward -> pushes the body forward (toward the FL/FR end).
  - Swing (1/4 of the cycle): femur lifts the foot while the coxa brings it
    forward again.
  - All targets are cosine-shaped, so joints never jump.
  - Gait amplitude ramps in over the first second from a standing pose.
"""

import os
import sys
import time

import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
XML = os.path.join(HERE, "quadruped.xml")

# ---- gait parameters (degrees / seconds) ----
STAND_FEMUR = 20.0     # femur slightly up
STAND_TIBIA = -110.0   # tibia pointing straight down (femur + tibia = -90)
COXA_SWING = 15.0      # +- coxa sweep around the neutral angle
SPLAY = 35.0           # X-stance: front feet angled forward, rear feet back
                       # (legs straight out sideways give a support polygon
                       #  only ~7 cm long, and the body tips over)
LIFT = 20.0            # extra femur lift during swing
PERIOD = 2.0           # seconds per full gait cycle
DUTY = 0.75            # fraction of the cycle each foot is on the ground
RAMP = 1.0             # seconds to ramp the gait in

LEGS = ["FL", "RL", "RR", "FR"]   # same order as the actuators
# a leg swings while its phase is in [DUTY, 1); offsets give RL, FL, RR, FR
PHASE = {"RL": 0.75, "FL": 0.50, "RR": 0.25, "FR": 0.00}
# coxa sign so that +1 moves every foot toward trunk -Z (backward):
# mirrored legs and 180-deg-rotated legs turn the other way.
COXA_SIGN = {"FL": +1, "RL": -1, "RR": -1, "FR": +1}
# left/right side (trunk +X = left, per build_quadruped_model.py). Used only
# for turning: left and right legs get opposite stride commands so the robot
# can rotate in place (tank-steer) instead of only walking straight.
SIDE_SIGN = {"FL": +1, "RL": +1, "RR": -1, "FR": -1}
TURN_GAIN = 1.0        # how much a turn command biases stride (see gait_targets)


def leg_cycle(phi):
    """Return (u, lift_fraction) for gait phase phi in [0, 1).
    u goes -1 (foot forward) -> +1 (foot back) during stance and back during swing."""
    if phi < DUTY:                                   # stance
        return -np.cos(np.pi * phi / DUTY), 0.0
    s = (phi - DUTY) / (1 - DUTY)                    # swing progress 0 -> 1
    return np.cos(np.pi * s), np.sin(np.pi * s)


def gait_targets(t, forward=1.0, turn=0.0):
    """Servo targets (radians) for all 12 actuators at time t.

    forward: -1 (walk backward) .. 0 (stand in place) .. +1 (walk forward).
    turn:    -1 .. +1, rotates the robot in place (or on an arc when combined
             with forward) by driving left-side and right-side legs with
             opposite stride, like a tank-steer skid turn. Swing timing
             (which leg is in the air, and when) never changes, so the
             three-feet-down stability guarantee holds regardless of
             forward/turn.
    """
    ramp = min(1.0, t / RAMP)
    ctrl = []
    for leg in LEGS:
        u, lift = leg_cycle((t / PERIOD + PHASE[leg]) % 1.0)
        stride = np.clip(forward + TURN_GAIN * turn * SIDE_SIGN[leg], -1.0, 1.0)
        # neutral: front legs (F*) splayed forward (-Z sign), rear legs back
        neutral = -SPLAY if leg.startswith("F") else SPLAY
        coxa = COXA_SIGN[leg] * (neutral + COXA_SWING * u * stride * ramp)
        femur = STAND_FEMUR + LIFT * lift * ramp
        tibia = STAND_TIBIA
        ctrl += [coxa, femur, tibia]
    return np.radians(ctrl)


def load():
    model = mujoco.MjModel.from_xml_path(XML)
    data = mujoco.MjData(model)
    # start in the standing pose
    data.ctrl[:] = gait_targets(0.0)
    for i, leg in enumerate(LEGS):
        for j, a in enumerate(data.ctrl[3 * i:3 * i + 3]):
            data.joint(f"{leg}_{['coxa', 'femur', 'tibia'][j]}").qpos = a
    mujoco.mj_forward(model, data)
    return model, data


def headless_test(seconds=20.0):
    model, data = load()
    trunk = model.body("trunk").id
    start = data.xpos[trunk].copy()
    print(f"{'t (s)':>6} {'x (cm)':>8} {'y (cm)':>8} {'height (cm)':>11} "
          f"{'tilt (deg)':>10} {'max servo torque':>17}")
    max_tilt, max_torque = 0.0, 0.0
    while data.time < seconds:
        data.ctrl[:] = gait_targets(data.time)
        mujoco.mj_step(model, data)
        up = data.xmat[trunk].reshape(3, 3)[:, 1]          # trunk +Y (up)
        tilt = np.degrees(np.arccos(np.clip(up[2], -1, 1)))
        max_tilt = max(max_tilt, tilt)
        max_torque = max(max_torque, np.abs(data.actuator_force).max())
        if abs(data.time % 1.0) < model.opt.timestep / 2:
            p = data.xpos[trunk] - start
            print(f"{data.time:6.1f} {p[0]*100:8.2f} {p[1]*100:8.2f} "
                  f"{data.xpos[trunk][2]*100:11.2f} {tilt:10.2f} "
                  f"{np.abs(data.actuator_force).max():14.3f} N*m")
    p = data.xpos[trunk] - start
    dist = np.linalg.norm(p[:2])
    print(f"\nwalked {dist*100:.1f} cm in {seconds:.0f} s "
          f"({dist/seconds*100:.2f} cm/s); max tilt {max_tilt:.1f} deg; "
          f"max servo torque {max_torque:.3f} N*m (SG90 stall 0.176)")


def view():
    import mujoco.viewer
    model, data = load()
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.trackbodyid = model.body("trunk").id
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.distance, viewer.cam.elevation, viewer.cam.azimuth = 0.6, -25, 135
        while viewer.is_running():
            t0 = time.time()
            data.ctrl[:] = gait_targets(data.time)
            mujoco.mj_step(model, data)
            viewer.sync()
            dt = model.opt.timestep - (time.time() - t0)
            if dt > 0:
                time.sleep(dt)


if __name__ == "__main__":
    headless_test() if "--test" in sys.argv else view()
