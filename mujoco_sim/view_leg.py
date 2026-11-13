"""
Open the single-leg MuJoCo model in the interactive viewer.

Run (from the project folder):
    mujoco_env/bin/python mujoco_sim/view_leg.py            # manual control
    mujoco_env/bin/mjpython mujoco_sim/view_leg.py --demo   # scripted motion

Manual control (default):
  - Right panel -> "Control": three sliders (coxa_servo, femur_servo,
    tibia_servo) are the SG90 position servos. Values are target angles in
    radians (-1.57 .. 1.57 = -90 .. +90 deg), same sign convention as the FK.
  - Mouse: left-drag rotate, right-drag pan, scroll zoom.
  - Left panel -> "Rendering": tick Group 3 to see the simple collision shapes.
  - Load keyframe "fk_initial_45_45_30" (Simulation panel) to see the same
    pose as the Matplotlib visualizer's initial angles.

--demo: smooth sine-wave targets on all three servos (same idea as the
Matplotlib animation). On macOS this needs `mjpython` (bundled in the venv).
"""

import os
import sys
import time

import numpy as np
import mujoco
import mujoco.viewer

HERE = os.path.dirname(os.path.abspath(__file__))
XML = os.path.join(HERE, "quadruped_leg.xml")


def demo_targets(t):
    """Smooth periodic servo targets (radians)."""
    w = 2 * np.pi / 4.0
    return np.radians([
        20 * np.sin(w * t),
        40 + 25 * np.sin(w * t + np.pi / 2),
        20 + 30 * np.sin(w * t + np.pi),
    ])


def main():
    model = mujoco.MjModel.from_xml_path(XML)
    data = mujoco.MjData(model)

    if "--demo" not in sys.argv:
        # Managed viewer: built-in Control sliders drive the actuators.
        mujoco.viewer.launch(model, data)
        return

    # Passive viewer: we step the physics and set ctrl ourselves.
    with mujoco.viewer.launch_passive(model, data) as viewer:
        start = time.time()
        while viewer.is_running():
            step_start = time.time()
            data.ctrl[:] = demo_targets(data.time)
            mujoco.mj_step(model, data)
            viewer.sync()
            # run roughly in real time
            dt = model.opt.timestep - (time.time() - step_start)
            if dt > 0:
                time.sleep(dt)


if __name__ == "__main__":
    main()
