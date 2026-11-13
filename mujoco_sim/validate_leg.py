"""
Validate the MuJoCo leg against the existing NumPy FK (numpyfk3d.Leg).

Run (from the project folder):
    mujoco_env/bin/python mujoco_sim/validate_leg.py

Checks
  1. Kinematics: for many joint-angle sets, MuJoCo site positions
     (femur servo centre, knee, foot) are compared with
       a) Leg(3, 5.6, 8)            - your FK as written (cm)
       b) Leg(<STL lengths>)        - the SAME FK code with the STL lengths
     (b) isolates convention errors from the known length differences.
  2. Link lengths stay constant.
  3. Actuators: servos reach their targets under gravity; the body,
     coxa servo axis and body origin do not move.

numpyfk3d.py is imported unchanged.
"""

import os
import sys

import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
# numpyfk3d.py lives one folder up locally, or in ../fk3d in the GitHub repo
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "fk3d"))
from numpyfk3d import Leg  # noqa: E402  (existing FK, unchanged)

XML = os.path.join(HERE, "quadruped_leg.xml")
model = mujoco.MjModel.from_xml_path(XML)
data = mujoco.MjData(model)

FRAME = model.body("leg_frame").id
SITES = ["femur_servo_center", "knee", "foot"]


def sites_in_leg_frame_cm():
    """Site positions expressed in the FK frame (leg_frame), in cm."""
    p0 = data.xpos[FRAME]
    R = data.xmat[FRAME].reshape(3, 3)
    out = {"body": R.T @ (data.site("body_origin").xpos - p0) * 100}
    for s in SITES:
        out[s] = R.T @ (data.site(s).xpos - p0) * 100
    return out


def set_angles_deg(c, f, t):
    data.qpos[:] = np.radians([c, f, t])
    mujoco.mj_kinematics(model, data)


# ---- STL lengths, measured from the model itself at zero pose ----
set_angles_deg(0, 0, 0)
p = sites_in_leg_frame_cm()
LC = np.linalg.norm(p["femur_servo_center"] - p["body"])
LF = np.linalg.norm(p["knee"] - p["femur_servo_center"])
foot_rel = p["foot"] - p["knee"]
LT_INPLANE = np.linalg.norm(foot_rel[:2])      # tibia length in the leg plane
FOOT_Z_OFF = foot_rel[2]                        # out-of-plane foot offset

fk_user = Leg(3, 5.6, 8)
fk_stl = Leg(LC, LF, LT_INPLANE)

print("=" * 70)
print("MuJoCo leg vs. numpyfk3d.Leg")
print("=" * 70)
print(f"STL link lengths (cm): coxa {LC:.4f}  femur {LF:.4f}  "
      f"tibia {LT_INPLANE:.4f} in-plane")
print(f"Foot out-of-plane offset (local Z of tibia): {FOOT_Z_OFF*10:.2f} mm")

rng = np.random.default_rng(0)
tests = [(0, 0, 0), (45, 45, 30), (90, 0, 0), (-90, 0, 0), (0, 90, 0),
         (0, 30, 40), (30, -45, 60), (-60, 20, -80)]
tests += [tuple(rng.uniform(-90, 90, 3)) for _ in range(200)]

err_stl, err_user, len_dev = [], [], []
print("\nangles (deg)          point        MuJoCo (cm)               "
      "FK STL-len diff  FK 3/5.6/8 diff")
for i, (c, f, t) in enumerate(tests):
    set_angles_deg(c, f, t)
    p = sites_in_leg_frame_cm()
    # FK ignores the out-of-plane foot offset -> compare foot with that offset
    # removed (tibia-local Z) for the STL-length check, and report it separately.
    R_c = data.xmat[model.body("tibia").id].reshape(3, 3)
    R0 = data.xmat[FRAME].reshape(3, 3)
    z_tibia = R0.T @ R_c[:, 2]
    foot_inplane = p["foot"] - z_tibia * FOOT_Z_OFF
    mj = {"femur servo": p["femur_servo_center"], "knee": p["knee"],
          "foot": foot_inplane}
    fk_s = {"femur servo": fk_stl.coxa(c), "knee": fk_stl.femur(c, f),
            "foot": fk_stl.tibia(c, f, t)}
    fk_u = {"femur servo": fk_user.coxa(c), "knee": fk_user.femur(c, f),
            "foot": fk_user.tibia(c, f, t)}
    for k in mj:
        es, eu = np.linalg.norm(mj[k] - fk_s[k]), np.linalg.norm(mj[k] - fk_u[k])
        err_stl.append(es)
        err_user.append((k, eu))
        if i < 8:
            print(f"({c:5.0f},{f:5.0f},{t:5.0f})  {k:11s}  "
                  f"{np.array2string(mj[k], precision=3, floatmode='fixed'):26s}"
                  f"  {es:9.2e}       {eu:6.3f}")
    len_dev.append([
        np.linalg.norm(p["femur_servo_center"] - p["body"]) - LC,
        np.linalg.norm(p["knee"] - p["femur_servo_center"]) - LF,
        np.linalg.norm(p["foot"] - p["knee"]) - np.linalg.norm(foot_rel),
    ])
    assert np.allclose(p["body"], 0, atol=1e-9), "body origin moved!"

err_joint = [e for e, (n, _) in zip(err_stl, err_user) if n != "foot"]
err_foot = [e for e, (n, _) in zip(err_stl, err_user) if n == "foot"]
print(f"\n[1] Max |MuJoCo - FK(STL lengths)| over {len(tests)} poses:")
print(f"    femur servo + knee: {max(err_joint):.2e} cm -> conventions "
      f"{'MATCH' if max(err_joint) < 1e-9 else 'DO NOT MATCH'}")
print(f"    foot:               {max(err_foot):.2e} cm "
      f"(measured STL tip is 0.01 mm off the tibia centreline in-plane; "
      f"{'OK' if max(err_foot) < 2e-3 else 'UNEXPECTED'})")
for k in ("femur servo", "knee", "foot"):
    e = [v for n, v in err_user if n == k]
    print(f"    vs FK Leg(3,5.6,8) {k:11s}: max {max(e):.3f} cm, "
          f"mean {np.mean(e):.3f} cm  (expected: length differences)")
print(f"[2] Max link-length drift over all poses: "
      f"{np.abs(len_dev).max():.2e} cm")

# ---- 3. actuator test under gravity ----
print("\n[3] Actuator test (2 s per target, gravity on)")
mujoco.mj_resetData(model, data)
mujoco.mj_forward(model, data)
body_pos0 = data.xpos[FRAME].copy()
axis0 = data.site("coxa_servo_axis").xpos.copy()
for target in [(0, 0, 0), (45, 45, 30), (-30, 60, -45), (60, -30, 70)]:
    data.ctrl[:] = np.radians(target)
    for _ in range(2000):
        mujoco.mj_step(model, data)
    reached = np.degrees(data.qpos)
    print(f"    target {target} -> reached "
          f"{np.array2string(reached, precision=2, floatmode='fixed')} deg, "
          f"servo torque {np.array2string(data.actuator_force, precision=4)} N*m")
print(f"    leg_frame moved:       {np.linalg.norm(data.xpos[FRAME]-body_pos0):.2e} m")
print(f"    coxa servo axis moved: "
      f"{np.linalg.norm(data.site('coxa_servo_axis').xpos-axis0):.2e} m")
print(f"    contacts at end: {data.ncon}")
