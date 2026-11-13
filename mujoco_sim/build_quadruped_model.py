"""
Build the full 4-leg quadruped MJCF from the same STL parts.

Run:    ../mujoco_env/bin/python build_quadruped_model.py
Output: quadruped.xml

Reuses every measured constant and helper from build_leg_model.py (that file
is NOT modified; the single-leg model stays as it was).

Leg placement (all from body_d pivot bosses = coxa axes, body mesh mm):
  leg   corner (x, y)    in trunk frame        how it is made
  FL    (21.00,  92.43)  (+35.5, 0, +35.5)     validated single leg as-is
  RL    (92.00,  92.43)  (+35.5, 0, -35.5)     mirror image (left/right part)
  RR    (92.00, 163.43)  (-35.5, 0, -35.5)     FL rotated 180 deg about Y
  FR    (21.00, 163.43)  (-35.5, 0, +35.5)     RL rotated 180 deg about Y
Forward = trunk +Z, up = +Y, so left = +X and right = -X.

Mirrored legs use the same STL geometry with a mirrored mesh scale. This is
the left/right counterpart of a printed part (coxa_r is an exact mirror of
coxa_l). NOTE: tibia_r.stl is a slightly different variant (1.6 mm pilot
holes instead of 2.0 mm), so the mirrored tibia_l is used instead of it.

Trunk frame = same convention as the single leg / FK: +Y up, legs stick out
along +-X. The trunk has a free joint, so the robot can actually walk.
"""

import os
import numpy as np

import build_leg_model as L   # measured constants + helpers (unchanged)

HERE = os.path.dirname(os.path.abspath(__file__))
MM = L.MM
M = np.diag([1.0, 1.0, -1.0])        # mirror across the leg's XY plane
FLIP_MESH_X = np.diag([-1.0, 1.0, 1.0])

# ---------------- single-leg geometry (identical to build_leg_model) -------
coxa_ref = np.array([L.COXA_AXIS_XY[0], L.COXA_AXIS_XY[1], L.FEMUR_AXIS_YZ[1]])
body_z = L.BODY_TONGUE_TOP_Z + (coxa_ref[2] - L.COXA_BOTTOM_Z)
body_center = np.array([56.5, 127.93, body_z])   # body plate bbox centre (x, y)

femur_body_pos = L.R_COXA @ ((np.array([L.LEG_PLANE_X, *L.FEMUR_AXIS_YZ]) - coxa_ref) * MM)
femur_ref = np.array([L.LEG_PLANE_X + L.FEMUR_X_SHIFT, *L.FEMUR_PIVOT_PROX])
knee_body_pos = L.R_FEMUR @ ((np.array([L.LEG_PLANE_X + L.FEMUR_X_SHIFT,
                                        *L.FEMUR_PIVOT_DIST]) - femur_ref) * MM)
tibia_ref = np.array([L.LEG_PLANE_X, *L.KNEE_AXIS_YZ])
foot_pos = L.R_TIBIA @ ((np.array(L.FOOT_TIP) - tibia_ref) * MM)
shold_on_coxa = coxa_ref + np.array(
    [0, L.SHOLD_HOLES_YZ[0] - L.COXA_TAB_YZ[0], L.SHOLD_HOLES_YZ[1] - L.COXA_TAB_YZ[1]])
FLIP_X = np.diag([1.0, -1.0, -1.0])
R_SHOLD_TIBIA = L.R_TIBIA @ FLIP_X
shold_pin = np.array([L.LEG_PLANE_X, L.SHOLD_HOLES_YZ[0] - 8.24, L.SHOLD_HOLES_YZ[1]])
knee_pin = np.array([L.LEG_PLANE_X, *L.KNEE_AXIS_YZ])
shold_on_tibia = shold_pin - FLIP_X @ (knee_pin - tibia_ref)

body_ref_leg = np.array([*L.BODY_COXA_AXIS_XY, body_z])
coxa_servo_c = (L.BODY_COXA_AXIS_XY[0] + L.SG90_SHAFT_OFFSET, L.BODY_COXA_AXIS_XY[1],
                77.27 - L.SG90_EAR_FROM_BOTTOM + L.SG90_H / 2)
coxa_servo_h = (L.SG90_L / 2, L.SG90_W / 2, L.SG90_H / 2)
sx = 167.86 - L.SG90_EAR_FROM_BOTTOM + L.SG90_H / 2
femur_servo_c = (sx, L.FEMUR_AXIS_YZ[0], L.FEMUR_AXIS_YZ[1] - L.SG90_SHAFT_OFFSET)
knee_servo_c = (sx, L.KNEE_AXIS_YZ[0], L.KNEE_AXIS_YZ[1] - L.SG90_SHAFT_OFFSET)
side_servo_h = (L.SG90_H / 2, L.SG90_W / 2, L.SG90_L / 2)

fmt = L.fmt


class LegWriter:
    """Writes geoms in a leg frame, optionally mirrored (z -> -z)."""

    def __init__(self, mirror):
        self.mirror = mirror
        self.Mx = M if mirror else np.eye(3)
        self.sfx = "_m" if mirror else ""

    def p(self, v):
        return self.Mx @ np.asarray(v, float)

    def mesh(self, mesh, R, ref_mm, rgba):
        pos = -R @ (np.asarray(ref_mm) * MM)
        Rg = R if not self.mirror else M @ R @ FLIP_MESH_X
        return (f'<geom class="visual" mesh="{mesh}{self.sfx}" pos="{fmt(self.p(pos))}" '
                f'quat="{fmt(L.mat2quat(Rg))}" rgba="{rgba}"/>')

    def box(self, name, cls, R, ref_mm, center_mm, half_mm, extra=""):
        pos = R @ ((np.asarray(center_mm) - ref_mm) * MM)
        size = np.abs(R) @ (np.asarray(half_mm) * MM)
        return (f'<geom name="{name}" class="{cls}" type="box" pos="{fmt(self.p(pos))}" '
                f'size="{fmt(size)}" {extra}/>')

    def massbox(self, name, part, R, ref_mm):
        vol, lo, hi = L.part_volume_and_bbox(part)
        return self.box(name, "massproxy", R, ref_mm, (lo + hi) / 2, (hi - lo) / 2,
                        f'mass="{vol * 1e-9 * L.PLA_DENSITY:.5f}"')


def leg_xml(name, pos, yaw180, mirror):
    w = LegWriter(mirror)
    quat = "0 0 1 0" if yaw180 else "1 0 0 0"
    coxa_axis = "0 -1 0" if mirror else "0 1 0"   # mirrored rotation sense
    fz = foot_pos[2] * (-1 if mirror else 1)
    return f"""
    <!-- ===== leg {name} ({'mirrored' if mirror else 'as built'}{', rotated 180' if yaw180 else ''}) ===== -->
    <body name="{name}_base" pos="{fmt(pos)}" quat="{quat}">
      {w.box(f"{name}_coxa_servo_housing", "servo", L.R_COXA, body_ref_leg, coxa_servo_c, coxa_servo_h)}
      <body name="{name}_coxa">
        <joint name="{name}_coxa" class="sg90" axis="{coxa_axis}" range="-1.5708 1.5708"/>
        {w.mesh("coxa_l", L.R_COXA, coxa_ref, "0.9 0.9 0.9 1")}
        {w.mesh("s_hold", L.R_COXA, shold_on_coxa, "0.3 0.3 0.3 1")}
        {w.massbox(f"{name}_coxa_massproxy", "coxa_l", L.R_COXA, coxa_ref)}
        {w.massbox(f"{name}_shold1_massproxy", "s_hold", L.R_COXA, shold_on_coxa)}
        {w.box(f"{name}_femur_servo_housing", "servo", L.R_COXA, coxa_ref, femur_servo_c, side_servo_h)}
        <body name="{name}_femur" pos="{fmt(w.p(femur_body_pos))}">
          <joint name="{name}_femur" class="sg90" axis="0 0 1" range="-1.5708 1.5708"/>
          {w.mesh("femur_1", L.R_FEMUR, femur_ref, "0.2 0.7 0.3 1")}
          {w.massbox(f"{name}_femur_massproxy", "femur_1", L.R_FEMUR, femur_ref)}
          <geom name="{name}_femur_col" class="collision" type="capsule"
                fromto="0.008 0 0 {knee_body_pos[0]-0.008:.6g} 0 0" size="0.004"/>
          <body name="{name}_tibia" pos="{fmt(w.p(knee_body_pos))}">
            <joint name="{name}_tibia" class="sg90" axis="0 0 1" range="-1.5708 1.5708"/>
            {w.mesh("tibia_l", L.R_TIBIA, tibia_ref, "0.9 0.9 0.9 1")}
            {w.mesh("s_hold", R_SHOLD_TIBIA, shold_on_tibia, "0.3 0.3 0.3 1")}
            {w.massbox(f"{name}_tibia_massproxy", "tibia_l", L.R_TIBIA, tibia_ref)}
            {w.massbox(f"{name}_shold2_massproxy", "s_hold", R_SHOLD_TIBIA, shold_on_tibia)}
            {w.box(f"{name}_knee_servo_housing", "servo", L.R_TIBIA, tibia_ref, knee_servo_c, side_servo_h)}
            <geom name="{name}_tibia_col" class="collision" type="capsule"
                  fromto="0.02 0 {fz:.6g} {foot_pos[0]-0.006:.6g} 0 {fz:.6g}" size="0.003"/>
            <geom name="{name}_foot" class="collision" type="sphere"
                  pos="{fmt(w.p(foot_pos))}" size="0.0035" friction="1.2 0.02 0.001"/>
            <site name="{name}_foot" pos="{fmt(w.p(foot_pos))}" size="0.002" rgba="1 0.5 0 1"/>
          </body>
        </body>
      </body>
    </body>"""


def leg_base_pos(corner_xy):
    return L.R_COXA @ ((np.array([*corner_xy, body_z]) - body_center) * MM)


LEGS = [  # name, corner (body mesh mm), rotated 180 about Y, mirrored
    ("FL", (21.00, 92.43), False, False),
    ("RL", (92.00, 92.43), False, True),
    ("RR", (92.00, 163.43), True, False),
    ("FR", (21.00, 163.43), True, True),
]


def build():
    L.convert_ascii_stls()
    meshes = []
    for part in ("body_u", "body_d", "coxa_l", "femur_1", "tibia_l", "s_hold"):
        for sfx, sx_ in (("", MM), ("_m", -MM)):
            if part.startswith("body") and sfx:
                continue
            meshes.append(f'<mesh name="{part}{sfx}" file="{part}.stl" '
                          f'scale="{sx_} {MM} {MM}" inertia="legacy"/>')

    # trunk collision box = body plate bounding box (trunk frame)
    _, lo_u, hi_u = L.part_volume_and_bbox("body_u")
    _, lo_d, hi_d = L.part_volume_and_bbox("body_d")
    lo, hi = np.minimum(lo_u, lo_d), np.maximum(hi_u, hi_d)
    trunk_c = L.R_COXA @ (((lo + hi) / 2 - body_center) * MM)
    trunk_h = np.abs(L.R_COXA) @ ((hi - lo) / 2 * MM)

    legs = "".join(leg_xml(n, leg_base_pos(c), y, m) for n, c, y, m in LEGS)
    acts = "\n".join(
        f'    <position name="{n}_{j}_servo" class="sg90" joint="{n}_{j}"/>'
        for n, *_ in LEGS for j in ("coxa", "femur", "tibia"))

    xml = f"""<!--
  Four-legged robot built from the STL parts. GENERATED by
  build_quadruped_model.py - edit that file, not this one.
  Same geometry, approximations and FK convention as quadruped_leg.xml.
-->
<mujoco model="quadruped">
  <compiler angle="radian" meshdir="meshes"/>
  <option timestep="0.001" gravity="0 0 -9.81" cone="elliptic" impratio="10"/>

  <visual>
    <global offwidth="1280" offheight="960"/>
    <headlight ambient="0.4 0.4 0.4" diffuse="0.6 0.6 0.6"/>
  </visual>

  <asset>
    {chr(10).join("    " + s for s in meshes).strip()}
    <texture name="grid" type="2d" builtin="checker" width="512" height="512"
             rgb1="0.85 0.85 0.85" rgb2="0.75 0.75 0.75"/>
    <material name="grid" texture="grid" texrepeat="40 40"/>
  </asset>

  <default>
    <default class="visual">
      <geom type="mesh" contype="0" conaffinity="0" group="2" density="0"/>
    </default>
    <default class="massproxy">
      <geom type="box" contype="0" conaffinity="0" group="5" rgba="0 0 0 0"/>
    </default>
    <default class="servo">
      <geom contype="0" conaffinity="0" group="2" mass="{L.SG90_MASS}" rgba="0.15 0.3 0.75 1"/>
    </default>
    <default class="collision">
      <!-- robot geoms touch the floor only (contype 1 / conaffinity 2) -->
      <geom group="3" rgba="1 0.3 0.3 0.4" density="0" contype="1" conaffinity="2"/>
    </default>
    <default class="sg90">
      <joint damping="0.002" armature="0.0002" frictionloss="0.001"/>
      <position kp="8.0" kv="0.15" forcerange="-{L.SG90_STALL_TORQUE} {L.SG90_STALL_TORQUE}"
                ctrlrange="-1.5708 1.5708"/>
    </default>
  </default>

  <worldbody>
    <light pos="0 -0.5 1.0" dir="0 0.4 -1" diffuse="0.6 0.6 0.6"/>
    <geom name="floor" type="plane" size="3 3 0.01" material="grid" friction="1 0.02 0.001"/>

    <!-- trunk: +90 deg about X so the FK frame's +Y (up) is world +Z -->
    <body name="trunk" pos="0 0 0.075" quat="0.707107 0.707107 0 0">
      <freejoint name="trunk"/>
      <site name="trunk_center" size="0.003" rgba="1 0 0 1"/>
      {L.mesh_geom("body_d", L.R_COXA, body_center, "0.95 0.6 0.2 1")}
      {L.mesh_geom("body_u", L.R_COXA, body_center, "0.95 0.7 0.3 1")}
      {L.mass_box("body_d", L.R_COXA, body_center)}
      {L.mass_box("body_u", L.R_COXA, body_center)}
      <geom name="trunk_col" class="collision" type="box" pos="{fmt(trunk_c)}" size="{fmt(trunk_h)}"/>
{legs}
    </body>
  </worldbody>

  <actuator>
{acts}
  </actuator>
</mujoco>
"""
    out = os.path.join(HERE, "quadruped.xml")
    with open(out, "w") as f:
        f.write(xml)
    print("wrote", out)


if __name__ == "__main__":
    build()
