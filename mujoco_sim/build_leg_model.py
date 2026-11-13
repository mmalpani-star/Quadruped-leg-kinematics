"""
Build the MJCF model of ONE quadruped leg from the STL parts in ../stl_parts.

Run:   ../mujoco_env/bin/python build_leg_model.py
Output: quadruped_leg.xml (in this folder)

------------------------------------------------------------------------
WHERE THE NUMBERS COME FROM
------------------------------------------------------------------------
All STL files are in millimetres and were exported "laid out", not assembled.
Every pivot below was measured from hole/boss features in the meshes
(see the comments next to each constant). Nothing is taken from the FK code.

Kinematic convention (same as numpyfk3d.py):
  +X  initial coxa direction, +Y up (coxa servo axis), +Z completes RH frame.
  coxa  hinge axis +Y   (positive: +X -> -Z, i.e. R_y(theta_c))
  femur hinge axis +Z of the coxa frame   (positive: toward local +Y)
  tibia hinge axis +Z of the femur frame  (same plane as femur - provisional
                                            FK assumption; the STL holes for
                                            femur and knee ARE parallel)

MuJoCo's viewer orbits around world +Z, so the whole leg sits inside a
static body "leg_frame" that is rotated +90 deg about X:
  leg-frame X -> world X,  leg-frame Y (up) -> world Z,  leg-frame Z -> world -Y
All positions inside leg_frame are in YOUR FK coordinate system.
Gravity is world -Z, which is leg-frame -Y (down).
"""

import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MM = 0.001  # mm -> m

# ----------------------------------------------------------------------
# Measured STL features (mesh coordinates, mm)
# ----------------------------------------------------------------------
# coxa_l: coxa axis = mesh Z line through the coaxial 6 mm pin hole (bottom
# flange) and 7.5 mm horn hole (top flange).
COXA_AXIS_XY = (161.86, 24.40)
# coxa_l: femur axis = mesh X line. y from the s_hold pin (8.24 mm from its
# screw holes) = servo ear-hole centreline (-3.10); z = horn-tab hole centre.
FEMUR_AXIS_YZ = (-3.09, 19.65)
# Plane of the leg in coxa_l mesh X: the plane containing the coxa axis
# (this is where the FK places everything).
LEG_PLANE_X = 161.86

# femur_1: two parallel pivots along mesh X, 55.00 mm apart (6 mm pin holes on
# the x=186.84..190.84 arm, 7.5 mm horn holes on the x=220.34..224.34 arm).
FEMUR_PIVOT_PROX = (178.01, 19.63)   # (y, z)
FEMUR_PIVOT_DIST = (233.01, 19.63)
# Femur pin arm rests against the s_hold plate face: femur mesh x 190.84 sits
# at coxa mesh x 149.80  ->  femur_x = coxa_x + 41.04
FEMUR_X_SHIFT = 190.84 - 149.80

# tibia_l: knee axis = mesh X line (same feature pattern as coxa_l).
KNEE_AXIS_YZ = (-58.03, 19.62)
# tibia_l foot tip: lowest point of the rounded tip (centre of tip region).
FOOT_TIP = (165.52, -58.04, -60.36)

# s_hold: screw-hole centre (y, z) and its 5.8 mm pin (bearing) centre.
SHOLD_HOLES_YZ = (185.87, 19.62)
# coxa_l / tibia_l horn-tab screw-hole centres (y, z) where s_hold is screwed.
COXA_TAB_YZ = (5.15, 19.65)
TIBIA_TAB_YZ = (-66.27, 19.62)

# body_d boss (5.6 mm, fits coxa 6 mm pin hole) = coxa axis on the body,
# 5.25 mm outboard of the body_u servo ear-hole pair centre (26.25, 92.43).
BODY_COXA_AXIS_XY = (21.00, 92.43)
# coxa bottom flange (coxa mesh z = -3.20) sits on the body_d tongue (z 56.67).
BODY_TONGUE_TOP_Z = 56.67
COXA_BOTTOM_Z = -3.20

# ----------------------------------------------------------------------
# SG90 servo proxy (no servo STL was provided).
# Datasheet nominal values - APPROXIMATE.
# ----------------------------------------------------------------------
SG90_L, SG90_W, SG90_H = 22.5, 11.8, 22.7   # body length, width, height (mm)
SG90_EAR_FROM_BOTTOM = 15.9                 # ear underside above case bottom
SG90_SHAFT_OFFSET = 5.25                    # shaft from case centre (matches
                                            # body_d boss vs. body_u ear holes)
SG90_MASS = 0.009                           # kg (datasheet)
SG90_STALL_TORQUE = 0.176                   # N*m (1.8 kgf*cm @ 4.8 V)

# Printed parts: mass from mesh volume x PLA density, assuming 100 % infill.
# APPROXIMATE - real prints with partial infill are lighter.
PLA_DENSITY = 1240.0  # kg/m^3

# ----------------------------------------------------------------------
# Mesh -> body-frame rotations (columns = where mesh X, Y, Z end up)
# ----------------------------------------------------------------------
# coxa_l, s_hold, body plates: mesh X -> -Z, mesh Y -> -X, mesh Z -> +Y
R_COXA = np.array([[0, -1, 0],
                   [0,  0, 1],
                   [-1, 0, 0]], float)
# femur_1: mesh X -> -Z (same side as coxa), mesh Y -> +X (along femur), Z -> -Y
R_FEMUR = np.array([[0, 1, 0],
                    [0, 0, -1],
                    [-1, 0, 0]], float)
# tibia_l: mesh X -> -Z, mesh Z (knee -> foot is -Z) -> -X, mesh Y -> -Y
R_TIBIA = np.array([[0, 0, -1],
                    [0, -1, 0],
                    [-1, 0, 0]], float)


STL_SRC = os.path.join(HERE, "..", "stl_parts")
MESH_DIR = os.path.join(HERE, "meshes")
PARTS = ["body_u", "body_d", "coxa_l", "femur_1", "tibia_l", "s_hold"]


def convert_ascii_stls():
    """MuJoCo only reads BINARY STL; the originals are ASCII.
    Write binary copies into mujoco_sim/meshes/ (originals are untouched).
    Geometry is copied exactly - no scaling or moving (scale is set in MJCF)."""
    import struct
    os.makedirs(MESH_DIR, exist_ok=True)
    for part in PARTS:
        src = os.path.join(STL_SRC, f"{part}_PCBWay Community.stl")
        with open(src) as f:
            v = [list(map(float, ln.split()[1:4])) for ln in f
                 if ln.strip().startswith("vertex")]
        tris = np.array(v, dtype=np.float32).reshape(-1, 3, 3)
        n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
        rec = np.zeros(len(tris), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
        rec["n"], rec["v"] = n, tris
        with open(os.path.join(MESH_DIR, f"{part}.stl"), "wb") as f:
            f.write(b"binary copy of " + part.encode() + b" (mm)".ljust(80 - 15 - len(part)))
            f.write(struct.pack("<I", len(tris)))
            f.write(rec.tobytes())


def mat2quat(R):
    """Rotation matrix -> MuJoCo quaternion (w x y z), using MuJoCo itself
    (robust for 180-degree rotations such as R_TIBIA)."""
    import mujoco
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).flatten())
    return q


def fmt(v):
    return " ".join(f"{x:.6g}" if abs(x) > 1e-12 else "0" for x in v)


def mesh_geom(mesh, R, ref_mm, rgba, extra=""):
    """Visual mesh geom so that mesh point `ref_mm` lands on the body origin."""
    pos = -R @ (np.asarray(ref_mm) * MM)
    return (f'<geom class="visual" mesh="{mesh}" pos="{fmt(pos)}" '
            f'quat="{fmt(mat2quat(R))}" rgba="{rgba}" {extra}/>')


def servo_box(name, R, ref_mm, center_mm, half_mm):
    """SG90 proxy box. center/half given in mesh coordinates."""
    pos = R @ ((np.asarray(center_mm) - ref_mm) * MM)
    size = np.abs(R) @ (np.asarray(half_mm) * MM)
    return (f'<geom name="{name}" class="servo" type="box" pos="{fmt(pos)}" '
            f'size="{fmt(size)}"/>')


def part_volume_and_bbox(part):
    """Signed mesh volume (mm^3) and bounding box of a binary STL copy."""
    import struct
    with open(os.path.join(MESH_DIR, f"{part}.stl"), "rb") as f:
        b = f.read()
    n = struct.unpack("<I", b[80:84])[0]
    t = np.frombuffer(b[84:], dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")],
                      count=n)["v"].astype(float)
    vol = np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])).sum() / 6
    v = t.reshape(-1, 3)
    return abs(vol), v.min(0), v.max(0)


def mass_box(part, R, ref_mm, tag=""):
    """Invisible mass proxy for a printed part (APPROXIMATE: 100 % infill PLA)."""
    vol, lo, hi = part_volume_and_bbox(part)
    mass = vol * 1e-9 * PLA_DENSITY
    pos = R @ (((lo + hi) / 2 - ref_mm) * MM)
    size = np.abs(R) @ ((hi - lo) / 2 * MM)
    return (f'<geom name="{part}{tag}_massproxy" class="massproxy" pos="{fmt(pos)}" '
            f'size="{fmt(size)}" mass="{mass:.5f}"/>')


def build():
    # ---------------- reference points ----------------
    coxa_ref = np.array([COXA_AXIS_XY[0], COXA_AXIS_XY[1], FEMUR_AXIS_YZ[1]])
    # Body plates: body coxa-axis point at the height that matches coxa_ref.
    body_z = BODY_TONGUE_TOP_Z + (coxa_ref[2] - COXA_BOTTOM_Z)
    body_ref = np.array([*BODY_COXA_AXIS_XY, body_z])

    femur_servo_mesh = np.array([LEG_PLANE_X, *FEMUR_AXIS_YZ])
    femur_body_pos = R_COXA @ ((femur_servo_mesh - coxa_ref) * MM)

    femur_ref = np.array([LEG_PLANE_X + FEMUR_X_SHIFT, *FEMUR_PIVOT_PROX])
    knee_mesh_f = np.array([LEG_PLANE_X + FEMUR_X_SHIFT, *FEMUR_PIVOT_DIST])
    knee_body_pos = R_FEMUR @ ((knee_mesh_f - femur_ref) * MM)

    tibia_ref = np.array([LEG_PLANE_X, *KNEE_AXIS_YZ])
    foot_pos = R_TIBIA @ ((np.array(FOOT_TIP) - tibia_ref) * MM)

    # s_hold placed by matching its screw holes to the tab holes (no rotation).
    shold_on_coxa = coxa_ref + np.array(
        [0, SHOLD_HOLES_YZ[0] - COXA_TAB_YZ[0], SHOLD_HOLES_YZ[1] - COXA_TAB_YZ[1]])
    # On tibia_l the horn tab lies on the OTHER side of the servo axis than on
    # coxa_l (tab y -66.27 < axis y -58.03), so s_hold is flipped 180 deg about
    # its pin axis (mesh X). Then its pin (y 177.63) lands on the knee axis and
    # its screw holes (8.24 mm away) land on the tab holes.
    FLIP_X = np.diag([1.0, -1.0, -1.0])
    R_SHOLD_TIBIA = R_TIBIA @ FLIP_X
    shold_pin = np.array([LEG_PLANE_X, SHOLD_HOLES_YZ[0] - 8.24, SHOLD_HOLES_YZ[1]])
    knee_pin = np.array([LEG_PLANE_X, KNEE_AXIS_YZ[0], KNEE_AXIS_YZ[1]])
    # mesh point of s_hold that must sit at the tibia body origin:
    shold_on_tibia = shold_pin - FLIP_X @ (knee_pin - tibia_ref)

    # ---------------- servo proxies (mesh coordinates) ----------------
    # Coxa servo: ears rest on body_u top (z 77.27), shaft up, case offset +X.
    ear_z = 77.27
    coxa_servo_c = (BODY_COXA_AXIS_XY[0] + SG90_SHAFT_OFFSET, BODY_COXA_AXIS_XY[1],
                    ear_z - SG90_EAR_FROM_BOTTOM + SG90_H / 2)
    coxa_servo_h = (SG90_L / 2, SG90_W / 2, SG90_H / 2)
    # Femur / knee servos: ears on the frame face x = 167.86, shaft along +X,
    # case offset -Z from the shaft (ear-hole centre z ~14.4, shaft z ~19.6).
    ear_x = 167.86
    sx = ear_x - SG90_EAR_FROM_BOTTOM + SG90_H / 2
    femur_servo_c = (sx, FEMUR_AXIS_YZ[0], FEMUR_AXIS_YZ[1] - SG90_SHAFT_OFFSET)
    knee_servo_c = (sx, KNEE_AXIS_YZ[0], KNEE_AXIS_YZ[1] - SG90_SHAFT_OFFSET)
    side_servo_h = (SG90_H / 2, SG90_W / 2, SG90_L / 2)

    q_root = "0.707107 0.707107 0 0"  # +90 deg about X: leg Y (up) -> world Z

    xml = f"""<!--
  Single quadruped leg built from the STL parts in ../stl_parts.
  GENERATED by build_leg_model.py - edit that file, not this one.
  Units: metres, radians. Coordinates inside "leg_frame" use the FK
  convention of numpyfk3d.py (+X coxa, +Y up, +Z right-handed).
  Joint geometry from STL (STL lengths, user-approved):
    coxa {np.linalg.norm(femur_body_pos)*1000:.2f} mm, femur {np.linalg.norm(knee_body_pos)*1000:.2f} mm,
    tibia (knee->foot tip) {np.linalg.norm(foot_pos)*1000:.2f} mm
  APPROXIMATIONS: servo housings are SG90 datasheet boxes (9 g); printed-part
  masses assume PLA at 100 % infill; armature/gains are estimates.
-->
<mujoco model="quadruped_leg">
  <compiler angle="radian" meshdir="meshes"/>
  <option timestep="0.001" gravity="0 0 -9.81"/>

  <visual>
    <global offwidth="1280" offheight="960"/>
    <headlight ambient="0.4 0.4 0.4" diffuse="0.6 0.6 0.6"/>
  </visual>

  <asset>
    <mesh name="body_u"  file="body_u.stl"  scale="{MM} {MM} {MM}" inertia="legacy"/>
    <mesh name="body_d"  file="body_d.stl"  scale="{MM} {MM} {MM}" inertia="legacy"/>
    <mesh name="coxa_l"  file="coxa_l.stl"  scale="{MM} {MM} {MM}" inertia="legacy"/>
    <mesh name="femur_1" file="femur_1.stl" scale="{MM} {MM} {MM}" inertia="legacy"/>
    <mesh name="tibia_l" file="tibia_l.stl" scale="{MM} {MM} {MM}" inertia="legacy"/>
    <mesh name="s_hold"  file="s_hold.stl"  scale="{MM} {MM} {MM}" inertia="legacy"/>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512"
             rgb1="0.85 0.85 0.85" rgb2="0.75 0.75 0.75"/>
    <material name="grid" texture="grid" texrepeat="20 20"/>
  </asset>

  <default>
    <default class="visual">
      <!-- printed PLA part: VISUAL ONLY, no mass (meshes are not watertight;
           mass is carried by the "massproxy" boxes instead) -->
      <geom type="mesh" contype="0" conaffinity="0" group="2" density="0"/>
    </default>
    <default class="massproxy">
      <!-- invisible box over the part bounding box; mass = mesh volume x PLA -->
      <geom type="box" contype="0" conaffinity="0" group="5" rgba="0 0 0 0"/>
    </default>
    <default class="servo">
      <!-- SG90 proxy box: visual + mass only (datasheet, approx.) -->
      <geom contype="0" conaffinity="0" group="2" mass="{SG90_MASS}" rgba="0.15 0.3 0.75 1"/>
    </default>
    <default class="collision">
      <geom group="3" rgba="1 0.3 0.3 0.4" density="0"/>
    </default>
    <default class="sg90">
      <joint damping="0.002" armature="0.0002" frictionloss="0.001"/>
      <position kp="2.0" kv="0.04" forcerange="-{SG90_STALL_TORQUE} {SG90_STALL_TORQUE}"
                ctrlrange="-1.5708 1.5708"/>
    </default>
  </default>

  <worldbody>
    <light pos="0 -0.3 0.6" dir="0 0.4 -1" diffuse="0.6 0.6 0.6"/>
    <geom name="floor" type="plane" size="0.5 0.5 0.01" material="grid"/>

    <!-- Static frame in YOUR FK coordinates (Y up). Origin = coxa axis at the
         height of the femur servo axis (FK body point). -->
    <body name="leg_frame" pos="0 0 0.14" quat="{q_root}">
      <site name="body_origin" size="0.002" rgba="1 0 0 1"/>
      <!-- Fixed body: printed plates + body-mounted coxa servo housing -->
      {mesh_geom("body_d", R_COXA, body_ref, "0.95 0.6 0.2 1")}
      {mesh_geom("body_u", R_COXA, body_ref, "0.95 0.7 0.3 1")}
      {servo_box("coxa_servo_housing", R_COXA, body_ref, coxa_servo_c, coxa_servo_h)}
      <!-- coxa servo axis (global Y), purple like the Matplotlib visualizer -->
      <site name="coxa_servo_axis" type="cylinder" fromto="0 -0.02 0 0 0.02 0"
            size="0.0008" rgba="0.6 0.1 0.8 1"/>

      <body name="coxa">
        <joint name="coxa" class="sg90" axis="0 1 0" range="-1.5708 1.5708"/>
        {mesh_geom("coxa_l", R_COXA, coxa_ref, "0.9 0.9 0.9 1")}
        {mass_box("coxa_l", R_COXA, coxa_ref)}
        {mesh_geom("s_hold", R_COXA, shold_on_coxa, "0.3 0.3 0.3 1")}
        {mass_box("s_hold", R_COXA, shold_on_coxa, "_coxa")}
        <!-- femur servo housing is FIXED to the coxa bracket -->
        {servo_box("femur_servo_housing", R_COXA, coxa_ref, femur_servo_c, side_servo_h)}

        <body name="femur" pos="{fmt(femur_body_pos)}">
          <joint name="femur" class="sg90" axis="0 0 1" range="-1.5708 1.5708"/>
          <site name="femur_servo_center" size="0.002" rgba="0 0 1 1"/>
          {mesh_geom("femur_1", R_FEMUR, femur_ref, "0.2 0.7 0.3 1")}
          {mass_box("femur_1", R_FEMUR, femur_ref)}
          <geom name="femur_col" class="collision" type="capsule"
                fromto="0.008 0 0 {knee_body_pos[0]-0.008:.6g} 0 0" size="0.004"/>

          <body name="tibia" pos="{fmt(knee_body_pos)}">
            <joint name="tibia" class="sg90" axis="0 0 1" range="-1.5708 1.5708"/>
            <site name="knee" size="0.002" rgba="0 0.6 0 1"/>
            {mesh_geom("tibia_l", R_TIBIA, tibia_ref, "0.9 0.9 0.9 1")}
            {mass_box("tibia_l", R_TIBIA, tibia_ref)}
            {mesh_geom("s_hold", R_SHOLD_TIBIA, shold_on_tibia, "0.3 0.3 0.3 1")}
            {mass_box("s_hold", R_SHOLD_TIBIA, shold_on_tibia, "_tibia")}
            <!-- knee servo housing sits in the tibia window: FIXED to the tibia -->
            {servo_box("knee_servo_housing", R_TIBIA, tibia_ref, knee_servo_c, side_servo_h)}
            <geom name="tibia_col" class="collision" type="capsule"
                  fromto="0.02 0 {foot_pos[2]:.6g} {foot_pos[0]-0.004:.6g} 0 {foot_pos[2]:.6g}" size="0.003"/>
            <geom name="foot_col" class="collision" type="sphere"
                  pos="{fmt(foot_pos)}" size="0.0025" friction="1 0.01 0.001"/>
            <site name="foot" pos="{fmt(foot_pos)}" size="0.002" rgba="1 0.5 0 1"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>

  <actuator>
    <position name="coxa_servo"  class="sg90" joint="coxa"/>
    <position name="femur_servo" class="sg90" joint="femur"/>
    <position name="tibia_servo" class="sg90" joint="tibia"/>
  </actuator>

  <keyframe>
    <key name="zero" qpos="0 0 0" ctrl="0 0 0"/>
    <key name="fk_initial_45_45_30" qpos="0.785398 0.785398 0.523599"
         ctrl="0.785398 0.785398 0.523599"/>
  </keyframe>
</mujoco>
"""
    out = os.path.join(HERE, "quadruped_leg.xml")
    with open(out, "w") as f:
        f.write(xml)
    print("wrote", out)
    print(f"  coxa  length (coxa axis -> femur axis): {np.linalg.norm(femur_body_pos)*1000:.2f} mm")
    print(f"  femur length (femur axis -> knee axis): {np.linalg.norm(knee_body_pos)*1000:.2f} mm")
    print(f"  tibia length (knee axis -> foot tip):   {np.linalg.norm(foot_pos)*1000:.2f} mm "
          f"(out-of-plane foot offset {foot_pos[2]*1000:.2f} mm)")


if __name__ == "__main__":
    convert_ascii_stls()
    build()
