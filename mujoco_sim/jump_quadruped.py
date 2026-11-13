"""
Make the 4-leg robot attempt a vertical jump in MuJoCo.

Run (from the project folder):
    mujoco_env/bin/mjpython mujoco_sim/jump_quadruped.py          # viewer (loops)
    mujoco_env/bin/python   mujoco_sim/jump_quadruped.py --test   # headless check

Builds on walk_quadruped.py (STAND_FEMUR/STAND_TIBIA, COXA_SIGN, SPLAY,
load()) without changing it. The crawl-gait file is untouched; this is a
separate, time-triggered open-loop sequence, not a periodic gait.

Jump sequence (open-loop, same pose on all 4 legs at once, coxa held at the
walking stance's splay angle throughout - no sideways motion, straight up):

  1. CROUCH   ramp from the standing pose down to a deep crouch (both knee
              joints near their limits; see constants below).
  2. PRELOAD  hold the crouch briefly so the legs actually get there.
  3. THRUST   an instantaneous step (not a ramp) to a near-straight-leg
              "extend" pose - the explosive push. This is the one place a
              smooth ramp would defeat the point: a real jump thrust is as
              fast as the actuator can move, so the commanded target jumps
              immediately and lets the servo's own dynamics (kp/kv/torque
              limit) govern how fast it can actually follow.
  4. EXTEND   hold the extended pose (legs stay long and ready to catch the
              landing).
  5. LANDING  ramp back down to the crouch to absorb the impact.
  6. RECOVER  ramp from crouch back up to the standing pose.
  7. REST     hold standing before the next cycle.

Crouch and extend poses were found empirically (see mujoco_sim/, scan of
femur/tibia pairs vs settled trunk height): crouch settles at ~2.7 cm,
extended at ~13.1 cm - about 5x stroke, both inside the +-90 deg joint
limits with margin, both settle with 0 deg tilt when splayed.

This uses the SAME SG90 position-actuator model (kp=8, stall torque 0.176
N*m) as the walking gait - no servo parameters are changed to make the jump
"work". Whatever liftoff happens (or doesn't) in the --test output is what
these actuators can actually do.
"""

import os
import sys
import time

import numpy as np
import mujoco

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import walk_quadruped as W  # noqa: E402  (reuses load(), COXA_SIGN, SPLAY, LEGS)

# ---- jump pose constants (degrees), found by scanning femur/tibia pairs ----
CROUCH_FEMUR, CROUCH_TIBIA = 70.0, -80.0     # settles at ~2.7 cm
EXTEND_FEMUR, EXTEND_TIBIA = -60.0, -30.0    # settles at ~13.1 cm

# ---- phase timing (seconds, within one jump cycle) ----
T_CROUCH_START = 0.00
T_CROUCH_RAMP = 0.35     # stand -> crouch
T_PRELOAD = 0.15     # hold crouch
T_EXTEND_HOLD = 0.40     # hold extended pose after the thrust step
T_LAND_RAMP = 0.35     # extend -> crouch (absorb)
T_SETTLE = 0.20     # hold crouch after landing
T_RECOVER_RAMP = 0.35     # crouch -> stand
CYCLE = 2.20     # total cycle length (includes rest at the end)

_T_THRUST = T_CROUCH_START + T_CROUCH_RAMP + T_PRELOAD           # 0.50
_T_LAND_START = _T_THRUST + T_EXTEND_HOLD                        # 0.90
_T_SETTLE_START = _T_LAND_START + T_LAND_RAMP                    # 1.25
_T_RECOVER_START = _T_SETTLE_START + T_SETTLE                    # 1.45
_T_REST_START = _T_RECOVER_START + T_RECOVER_RAMP                # 1.80


def _ease(s):
    """Cosine ease 0 -> 1, used for every ramp except the thrust step itself."""
    return 0.5 - 0.5 * np.cos(np.pi * np.clip(s, 0.0, 1.0))


def jump_targets(t_in_cycle):
    """(femur_deg, tibia_deg) for all legs at this point in one jump cycle."""
    t = t_in_cycle
    stand = (W.STAND_FEMUR, W.STAND_TIBIA)
    crouch = (CROUCH_FEMUR, CROUCH_TIBIA)
    extend = (EXTEND_FEMUR, EXTEND_TIBIA)

    if t < T_CROUCH_START + T_CROUCH_RAMP:                       # 1. CROUCH
        s = _ease((t - T_CROUCH_START) / T_CROUCH_RAMP)
        return tuple(a + s * (b - a) for a, b in zip(stand, crouch))
    if t < _T_THRUST:                                            # 2. PRELOAD
        return crouch
    if t < _T_LAND_START:                                        # 3+4. THRUST + EXTEND
        return extend                                            # (instant step, no ramp)
    if t < _T_SETTLE_START:                                      # 5. LANDING
        s = _ease((t - _T_LAND_START) / T_LAND_RAMP)
        return tuple(a + s * (b - a) for a, b in zip(extend, crouch))
    if t < _T_RECOVER_START:                                     # 6. SETTLE
        return crouch
    if t < _T_REST_START:                                        # 7. RECOVER
        s = _ease((t - _T_RECOVER_START) / T_RECOVER_RAMP)
        return tuple(a + s * (b - a) for a, b in zip(crouch, stand))
    return stand                                                 # REST


def jump_ctrl(t_in_cycle):
    """Full 12-actuator ctrl vector for one instant within a jump cycle.
    Coxa stays at the walking stance's splay angle on every leg (no yaw)."""
    femur, tibia = jump_targets(t_in_cycle)
    ctrl = []
    for leg in W.LEGS:
        neutral = -W.SPLAY if leg.startswith("F") else W.SPLAY
        coxa = W.COXA_SIGN[leg] * neutral
        ctrl += [coxa, femur, tibia]
    return np.radians(ctrl)


def load():
    model = mujoco.MjModel.from_xml_path(W.XML)
    data = mujoco.MjData(model)
    data.ctrl[:] = jump_ctrl(CYCLE)          # start at REST (= standing pose)
    for i, leg in enumerate(W.LEGS):
        for j, a in enumerate(data.ctrl[3 * i:3 * i + 3]):
            data.joint(f"{leg}_{['coxa', 'femur', 'tibia'][j]}").qpos = a
    mujoco.mj_forward(model, data)
    return model, data


MIN_FLIGHT_SAMPLES = 15   # contact chatter shorter than this (15 ms) doesn't count


def _flight_events(all_off_trace, height_trace, dt):
    """Collapse a per-step all-feet-off boolean trace into genuine contiguous
    flight events (runs at least MIN_FLIGHT_SAMPLES long), each with its
    duration and the peak height reached during that run. Short runs are
    contact chatter (the stiff legs flickering against the floor while
    settling), not real flight, and are dropped."""
    events = []
    i = 0
    n = len(all_off_trace)
    while i < n:
        if all_off_trace[i]:
            j = i
            while j < n and all_off_trace[j]:
                j += 1
            if j - i >= MIN_FLIGHT_SAMPLES:
                events.append(((j - i) * dt, max(height_trace[i:j])))
            i = j
        else:
            i += 1
    return events


def headless_test(n_jumps=4, settle_secs=1.0):
    model, data = load()
    trunk = model.body("trunk").id
    feet = [model.geom(f"{leg}_foot").id for leg in W.LEGS]
    floor = model.geom("floor").id
    dt = model.opt.timestep

    # let it settle onto the floor from the loaded pose before jumping
    while data.time < settle_secs:
        mujoco.mj_step(model, data)
    stand_height = data.xpos[trunk][2]
    print(f"standing height before jumping: {stand_height*100:.2f} cm\n")

    print(f"{'jump':>4} {'peak height':>12} {'liftoff':>8} {'flight time':>12} "
          f"{'chatter':>8} {'land tilt':>10} {'max torque':>11}")

    t_cycle_start = data.time
    for n in range(1, n_jumps + 1):
        all_off_trace, height_trace = [], []
        max_tilt = 0.0
        max_torque = 0.0
        cycle_end = t_cycle_start + CYCLE
        while data.time < cycle_end:
            tc = data.time - t_cycle_start
            data.ctrl[:] = jump_ctrl(tc)
            mujoco.mj_step(model, data)

            h = data.xpos[trunk][2]
            height_trace.append(h)
            down = {g for g in feet
                    if any((c.geom1 == floor and c.geom2 == g) or
                           (c.geom2 == floor and c.geom1 == g)
                           for c in data.contact[:data.ncon])}
            all_off_trace.append(len(down) == 0)
            up = data.xmat[trunk].reshape(3, 3)[:, 1]
            max_tilt = max(max_tilt, np.degrees(np.arccos(np.clip(up[2], -1, 1))))
            max_torque = max(max_torque, np.abs(data.actuator_force).max())

        events = _flight_events(all_off_trace, height_trace, dt)
        n_chatter = sum(all_off_trace) - sum(round(d / dt) for d, _ in events)
        if events:
            # report the highest genuine flight event this cycle
            flight_t, peak_h = max(events, key=lambda e: e[1])
            liftoff_cm = (peak_h - stand_height) * 100
            print(f"{n:4d} {peak_h*100:9.2f} cm {liftoff_cm:7.2f} cm "
                  f"{flight_t*1000:9.0f} ms {n_chatter*dt*1000:6.0f} ms "
                  f"{max_tilt:9.1f} deg {max_torque:10.3f} N*m")
        else:
            print(f"{n:4d} {'--':>9}    {'--':>7}    {'--':>9}    "
                  f"{n_chatter*dt*1000:6.0f} ms {max_tilt:9.1f} deg "
                  f"{max_torque:10.3f} N*m   (no feet-off run >= "
                  f"{MIN_FLIGHT_SAMPLES} ms - no real liftoff)")
        t_cycle_start = cycle_end

    final_h = data.xpos[trunk][2]
    print(f"\nfinal height after {n_jumps} jumps: {final_h*100:.2f} cm "
          f"(started at {stand_height*100:.2f} cm)")
    print(f"'flight time' / 'peak height' = the single longest contiguous "
          f"all-feet-off run (>= {MIN_FLIGHT_SAMPLES} ms) each cycle - a real jump, "
          "not contact noise.")
    print("'chatter' = total time feet registered off the ground in runs shorter "
          "than that (stiff legs flickering against the floor while settling, "
          "not actual flight) - included for transparency, not counted as airborne.")


def view():
    import mujoco.viewer
    model, data = load()
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.trackbodyid = model.body("trunk").id
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.distance, viewer.cam.elevation, viewer.cam.azimuth = 0.5, -15, 120
        t0_cycle = data.time
        while viewer.is_running():
            t0 = time.time()
            tc = (data.time - t0_cycle) % CYCLE
            data.ctrl[:] = jump_ctrl(tc)
            mujoco.mj_step(model, data)
            viewer.sync()
            dt = model.opt.timestep - (time.time() - t0)
            if dt > 0:
                time.sleep(dt)


if __name__ == "__main__":
    headless_test() if "--test" in sys.argv else view()
