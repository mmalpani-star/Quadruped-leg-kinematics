
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.animation import FuncAnimation
from collections import deque

from numpyfk3d import Leg

# Initialize leg
leg = Leg(3, 5.6, 8)

# Initial joint angles
theta_c = 45
theta_f = 45
theta_t = 30

# Fixed body attachment point (coxa servo center)
body = np.array([0.0, 0.0, 0.0])

# Initial forward kinematics
femur_servo = leg.coxa(theta_c)
knee = leg.femur(theta_c, theta_f)
foot = leg.tibia(theta_c, theta_f, theta_t)


# Function to calculate and display link lengths
def validate_lengths(body, femur_servo, knee, foot):

    coxa_length = np.linalg.norm(femur_servo - body)
    femur_length = np.linalg.norm(knee - femur_servo)
    tibia_length = np.linalg.norm(foot - knee)

    print("\n--- Link Length Validation ---")
    print(f"Coxa:  {coxa_length:.4f} cm (Expected: 3.0 cm)")
    print(f"Femur: {femur_length:.4f} cm (Expected: 5.6 cm)")
    print(f"Tibia: {tibia_length:.4f} cm (Expected: 8.0 cm)")


# Validate initial configuration
validate_lengths(body, femur_servo, knee, foot)


# Create figure
fig = plt.figure(figsize=(11, 9))
ax = fig.add_axes([0.1, 0.3, 0.8, 0.65], projection='3d')

# Plot leg segments: body -> femur servo -> knee -> foot
line, = ax.plot(
    [body[0], femur_servo[0], knee[0], foot[0]],
    [body[1], femur_servo[1], knee[1], foot[1]],
    [body[2], femur_servo[2], knee[2], foot[2]],
    marker='o',
    linewidth=3
)

# Joint markers
body_point = ax.scatter(
    *body, color='red', s=80, label='Body / Coxa servo'
)

femur_servo_point = ax.scatter(
    *femur_servo, color='blue', s=60, label='Femur servo'
)

knee_point = ax.scatter(
    *knee, color='green', s=60, label='Knee'
)

foot_point = ax.scatter(
    *foot, color='orange', s=60, label='Foot'
)

# Fixed coxa servo axis: parallel to global Y
servo_axis, = ax.plot(
    [0, 0],
    [-2, 2],
    [0, 0],
    color='purple',
    linewidth=4,
    label='Coxa servo axis (Y)'
)

# Fixed global coordinate frame
axis_length = 5

ax.quiver(0, 0, 0, axis_length, 0, 0,
          color='red', linewidth=2)

ax.quiver(0, 0, 0, 0, axis_length, 0,
          color='green', linewidth=2)

ax.quiver(0, 0, 0, 0, 0, axis_length,
          color='blue', linewidth=2)

# Axis labels
ax.text(axis_length + 1, 0, 0, 'X', color='red')
ax.text(0, axis_length + 1, 0, 'Y', color='green')
ax.text(0, 0, axis_length + 1, 'Z', color='blue')

# Graph settings
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Quadruped Leg - Forward Kinematics")

ax.set_xlim(-18, 18)
ax.set_ylim(-18, 18)
ax.set_zlim(-18, 18)
ax.set_box_aspect([1, 1, 1])

ax.legend()

# Slider positions
ax_coxa = fig.add_axes([0.2, 0.19, 0.65, 0.025])
ax_femur = fig.add_axes([0.2, 0.12, 0.65, 0.025])
ax_tibia = fig.add_axes([0.2, 0.05, 0.65, 0.025])

# Sliders
coxa_slider = Slider(
    ax_coxa, 'Coxa', -90, 90, valinit=theta_c
)

femur_slider = Slider(
    ax_femur, 'Femur', -90, 90, valinit=theta_f
)

tibia_slider = Slider(
    ax_tibia, 'Tibia', -90, 90, valinit=theta_t
)


# Update function
def update(val):

    theta_c_new = coxa_slider.val
    theta_f_new = femur_slider.val
    theta_t_new = tibia_slider.val

    # Recalculate joint positions
    femur_servo_new = leg.coxa(theta_c_new)

    knee_new = leg.femur(
        theta_c_new, theta_f_new
    )

    foot_new = leg.tibia(
        theta_c_new, theta_f_new, theta_t_new
    )

    # Validate link lengths
    validate_lengths(
        body, femur_servo_new, knee_new, foot_new
    )

    # Update leg segments
    line.set_data_3d(
        [body[0], femur_servo_new[0], knee_new[0], foot_new[0]],
        [body[1], femur_servo_new[1], knee_new[1], foot_new[1]],
        [body[2], femur_servo_new[2], knee_new[2], foot_new[2]]
    )

    # Update moving joint markers
    femur_servo_point._offsets3d = (
        [femur_servo_new[0]],
        [femur_servo_new[1]],
        [femur_servo_new[2]]
    )

    knee_point._offsets3d = (
        [knee_new[0]], [knee_new[1]], [knee_new[2]]
    )

    foot_point._offsets3d = (
        [foot_new[0]], [foot_new[1]], [foot_new[2]]
    )

    # Body marker and servo axis stay fixed
    fig.canvas.draw_idle()


# Connect sliders
coxa_slider.on_changed(update)
femur_slider.on_changed(update)
tibia_slider.on_changed(update)


# ============================================================
# ANIMATION (added on top of the existing slider visualization)
# ============================================================
#
# How it works:
#   - Each animation frame computes three smooth sine-wave joint angles.
#   - The EXISTING FK methods (leg.coxa / leg.femur / leg.tibia) turn those
#     angles into femur-servo, knee and foot positions.
#   - The existing leg line and joint markers are updated with those positions.
#   - The sliders are moved silently (eventson=False) so they show the current
#     angles without triggering the slider update() callback.
#   - Touching a slider pauses the animation, so manual control always wins.
#   - Pressing "Resume" smoothly blends from the slider pose back into the
#     animated motion (no sudden jump).

# --- Animation settings ---
ANIM_INTERVAL_MS = 33        # ~30 frames per second
CYCLE_SECONDS = 4.0          # time for one full motion cycle
BLEND_FRAMES = 30            # frames used to ease from slider pose into motion
TRAIL_LENGTH = 400           # number of recent foot positions kept in the trail
LENGTH_TOLERANCE = 1e-6      # allowed link-length error (cm)


def animated_angles(t):
    """Smooth, periodic joint angles (degrees) at time t (seconds).

    All values stay well inside the slider limits of -90..90 degrees.
    """
    w = 2 * np.pi / CYCLE_SECONDS
    theta_c_anim = 20 * np.sin(w * t)                      # -20 .. 20
    theta_f_anim = 40 + 25 * np.sin(w * t + np.pi / 2)     #  15 .. 65
    theta_t_anim = 20 + 30 * np.sin(w * t + np.pi)         # -10 .. 50
    return theta_c_anim, theta_f_anim, theta_t_anim


# Foot trajectory trail (bounded: deque drops the oldest points automatically)
trail_points = deque(maxlen=TRAIL_LENGTH)
trail_line, = ax.plot(
    [], [], [],
    color='magenta', linewidth=1.5, alpha=0.6,
    label='Foot trail'
)
ax.legend()  # refresh legend so the trail appears in it

# Animation state
anim_state = {
    'running': True,
    'time': 0.0,           # animation clock (only advances while running)
    'blend_frame': 0,      # counts up to BLEND_FRAMES after start/resume
    'blend_start': (theta_c, theta_f, theta_t),  # pose to blend from
}


def set_sliders_silently(c, f, t):
    """Move the sliders to show the current angles WITHOUT firing update()."""
    for slider, value in ((coxa_slider, c), (femur_slider, f), (tibia_slider, t)):
        slider.eventson = False
        slider.set_val(value)
        slider.eventson = True


def check_lengths_silently(femur_servo_pos, knee_pos, foot_pos):
    """Check link lengths every frame; only print if something is wrong."""
    lengths = (
        np.linalg.norm(femur_servo_pos - body),
        np.linalg.norm(knee_pos - femur_servo_pos),
        np.linalg.norm(foot_pos - knee_pos),
    )
    for name, measured, expected in zip(
        ('Coxa', 'Femur', 'Tibia'), lengths, (leg.lc, leg.lf, leg.lt)
    ):
        if abs(measured - expected) > LENGTH_TOLERANCE:
            print(f"WARNING: {name} length {measured:.6f} cm "
                  f"!= expected {expected} cm")


def animate(frame):
    """Called by FuncAnimation for every frame."""
    if not anim_state['running']:
        return

    # 1) Target angles from the periodic trajectory
    anim_state['time'] += ANIM_INTERVAL_MS / 1000.0
    target = animated_angles(anim_state['time'])

    # 2) Ease in from the blend-start pose (avoids jumps on start/resume)
    if anim_state['blend_frame'] < BLEND_FRAMES:
        anim_state['blend_frame'] += 1
        s = anim_state['blend_frame'] / BLEND_FRAMES
        s = 0.5 - 0.5 * np.cos(np.pi * s)   # smooth 0 -> 1 ease curve
        start = anim_state['blend_start']
        c, f, t = (a + s * (b - a) for a, b in zip(start, target))
    else:
        c, f, t = target

    # 3) Forward kinematics using the EXISTING Leg methods
    femur_servo_a = leg.coxa(c)
    knee_a = leg.femur(c, f)
    foot_a = leg.tibia(c, f, t)

    check_lengths_silently(femur_servo_a, knee_a, foot_a)

    # 4) Update the existing leg line and moving joint markers
    line.set_data_3d(
        [body[0], femur_servo_a[0], knee_a[0], foot_a[0]],
        [body[1], femur_servo_a[1], knee_a[1], foot_a[1]],
        [body[2], femur_servo_a[2], knee_a[2], foot_a[2]]
    )
    femur_servo_point._offsets3d = (
        [femur_servo_a[0]], [femur_servo_a[1]], [femur_servo_a[2]]
    )
    knee_point._offsets3d = ([knee_a[0]], [knee_a[1]], [knee_a[2]])
    foot_point._offsets3d = ([foot_a[0]], [foot_a[1]], [foot_a[2]])

    # 5) Foot trail
    trail_points.append(foot_a)
    trail = np.array(trail_points)
    trail_line.set_data_3d(trail[:, 0], trail[:, 1], trail[:, 2])

    # 6) Keep the sliders in sync with the animated angles
    set_sliders_silently(c, f, t)

    # Body marker and coxa servo axis are never touched -> they stay fixed.


# --- Pause / Resume ---
ax_button = fig.add_axes([0.42, 0.235, 0.16, 0.04])
pause_button = Button(ax_button, 'Pause')


def pause_animation():
    if anim_state['running']:
        anim_state['running'] = False
        anim.pause()
        pause_button.label.set_text('Resume')
        fig.canvas.draw_idle()


def resume_animation():
    if not anim_state['running']:
        # Start blending from wherever the sliders currently are
        anim_state['blend_start'] = (
            coxa_slider.val, femur_slider.val, tibia_slider.val
        )
        anim_state['blend_frame'] = 0
        anim_state['running'] = True
        pause_button.label.set_text('Pause')
        anim.resume()


def toggle_animation(event):
    if anim_state['running']:
        pause_animation()
    else:
        resume_animation()


pause_button.on_clicked(toggle_animation)


def pause_on_manual_slider(val):
    """Runs only for real user slider moves (animation moves them silently)."""
    pause_animation()


coxa_slider.on_changed(pause_on_manual_slider)
femur_slider.on_changed(pause_on_manual_slider)
tibia_slider.on_changed(pause_on_manual_slider)

# Keep a reference to the animation so it is not garbage-collected.
anim = FuncAnimation(
    fig, animate,
    interval=ANIM_INTERVAL_MS,
    blit=False,
    cache_frame_data=False
)

plt.show()
