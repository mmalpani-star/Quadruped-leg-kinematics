"""
Visualizes the leg's reachable workspace using the real Leg.forward_kinematics()
code, swept across the joint limits already defined in leg.py's Servo objects
(coxa: 40-145deg, femur: 30-90deg, tibia: 15-90deg).

Run from the repo root with:
    python examples/visualize_workspace.py

Writes diagrams/workspace_visualization.png
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from leg import Leg

COXA_LEN, FEMUR_LEN, TIBIA_LEN = 3, 5.6, 8

# Joint limits, taken directly from the Servo(...) definitions in leg.py
COXA_RANGE = (40, 145)
FEMUR_RANGE = (30, 90)
TIBIA_RANGE = (15, 90)


def foot_xyz(coxa_angle, femur_angle, tibia_angle):
    leg = Leg(coxa_angle, femur_angle, tibia_angle, COXA_LEN, FEMUR_LEN, TIBIA_LEN)
    x, y, z = (float(v) for v in leg.forward_kinematics().strip("()").split(","))
    return x, y, z


fig, axes = plt.subplots(1, 2, figsize=(11, 5))

# --- Left: side-view envelope. Sweep femur & tibia through their full range
#     at a fixed mid-range coxa angle, and plot radial reach vs. height. ---
coxa_fixed = sum(COXA_RANGE) / 2
radials, heights = [], []
for fa in np.linspace(*FEMUR_RANGE, 30):
    for ta in np.linspace(*TIBIA_RANGE, 30):
        x, y, z = foot_xyz(coxa_fixed, fa, ta)
        radials.append((x**2 + y**2) ** 0.5)
        heights.append(z)

axes[0].scatter(radials, heights, s=4, alpha=0.35, color="#2f6feb")
axes[0].set_xlabel("radial reach")
axes[0].set_ylabel("height (z)")
axes[0].set_title(f"Reachable envelope\n(femur/tibia swept, coxa fixed at {coxa_fixed:.0f}°)")
axes[0].grid(alpha=0.2)

# --- Right: top-view sweep. Fix femur/tibia at mid-range and sweep coxa
#     through its full range to trace the foot's arc in the XY plane. ---
femur_mid = sum(FEMUR_RANGE) / 2
tibia_mid = sum(TIBIA_RANGE) / 2
xs, ys = [], []
for ca in np.linspace(*COXA_RANGE, 60):
    x, y, z = foot_xyz(ca, femur_mid, tibia_mid)
    xs.append(x)
    ys.append(y)

axes[1].plot(xs, ys, color="#c2410c", linewidth=2)
axes[1].scatter([0], [0], color="black", zorder=5, label="coxa joint")
axes[1].set_xlabel("x")
axes[1].set_ylabel("y")
axes[1].set_title(f"Foot arc as coxa rotates\n({COXA_RANGE[0]}°\N{EN DASH}{COXA_RANGE[1]}°, femur/tibia fixed mid-range)")
axes[1].set_aspect("equal")
axes[1].legend(loc="upper left", fontsize=8)
axes[1].grid(alpha=0.2)

plt.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), "..", "diagrams", "workspace_visualization.png")
plt.savefig(out_path, dpi=150)
print(f"Saved: {out_path}")
