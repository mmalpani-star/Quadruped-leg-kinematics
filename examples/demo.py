"""
Basic usage demo for the quadruped leg kinematics framework.

Run from the repo root with:
    python examples/demo.py
"""
import os
import sys

# Make src/ importable without installing the package
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from leg import Leg

# A single leg with 3 cm coxa, 5.6 cm femur, 8 cm tibia links,
# starting at coxa=0deg, femur=45deg, tibia=90deg.
# NOTE: Leg's constructor takes the three JOINT ANGLES first, then the
# three LINK LENGTHS: Leg(coxa_angle, femur_angle, tibia_angle, coxa, femur, tibia)
leg = Leg(coxa_angle=0, femur_angle=45, tibia_angle=90, coxa=3, femur=5.6, tibia=8)

print("=== Forward kinematics ===")
foot_position = leg.forward_kinematics()
print(f"Foot position given current joint angles: {foot_position}")

print("\n=== Inverse kinematics (reachable target) ===")
result = leg.inverse_kinematics(x=8, y=2, z=-4)
print(f"Joint angles for foot target (8, 2, -4): {result}")

print("\n=== Inverse kinematics (unreachable target) ===")
result = leg.inverse_kinematics(x=50, y=0, z=0)
print(f"Result for an out-of-range target (50, 0, 0): {result}")
