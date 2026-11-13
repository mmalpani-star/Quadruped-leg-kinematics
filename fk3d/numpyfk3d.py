
import numpy as np

class Leg:
    def __init__(self, lc, lf, lt):
        self.lc = lc
        self.lf = lf
        self.lt = lt

    # Coxa endpoint in global coordinates
    def coxa(self, theta_c):
        return np.array([
            self.lc * np.cos(np.radians(theta_c)),
            0,
            -self.lc * np.sin(np.radians(theta_c))
        ])

    # Knee position in global coordinates
    def femur(self, theta_c, theta_f):

        hip_position = self.coxa(theta_c)

        theta_c_rad = np.radians(theta_c)
        theta_f_rad = np.radians(theta_f)

        femur_rel_coxa = np.array([
            [np.cos(theta_c_rad), 0, np.sin(theta_c_rad)],
            [0, 1, 0],
            [-np.sin(theta_c_rad), 0, np.cos(theta_c_rad)]
        ])

        femur_local = np.array([
            self.lf * np.cos(theta_f_rad),
            self.lf * np.sin(theta_f_rad),
            0
        ])

        knee_position = hip_position + femur_rel_coxa @ femur_local

        return knee_position

    # Foot position in global coordinates
    def tibia(self, theta_c, theta_f, theta_t):

        knee_position = self.femur(theta_c, theta_f)

        theta_c_rad = np.radians(theta_c)
        theta_f_rad = np.radians(theta_f)
        theta_t_rad = np.radians(theta_t)

        femur_rel_coxa = np.array([
            [np.cos(theta_c_rad), 0, np.sin(theta_c_rad)],
            [0, 1, 0],
            [-np.sin(theta_c_rad), 0, np.cos(theta_c_rad)]
        ])

        femur_rotation = np.array([
            [np.cos(theta_f_rad), -np.sin(theta_f_rad), 0],
            [np.sin(theta_f_rad), np.cos(theta_f_rad), 0],
            [0, 0, 1]
        ])

        tibia_rel_femur = np.array([
            [np.cos(theta_t_rad), -np.sin(theta_t_rad), 0],
            [np.sin(theta_t_rad), np.cos(theta_t_rad), 0],
            [0, 0, 1]
        ])

        tibia_local = np.array([self.lt, 0, 0])

        foot_position = (
            knee_position
            + femur_rel_coxa @ femur_rotation
            @ tibia_rel_femur @ tibia_local
        )

        return foot_position


# Testing
if __name__ == "__main__":

    leg = Leg(3, 5.6, 8)

    print("Test 1:", leg.tibia(0, 0, 0))
    print("Test 2:", leg.tibia(0, 90, 0))
    print("Test 3:", leg.tibia(0, 30, 40))
    