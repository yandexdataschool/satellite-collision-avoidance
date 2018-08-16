import numpy as np


def rotate_velocity(vel, pos, angle):
    w = np.cross(pos, vel)
    norm_vel = np.linalg.norm(vel)
    x1 = np.cos(angle) / norm_vel
    x2 = np.sin(angle) / np.linalg.norm(w)
    rotated_vel = norm_vel * (x1 * np.array(vel) + x2 * w)
    return rotated_vel
