import numpy as np

from space_navigator.api import SpaceObject


def SpaceObject2srt(space_object, collision_time):
    # TODO - different formats?
    name = space_object.get_name()
    time = collision_time.mjd2000
    elements = space_object.get_orbital_elements()

    # pykep debug
    if np.isnan(elements[5]):
        elements = list(elements)
        elements[5] = -np.pi
        elements = tuple(elements)

    mu_central_body = space_object.get_mu_central_body()
    mu_self = space_object.get_mu_self()
    radius = space_object.get_radius()
    safe_radius = space_object.get_safe_radius()

    fuel = space_object.get_fuel()

    s = name + "\n"
    s += f"{time}\n"
    s += str(elements).strip('()') + "\n"
    s += f"{mu_central_body}, {mu_self}, {radius}, {safe_radius}\n"
    s += f"{fuel}\n"

    return s


def rotate_velocity(vel, pos, angle):
    w = np.cross(pos, vel)
    norm_vel = np.linalg.norm(vel)
    x1 = np.cos(angle) / norm_vel
    x2 = np.sin(angle) / np.linalg.norm(w)
    rotated_vel = norm_vel * (x1 * np.array(vel) + x2 * w)
    return rotated_vel
