# Generate random collision situation.
import argparse
import sys

import numpy as np
import pykep as pk


from .generator_utils import rotate_velocity
from ..api import SpaceObject, Environment
from ..utils import write_environment


class Generator:
    """Generates random collision situation.

    TODO:
        better distributions parameters
        add user protected object?
        add user collision times?
    """

    def __init__(self, start_time, end_time):
        # TODO - random start/end time with duration?
        # TODO - random duration?
        if isinstance(start_time, (int, float)):
            start_time = pk.epoch(start_time, "mjd2000")
        elif isinstance(start_time, pk.core._core.epoch):
            pass
        else:
            raise TypeError("Invalid start_time type")

        if isinstance(end_time, (int, float)):
            end_time = pk.epoch(end_time, "mjd2000")
        elif isinstance(end_time, pk.core._core.epoch):
            pass
        else:
            raise TypeError("Invalid end_time type")

        self.start_time = start_time
        self.end_time = end_time

        self.protected = None
        self.debris = []

        self.collision_epochs = []

    def add_protected(self):
        """Add protected object."""

        # gravity parameter of the object (m^2/s^3)
        # TODO - add distribution?
        mu_self = 0.1

        # objects radii (meters)
        """
        http://www.businessinsider.com/size-of-most-famous-satellites-2015-10
        """
        radius = np.random.uniform(0.3, 55)

        # mimimual radius that is safe during a fly-by of the object (meters)
        # TODO - differs from radius?
        safe_radius = radius

        # protected fuel
        # TODO - add distribution?
        fuel = 10

        # six osculating keplerian elements (a,e,i,W,w,M) at the reference epoch
        # a (semi-major axis): meters
        # https://upload.wikimedia.org/wikipedia/commons/b/b4/Comparison_satellite_navigation_orbits.svg
        a = np.random.uniform(7e6, 8e6)
        # e (eccentricity): in interval [0, 1)
        e = np.random.uniform(0, 0.003)
        # i (inclination): radians
        # TODO - fix if i > pi? i > 2*pi?
        i = np.random.uniform(0, 2 * np.pi)
        # W (Longitude of the ascending node): radians
        W = np.random.uniform(0, 2 * np.pi)
        # w (Argument of periapsis): radians
        w = np.random.uniform(0, 2 * np.pi)
        # M (mean anomaly): radians
        M = np.random.uniform(0, 2 * np.pi)
        # Keplerian elements
        elements = [a, e, i, W, w, M]

        # protected object parameters
        params = {
            "epoch": self.start_time,
            "elements": elements,
            "mu_central_body": pk.MU_EARTH,
            "mu_self": mu_self,
            "radius": radius,
            "safe_radius": safe_radius,
            "fuel": fuel
        }

        self.protected = SpaceObject("PROTECTED", "osc", params)

    def add_debris(self, pos_sigma=0, vel_ratio_sigma=0.05,
                   i_threshold=0.5):
        """Add debris object to the orbit of protected object.

        Args:
            pos_sigma (float): standard deviation of debris position
                from protected one (meters).
            vel_ratio_sigma (float): standard deviation of debris and protected
                velocities ratio (m/s).
            i_threshold (float): minimum angle between debris and protected
                at collision time (radians) (<=pi/4).

        Raises:
            Exception: if the protected object was not added.

        TODO:
            add ValueErrors for args.

        """
        if not self.protected:
            raise Exception("no protected object")

        # TODO - indent?
        collision_time = np.random.uniform(
            self.start_time.mjd2000, self.end_time.mjd2000
        )

        collision_time = pk.epoch(collision_time, "mjd2000")
        self.collision_epochs.append(collision_time)

        # position (x, y, z) and velocity (Vx, Vy, Vz) of protected object
        pos_prot, vel_prot = self.protected.position(collision_time)

        # position and velocity of debris at collision time
        # TODO -  truncated normal?
        pos = np.random.normal(pos_prot, pos_sigma)
        rotate_angle = np.random.choice([
            np.random.uniform(i_threshold, np.pi - i_threshold),
            np.random.uniform(np.pi + i_threshold, 2 * np.pi - i_threshold)
        ])
        vel = rotate_velocity(vel_prot, pos, rotate_angle)
        vel = vel * np.random.normal(1, vel_ratio_sigma)

        # gravity parameter of the object (m^2/s^3)
        # TODO - add distribution?
        mu_self = 0.1

        # objects radii (meters)
        """
        https://www.nasa.gov/mission_pages/station/news/orbital_debris.html
        https://m.esa.int/Our_Activities/Operations/Space_Debris/Space_debris_by_the_numbers
        """
        radius = np.random.uniform(0.05, 1)

        # mimimual radius that is safe during a fly-by of the object (meters)
        # TODO - differs from radius?
        safe_radius = radius

        name = "DEBRIS" + str(len(self.debris))

        # protected object parameters
        params = {
            "epoch": collision_time,
            "pos": pos,
            "vel": vel,
            "mu_central_body": pk.MU_EARTH,
            "mu_self": mu_self,
            "radius": radius,
            "safe_radius": safe_radius,
            "fuel": 0
        }

        self.debris.append(SpaceObject(name, "eph", params))

    def get_env(self, *args, **kwargs):
        env = Environment(self.protected, self.debris,
                          self.start_time, self.end_time, *args, **kwargs)
        return env

    def save_env(self, save_path, *args, **kwargs):
        env = self.get_env(*args, **kwargs)
        write_environment(env, path)
        # with open(save_path, 'w') as f:
        #     f.write(f'{self.start_time.mjd2000}, {self.end_time.mjd2000}\n')
        #     f.write('osc\n')
        #     f.write(SpaceObject2srt(self.protected, self.start_time))
        #     for debr, epoch in zip(self.debris, self.collision_epochs):
        #         f.write(SpaceObject2srt(debr, epoch))

    def print_info(self):
        pass
