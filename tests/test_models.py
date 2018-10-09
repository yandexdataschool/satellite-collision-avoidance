# We need to test basic funcitons, Environment, SpaceObject
import unittest

import pykep as pk
import numpy as np

from space_navigator.api import Environment, SpaceObject
from space_navigator.models import time_before_first_collision, time_before_early_first_maneuver


class TestTrainUtils(unittest.TestCase):
    # TODO: several debris objects.

    def setUp(self):
        self.start_time = pk.epoch(6600.0, "mjd2000")
        self.end_time = pk.epoch(6601.01, "mjd2000")
        self.step = 10e-6
        self.collision_epoch = 6601.008917730008

        osculating_elements = (7715189.3663724195, 0.0018082901282149317, 3.4236020095353483,
                               2.661901610522322, 4.058272401214204, 2.749441536415439)
        mu_central_body, mu_self, radius, safe_radius = 398600441800000.0, 0.1, 30., 30.
        fuel = 10
        params = dict(
            elements=osculating_elements, epoch=pk.epoch(6601.0, "mjd2000"),
            mu_central_body=mu_central_body,
            mu_self=mu_self,
            radius=radius,
            safe_radius=safe_radius,
            fuel=fuel,
        )
        self.protected = SpaceObject("protected", "osc", params)
        self.protected_orbital_period = self.protected.get_orbital_period()

        osculating_elements = (9078818.79566534, 0.14874348046655975, 2.522406378437988,
                               1.0402321037629632, 5.815814445605525, -0.002583510143235598)
        mu_central_body, mu_self, radius, safe_radius = 398600441800000.0, 0.1, 0.37, 0.37
        fuel = 0
        params = dict(
            elements=osculating_elements, epoch=pk.epoch(
                self.collision_epoch, "mjd2000"),
            mu_central_body=mu_central_body,
            mu_self=mu_self,
            radius=radius,
            safe_radius=safe_radius,
            fuel=fuel,
        )
        self.debris = SpaceObject("protected", "osc", params)

    def test_time_before_first_collision(self):

        env = Environment(self.protected, [], self.start_time, self.end_time)
        self.assertIsNone(time_before_first_collision(
            env, self.step))

        env = Environment(
            self.protected, [self.debris], self.start_time, self.end_time)
        expected = self.collision_epoch - self.start_time.mjd2000
        self.assertAlmostEqual(time_before_first_collision(
            env, self.step), expected, 5)

    def test_time_before_early_first_maneuver(self):
        time_before_collision = self.collision_epoch - self.start_time.mjd2000

        # no collisions
        env = Environment(self.protected, [], self.start_time, self.end_time)
        self.assertIsNone(time_before_early_first_maneuver(
            env, self.step))

        # low max_n_orbits
        max_n_orbits = 0.01
        max_time = max_n_orbits * self.protected_orbital_period
        env = Environment(
            self.protected,
            [self.debris],
            self.start_time,
            self.end_time
        )
        expected = time_before_collision - max_time
        self.assertAlmostEqual(time_before_early_first_maneuver(env, self.step, max_n_orbits),
                               expected, 5)

        # max time >= time before collision
        max_n_orbits = 999
        delta = 0.0001
        env = Environment(
            self.protected,
            [self.debris],
            pk.epoch(self.collision_epoch - delta),
            self.end_time
        )
        expected = 0
        self.assertEqual(time_before_early_first_maneuver(env, self.step, max_n_orbits),
                         expected)

        # max_n_orbits = 999
        env = Environment(
            self.protected,
            [self.debris],
            self.start_time,
            self.end_time
        )
        expected = (time_before_collision - self.protected_orbital_period /
                    2) % self.protected_orbital_period
        self.assertAlmostEqual(time_before_early_first_maneuver(env, self.step, max_n_orbits),
                               expected, 5)

        # max time < time before collision
        max_n_orbits = 1.7
        env = Environment(
            self.protected,
            [self.debris],
            self.start_time,
            self.end_time
        )
        expected = time_before_collision - 1.5 * self.protected_orbital_period
        self.assertAlmostEqual(
            time_before_early_first_maneuver(env, self.step, max_n_orbits),
            expected, 5)

if __name__ == '__main__':
    unittest.main()
