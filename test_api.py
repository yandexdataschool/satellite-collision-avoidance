# We need to test basic funcitons, Environment, SpaceObject
import unittest

import pykep as pk
import numpy as np

from api import Environment, SpaceObject
from api import fuel_consumption, sum_coll_prob
from api import MAX_PROPAGATION_STEP, MAX_FUEL_CONSUMPTION


class TestBasicFunctions(unittest.TestCase):

    def setUp(self):
        self.start_time = pk.epoch(1, "mjd2000")
        osculating_elements = (7800, 0.001, 1, 0, 0, 0)
        mu_central_body, mu_self, radius, safe_radius = 0.1, 0.1, 0.1, 0.1
        fuel = 10
        params = dict(
            elements=osculating_elements, epoch=self.start_time,
            mu_central_body=mu_central_body,
            mu_self=mu_self,
            radius=radius,
            safe_radius=safe_radius,
            fuel=fuel,
        )
        self.protected = SpaceObject("protected", "osc", params)

        debris_osculating_elements = (7800, 0.001, 90, 0, 0, 0)
        params["elements"] = debris_osculating_elements
        self.debris = [SpaceObject("Debris 1", "osc", params)]

    def test_sum_coll_prob(self):
        p = np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.3, 0.2],
        ])
        axises = [0, 1]
        wants = [
            np.array([0.46,  0.44,  0.44]),
            np.array([0.496,  0.664]),
        ]
        for axis, want in zip(axises, wants):
            self.assertTrue(np.allclose(sum_coll_prob(p, axis=axis), want))

    def test_coll_prob_estimation(self):
        # TODO: implement test after new approach will be added.
        self.assertTrue(True)

    def test_danger_debr_and_collision_prob(self):
        # TODO: implement test after new approach will be added.
        self.assertTrue(True)


class TestEnvironment(unittest.TestCase):

    def setUp(self):
        self.start_time = pk.epoch(1, "mjd2000")
        osculating_elements = (7800, 0.001, 1, 0, 0, 0)
        mu_central_body, mu_self, radius, safe_radius = 0.1, 0.1, 0.1, 0.1
        fuel = 10
        params = dict(
            elements=osculating_elements, epoch=self.start_time,
            mu_central_body=mu_central_body,
            mu_self=mu_self,
            radius=radius,
            safe_radius=safe_radius,
            fuel=fuel,
        )
        self.protected = SpaceObject("protected", "osc", params)

    def test_propagate_forward(self):
        env = Environment(self.protected, [], self.start_time)
        end_times = [
            self.start_time.mjd2000 + MAX_PROPAGATION_STEP * 100,
            self.start_time.mjd2000 + MAX_PROPAGATION_STEP,
            self.start_time.mjd2000 + MAX_PROPAGATION_STEP * 1.5,
        ]

        for end_time in end_times:
            env.propagate_forward(end_time)
            self.assertEqual(env.state["epoch"].mjd2000, end_time)
            env = Environment(self.protected, [], self.start_time)

    def test_update_collision_probability(self):
        # TODO: implement test after new approach will be added.
        self.assertTrue(True)

    def get_reward(self):
        # TODO: implement test after new approach will be added.
        self.assertTrue(True)

    def test_act_normal(self):
        env = Environment(self.protected, [], self.start_time)

        time_to_req = 2
        action = np.array([1, 1, 1, self.start_time.mjd2000, time_to_req])
        dV = action[:3]
        fuel_cons = fuel_consumption(dV)

        prev_fuel = env.protected.get_fuel()
        want_fuel = prev_fuel - fuel_cons
        want_next_action = self.start_time.mjd2000 + time_to_req
        want_osculating_elements = (
            0.033223781639051674, 191657.01087007116,
            0.7859365480471975, 0.0, 5.668868048507697, -132706.35091427292
        )

        env.act(action)

        self.assertEqual(want_osculating_elements,
                         env.protected.satellite.osculating_elements(self.start_time))
        self.assertEqual(env.protected.get_fuel(), want_fuel)
        self.assertEqual(env.next_action.mjd2000, want_next_action)

    def test_act_no_fuel(self):
        # TODO: implement test after decision on proper behavior.
        self.assertTrue(True)

    def test_act_impossible_action(self):
        # TODO: implement test after decision on proper behavior.
        self.assertTrue(True)


class TestSpaceObject(unittest.TestCase):

    def setUp(self):
        self.start_time = pk.epoch(1, "mjd2000")
        osculating_elements = (7800, 0.001, 1, 0, 0, 0)
        mu_central_body, mu_self, radius, safe_radius = 0.1, 0.1, 0.1, 0.1
        fuel = 10
        self.params = dict(
            elements=osculating_elements, epoch=self.start_time,
            mu_central_body=mu_central_body,
            mu_self=mu_self,
            radius=radius,
            safe_radius=safe_radius,
            fuel=fuel,
        )

    def test_maneuver(self):
        satellite = SpaceObject("satellite", "osc", self.params)

        time_to_req = 2
        action = np.array([1, 1, 1, self.start_time.mjd2000, time_to_req])
        dV = action[:3]
        fuel_cons = fuel_consumption(dV)

        prev_fuel = satellite.get_fuel()
        want_fuel = prev_fuel - fuel_cons

        want_osculating_elements = (
            0.033223781639051674, 191657.01087007116,
            0.7859365480471975, 0.0, 5.668868048507697, -132706.35091427292
        )

        satellite.maneuver(action[:4])

        self.assertEqual(want_osculating_elements,
                         satellite.satellite.osculating_elements(self.start_time))
        self.assertEqual(satellite.get_fuel(), want_fuel)

    def test_position(self):
        satellite = SpaceObject("satellite", "osc", self.params)

        want_pos = (
            7792.2, 0.0, 0.0
        )
        want_vel = (
            0.0, 0.0019365281494027572, 0.0030159638988915163
        )

        pos, vel = satellite.position(self.start_time)

        self.assertEqual(pos, want_pos)
        self.assertEqual(vel, want_vel)


if __name__ == '__main__':
    unittest.main()
