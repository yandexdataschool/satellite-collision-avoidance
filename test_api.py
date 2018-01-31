# We need to test basic funcitons, Environment, SpaceObject
import unittest

import pykep as pk
import numpy as np

from api import Environment, SpaceObject
from api import fuel_consumption
from api import MAX_PROPAGATION_STEP, MAX_FUEL_CONSUMPTION


class TestBasicFunctions(unittest.TestCase):

    def test_euclidean_distance(self):
        self.assertTrue(True)

    def test_fuel_consumption(self):
        self.assertTrue(True)

    def test_sum_coll_prob(self):
        self.assertTrue(True)

    def test_rV2ocs(self):
        self.assertTrue(True)

    def test_coll_prob_estimation_with_normal_assumption(self):
        self.assertTrue(True)

    def test_coll_prob_estimation_hutorovski_approach(self):
        self.assertTrue(True)

    def test_danger_debr_and_collision_prob(self):
        self.assertTrue(True)


class TestEnvironment(unittest.TestCase):

    def setUp(self):
        self.start_time = pk.epoch(1, "mjd2000")
        osculating_elements = (7800, 0.001, 1, 0, 0, 0)
        mu_central_body, mu_self, radius, safe_radius = 0.1, 0.1, 0.1, 0.1
        fuel = 10
        params = dict(elements=osculating_elements, epoch=self.start_time,
                      mu_central_body=mu_central_body,
                      mu_self=mu_self,
                      radius=radius,
                      safe_radius=safe_radius,
                      fuel=fuel)
        self.protected = SpaceObject("protected", "osc", params)

    def test_propagate_forward(self):
        env = Environment(self.protected, [], self.start_time)
        end_times = [self.start_time.mjd2000 + 0.1,
                     self.start_time.mjd2000 + MAX_PROPAGATION_STEP,
                     self.start_time.mjd2000 + MAX_PROPAGATION_STEP * 1.5]
        for end_time in end_times:
            env.propagate_forward(end_time)
            self.assertEqual(env.state["epoch"].mjd2000, end_time)
            env = Environment(self.protected, [], self.start_time)

        self.assertTrue(True)

    def test_update_collision_probability(self):
        self.assertTrue(True)

    def test_act_normal(self):
        env = Environment(self.protected, [], self.start_time)

        time_to_req = 2

        action = np.array([1, 1, 1, self.start_time.mjd2000, time_to_req])
        dV = action[:3]

        prev_fuel = env.protected.get_fuel()
        fuel_cons = fuel_consumption(dV)
        env.act(action)

        new_osculating_elements = (0.033223781639051674, 191657.01087007116,
                                   0.7859365480471975, 0.0, 5.668868048507697, -132706.35091427292)

        self.assertEqual(new_osculating_elements,
                         env.protected.satellite.osculating_elements(self.start_time))

        self.assertEqual(env.protected.get_fuel(), prev_fuel - fuel_cons)

        self.assertEqual(env.next_action.mjd2000,
                         self.start_time.mjd2000 + time_to_req)


class TestSpaceObject(unittest.TestCase):

    def setUp(self):
        self.start_time = pk.epoch(1, "mjd2000")
        osculating_elements = (7800, 0.001, 1, 0, 0, 0)
        mu_central_body, mu_self, radius, safe_radius = 0.1, 0.1, 0.1, 0.1
        fuel = 10
        self.params = dict(elements=osculating_elements, epoch=self.start_time,
                           mu_central_body=mu_central_body,
                           mu_self=mu_self,
                           radius=radius,
                           safe_radius=safe_radius,
                           fuel=fuel)

    def test_maneuver(self):
        satellite = SpaceObject("satellite", "osc", self.params)

        time_to_req = 2
        action = np.array([1, 1, 1, self.start_time.mjd2000, time_to_req])
        dV = action[:3]

        prev_fuel = satellite.get_fuel()
        fuel_cons = fuel_consumption(dV)
        satellite.maneuver(action[:4])

        new_osculating_elements = (0.033223781639051674, 191657.01087007116,
                                   0.7859365480471975, 0.0, 5.668868048507697, -132706.35091427292)

        self.assertEqual(new_osculating_elements,
                         satellite.satellite.osculating_elements(self.start_time))
        self.assertEqual(satellite.get_fuel(), prev_fuel - fuel_cons)

    def test_position(self):
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
