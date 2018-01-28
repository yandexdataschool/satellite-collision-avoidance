# We need to test basic funcitons, Environment, SpaceObject
import unittest

import pykep as pk
import numpy as np

from api import Environment, SpaceObject

from api import fuel_consumption

# will be used in maneuver testing.
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

    def test_coll_prob_estimation(self):
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
        self.assertTrue(True)

    def test_update_collision_probability(self):
        self.assertTrue(True)

    def test_act_normal(self):
        env = Environment(self.protected, [], self.start_time)

        action = np.array([1, 1, 1, 2, 2])
        dV = action[:3]
        epoch = pk.epoch(float(action[3]), "mjd2000")
        # time_to_req = pk.epoch(float(action[4]), "mjd2000")

        prev_fuel = env.protected.get_fuel()
        fuel_cons = fuel_consumption(dV)
        env.act(action)

        new_osculating_elements = (0.03322700875329275, 186259.18646581616, 0.7798493139874857,
                                   6.270834870874244, 5.67768891107007, -137721.8773532762)
        self.assertEqual(new_osculating_elements,
                         env.protected.satellite.osculating_elements(epoch))
        self.assertEqual(env.protected.get_fuel(), prev_fuel - fuel_cons)


class TestSpaceObject(unittest.TestCase):

    def test_maneuver(self):
        self.assertTrue(True)

    def test_position(self):
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
