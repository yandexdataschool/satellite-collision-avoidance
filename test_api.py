# We need to test basic funcitons, Environment, SpaceObject
import unittest

import pykep as pk
import numpy as np

from api import Environment, SpaceObject
from api import fuel_consumption, sum_coll_prob
from api import MAX_PROPAGATION_STEP, MAX_FUEL_CONSUMPTION
from api import CollProbEstimation


class TestBasicFunctions(unittest.TestCase):

    def setUp(self):
        self.start_time = pk.epoch(1, "mjd2000")
        osculating_elements = (7800000, 0.001, 0.017453292519943295, 0, 0, 0)
        mu_central_body, mu_self, radius, safe_radius = 398600800000000, 0.1, 0.1, 0.1
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

        debris_osculating_elements = (
            7800000, 0.001, 1.5707963267948966, 0, 0, 0)
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

    def test_danger_debr_and_collision_prob(self):
        # TODO: implement test after new approach will be added.
        self.assertTrue(True)


class TestCollProbEstimation(unittest.TestCase):

    def test_ChenBai_approach(self):
        estimator = CollProbEstimation()
        # collision cross-section radii of ISS and the debris
        rV1 = np.array([
            3126018.8, 5227146.1, -2891302.9, -3298.0, 4758.7, 5054.3
        ])  # meters, m/s
        rV2 = np.array([
            3124368.5, 5226004.2, -2889944.6, -7772.6, 1930.8, -2758.0
        ])  # meters, m/s
        # sizes
        cs_r1 = 100  # meters
        cs_r2 = 0.13  # meters
        # sigma/ meters
        sigma_1N = 554.8968
        sigma_1T = 6185.655
        sigma_1W = 1943.3925
        sigma_2N = 871.7616
        sigma_2T = 12306.207
        sigma_2W = 921.0618
        probability = estimator.ChenBai_approach(
            rV1, rV2,
            cs_r1, cs_r2,
            sigma_1N, sigma_1T, sigma_1W,
            sigma_2N, sigma_2T, sigma_2W
        )
        self.assertAlmostEqual(probability, 4.749411e-5)


class TestEnvironment(unittest.TestCase):

    def setUp(self):
        self.start_time = pk.epoch(1, "mjd2000")
        osculating_elements = (7800000, 0.001, 0.017453292519943295, 0, 0, 0)
        mu_central_body, mu_self, radius, safe_radius = 398600800000000, 0.1, 0.1, 0.1
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
            7802223.307087888, 0.0012922495763325007, 0.017590560250921223,
            0.0, 6.174706362516025, 0.10819939983993268
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
        osculating_elements = (7800000, 0.001, 0.017453292519943295, 0, 0, 0)
        mu_central_body, mu_self, radius, safe_radius = 398600800000000, 0.1, 0.1, 0.1
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
            7802223.307087888, 0.0012922495763325007, 0.017590560250921223,
            0.0, 6.174706362516025, 0.10819939983993268
        )

        satellite.maneuver(action[:4])

        self.assertEqual(want_osculating_elements,
                         satellite.satellite.osculating_elements(self.start_time))
        self.assertEqual(satellite.get_fuel(), want_fuel)

    def test_position(self):
        satellite = SpaceObject("satellite", "osc", self.params)

        want_pos = (
            7792200, 0, 0
        )
        want_vel = (
            0.0, 7154.675252184611, 124.88532106719391
        )

        pos, vel = satellite.position(self.start_time)

        self.assertEqual(pos, want_pos)
        self.assertEqual(vel, want_vel)


if __name__ == '__main__':
    unittest.main()
