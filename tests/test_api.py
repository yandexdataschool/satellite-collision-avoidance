# We need to test basic funcitons, Environment, SpaceObject
import unittest

import pykep as pk
import numpy as np

from space_navigator.api import Environment, SpaceObject
from space_navigator.api import MAX_FUEL_CONSUMPTION
from space_navigator.api import (
    fuel_consumption, sum_coll_prob, lower_estimate_of_time_to_conjunction,
    reward_func_0, reward_func, correct_angular_deviations, SEC_IN_DAY,
)

from space_navigator.collision import CollProbEstimator


class TestBasicFunctions(unittest.TestCase):

    def test_fuel_consumption(self):
        vector = [1, 2, 3]
        self.assertEqual(fuel_consumption(vector),
                         sum([i**2 for i in vector])**0.5)

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

    def test_lower_estimate_of_time_to_conjunction(self):
        # toy tests
        prot_rV = np.array([
            [0, 0, 0, 10, 0, 0]
        ])
        debr_rV = np.array([
            [0, 0, 100, 10, 0, 0],
            [0, 0, 100, 10, 0, 0],
            [0, 0, 100, 5, 5, 5],
        ])

        crit_distance = 10
        dangerous_debris, distances, time_to_conjunction = lower_estimate_of_time_to_conjunction(
            prot_rV, debr_rV, crit_distance)
        self.assertFalse(dangerous_debris)
        self.assertFalse(distances)
        self.assertEqual(time_to_conjunction * SEC_IN_DAY, 4.5)

        crit_distance = 100
        dangerous_debris, distances, time_to_conjunction = lower_estimate_of_time_to_conjunction(
            prot_rV, debr_rV, crit_distance)
        self.assertEqual(list(dangerous_debris), [0, 1, 2])
        self.assertEqual(list(distances), [100.] * 3)
        self.assertEqual(time_to_conjunction, 0.)

        # test zeros
        prot_rV = np.array([
            [0, 0, 0, 0, 0, 0]
        ])
        debr_rV = np.array([
            [0, 0, 100, 0, 0, 0],
        ])
        crit_distance = 10
        dangerous_debris, distances, time_to_conjunction = lower_estimate_of_time_to_conjunction(
            prot_rV, debr_rV, crit_distance)
        self.assertEqual(time_to_conjunction, float("inf"))

        # test empty
        debr_rV = np.empty((0, 6))
        dangerous_debris, distances, time_to_conjunction = lower_estimate_of_time_to_conjunction(
            prot_rV, debr_rV, crit_distance)
        self.assertFalse(dangerous_debris)
        self.assertFalse(distances)
        self.assertEqual(time_to_conjunction, float("inf"))

    def test_reward_func(self):
        thr = 10
        y_thr = -1
        thr_times_exceeded = 2
        y_thr_times_exceeded = -100

        # reward_func_0
        def func(value, thr=thr):
            return reward_func_0(
                value, thr, y_thr, thr_times_exceeded, y_thr_times_exceeded)
        self.assertEqual(func(0), 0)
        self.assertEqual(func(thr), y_thr)
        self.assertEqual(func(thr * thr_times_exceeded), y_thr_times_exceeded)

        # reward_func
        n = 100
        value_arr = np.linspace(0, thr * thr_times_exceeded, n)
        thr_arr = np.full(n, thr)
        self.assertTrue(np.array_equal(
            reward_func(value_arr, thr_arr, func),
            np.array([func(v, thr) for v in value_arr]),
        ))
        n = 2
        self.assertTrue(np.array_equal(
            reward_func(np.ones(n), np.full(n, np.nan), func), np.zeros(n))
        )

    def test_correct_angular_deviations(self):
        delta = 0.01
        test_angles = np.array(
            [np.pi * 2 - delta, -np.pi * 2 + delta, delta, -delta],
        )
        corrected_angles = np.array(
            [-delta, delta, delta, -delta],
        )
        correct_angular_deviations(test_angles)
        self.assertTrue(np.all(np.isclose(
            test_angles, corrected_angles
        )))


class TestCollProbEstimation(unittest.TestCase):

    def test_ChenBai_approach(self):
        """
        Example from book:
            page 177,
            Lei Chen, Xian-Zong Bai, Yan-Gang Liang, Ke-Bo Li,
            "Orbital Data Applications for Space Objects",
            2017.
        """
        estimator = CollProbEstimator.ChenBai_approach
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
        probability = estimator(
            rV1, rV2,
            cs_r1, cs_r2,
            sigma_1N, sigma_1T, sigma_1W,
            sigma_2N, sigma_2T, sigma_2W
        )
        self.assertAlmostEqual(probability, 4.749411e-5)
        self.assertEqual(1, estimator(np.ones(6), np.ones(6)))


class TestEnvironment(unittest.TestCase):

    def setUp(self):
        self.start_time = pk.epoch(1, "mjd2000")
        self.end_time = None

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
        env = Environment(self.protected, [], self.start_time, self.end_time)
        step = 10e-6

        end_times = [
            self.start_time.mjd2000 + step * 100,
            self.start_time.mjd2000 + step,
            self.start_time.mjd2000 + step * 1.5,
        ]

        for end_time in end_times:
            env.end_time = pk.epoch(end_time, "mjd2000")
            env.propagate_forward(end_time, step)
            self.assertEqual(env.state["epoch"].mjd2000, end_time)
            env.reset()

    def test_update_distances_and_probabilities_prior_to_current_conjunction(self):
        # TODO: implement test after new approach will be added.
        self.assertTrue(True)

    def test_get_collision_probability(self):
        # TODO: implement test after new approach will be added.
        self.assertTrue(True)

    def get_reward(self):
        # TODO: implement test after new approach will be added.
        self.assertTrue(True)

    def test_act_normal(self):
        env = Environment(self.protected, [], self.start_time, self.end_time)

        time_to_req = 2
        action = np.array([1, 1, 1, time_to_req])
        dV = action[: 3]
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

    def test_reset(self):
        # TODO: implement test.
        self.assertTrue(True)

    def test__update_distances_and_probabilities_prior_to_current_conjunction(self):
        # TODO: implement test.
        self.assertTrue(True)

    def test__update_total_collision_probability(self):
        # TODO: implement test.
        self.assertTrue(True)

    def test__update_trajectory_deviation(self):
        # TODO: implement test.
        self.assertTrue(True)

    def test__update_reward(self):
        # TODO: implement test.
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
        action = np.array([1, 1, 1, time_to_req])
        dV = action[:3]
        fuel_cons = fuel_consumption(dV)

        prev_fuel = satellite.get_fuel()
        want_fuel = prev_fuel - fuel_cons

        want_osculating_elements = (
            7802223.307087888, 0.0012922495763325007, 0.017590560250921223,
            0.0, 6.174706362516025, 0.10819939983993268
        )

        satellite.maneuver(action[:3], self.start_time)

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

    def test_osculating_elements(self):
        # TODO: implement test (test time changing).
        self.assertTrue(True)

    def test_get_orbital_period(self):
        # TODO: implement test (does pykep have such method?).
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
