# We need to test basic funcitons, Environment, SpaceObject
import unittest


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

    def test_propagate_forward(self):
        self.assertTrue(True)

    def test_update_collision_probability(self):
        self.assertTrue(True)

    def test_act(self):
        self.assertTrue(True)


class TestSpaceObject(unittest.TestCase):

    def test_maneuver(self):
        self.assertTrue(True)

    def test_position(self):
        self.assertTrue(True)

    def test_(self):
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
