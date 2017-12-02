# Module api provides functionality for simulation of
# object movement in space , communication with the space environment
# by the Agent and state/reward exchanging.
#
# In the first implementation wi will have only one protected
# object. All other objects will be treated as space debris.
# As a first, we will observe only ideal satellite's trajectories,
# so that we can describe any object location at time t after the
# simulation has been started.

import pykep as pk
import numpy as np


def to_xyz(coordinates):
    """
    ф-я переводит координаты в xyz
    (пока случайно)
    """
    result = np.random.normal(size=[coordinates.shape[0], 3])
    return result


class Agent:
    """ Agent implements an agent to communicate with space Environment.
        Agent can make actions to the space environment by taking it's state
        and reward after the last action.
    """

    def __init__(self):
        """"""

    def get_action(self, s, r):
        """ Provides action  for protected object.
        Args:
            s - np.array, state of the environment as matrix.
            r - reward after last action.
        Returns:
            np.array([dVx, dVy, dVz, pk.epoch]) - vector of deltas for
            protected object and maneuver time.
        """
        action = np.zeros(4)
        return action


class Environment:
    """ Environment provides the space environment with space objects:
        satellites and debris, in it.
    """

    def __init__(self, protected, debris):
        """
            protected - SpaceObject, protected space object in Environment.
            debris - [SpaceObject], list of other space objects.
        """
        self.protected = protected
        self.debris = debris
        self.reward = 0
        self.state = dict()

    def get_reward(self, state, current_reward):
        """
        state: dict where keys:
            'coord': dict where:
                {'st': np.array shape (1, 6)},  satellite coordinates
                {'db': np.array shape (n_items, 6)},  debris coordinates
            'trajectory_deviation_coef': float
        current_reward: float
        ---
        output: float
        """
        sat_coordinates = to_xyz(state['coord']['st'])
        debr_coordinates = to_xyz(state['coord']['db'][0])

        # Euclidean distance
        # distances array
        distances = np.sum(
            (sat_coordinates - debr_coordinates) ** 2,
            axis=1) ** 0.5

        def distance_to_reward(dist_array):
            result = -1. / (dist_array + 0.001)
            return np.sum(result)
        collision_danger = distance_to_reward(distances)

        # fuel reward
        fuel_consumption = self.protected.fuel

        # trajectory reward
        traj_reward = -state['trajectory_deviation_coef']

        # whole reward
        reward = (
            collision_danger
            + fuel_consumption
            + traj_reward
        )
        new_reward = current_reward + reward
        self.reward = new_reward
        return new_reward

    def get_curr_reward(self):
        """ Provide the last calculated reward. """
        return self.reward

    def act(self, action):
        """ Change velocity for protected object.
        Args:
            action -- np.array([dVx, dVy, dVz, pk.epoch]), vector of deltas
            and maneuver time.
        """
        # TODO(dsdubov): populate the function.
        # Learn how to make action for pykep.planet [tle or keplerian] object.
        self.protected.act(action)

    def get_state(self, epoch):
        """ Provides environment state as dictionary
            and is_end flag.
        Args:
            epoch -- pk.epoch, at which time to return environment state.
        ---
        output: bool, dict()
        """
        st_pos, st_v = self.protected.position(epoch)
        st = np.hstack((np.array(st_pos), np.array(st_v)))
        N = len(self.debris)
        db = np.zeros((N, 6))
        for i in range(N):
            pos, v = self.debris[i].position(epoch)
            db[i] = np.hstack((np.array(pos), np.array(v)))

        coord = dict(st=st, db=db)
        self.state = dict(coord=coord, trajectory_deviation_coef=0.0)
        self.is_end = self.check_collision()
        # Provide state for reward function.
        return self.is_end, self.state

    def check_collision(self):
        """ Return True if collision with protected object appears. """
        # TODO: populate function. Compare protected and debris positions.
        return False


class SpaceObject:
    """ SpaceObject represents a satellite or a space debris. """

    def __init__(self, name, is_tle, params):
        """
        Args:
            name -- str, name of satellite or a space debris.
            is_tle -- bool, whether tle parameteres are provided.
            params -- dict, dictionary of space object coordinates. Keys are:
                "fuel" -- float, initial fuel capacity.

                for TLE cooridantes:
                    "tle1" -- str, tle line1
                    "tle2" -- str, tle line2

                otherwise:
                    "pos" -- [x, y, z], position (cartesian).
                    "vel" -- [Vx, Vy, Vz], velocity (cartesian).
                    "epoch" -- pykep.epoch, start time in epoch format.
                    "mu" -- float, gravity parameter.
        """

        self.fuel = params["fuel"]

        if is_tle:
            self.type = "tle"
            self.satellite = pk.planet.tle(
                params["tle_line1"], params["tle_line2"])
        else:
            self.type = "keplerian"
            self.satellite = pk.planet.keplerian(params["epoch"],
                                                 pk.ic2eq(r=params["pos"],
                                                          v=params["vel"],
                                                          mu=params["mu"]))
        self.satellite.name = name

    def act(self, action):
        """ Make manoeuvre for the object. """
        return

    def position(self, epoch):
        """ Provide SpaceObject position:
            (x, y, z) and (Vx, Vy, Vz), at given epoch.
        """
        pos, vel = self.satellite.eph(epoch)
        return pos, vel

    def get_name(self):
        return self.satellite.name
