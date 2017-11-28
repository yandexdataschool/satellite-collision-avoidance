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
            Vector of ΔV for protected object.
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
        self.state = EnvState()

    def get_state(self, params):
        """ Provides environment state.
            params -- dict(), which parameters to return in state.
        """
        objects = [self.protected] + self.debris
        return self.state.get_state(params, objects)

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
        debr_coordinates = to_xyz(state['coord']['db'])

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
        fuel_consumption = self.protected.f

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
            action -- np.array([dVx, dVy, dVz, pk.epoch]), vector of deltas.
        """
        # TODO(dsdubov): populate the function.
        # Learn how to make action for pykep.planet [tle or keplerian] object.
        self.protected.act(action)


class EnvState:
    """ Describes Environment state."""

    def __init__(self):
        """"""
        self.is_end = False

    def get_state(self, params, objects):
        """ Provides the state of the environment as matrix.
        Args:
            params - dict(), which parameteres to include into state.
            in the first implementation may consist of keys: "coord", "v".
        """
        # TODO(dsdubov): populate the function.
        # Provide state for reward function.
        return dict(coord=dict(st=np.zeros((1, 6)),
                               db=np.zeros((10, 6))),
                    trajectory_deviation_coef=0.0)

    def get_coordinates(self, t, objects):
        """"""

    def get_v(self):
        """"""


class SpaceObject:
    """ SpaceObject represents a satellite or a space debris. """

    def __init__(self, name, is_tle, params):
        """
        Args:
            name -- str, name of satellite or a space debris.
            is_tle -- bool, whether tle parameteres are provided.
            params -- dict, dictionary of space object coordinates. Keys are:
                "f" -- float, initial fuel capacity.

                for TLE cooridantes:
                    "tle1" -- str, tle line1
                    "tle2" -- str, tle line2

                otherwise:
                    "pos" -- [x, y, z], position (cartesian).
                    "v" -- [Vx, Vy, Vz], velocity (cartesian).
                    "epoch" -- pykep.epoch, start time in epoch format.
                    "mu" -- float, gravity parameter.
        """

        self.f = params["f"]

        if is_tle:
            self.type = "tle"
            self.satellite = pk.planet.tle(
                params["tle_line1"], params["tle_line2"])
        else:
            self.type = "keplerian"
            self.satellite = pk.planet.keplerian(params["epoch"],
                                                 pk.ic2eq(r=params["pos"],
                                                          v=params["v"],
                                                          mu=params["mu"]))
        self.satellite.name = name

    def act(self, action):
        """ Make manoeuvre for the object. """
        return

    def position(self, epoch):
        """ Provide SpaceObject position:
            (x, y, z) and (Vx, Vy, Vz), at given epoch.
        """
        pos, v = self.satellite.eph(epoch)
        return pos, v

    def get_name(self):
        return self.satellite.name
