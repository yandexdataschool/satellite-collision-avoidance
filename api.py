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
            devris - [SpaceObject], list of other space objects.
        """
        self.protected = protected
        self.debris = debris
        self.state = EnvState()

    def get_state(self, params):
        """ Provides environment state.
            params -- dict(), which parameters to return in state.
        """
        objects = [self.protected] + self.debris
        return self.state.get_state(params, objects)

    def get_reward(self):
        """"""
        return 0

    def act(self, action):
        """ Change velocity for protected object.
        Args:
            action -- np.array([dVx, dVy, dVz, pk.epoch]), vector of deltas.
        """
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
        return

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
